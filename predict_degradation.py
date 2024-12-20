from Bio import SeqIO
import esm
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict
import pandas as pd
import argparse

class EfficientRNAPredictor(nn.Module):
    def __init__(self, embedding_dim: int = 1280, hidden_dim: int = 256):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Main prediction network
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(self.device)
        
        # Separate heads for different degradation conditions with identical architecture
        self.heads = nn.ModuleDict({
            'reactivity': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),  # 0
                nn.ReLU(),                         # 1
                nn.Dropout(0.2),                   # 2
                nn.Linear(hidden_dim, 68)          # 3
            ),
            'deg_Mg_pH10': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),  # 0
                nn.ReLU(),                         # 1
                nn.Dropout(0.2),                   # 2
                nn.Linear(hidden_dim, 68)          # 3
            ),
            'deg_pH10': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),  # 0
                nn.ReLU(),                         # 1
                nn.Dropout(0.2),                   # 2
                nn.Linear(hidden_dim, 68)          # 3
            ),
            'deg_Mg_50C': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),  # 0
                nn.ReLU(),                         # 1
                nn.Dropout(0.2),                   # 2
                nn.Linear(hidden_dim, 68)          # 3
            ),
            'deg_50C': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),  # 0
                nn.ReLU(),                         # 1
                nn.Dropout(0.2),                   # 2
                nn.Linear(hidden_dim, 68)          # 3
            )
        }).to(self.device)
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.predictor(embeddings)
        return {name: head(features) for name, head in self.heads.items()}

class RNADegradationPredictor:
    def __init__(self, model_path: str):
        """Initialize predictor with model only"""
        # Load ESM model
        print("Loading ESM model...")
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.esm_model = self.esm_model.to(self.device)
        self.esm_model.eval()
        
        # Load prediction model
        self.model = EfficientRNAPredictor()
        try:
            print(f"Loading prediction model from {model_path}")
            if torch.cuda.is_available():
                checkpoint = torch.load(model_path, map_location='cuda', weights_only=True)
            else:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            head_state_dict = {k: v for k, v in state_dict.items() if 'heads.' in k}
            self.model.load_state_dict(head_state_dict, strict=False)
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
        self.model.eval()
    
    def generate_embedding(self, sequence: str) -> torch.Tensor:
        """Generate ESM embedding for a sequence"""
        batch_converter = self.alphabet.get_batch_converter()
        with torch.no_grad():
            _, _, tokens = batch_converter([("", sequence)])
            tokens = tokens.to(self.device)
            results = self.esm_model(tokens, repr_layers=[33])
            embeddings = results["representations"][33]
        return embeddings[0]  # Return sequence embedding
    
    def process_predictions(self, predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Process predictions into interpretable metrics"""
        processed = {}
        for condition, values in predictions.items():
            processed[condition] = {
                'mean_degradation': float(np.mean(values)),
                'max_degradation': float(np.max(values)),
                'min_degradation': float(np.min(values)),
                'std_degradation': float(np.std(values)),
                'position_wise': values.tolist()
            }
        return processed
    
    def predict(self, sequence: str) -> Dict[str, np.ndarray]:
        """Predict degradation rates for a new sequence"""
        # Generate embedding
        print(f"\nGenerating embedding for sequence: {sequence[:50]}...")
        embedding = self.generate_embedding(sequence)
        
        # Average pooling over sequence length
        embedding = embedding.mean(dim=0, keepdim=True)  # Shape: [1, embed_dim]
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(embedding)
            return {k: v.cpu().numpy()[0] for k, v in predictions.items()}

def process_fasta(fasta_file: str, model_path: str, output_dir: str) -> None:
    """Process all sequences in a FASTA file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize predictor
    predictor = RNADegradationPredictor(model_path)
    
    # Process each sequence
    results = []
    print("\nProcessing sequences from FASTA file...")
    for record in SeqIO.parse(fasta_file, "fasta"):
        print(f"\nProcessing sequence: {record.id}")
        sequence = str(record.seq).upper()
        try:
            predictions = predictor.predict(sequence)
            processed = predictor.process_predictions(predictions)
            
            row = {
                'sequence_id': record.id,
                'sequence': sequence,
                'length': len(sequence)
            }
            
            for condition, metrics in processed.items():
                row.update({
                    f'{condition}_mean': metrics['mean_degradation'],
                    f'{condition}_max': metrics['max_degradation'],
                    f'{condition}_min': metrics['min_degradation'],
                    f'{condition}_std': metrics['std_degradation']
                })
            
            results.append(row)
            print(f"Successfully processed sequence {record.id}")
            
        except Exception as e:
            print(f"Error processing sequence {record.id}: {str(e)}")
    
    if results:
        df = pd.DataFrame(results)
        output_file = output_dir / f"{Path(fasta_file).stem}_predictions.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\nResults saved to: {output_file}")
        print("\nPrediction Statistics:")
        print("-" * 50)
        
        for condition in ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']:
            mean_col = f'{condition}_mean'
            print(f"\n{condition}:")
            print(f"Average degradation: {df[mean_col].mean():.3f} ± {df[mean_col].std():.3f}")
            print(f"Range: {df[mean_col].min():.3f} to {df[mean_col].max():.3f}")

def main():
    parser = argparse.ArgumentParser(description='Predict RNA degradation from FASTA file')
    parser.add_argument('--fasta', type=str, required=True, help='Input FASTA file')
    parser.add_argument('--model', type=str, default='runs/best_model.pt', help='Path to model weights')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    process_fasta(args.fasta, args.model, args.output)

if __name__ == "__main__":
    main()
