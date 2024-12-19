from Bio import SeqIO
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
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
        
        # Separate heads for different degradation conditions
        self.heads = nn.ModuleDict({
            'reactivity': nn.Linear(hidden_dim, 68),
            'deg_Mg_pH10': nn.Linear(hidden_dim, 68),
            'deg_pH10': nn.Linear(hidden_dim, 68),
            'deg_Mg_50C': nn.Linear(hidden_dim, 68),
            'deg_50C': nn.Linear(hidden_dim, 68)
        }).to(self.device)
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.predictor(embeddings)
        return {name: head(features) for name, head in self.heads.items()}

class RNADegradationPredictor:
    def __init__(self, model_path: str, embeddings_dir: str):
        """Initialize predictor with pre-trained model and embeddings"""
        # Load the model
        self.model = EfficientRNAPredictor()
        try:
            # Load model with weights_only=True for security
            if torch.cuda.is_available():
                state_dict = torch.load(model_path, map_location='cuda', weights_only=True)
            elif torch.backends.mps.is_available():
                state_dict = torch.load(model_path, map_location='mps', weights_only=True)
            else:
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model from {model_path}")
            print(f"Error details: {str(e)}")
            print("\nPlease ensure:")
            print("1. The model file exists and is not corrupted")
            print("2. The model file is in the correct location (default: runs/*/best_model.pt)")
            print("3. You have trained the model first")
            raise

        self.model.eval()
        
        # Load embeddings
        self.embeddings_path = Path(embeddings_dir)
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
            
        try:
            self.sequence_to_idx = self._load_sequence_mapping()
            self.embeddings = np.load(self.embeddings_path / "esm_embeddings.npy")
        except Exception as e:
            print(f"Error loading embeddings from {embeddings_dir}")
            print(f"Error details: {str(e)}")
            print("\nPlease ensure:")
            print("1. The embeddings directory exists")
            print("2. It contains esm_embeddings.npy and sequence_ids.txt")
            print("3. You have generated embeddings first using the ESM model")
            raise
        
    def _load_sequence_mapping(self) -> Dict[str, int]:
        """Load mapping of sequences to embedding indices"""
        mapping = {}
        with open(self.embeddings_path / "sequence_ids.txt") as f:
            for line in f:
                idx, seq = line.strip().split('\t')
                mapping[seq] = int(idx)
        return mapping
    
    def predict(self, sequence: str) -> Dict[str, np.ndarray]:
        """Predict degradation rates for a sequence"""
        # Find sequence in mapping
        if sequence not in self.sequence_to_idx:
            raise ValueError("Sequence not found in pre-computed embeddings")
            
        # Get embedding
        idx = self.sequence_to_idx[sequence]
        embedding = torch.tensor(self.embeddings[idx]).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(embedding)
            return {k: v.numpy()[0] for k, v in predictions.items()}
    
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

def process_fasta(fasta_file: str, model_path: str, embeddings_dir: str, output_dir: str) -> None:
    """Process all sequences in a FASTA file and save results as CSV"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize predictor
    predictor = RNADegradationPredictor(model_path, embeddings_dir)
    
    # Process each sequence
    results = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        try:
            predictions = predictor.predict(sequence)
            processed = predictor.process_predictions(predictions)
            
            # Create row for each sequence
            row = {
                'sequence_id': record.id,
                'sequence': sequence,
                'length': len(sequence)
            }
            
            # Add predictions for each condition
            for condition, metrics in processed.items():
                row.update({
                    f'{condition}_mean': metrics['mean_degradation'],
                    f'{condition}_max': metrics['max_degradation'],
                    f'{condition}_min': metrics['min_degradation'],
                    f'{condition}_std': metrics['std_degradation']
                })
            
            results.append(row)
            print(f"Processed sequence: {record.id}")
            
        except Exception as e:
            print(f"Error processing sequence {record.id}: {str(e)}")
    
    if results:
        # Save main results
        df = pd.DataFrame(results)
        output_file = output_dir / f"{Path(fasta_file).stem}_predictions.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\nResults saved to: {output_file}")
        print("\nPrediction Statistics:")
        print("-" * 50)
        
        # Print summary statistics
        for condition in ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']:
            mean_col = f'{condition}_mean'
            print(f"\n{condition}:")
            print(f"Average degradation: {df[mean_col].mean():.3f} ± {df[mean_col].std():.3f}")
            print(f"Range: {df[mean_col].min():.3f} to {df[mean_col].max():.3f}")

def main():
    parser = argparse.ArgumentParser(description='Predict RNA degradation from FASTA file')
    parser.add_argument('--fasta', type=str, required=True, help='Input FASTA file')
    parser.add_argument('--model', type=str, default='runs/experiment_1/best_model.pt', help='Path to model weights')
    parser.add_argument('--embeddings', type=str, default='data/embeddings', help='Directory containing embeddings')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    process_fasta(args.fasta, args.model, args.embeddings, args.output)

if __name__ == "__main__":
    main()
