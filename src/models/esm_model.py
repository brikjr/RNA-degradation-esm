import torch
import torch.nn as nn
import esm
from typing import Dict, List

class ESMRNAPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Load ESM model and move to correct device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(config.model.esm_model)
        self.esm_model = self.esm_model.to(self.device)
        self.esm_model.eval()  # Freeze ESM weights
        
        # Store dimensions
        self.embedding_dim = config.model.embedding_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_length = 68  # Fixed output length to match targets
        
        # Feature processing
        self.feature_projection = nn.Linear(1286, self.hidden_dim).to(self.device)
        self.embedding_projection = nn.Linear(self.embedding_dim, self.hidden_dim).to(self.device)
        
        self.feature_combiner = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout)
        ).to(self.device)
        
        # Prediction heads - directly output the correct sequence length
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.model.dropout),
                nn.Linear(self.hidden_dim, self.output_length)  # Direct output to target length
            ).to(self.device)
            for name in ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 
                        'deg_Mg_50C', 'deg_50C']
        })
        
        # Store the batch converter
        self.batch_converter = self.alphabet.get_batch_converter()
    
    def forward(self, sequences: List[str], features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Ensure features are on the correct device
        features = features.to(self.device)  # Shape: [batch_size, feature_dim]
        
        # Prepare sequences for ESM model
        batch_labels, batch_strs, batch_tokens = self.batch_converter(
            [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        )
        batch_tokens = batch_tokens.to(self.device)
        
        # Get ESM embeddings
        with torch.no_grad():
            esm_results = self.esm_model(batch_tokens, repr_layers=[33])
            embeddings = esm_results["representations"][33]  # Shape: [batch_size, seq_len, embedding_dim]
            embeddings = embeddings.mean(dim=1)  # Shape: [batch_size, embedding_dim]
        
        # Project features and embeddings to same dimension
        projected_features = self.feature_projection(features)  # Shape: [batch_size, hidden_dim]
        projected_embeddings = self.embedding_projection(embeddings)  # Shape: [batch_size, hidden_dim]
        
        # Combine projected features and embeddings
        combined = torch.cat([projected_features, projected_embeddings], dim=-1)  # Shape: [batch_size, hidden_dim * 2]
        processed = self.feature_combiner(combined)  # Shape: [batch_size, hidden_dim]
        
        # Generate predictions
        predictions = {}
        for name, head in self.heads.items():
            pred = head(processed)  # Shape: [batch_size, 68]
            predictions[name] = pred
        
        return predictions
