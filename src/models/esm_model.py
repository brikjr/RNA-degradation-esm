import torch
import torch.nn as nn
import esm
from typing import Dict, Tuple

class ESMRNAPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Load ESM model
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(config.model.esm_model)
        self.esm_model.eval()  # Freeze ESM weights
        
        # Feature processing
        self.feature_network = nn.Sequential(
            nn.Linear(self.esm_model.dim_output + config.model.hidden_dim, 
                     config.model.hidden_dim),
            nn.LayerNorm(config.model.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout)
        )
        
        # RNA-specific layers
        self.rna_encoder = nn.LSTM(
            config.model.hidden_dim,
            config.model.hidden_dim,
            num_layers=config.model.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.model.dropout if config.model.num_layers > 1 else 0
        )
        
        # Prediction heads
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(config.model.hidden_dim * 2, config.model.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.model.dropout),
                nn.Linear(config.model.hidden_dim, 68)
            )
            for name in ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 
                        'deg_Mg_50C', 'deg_50C']
        })
        
    def forward(self, sequences, features) -> Dict[str, torch.Tensor]:
        # Get ESM embeddings
        with torch.no_grad():
            batch_tokens = self.alphabet.batch_encode(sequences)
            esm_results = self.esm_model(batch_tokens, repr_layers=[33])
            embeddings = esm_results["representations"][33]
        
        # Process features
        combined = torch.cat([embeddings, features], dim=-1)
        processed = self.feature_network(combined)
        
        # RNA-specific encoding
        rna_features, _ = self.rna_encoder(processed)
        
        # Generate predictions
        predictions = {}
        for name, head in self.heads.items():
            predictions[name] = head(rna_features)
            
        return predictions
