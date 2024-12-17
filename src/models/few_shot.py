import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class FewShotAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.n_support = config.few_shot.n_support
        self.n_query = config.few_shot.n_query
        self.adaptation_steps = config.few_shot.adaptation_steps
        
        self.prototype_network = nn.Sequential(
            nn.Linear(config.model.hidden_dim, config.few_shot.prototype_dim),
            nn.ReLU(),
            nn.Linear(config.few_shot.prototype_dim, config.few_shot.prototype_dim)
        ) if config.model.use_prototypes else None
        
    def adapt_to_support(self, support_features: torch.Tensor, 
                        support_targets: Dict[str, torch.Tensor]) -> Dict:
        """Adapt model based on support set"""
        if self.prototype_network is not None:
            # Compute prototypes
            prototypes = self.prototype_network(support_features)
            return {'prototypes': prototypes, 'targets': support_targets}
        return {'features': support_features, 'targets': support_targets}
    
    def compute_similarities(self, query_features: torch.Tensor, 
                           support_dict: Dict) -> torch.Tensor:
        """Compute similarity between query and support features"""
        if 'prototypes' in support_dict:
            query_protos = self.prototype_network(query_features)
            return F.cosine_similarity(
                query_protos.unsqueeze(1),
                support_dict['prototypes'].unsqueeze(0),
                dim=-1
            )
        return F.cosine_similarity(
            query_features.unsqueeze(1),
            support_dict['features'].unsqueeze(0),
            dim=-1
        )
