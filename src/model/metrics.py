import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from typing import Dict, List

class RNAMetrics:
    @staticmethod
    def compute_all_metrics(predictions: Dict[str, torch.Tensor], 
                          targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute all metrics for RNA predictions"""
        metrics = {}
        
        for name in predictions:
            pred = predictions[name].detach().cpu().numpy()
            target = targets[name].detach().cpu().numpy()
            
            metrics.update({
                f"{name}_mse": np.mean((pred - target) ** 2),
                f"{name}_correlation": np.corrcoef(pred.flatten(), target.flatten())[0,1],
                f"{name}_auroc": RNAMetrics.compute_auroc(pred, target),
                f"{name}_auprc": RNAMetrics.compute_auprc(pred, target)
            })
        
        return metrics
    
    @staticmethod
    def compute_auroc(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Area Under ROC Curve"""
        try:
            return roc_auc_score(targets.flatten() > 0.5, predictions.flatten())
        except:
            return 0.0
    
    @staticmethod
    def compute_auprc(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Area Under Precision-Recall Curve"""
        try:
            return average_precision_score(targets.flatten() > 0.5, predictions.flatten())
        except:
            return 0.0
