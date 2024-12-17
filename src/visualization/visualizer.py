import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd

class MetricsVisualizer:
    def __init__(self, log_dir: str):
        """Initialize TensorBoard writer and metrics tracking"""
        self.writer = SummaryWriter(log_dir)
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'f1_scores': defaultdict(list),
            'precision': defaultdict(list),
            'recall': defaultdict(list)
        }
    
    def log_metrics(self, metrics: Dict, step: int):
        """Log metrics to TensorBoard"""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
    
    def log_predictions(self, predictions: torch.Tensor, targets: torch.Tensor, 
                       threshold: float = 0.5):
        """Calculate and log detailed prediction metrics"""
        pred_binary = (predictions > threshold).float()
        target_binary = (targets > threshold).float()
        
        # Calculate metrics
        true_pos = (pred_binary * target_binary).sum().item()
        false_pos = (pred_binary * (1 - target_binary)).sum().item()
        false_neg = ((1 - pred_binary) * target_binary).sum().item()
        true_neg = ((1 - pred_binary) * (1 - target_binary)).sum().item()
        
        # Calculate rates
        precision = true_pos / (true_pos + false_pos + 1e-10)
        recall = true_pos / (true_pos + false_neg + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': false_pos / (false_pos + true_neg + 1e-10),
            'false_negative_rate': false_neg / (false_neg + true_pos + 1e-10)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, predictions: torch.Tensor, targets: torch.Tensor, 
                            threshold: float = 0.5):
        """Plot confusion matrix"""
        pred_binary = (predictions > threshold).float()
        target_binary = (targets > threshold).float()
        
        conf_matrix = torch.zeros(2, 2)
        for p, t in zip(pred_binary.flatten(), target_binary.flatten()):
            conf_matrix[int(p), int(t)] += 1
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix.numpy(), annot=True, fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save to TensorBoard
        self.writer.add_figure('confusion_matrix', plt.gcf())
        plt.close()
    
    def plot_roc_curve(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(targets.numpy().flatten(), 
                               predictions.numpy().flatten())
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        self.writer.add_figure('roc_curve', plt.gcf())
        plt.close()
    
    def plot_learning_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_history['train_loss'], label='Train')
        plt.plot(self.metrics_history['val_loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        
        self.writer.add_figure('learning_curves', plt.gcf())
        plt.close()
    
    def plot_attention_weights(self, attention_weights: torch.Tensor, sequences: List[str]):
        """Plot attention weights for sequences"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_weights.cpu().numpy(), xticklabels=list(sequences[0]), 
                   yticklabels=list(sequences[0]))
        plt.title('Attention Weights')
        
        self.writer.add_figure('attention_weights', plt.gcf())
        plt.close()
    
    def save_metrics_summary(self, output_path: str):
        """Save metrics summary to CSV"""
        df = pd.DataFrame({
            'Epoch': range(len(self.metrics_history['train_loss'])),
            'Train_Loss': self.metrics_history['train_loss'],
            'Val_Loss': self.metrics_history['val_loss'],
            **{f'F1_{k}': v for k, v in self.metrics_history['f1_scores'].items()},
            **{f'Precision_{k}': v for k, v in self.metrics_history['precision'].items()},
            **{f'Recall_{k}': v for k, v in self.metrics_history['recall'].items()}
        })
        df.to_csv(output_path, index=False)

class TrainingVisualizer:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.log_dir = Path(log_dir)
        
    def plot_attention_weights(self, weights: torch.Tensor, sequence: str):
        """Plot attention weight heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights.cpu().numpy(), 
                   xticklabels=list(sequence), 
                   yticklabels=list(sequence))
        plt.title('Attention Weights')
        self.writer.add_figure('attention_weights', plt.gcf())
        plt.close()
    
    def plot_feature_importance(self, model, features: torch.Tensor, 
                              feature_names: List[str]):
        """Plot feature importance based on gradient analysis"""
        model.eval()
        features.requires_grad_(True)
        
        outputs = model(features)
        importance = torch.zeros(features.size(1))
        
        for output in outputs.values():
            output.sum().backward(retain_graph=True)
            importance += features.grad.abs().mean(0)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=feature_names, y=importance.cpu().numpy())
        plt.xticks(rotation=45)
        plt.title('Feature Importance')
        self.writer.add_figure('feature_importance', plt.gcf())
        plt.close()
    
    def plot_prediction_distribution(self, predictions: Dict[str, torch.Tensor], 
                                   targets: Dict[str, torch.Tensor]):
        """Plot distribution of predictions vs targets"""
        plt.figure(figsize=(15, 5))
        for i, (name, pred) in enumerate(predictions.items(), 1):
            plt.subplot(1, len(predictions), i)
            sns.kdeplot(pred.cpu().numpy().flatten(), label='Predicted')
            sns.kdeplot(targets[name].cpu().numpy().flatten(), label='Actual')
            plt.title(f'{name} Distribution')
            plt.legend()
        plt.tight_layout()
        self.writer.add_figure('prediction_distribution', plt.gcf())
        plt.close()
    
    def plot_learning_curves(self, train_losses: List[float], 
                           val_losses: List[float]):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        self.writer.add_figure('learning_curves', plt.gcf())
        plt.close()
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to TensorBoard"""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
    
    def save_predictions(self, predictions: Dict[str, torch.Tensor], 
                        output_path: str):
        """Save predictions to file"""
        output_path = self.log_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        pred_dict = {k: v.cpu().numpy() for k, v in predictions.items()}
        np.save(output_path, pred_dict)