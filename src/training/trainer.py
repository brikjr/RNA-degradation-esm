import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import logging
from ..models.metrics import RNAMetrics
from ..visualization.visualizer import TrainingVisualizer

class RNATrainer:
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience
        )
        
        # Setup visualizer
        self.visualizer = TrainingVisualizer(config.logging.tensorboard_dir)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Forward pass
            sequences = batch['sequences']
            features = batch['features'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            predictions = self.model(sequences, features)
            
            # Calculate loss
            loss = 0
            for name in predictions:
                loss += nn.MSELoss()(predictions[name], targets[name])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.gradient_clip
            )
            
            self.optimizer.step()
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % self.config.logging.log_interval == 0:
                self.logger.info(f'Train Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequences']
                features = batch['features'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                
                predictions = self.model(sequences, features)
                
                # Calculate loss
                loss = 0
                for name in predictions:
                    loss += nn.MSELoss()(predictions[name], targets[name])
                
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Calculate metrics
        metrics = RNAMetrics.compute_all_metrics(
            self.aggregate_predictions(all_predictions),
            self.aggregate_predictions(all_targets)
        )
        
        return total_loss / len(val_loader), metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.epochs):
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log metrics
            self.visualizer.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                **metrics
            }, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 
                         f"{self.config.logging.save_dir}/best_model.pt")
            
            self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    @staticmethod
    def aggregate_predictions(predictions_list: list) -> Dict[str, torch.Tensor]:
        """Aggregate predictions from multiple batches"""
        aggregated = {}
        for key in predictions_list[0].keys():
            aggregated[key] = torch.cat([p[key] for p in predictions_list])
        return aggregated
