import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import logging
from pathlib import Path
from tqdm import tqdm, trange
from ..models.metrics import RNAMetrics
from ..visualization.visualizer import TrainingVisualizer

class RNATrainer:
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        self.model = self.model.to(self.device)
        
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
        
        # Create save directory if it doesn't exist
        Path(config.logging.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.training.epochs}")
        for batch_idx, batch in enumerate(pbar):
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
            
            # Update progress bar
            pbar.set_postfix({'batch_loss': f'{loss.item():.4f}', 
                            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
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
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_loss
            }
            
            # Save regular checkpoint
            checkpoint_path = Path(self.config.logging.save_dir) / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model if this is the best so far
            if is_best:
                best_model_path = Path(self.config.logging.save_dir) / 'best_model.pt'
                torch.save(checkpoint, best_model_path)
                print(f"New best model saved! Val Loss: {val_loss:.4f}")
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    # def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
    #     """Save model checkpoint with robust error handling"""
    #     try:
    #         # Create separate state dicts
    #         model_state = self.model.state_dict()
    #         optimizer_state = self.optimizer.state_dict()
    #         scheduler_state = self.scheduler.state_dict()
            
    #         # Save model state separately
    #         model_path = Path(self.config.logging.save_dir) / f'model_epoch_{epoch}.pt'
    #         torch.save(model_state, model_path)
            
    #         # Save optimizer and scheduler states
    #         training_state = {
    #             'epoch': epoch,
    #             'optimizer_state_dict': optimizer_state,
    #             'scheduler_state_dict': scheduler_state,
    #             'val_loss': val_loss
    #         }
    #         state_path = Path(self.config.logging.save_dir) / f'training_state_epoch_{epoch}.pt'
    #         torch.save(training_state, state_path)
            
    #         # Save best model if this is the best so far
    #         if is_best:
    #             best_model_path = Path(self.config.logging.save_dir) / 'best_model.pt'
    #             torch.save(model_state, best_model_path)
    #             best_state_path = Path(self.config.logging.save_dir) / 'best_training_state.pt'
    #             torch.save(training_state, best_state_path)
    #             print(f"New best model saved! Val Loss: {val_loss:.4f}")
                
    #         # Clean up old checkpoints to save space
    #         self._cleanup_old_checkpoints(epoch)
                
    #     except Exception as e:
    #         print(f"Warning: Failed to save checkpoint: {str(e)}")
    #         print("Attempting to save with different protocol...")
    #         try:
    #             # Try saving with older protocol version
    #             torch.save(model_state, model_path, _use_new_zipfile_serialization=False)
    #             torch.save(training_state, state_path, _use_new_zipfile_serialization=False)
    #             if is_best:
    #                 torch.save(model_state, best_model_path, _use_new_zipfile_serialization=False)
    #                 torch.save(training_state, best_state_path, _use_new_zipfile_serialization=False)
    #             print("Successfully saved using older protocol")
    #         except Exception as e2:
    #             print(f"Warning: Also failed with older protocol: {str(e2)}")
    #             print("Continuing training without saving checkpoint")

    # def _cleanup_old_checkpoints(self, current_epoch: int, keep_last_n: int = 3):
    #     """Clean up old checkpoints, keeping only the last n"""
    #     save_dir = Path(self.config.logging.save_dir)
        
    #     # Find all checkpoint files
    #     model_files = list(save_dir.glob('model_epoch_*.pt'))
    #     state_files = list(save_dir.glob('training_state_epoch_*.pt'))
        
    #     # Sort by epoch number
    #     def get_epoch_num(filepath):
    #         return int(str(filepath).split('_')[-1].split('.')[0])
        
    #     model_files.sort(key=get_epoch_num)
    #     state_files.sort(key=get_epoch_num)
        
    #     # Remove old files, keeping last n
    #     if len(model_files) > keep_last_n:
    #         for f in model_files[:-keep_last_n]:
    #             try:
    #                 f.unlink()
    #             except Exception as e:
    #                 print(f"Warning: Failed to delete old checkpoint {f}: {e}")
                    
    #     if len(state_files) > keep_last_n:
    #         for f in state_files[:-keep_last_n]:
    #             try:
    #                 f.unlink()
    #             except Exception as e:
    #                 print(f"Warning: Failed to delete old state file {f}: {e}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        best_val_loss = float('inf')
        
        print("Starting training...")
        for epoch in range(self.config.training.epochs):
            # Train epoch
            train_loss = self.train_epoch(train_loader, epoch)
            
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
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")
            
            # Save training state for recovery
            try:
                state_path = Path(self.config.logging.save_dir) / 'training_state.pt'
                torch.save({
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict()
                }, state_path)
            except Exception as e:
                print(f"Warning: Failed to save training state: {e}")
    
    @staticmethod
    def aggregate_predictions(predictions_list: list) -> Dict[str, torch.Tensor]:
        """Aggregate predictions from multiple batches"""
        aggregated = {}
        for key in predictions_list[0].keys():
            aggregated[key] = torch.cat([p[key] for p in predictions_list])
        return aggregated
