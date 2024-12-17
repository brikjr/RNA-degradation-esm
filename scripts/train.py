import argparse
import yaml
from pathlib import Path
import sys
from os.path import abspath, dirname

# Add project root to Python path
project_root = dirname(dirname(abspath(__file__)))
sys.path.append(project_root)

from src.training.config import Config
from src.data.loader import RNADataModule
from src.models.esm_model import ESMRNAPredictor
from src.training.trainer import RNATrainer

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'few_shot'])
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)
    
    # Update output directory in config
    config.logging.save_dir = args.output_dir
    config.logging.tensorboard_dir = str(Path(args.output_dir) / 'tensorboard')
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    data_module = RNADataModule(config)
    
    # Create dataloaders
    train_loader = data_module.create_dataloader('train', config.training.batch_size)
    val_loader = data_module.create_dataloader('val', config.training.batch_size)
    
    # Initialize model
    model = ESMRNAPredictor(config)
    
    # Initialize trainer
    trainer = RNATrainer(model, config)
    
    if args.mode == 'train':
        # Regular training
        trainer.train(train_loader, val_loader)
    else:
        # Few-shot training
        trainer.train_few_shot(data_module)

if __name__ == "__main__":
    main()
