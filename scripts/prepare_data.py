import argparse
from pathlib import Path
import sys
from os.path import abspath, dirname

# Add project root to Python path
project_root = dirname(dirname(abspath(__file__)))
sys.path.append(project_root)

from src.training.config import Config
from src.data.preprosessing import RNAPreprocessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)
    
    # Update paths in config
    config.data.processed_dir = args.output_dir
    
    # Initialize preprocessor
    preprocessor = RNAPreprocessor(config)
    
    # Process training data
    train_data = preprocessor.load_data(Path(args.input_dir) / 'train.csv')
    sequences, features, targets = preprocessor.process_data(train_data)
    preprocessor.save_processed_data(sequences, features, targets, 'train')
    
    # Process validation data
    val_data = preprocessor.load_data(Path(args.input_dir) / 'test.csv')
    sequences, features, targets = preprocessor.process_data(val_data)
    preprocessor.save_processed_data(sequences, features, targets, 'val')

if __name__ == "__main__":
    main()
