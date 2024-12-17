import argparse
from pathlib import Path
import sys
from os.path import abspath, dirname
import pandas as pd

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
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = RNAPreprocessor(config)
    
    print("Loading training data...")
    train_data = pd.read_json(Path(args.input_dir) / 'train.json', lines=True)
    print(f"Loaded {len(train_data)} training samples")
    
    print("Processing training data...")
    sequences, features, targets = preprocessor.process_data(train_data)
    
    print("Saving processed data...")
    preprocessor.save_processed_data(sequences, features, targets, 'train')
    
    print("Loading test data...")
    test_data = pd.read_json(Path(args.input_dir) / 'test.json', lines=True)
    print(f"Loaded {len(test_data)} test samples")
    
    print("Processing test data...")
    sequences, features, targets = preprocessor.process_data(test_data)
    
    print("Saving processed data...")
    preprocessor.save_processed_data(sequences, features, targets, 'test')
    
    print("Data processing completed!")

if __name__ == "__main__":
    main()