import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple

class RNADataset(Dataset):
    def __init__(self, sequences: List[str], features: np.ndarray, 
                 targets: Dict[str, np.ndarray]):
        self.sequences = sequences
        self.features = torch.FloatTensor(features)
        self.targets = {k: torch.FloatTensor(v) for k, v in targets.items()}
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': self.sequences[idx],
            'features': self.features[idx],
            'targets': {k: v[idx] for k, v in self.targets.items()}
        }

class RNADataModule:
    def __init__(self, config):
        self.config = config
        self.processed_dir = Path(config.data.processed_dir)
    
    def load_processed_data(self, split: str) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
        """Load processed data files"""
        print(f"Loading {split} data...")
        
        # Load sequences
        with open(self.processed_dir / f"{split}_sequences.txt") as f:
            sequences = [line.strip() for line in tqdm(f, desc="Loading sequences")]
        
        # Load features
        print("Loading features...")
        features = np.load(self.processed_dir / f"{split}_features.npy")
        
        # Load targets
        print("Loading targets...")
        targets = {}
        for target_name in tqdm(['reactivity', 'deg_Mg_pH10', 'deg_pH10', 
                               'deg_Mg_50C', 'deg_50C'], desc="Loading targets"):
            target_path = self.processed_dir / f"{split}_{target_name}.npy"
            if target_path.exists():
                target_array = np.load(target_path, allow_pickle=True)
                targets[target_name] = target_array.astype(np.float32)
        
        return sequences, features.astype(np.float32), targets
    
    def create_dataloader(self, split: str, batch_size: int, 
                         shuffle: bool = True) -> DataLoader:
        """Create DataLoader for specified split"""
        sequences, features, targets = self.load_processed_data(split)
        dataset = RNADataset(sequences, features, targets)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
    
    def get_few_shot_batch(self, n_support: int, n_query: int) -> Tuple[Dict, Dict]:
        """Get few-shot support and query sets"""
        # Load training data
        sequences, features, targets = self.load_processed_data('train')
        
        # Randomly select indices for support and query sets
        total_indices = np.random.permutation(len(sequences))
        support_indices = total_indices[:n_support]
        query_indices = total_indices[n_support:n_support + n_query]
        
        # Create support set
        support_set = {
            'sequences': [sequences[i] for i in support_indices],
            'features': features[support_indices],
            'targets': {k: v[support_indices] for k, v in targets.items()}
        }
        
        # Create query set
        query_set = {
            'sequences': [sequences[i] for i in query_indices],
            'features': features[query_indices],
            'targets': {k: v[query_indices] for k, v in targets.items()}
        }
        
        return support_set, query_set
