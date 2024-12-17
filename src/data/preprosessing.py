import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import esm
from collections import Counter

class RNAPreprocessor:
    def __init__(self, config):
        self.config = config
        
        # Load ESM model for embeddings
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            config.model.esm_model
        )
        self.esm_model.eval()
        
    def preprocess_sequence(self, sequence: str) -> Dict[str, float]:
        """Extract sequence features"""
        counts = Counter(sequence)
        length = len(sequence)
        
        return {
            'gc_content': (counts['G'] + counts['C']) / length,
            'au_content': (counts['A'] + counts['U']) / length,
            'sequence_length': length,
            'purine_content': (counts['A'] + counts['G']) / length
        }
    
    def preprocess_structure(self, structure: str) -> Dict[str, float]:
        """Extract structure features"""
        length = len(structure)
        paired = structure.count('(') + structure.count(')')
        unpaired = structure.count('.')
        
        return {
            'paired_ratio': paired / (2 * length),
            'unpaired_ratio': unpaired / length
        }
    
    def generate_esm_embeddings(self, sequence: str) -> np.ndarray:
        """Generate ESM embeddings for sequence"""
        with torch.no_grad():
            batch_tokens = self.alphabet.batch_encode([sequence])
            results = self.esm_model(batch_tokens, repr_layers=[33])
            embeddings = results["representations"][33].numpy()
        return embeddings[0]
    
    def process_data(self, df: pd.DataFrame) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
        """Process full dataset"""
        # Filter by signal-to-noise
        df = df[df['signal_to_noise'] >= self.config.data.sn_threshold].copy()
        
        sequences = []
        features = []
        targets = {
            'reactivity': [], 'deg_Mg_pH10': [], 'deg_pH10': [], 
            'deg_Mg_50C': [], 'deg_50C': []
        }
        
        for _, row in df.iterrows():
            # Process sequence
            seq_features = self.preprocess_sequence(row['sequence'])
            struct_features = self.preprocess_structure(row['structure'])
            
            # Get ESM embeddings
            embeddings = self.generate_esm_embeddings(row['sequence'])
            
            # Combine features
            combined_features = np.concatenate([
                embeddings,
                np.array(list(seq_features.values())),
                np.array(list(struct_features.values()))
            ])
            
            sequences.append(row['sequence'])
            features.append(combined_features)
            
            # Process targets
            for name in targets:
                targets[name].append(row[name][0])  # Take first value
        
        return (
            sequences,
            np.array(features),
            {k: np.array(v) for k, v in targets.items()}
        )
    
    def save_processed_data(self, sequences: List[str], features: np.ndarray, 
                          targets: Dict[str, np.ndarray], split: str):
        """Save processed data"""
        output_dir = Path(self.config.data.processed_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sequences
        with open(output_dir / f"{split}_sequences.txt", 'w') as f:
            for seq in sequences:
                f.write(f"{seq}\n")
        
        # Save features
        np.save(output_dir / f"{split}_features.npy", features)
        
        # Save targets
        for name, values in targets.items():
            np.save(output_dir / f"{split}_{name}.npy", values)
