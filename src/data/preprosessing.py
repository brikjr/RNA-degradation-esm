import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import esm
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class RNAPreprocessor:
    def __init__(self, config):
        self.config = config
        
        # Load ESM model for embeddings
        print("Loading ESM model...")
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            config.model.esm_model
        )
        self.esm_model.eval()
        print("ESM model loaded successfully")

    def check_files_exist(self, split: str) -> bool:
        """Check if processed files already exist for a given split"""
        output_dir = Path(self.config.data.processed_dir)
        
        # List of files that should exist
        required_files = [
            output_dir / f"{split}_sequences.txt",
            output_dir / f"{split}_features.npy"
        ]
        
        # Check target files
        for target in ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']:
            required_files.append(output_dir / f"{split}_{target}.npy")
        
        # For training data, also check validation files
        if split == 'train':
            required_files.extend([
                output_dir / "val_sequences.txt",
                output_dir / "val_features.npy"
            ])
            for target in ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']:
                required_files.append(output_dir / f"val_{target}.npy")
        
        # Check if all files exist
        exists = all(f.exists() for f in required_files)
        if exists:
            print(f"All files exist for {split} split")
        else:
            print(f"Some files missing for {split} split, preparing files:")
        return exists
    
    def preprocess_sequence(self, sequence: str) -> Dict[str, float]:
        """Extract sequence features"""
        # Convert T to U if present (DNA to RNA)
        sequence = sequence.replace('T', 'U')
        counts = Counter(sequence)
        length = len(sequence)
        
        return {
            'gc_content': (counts['G'] + counts['C']) / length if length > 0 else 0,
            'au_content': (counts['A'] + counts['U']) / length if length > 0 else 0,
            'sequence_length': length,
            'purine_content': (counts['A'] + counts['G']) / length if length > 0 else 0
        }
    
    def preprocess_structure(self, structure: str) -> Dict[str, float]:
        """Extract structure features"""
        # Handle cases where structure might be missing
        if not structure or not isinstance(structure, str):
            return {
                'paired_ratio': 0.0,
                'unpaired_ratio': 0.0
            }
            
        length = len(structure)
        if length == 0:
            return {
                'paired_ratio': 0.0,
                'unpaired_ratio': 0.0
            }
            
        paired = structure.count('(') + structure.count(')')
        unpaired = structure.count('.')
        
        return {
            'paired_ratio': paired / (2 * length),
            'unpaired_ratio': unpaired / length
        }
    
    def generate_esm_embeddings(self, sequence: str) -> np.ndarray:
        """Generate ESM embeddings for sequence."""
        batch_converter = self.alphabet.get_batch_converter()
        with torch.no_grad():
            # Prepare the sequence as input with a label
            batch_labels, batch_strs, batch_tokens = batch_converter([("seq", sequence)])
            
            # Pass through the model
            results = self.esm_model(batch_tokens, repr_layers=[33])
            embeddings = results["representations"][33].numpy()
        return embeddings[0]
    
    def process_data(self, df: pd.DataFrame) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
        """Process full dataset"""
        print("Processing data...")
        if 'signal_to_noise' in df.columns:
            df = df[df['signal_to_noise'] >= self.config.data.sn_threshold].copy()
        
        sequences = []
        features = []
        targets = {
            'reactivity': [],
            'deg_Mg_pH10': [],
            'deg_pH10': [],
            'deg_Mg_50C': [],
            'deg_50C': []
        }
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            try:
                # Validate sequence
                if not isinstance(row['sequence'], str) or len(row['sequence']) == 0:
                    print(f"Invalid sequence in row {idx}")
                    continue
                
                # Process sequence
                seq_features = self.preprocess_sequence(row['sequence'])
                struct_features = self.preprocess_structure(row.get('structure', ''))
                
                # Get ESM embeddings
                try:
                    embeddings = self.generate_esm_embeddings(row['sequence'])
                    embeddings = embeddings.mean(axis=0)
                except Exception as e:
                    print(f"Error generating embeddings for row {idx}: {e}")
                    continue
                
                # Convert feature dictionaries to arrays
                seq_feat_array = np.array(list(seq_features.values()))
                struct_feat_array = np.array(list(struct_features.values()))
                
                # Combine features
                combined_features = np.concatenate([
                    embeddings,
                    seq_feat_array,
                    struct_feat_array
                ])
                
                sequences.append(row['sequence'])
                features.append(combined_features)
                
                # Process targets
                for name in targets:
                    if name in row and isinstance(row[name], (list, np.ndarray)) and len(row[name]) > 0:
                        targets[name].append(np.array(row[name]).astype(np.float32))
                    else:
                        targets[name].append(np.zeros(68, dtype=np.float32))
                        
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        if not sequences:
            raise ValueError("No sequences were processed successfully")
        
        # Convert lists to numpy arrays
        features_array = np.stack(features)
        targets_dict = {k: np.stack(v) for k, v in targets.items()}
        
        print(f"Processed {len(sequences)} sequences")
        print(f"Features shape: {features_array.shape}")
        for k, v in targets_dict.items():
            print(f"{k} shape: {v.shape}")
        
        return sequences, features_array, targets_dict
    
    def save_processed_data(self, sequences: List[str], features: np.ndarray, 
                          targets: Dict[str, np.ndarray], split: str):
        """Save processed data with train/val split if it's training data"""
        output_dir = Path(self.config.data.processed_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if split == 'train':
            # Create train/val split
            indices = np.arange(len(sequences))
            train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
            
            # Save training data
            with open(output_dir / "train_sequences.txt", 'w') as f:
                for idx in train_idx:
                    f.write(f"{sequences[idx]}\n")
            np.save(output_dir / "train_features.npy", features[train_idx].astype(np.float32))
            for name, values in targets.items():
                np.save(output_dir / f"train_{name}.npy", values[train_idx].astype(np.float32))
            
            # Save validation data
            with open(output_dir / "val_sequences.txt", 'w') as f:
                for idx in val_idx:
                    f.write(f"{sequences[idx]}\n")
            np.save(output_dir / "val_features.npy", features[val_idx].astype(np.float32))
            for name, values in targets.items():
                np.save(output_dir / f"val_{name}.npy", values[val_idx].astype(np.float32))
                
        else:  # For test data
            with open(output_dir / f"test_sequences.txt", 'w') as f:
                for seq in sequences:
                    f.write(f"{seq}\n")
            np.save(output_dir / f"test_features.npy", features.astype(np.float32))
            for name, values in targets.items():
                np.save(output_dir / f"test_{name}.npy", values.astype(np.float32))
