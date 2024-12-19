import argparse
from pathlib import Path
import sys
from os.path import abspath, dirname
import pandas as pd
import torch
import esm
import numpy as np
from tqdm import tqdm

# Add project root to Python path
project_root = dirname(dirname(abspath(__file__)))
sys.path.append(project_root)

def generate_embeddings(model, alphabet, sequences, output_dir: Path):
    """Generate and save ESM embeddings for sequences"""
    print("Generating embeddings...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get batch converter
    batch_converter = alphabet.get_batch_converter()
    
    # Process sequences in batches
    batch_size = 32
    embeddings_list = []
    
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i:i + batch_size]
        batch_data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_sequences)]
        
        with torch.no_grad():
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            results = model(batch_tokens, repr_layers=[33])
            embeddings = results["representations"][33].numpy()
            embeddings_list.append(embeddings)
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    
    # Save embeddings
    embedding_path = output_dir / "esm_embeddings.npy"
    np.save(embedding_path, all_embeddings)
    
    # Save sequence mapping
    with open(output_dir / "sequence_ids.txt", 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f"{i}\t{seq}\n")
            
    print(f"Generated embeddings shape: {all_embeddings.shape}")
    print(f"Embeddings saved to: {embedding_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate ESM embeddings for RNA sequences')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing train.json and test.json')
    parser.add_argument('--output', type=str, required=True, help='Output directory for embeddings')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Load ESM model
    print("Loading ESM model...")
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully")
    
    # Process training data
    print("\nProcessing training data...")
    train_data = pd.read_json(input_dir / "train.json", lines=True)
    train_sequences = train_data['sequence'].tolist()
    generate_embeddings(model, alphabet, train_sequences, output_dir / "train")
    
    # Process test data
    print("\nProcessing test data...")
    test_data = pd.read_json(input_dir / "test.json", lines=True)
    test_sequences = test_data['sequence'].tolist()
    generate_embeddings(model, alphabet, test_sequences, output_dir / "test")
    
    print("\nEmbedding generation completed!")

if __name__ == "__main__":
    main()
