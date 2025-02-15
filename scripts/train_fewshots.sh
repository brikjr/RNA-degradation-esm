#!/bin/bash

# Activate conda environment (if using conda)
# conda activate rna_env

# Process data first
python scripts/prepare_data.py \
    --config configs/few_shot.yaml \
    --input_dir data/raw \
    --output_dir data/processed

# Train model
python scripts/train.py \
    --config configs/few_shot.yaml \
    --mode train \
    --output_dir runs