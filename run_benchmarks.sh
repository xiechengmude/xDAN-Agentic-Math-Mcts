#!/bin/bash

# Define the model name
MODEL_NAME="gpt-4o-mini"

# Define the base datasets
BASE_DATASETS=(
    'gsm8k'
    'gsmhard'
    'olympiadbench'
    'GAIC'
    'MATH'
    'AIME'
)

# Define the mcts info for each dataset
MCTS_INFO=(
    'new-mcts-8'
    'new-mcts-8'
    'new-mcts-8'
    'new-mcts-8'
    'mcts-2'
)

# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

# Iterate through each base dataset
for i in "${!BASE_DATASETS[@]}"; do
    DATASET="${BASE_DATASETS[$i]}-${MODEL_NAME}-${MCTS_INFO[$i]}-${TIMESTAMP}"
    # Run the command and ignore errors
    python run_with_earlystopping.py "$MODEL_NAME" "$DATASET" || echo "Error running dataset: $DATASET. Continuing to next dataset."
done