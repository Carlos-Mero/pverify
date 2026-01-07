#!/bin/bash

# Example script to run the iterative prover pipeline
# Using gemini-3-pro-preview as prover and progressive reviewer on MathArena/imo_2025

# Ensure PYTHONPATH includes the current directory
export PYTHONPATH=$PYTHONPATH:.

# Define variables
DATASET="MathArena/imo_2025"
PROVER="gemini-3-pro-preview"
REVIEWER="progressive"
REFINE_ITERS=10
LOG_DIR="eval_logs/imo_prove_experiment"

echo "Starting prover pipeline..."
echo "Dataset: $DATASET"
echo "Prover: $PROVER"
echo "Reviewer: $REVIEWER"
echo "Refine Iters: $REFINE_ITERS"

python main.py prove \
    --dataset "$DATASET" \
    --prover "$PROVER" \
    --reviewer "$REVIEWER" \
    --refine_iters "$REFINE_ITERS" \
    --log_dir "$LOG_DIR" \
    --progressive_max_iters 3 \
    --progressive_min_chunk_size 6 \
    --enable_thinking \
    --reasoning_effort medium \
    --prover_base_url <your_base_url_here> \
    --prover_api_key <you_api_key>

echo "Pipeline finished. Logs should be in $LOG_DIR"
