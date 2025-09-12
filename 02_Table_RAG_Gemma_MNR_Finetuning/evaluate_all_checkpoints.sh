#!/bin/bash

# --- Configuration ---
# The parent directory containing all your checkpoint folders
CHECKPOINT_PARENT_DIR="mnr-checkpoints"

# The file where all summary outputs will be saved
LOG_FILE="evaluation_summary_train_set_mnr_finetuning.log"


# This main function contains the logic that will be logged
run_evaluations() {
    echo "Starting evaluation for all checkpoints in: $CHECKPOINT_PARENT_DIR"
    echo "Detailed JSON results will be saved in the 'data/' directory."
    echo "==================================================================="

    # Check if the parent directory exists
    if [ ! -d "$CHECKPOINT_PARENT_DIR" ]; then
        echo "Error: Directory '$CHECKPOINT_PARENT_DIR' not found."
        exit 1
    fi

    # Loop through each subdirectory that starts with "checkpoint-"
    for checkpoint_path in "$CHECKPOINT_PARENT_DIR"/checkpoint-*; do
        # Check if the item found is a directory
        if [ -d "$checkpoint_path" ]; then
            # Extract just the name of the checkpoint folder (e.g., "checkpoint-474")
            checkpoint_name=$(basename "$checkpoint_path")
            
            # Define a unique name for the detailed JSON output file
            output_json="data/evaluation_results_test_mnr_finetuning_${checkpoint_name}.json"

            echo ""
            echo "--- Evaluating: $checkpoint_name ---"
            
            # Execute the evaluation script for the current checkpoint
            python retrieval_pipeline.py evaluate \
                --input data/test_set.json \
                --output "$output_json" \
                --k 10 \
                --embedding_model="$checkpoint_path"
        fi
    done

    echo ""
    echo "==================================================================="
    echo "✅ All checkpoint evaluations are complete."
}

# --- Script Execution ---
# 1. Clear the log file from any previous runs.
# 2. Run the main function and use 'tee' to show output on the screen AND save it to the log file.
> "$LOG_FILE"
run_evaluations | tee "$LOG_FILE"