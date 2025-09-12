#!/bin/bash

# --- Configuration ---
# Set the parent directory containing all your checkpoint folders.
# This should match the name of the folder in your screenshot.
CHECKPOINT_PARENT_DIR="mnr-checkpoints"

echo "Starting to process all checkpoints in: $CHECKPOINT_PARENT_DIR"
echo "============================================================"

# Check if the parent directory exists to avoid errors
if [ ! -d "$CHECKPOINT_PARENT_DIR" ]; then
    echo "Error: Directory '$CHECKPOINT_PARENT_DIR' not found."
    echo "Please make sure you are in the correct directory and the folder name is correct."
    exit 1
fi

# Loop through each subdirectory that starts with "checkpoint-"
for checkpoint_path in "$CHECKPOINT_PARENT_DIR"/checkpoint-*; do
    # Check if the item found is actually a directory
    if [ -d "$checkpoint_path" ]; then
        echo ""
        echo "--- Processing checkpoint: $checkpoint_path ---"
        
        # Execute the python script with the current checkpoint directory as the argument
        python vector_store.py --embedding_model="$checkpoint_path"
        
        echo "--- Finished processing $checkpoint_path ---"
    fi
done

echo ""
echo "============================================================"
echo "✅ All checkpoints have been processed."