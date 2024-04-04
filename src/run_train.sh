#!/bin/bash

# Define model types
MODEL_TYPES=("SVM" "MLP" "CNN" "LSTM" "GRU" "Transformer")


# Define learning rates
LEARNING_RATES=(0.001 0.0001 0.00001)

# Define output directory
OUTPUT_DIR="./training_results/"

# Redirect output to a file and terminal simultaneously
OUTPUT_FILE="./training_results/training_log.txt"

# Clear the log file if it exists
> "$OUTPUT_FILE"

# Iterate over model types
for model_type in "${MODEL_TYPES[@]}"; do
    # Create directory for model type
    MODEL_DIR="${OUTPUT_DIR}${model_type}/"
    mkdir -p "$MODEL_DIR"
    
    # Iterate over learning rates
    for lr in "${LEARNING_RATES[@]}"; do
        # Create directory for learning rate
        LR_DIR="${MODEL_DIR}${lr}/"
        mkdir -p "$LR_DIR"
        
        # Run the Python program with current model type and learning rate
        echo "Running training for $model_type with learning rate $lr..."
        (python3 /home/eo/Desktop/proj/GRU_Classifying_GTZAN-main/src/train_model.py "$model_type" "$LR_DIR" "$lr" | tee -a "$OUTPUT_FILE") 2>&1
        echo "Finished training for $model_type with learning rate $lr."
    done
done

