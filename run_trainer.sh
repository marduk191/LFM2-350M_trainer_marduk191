#!/bin/bash

# run_trainer.sh - Training launcher for Linux/macOS

echo "Activating environment..."
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

source venv/bin/activate

# Check for dataset
if [ ! -f "training data/dataset.jsonl" ]; then
    echo "Dataset not found. Generating initial data..."
    python3 tools/generate_data.py
fi

echo "Starting optimized training on GPU..."
python3 tools/train.py
