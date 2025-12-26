#!/bin/bash

# run_gui.sh - GUI launcher for Linux/macOS

echo "Activating environment..."
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

source venv/bin/activate

echo "Launching GUI..."
python3 tools/gui.py
