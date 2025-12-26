#!/bin/bash

# setup.sh - Environment setup for Linux/macOS
# Optimized for NVIDIA GPUs

echo "------------------------------------------------"
echo " LFM2-350M Trainer Setup (Linux/macOS) "
echo "------------------------------------------------"

# 1. Create Virtual Environment
echo "Creating virtual environment..."
python3 -m venv venv

# 2. Activate Environment
source venv/bin/activate

echo "Launching GUI..."
python3 tools/gui.py

# 3. Install PyTorch with CUDA support (targeting latest stable for 5090/4090)
echo "Installing CUDA-enabled PyTorch 2.6+..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install other dependencies
echo "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found!"
fi

# Check for dataset
if [ ! -f "dataset.jsonl" ]; then
    echo "Dataset not found. Generating initial data..."
    python3 tools/generate_data.py
fi

echo "Starting optimized training on GPU..."
python3 tools/train.py

# 5. Pre-download model
echo "Downloading LFM2-350M model from Hugging Face..."
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model_id = 'LiquidAI/LFM2-350M'; AutoTokenizer.from_pretrained(model_id); AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)"

# 6. Validate CUDA
echo "Validating CUDA Installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

echo "------------------------------------------------"
echo "Setup Complete!"
echo "To activate: source venv/bin/activate"
echo "------------------------------------------------"
