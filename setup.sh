#!/bin/bash

# setup.sh - Environment setup for Linux/macOS
# Optimized for NVIDIA GPUs

echo "------------------------------------------------"
echo " LFM2-350M Trainer Setup (Linux/macOS) "
echo "------------------------------------------------"

# 1. Create Virtual Environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# 2. Activate Environment
source venv/bin/activate

# 3. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install PyTorch with CUDA support
echo "Installing CUDA-enabled PyTorch..."
# Note: cu124 is a safe default for most modern Linux distros with 50/40 series cards
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

# 5. Install other dependencies
echo "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found!"
fi

# 6. Pre-download model
echo "Downloading LFM2-350M model from Hugging Face..."
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model_id = 'LiquidAI/LFM2-350M'; AutoTokenizer.from_pretrained(model_id); AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)"

# 7. Validate CUDA
echo "Validating CUDA Installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

echo "------------------------------------------------"
echo "Setup Complete!"
echo "To activate manually: source venv/bin/activate"
echo "------------------------------------------------"
