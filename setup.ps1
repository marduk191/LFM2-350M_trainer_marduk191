# PowerShell Setup Script for LFM2-350M Trainer
# Optimized for RTX GPUs and Windows

Write-Host "------------------------------------------------" -ForegroundColor Cyan
Write-Host " LFM2-350M Trainer Setup (Windows) " -ForegroundColor Cyan
Write-Host "------------------------------------------------" -ForegroundColor Cyan

# 1. Create Virtual Environment
Write-Host "Creating Virtual Environment..." -ForegroundColor Cyan
if (-Not (Test-Path ".\venv")) {
    python -m venv venv
}

# 2. Activate Virtual Environment
Write-Host "Activating Virtual Environment..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# 4. Install CUDA-enabled PyTorch (targeting RTX 5090/4090/3090)
Write-Host "Installing CUDA-enabled PyTorch 2.9 (CUDA 12.8)..." -ForegroundColor Cyan
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

# 5. Install other dependencies
Write-Host "Installing requirements..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

# 6. Pre-download model
Write-Host "Downloading LFM2-350M model from Hugging Face..." -ForegroundColor Cyan
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model_id = 'LiquidAI/LFM2-350M'; AutoTokenizer.from_pretrained(model_id); AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)"

# 7. Validate CUDA
Write-Host "Validating CUDA Installation..." -ForegroundColor Cyan
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

Write-Host "------------------------------------------------" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "To start the environment manually, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "Or launch the GUI with: .\run_gui.ps1" -ForegroundColor Yellow
Write-Host "------------------------------------------------" -ForegroundColor Cyan
