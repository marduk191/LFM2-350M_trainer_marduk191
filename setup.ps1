# PowerShell Setup Script for LFM2-350M Trainer

Write-Host "Creating Virtual Environment..." -ForegroundColor Cyan
if (-Not (Test-Path "dataset.jsonl")) {
    Write-Host "Dataset not found. Generating initial data..." -ForegroundColor Yellow
    python tools/generate_data.py
}

Write-Host "Starting optimized training on RTX 5090..." -ForegroundColor Green
python tools/train.py

Write-Host "Installing CUDA-enabled PyTorch 2.9.0 (CUDA 12.8)..." -ForegroundColor Cyan
python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

Write-Host "Installing other dependencies from requirements.txt..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

Write-Host "Downloading LFM2-350M model from Hugging Face..." -ForegroundColor Cyan
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model_id = 'LiquidAI/LFM2-350M'; AutoTokenizer.from_pretrained(model_id); AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)"

Write-Host "Validating CUDA Installation..." -ForegroundColor Cyan
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "To start the environment, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
