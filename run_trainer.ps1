# PowerShell script to run the LFM2-350M Trainer

Write-Host "Activating environment..." -ForegroundColor Cyan
if (-Not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "Error: Virtual environment not found. Please run .\setup.ps1 first." -ForegroundColor Red
    exit
}

.\venv\Scripts\Activate.ps1

# Check if dataset exists
if (-Not (Test-Path ".\training data\dataset.jsonl")) {
    Write-Host "Dataset not found. Generating synthetic data first..." -ForegroundColor Yellow
    python tools/generate_data.py
}

Write-Host "Starting optimized training on RTX 5090..." -ForegroundColor Green
python tools/train.py

Write-Host "Process finished." -ForegroundColor Cyan
