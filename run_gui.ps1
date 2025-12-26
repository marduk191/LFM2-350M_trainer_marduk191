# PowerShell script to launch the LFM2-350M GUI

Write-Host "Activating environment..." -ForegroundColor Cyan
if (-Not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "Error: Virtual environment not found. Please run .\setup.ps1 first." -ForegroundColor Red
    exit
}

.\venv\Scripts\Activate.ps1

Write-Host "Launching GUI..." -ForegroundColor Green
python tools/gui.py
