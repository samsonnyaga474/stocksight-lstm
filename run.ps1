# StockSight Run Script (Windows / PowerShell)
# Launches the Streamlit dashboard at http://localhost:8501
# Usage (from the project root):
#   .\run.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  StockSight - Starting Application" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "ERROR: Virtual environment not found." -ForegroundColor Red
    Write-Host "Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Verifying dependencies..." -ForegroundColor Yellow
& $venvPython -c "import torch, streamlit, yfinance, pandas, plotly, pyarrow" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Required packages are missing. Run .\setup.ps1 again." -ForegroundColor Red
    exit 1
}

Write-Host "Launching Streamlit..." -ForegroundColor Green
Write-Host "Dashboard: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the application." -ForegroundColor Gray
Write-Host ""

& $venvPython -m streamlit run app.py --server.port 8501

Write-Host ""
Write-Host "Application stopped." -ForegroundColor Yellow
