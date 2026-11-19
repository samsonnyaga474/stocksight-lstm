# StockSight Setup Script (Windows / PowerShell)
# Creates a project-local virtual environment and installs all dependencies.
# Usage (from the project root):
#   .\setup.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  StockSight - Installation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Locate and validate Python
Write-Host "[1/7] Checking Python installation..." -ForegroundColor Yellow
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "  ERROR: Python is not installed or not on PATH." -ForegroundColor Red
    Write-Host "  Install Python 3.11+ from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "  and enable 'Add Python to PATH' during installation." -ForegroundColor Red
    exit 1
}
$pythonExe = $pythonCmd.Source
$versionOk = & $pythonExe -c "import sys; print(sys.version.split()[0]); sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Python 3.11 or newer is required. Found: $versionOk" -ForegroundColor Red
    exit 1
}
Write-Host "  Found Python $versionOk at $pythonExe" -ForegroundColor Green

# 2. Create the project-local virtual environment
Write-Host "[2/7] Creating virtual environment (.venv)..." -ForegroundColor Yellow
$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    & $pythonExe -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to create the virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "  Existing virtual environment found." -ForegroundColor Gray
}

# 3. Upgrade pip inside the venv only
Write-Host "[3/7] Upgrading pip inside .venv..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip --disable-pip-version-check --progress-bar off
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARNING: pip upgrade failed; continuing with the existing pip." -ForegroundColor Yellow
}

# 4. Install dependencies (CPU-only PyTorch first)
Write-Host "[4/7] Installing dependencies (this can take several minutes)..." -ForegroundColor Yellow
& $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check --progress-bar off
if ($LASTEXITCODE -ne 0) {
    Write-Host "  CPU PyTorch index unavailable; trying the default package index..." -ForegroundColor Yellow
    & $venvPython -m pip install torch --disable-pip-version-check --progress-bar off
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: PyTorch installation failed." -ForegroundColor Red
        exit 1
    }
}
& $venvPython -m pip install -r requirements.txt --disable-pip-version-check --progress-bar off
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Failed to install project dependencies." -ForegroundColor Red
    exit 1
}
Write-Host "  Dependencies installed." -ForegroundColor Green

# 5. Create application directories
Write-Host "[5/7] Creating application directories..." -ForegroundColor Yellow
$dirs = @(
    "models\checkpoints",
    "models\scalers",
    "models\metadata",
    "data\raw",
    "data\processed",
    "data\external\imports",
    "data\external\uploads",
    "data\samples",
    "database",
    "logs",
    "experiments"
)
foreach ($dir in $dirs) {
    $path = Join-Path $ProjectRoot $dir
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
    $gitkeep = Join-Path $path ".gitkeep"
    if (-not (Test-Path $gitkeep)) {
        New-Item -ItemType File -Path $gitkeep -Force | Out-Null
    }
}
Write-Host "  Directories ready." -ForegroundColor Green

# 6. Initialise the SQLite database
Write-Host "[6/7] Initialising SQLite database..." -ForegroundColor Yellow
& $venvPython -c "from src.database.db import init_database; init_database(); print('Database initialised.')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARNING: Database initialisation failed; it will be retried on first launch." -ForegroundColor Yellow
} else {
    Write-Host "  Database ready." -ForegroundColor Green
}

# 7. Verify critical imports
Write-Host "[7/7] Verifying critical imports..." -ForegroundColor Yellow
& $venvPython -c @"
import sys
errors = []
for name in ('torch', 'pandas', 'numpy', 'sklearn', 'streamlit', 'plotly', 'yfinance', 'pyarrow', 'yaml', 'joblib'):
    try:
        module = __import__(name)
        print(f'  {name} {getattr(module, \"__version__\", \"OK\")}')
    except Exception as exc:
        errors.append(f'{name}: {exc}')
if errors:
    print('ERRORS:')
    for item in errors:
        print(f'  - {item}')
    sys.exit(1)
print('All critical imports successful.')
"@
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: One or more packages failed to import." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Installation complete" -ForegroundColor Green
Write-Host "  Next: .\run.ps1" -ForegroundColor White
Write-Host "  Dashboard: http://localhost:8501" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Predictions are educational estimates, not financial advice." -ForegroundColor Gray
