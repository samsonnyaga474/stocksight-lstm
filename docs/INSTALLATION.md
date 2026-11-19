# Installation Guide (Windows 11)

## Requirements

- Windows 11
- Python 3.11 or newer (64-bit recommended)
- Internet connection for the initial package download and for live market data
- ~2 GB free disk space (PyTorch + dependencies)

## Step-by-step

1. **Get the project**  
   Clone the repository (or extract the archive) so that you have a single top-level folder:
   ```powershell
   git clone https://github.com/samsonnyaga474/stocksight-lstm.git
   ```

2. **Open PowerShell**  
   Navigate into that folder:
   ```powershell
   cd C:\path\to\StockSight
   ```

3. **(Optional) Allow local scripts**  
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

4. **Run setup**  
   ```powershell
   .\setup.ps1
   ```
   The script will:
   - Check Python
   - Create `.venv`
   - Install CPU-compatible PyTorch
   - Install remaining requirements
   - Create data / model / log directories
   - Initialise the SQLite database
   - Verify critical imports

5. **Start the application**  
   ```powershell
   .\run.ps1
   ```
   Browser address: http://localhost:8501

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `python` not found | Python not on PATH | Re-install Python and tick “Add to PATH” |
| PyTorch install fails | Network / index | Retry or install manually with the CPU index URL shown in `setup.ps1` |
| Port 8501 in use | Another Streamlit process | Stop the other process or change the port in `run.ps1` |
| Empty Kenyan data | Provider gap | Use CSV import on the Stock Explorer page |
| Training very slow | CPU only, long history | Reduce period (e.g. 2y) or epochs for experiments |

## Uninstall

Simply delete the `StockSight` folder. The virtual environment and all generated files live inside it.
