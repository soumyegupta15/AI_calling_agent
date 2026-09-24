# Windows setup for the local CPU voice agent.
#   Run from this folder in PowerShell:
#       .\setup_windows.ps1
#
# Creates .venv, installs everything (including the packages that need a special
# index or a git URL), downloads the models, and runs the preflight check.

$ErrorActionPreference = "Stop"

Write-Host "`n=== Local CPU Voice Agent - Windows setup ===`n" -ForegroundColor Cyan

# --- Python version check -------------------------------------------------
$pyVersion = (python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
Write-Host "Python $pyVersion detected"
if ($pyVersion -notin @("3.10", "3.11", "3.12", "3.13")) {
    Write-Host "WARNING: Python 3.10-3.13 tested. onnxruntime / tokenizers wheels may not exist for $pyVersion." -ForegroundColor Yellow
}
Write-Host "Stack: Silero VAD -> Vosk (or Whisper) -> Qwen2.5-1.5B -> Piper" -ForegroundColor Gray
Write-Host "No compiler required: the LLM runs on a prebuilt llama-server binary." -ForegroundColor Gray

# --- virtualenv -----------------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "`nCreating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# --- core deps ------------------------------------------------------------
Write-Host "`nInstalling core dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# --- models + llama.cpp binary --------------------------------------------
# No Python LLM binding on purpose: we download llama.cpp's official prebuilt
# CPU binary instead of compiling a Python extension.
Write-Host "`nDownloading models and the llama.cpp CPU binary (~1.5 GB, one time)..." -ForegroundColor Cyan
python scripts\download_models.py --warm-stt

# --- preflight ------------------------------------------------------------
Write-Host "`nRunning preflight..." -ForegroundColor Cyan
python run_agent.py --check

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  .\run_vosk.ps1                 # live call, Vosk streaming STT (fast)"
Write-Host "  .\run_whisper.ps1              # live call, Whisper STT (accurate)"
Write-Host "  python run_agent.py --text     # typed input, spoken output"
Write-Host "  python benchmark.py            # latency against the budget"
Write-Host "  python diagnose.py             # per-stage check if something is wrong"
Write-Host "`nTip: wear headphones. On open speakers the agent hears itself and barge-in gets twitchy.`n" -ForegroundColor Yellow
