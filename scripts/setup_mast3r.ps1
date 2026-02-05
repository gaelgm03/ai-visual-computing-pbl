# ============================================================
# MASt3R Environment Setup Script (Windows PowerShell)
# ============================================================
# This script sets up the MASt3R (Matching And Stereo 3D Reconstruction)
# model for our face authentication system on Windows.
#
# WHAT THIS SCRIPT DOES:
#   1. Clones the MASt3R repository as a git submodule
#   2. Downloads the pre-trained model weights (~1GB)
#   3. Installs Python dependencies
#
# REQUIREMENTS:
#   - Git installed and in PATH
#   - Python 3.9+ with pip
#   - CUDA 11.x or 12.x (for GPU support)
#   - ~2GB free disk space
#
# USAGE:
#   Open PowerShell as Administrator, navigate to project folder:
#   cd face-auth-mast3r
#   .\scripts\setup_mast3r.ps1
#
# NOTE: If you get "running scripts is disabled" error, run:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#
# Author: CS-1
# ============================================================

# Stop on any error
$ErrorActionPreference = "Stop"

# ============================================================
# Helper Functions
# ============================================================
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# ============================================================
# Step 0: Check prerequisites
# ============================================================
Write-Status "Checking prerequisites..."

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Status "Git found: $gitVersion"
} catch {
    Write-Err "Git is not installed or not in PATH."
    Write-Err "Please install Git from https://git-scm.com/"
    exit 1
}

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Status "Python found: $pythonVersion"
} catch {
    Write-Err "Python is not installed or not in PATH."
    Write-Err "Please install Python 3.9+ from https://python.org/"
    exit 1
}

# Check if CUDA is available (optional)
try {
    $nvidiaSmi = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    if ($nvidiaSmi) {
        Write-Status "CUDA is available:"
        Write-Host $nvidiaSmi
    }
} catch {
    Write-Warn "CUDA not detected. MASt3R will run on CPU (very slow)."
    Write-Warn "For GPU support, install CUDA toolkit and NVIDIA drivers."
}

Write-Status "Prerequisites check passed!"
Write-Host ""

# ============================================================
# Step 1: Clone MASt3R repository
# ============================================================
Write-Status "Setting up MASt3R repository..."

# Create third_party directory if it doesn't exist
if (-not (Test-Path "third_party")) {
    New-Item -ItemType Directory -Path "third_party" | Out-Null
}

# Check if MASt3R is already cloned
if (Test-Path "third_party\mast3r") {
    Write-Warn "MASt3R directory already exists. Updating..."
    Push-Location "third_party\mast3r"
    git pull origin main
    git submodule update --init --recursive
    Pop-Location
} else {
    # Clone MASt3R as a submodule
    Write-Status "Cloning MASt3R repository (this may take a few minutes)..."
    git submodule add https://github.com/naver/mast3r.git third_party/mast3r 2>$null
    git submodule update --init --recursive
}

Write-Status "MASt3R repository setup complete!"
Write-Host ""

# ============================================================
# Step 2: Download model checkpoint
# ============================================================
Write-Status "Setting up model checkpoint..."

# Create checkpoints directory
if (-not (Test-Path "checkpoints")) {
    New-Item -ItemType Directory -Path "checkpoints" | Out-Null
}

$checkpointUrl = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
$checkpointPath = "checkpoints\MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

if (Test-Path $checkpointPath) {
    Write-Warn "Checkpoint already exists at $checkpointPath"
    Write-Warn "Skipping download. Delete the file to re-download."
} else {
    Write-Status "Downloading MASt3R checkpoint (~1GB, please wait)..."
    Write-Status "This may take several minutes depending on your internet speed."

    try {
        # Use .NET WebClient for downloading with progress
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($checkpointUrl, $checkpointPath)
        Write-Status "Checkpoint downloaded successfully!"
    } catch {
        Write-Err "Download failed. Error: $_"
        Write-Err "Please manually download from:"
        Write-Err $checkpointUrl
        Write-Err "And save to: $checkpointPath"
        exit 1
    }
}

Write-Host ""

# ============================================================
# Step 3: Install Python dependencies
# ============================================================
Write-Status "Installing Python dependencies..."

# Install MASt3R dependencies
if (Test-Path "third_party\mast3r\requirements.txt") {
    Write-Status "Installing MASt3R requirements..."
    pip install -r "third_party\mast3r\requirements.txt"
}

# Install DUSt3R dependencies
if (Test-Path "third_party\mast3r\dust3r\requirements.txt") {
    Write-Status "Installing DUSt3R requirements..."
    pip install -r "third_party\mast3r\dust3r\requirements.txt"
}

# Install our project's requirements
if (Test-Path "requirements.txt") {
    Write-Status "Installing project requirements..."
    pip install -r "requirements.txt"
}

Write-Status "Python dependencies installed!"
Write-Host ""

# ============================================================
# Step 4: Verify installation
# ============================================================
Write-Status "Verifying installation..."

$verifyScript = @"
import sys
sys.path.insert(0, 'third_party/mast3r')
sys.path.insert(0, 'third_party/mast3r/dust3r')
try:
    from mast3r.model import AsymmetricMASt3R
    print('MASt3R module imported successfully!')
except ImportError as e:
    print(f'Import failed: {e}')
    sys.exit(1)
"@

try {
    $result = python -c $verifyScript
    Write-Host $result
    Write-Status "MASt3R import verification passed!"
} catch {
    Write-Warn "MASt3R import verification failed."
    Write-Warn "You may need to add the paths to your PYTHONPATH."
}

Write-Host ""

# ============================================================
# Setup Complete!
# ============================================================
Write-Host "============================================================"
Write-Status "MASt3R setup complete!"
Write-Host "============================================================"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Run the smoke test to verify everything works:"
Write-Host "     python scripts/test_inference.py"
Write-Host ""
Write-Host "  2. For GPU usage, ensure CUDA is properly installed."
Write-Host "     Check with: nvidia-smi"
Write-Host ""
Write-Host "  3. If using virtual environment, activate it first:"
Write-Host "     .\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Troubleshooting:"
Write-Host "  - If imports fail, check that paths are correct"
Write-Host "  - For memory errors, try setting force_fp16=True in config.yaml"
Write-Host "  - See README.md for more detailed instructions"
Write-Host ""

# Note: CUDA kernel compilation is not directly supported on Windows
# Users should use WSL for that functionality if needed
