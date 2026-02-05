#!/bin/bash
# ============================================================
# MASt3R Environment Setup Script
# ============================================================
# This script sets up the MASt3R (Matching And Stereo 3D Reconstruction)
# model for our face authentication system.
#
# WHAT THIS SCRIPT DOES:
#   1. Clones the MASt3R repository as a git submodule
#   2. Downloads the pre-trained model weights (~1GB)
#   3. Installs Python dependencies
#   4. (Optional) Compiles CUDA kernels for faster inference
#
# REQUIREMENTS:
#   - Git installed
#   - Python 3.9+ with pip
#   - CUDA 11.x or 12.x (for GPU support)
#   - ~2GB free disk space
#
# USAGE:
#   cd face-auth-mast3r
#   bash scripts/setup_mast3r.sh
#
# For Windows users: Use WSL (Windows Subsystem for Linux) or
# see scripts/setup_mast3r.ps1 for PowerShell version.
#
# Author: CS-1
# ============================================================

set -e  # Exit immediately if any command fails

# Color codes for prettier output (makes terminal messages easier to read)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color (reset)

# Helper function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================
# Step 0: Check prerequisites
# ============================================================
print_status "Checking prerequisites..."

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install git first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check Python version (need 3.9+)
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
print_status "Python version: $PYTHON_VERSION"

# Check if CUDA is available (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    print_status "CUDA is available:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "CUDA not detected. MASt3R will run on CPU (very slow)."
    print_warning "For GPU support, install CUDA toolkit and NVIDIA drivers."
fi

print_status "Prerequisites check passed!"
echo ""

# ============================================================
# Step 1: Clone MASt3R repository as git submodule
# ============================================================
print_status "Setting up MASt3R repository..."

# Create third_party directory if it doesn't exist
mkdir -p third_party

# Check if MASt3R is already cloned
if [ -d "third_party/mast3r" ]; then
    print_warning "MASt3R directory already exists. Updating..."
    cd third_party/mast3r
    git pull origin main
    git submodule update --init --recursive
    cd ../..
else
    # Clone MASt3R as a submodule
    # --recursive flag also clones nested submodules (like dust3r)
    print_status "Cloning MASt3R repository (this may take a few minutes)..."
    git submodule add https://github.com/naver/mast3r.git third_party/mast3r 2>/dev/null || true
    git submodule update --init --recursive
fi

print_status "MASt3R repository setup complete!"
echo ""

# ============================================================
# Step 2: Download model checkpoint
# ============================================================
print_status "Setting up model checkpoint..."

# Create checkpoints directory
mkdir -p checkpoints

# The model file is about 1GB
CHECKPOINT_URL="https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
CHECKPOINT_PATH="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

if [ -f "$CHECKPOINT_PATH" ]; then
    print_warning "Checkpoint already exists at $CHECKPOINT_PATH"
    print_warning "Skipping download. Delete the file to re-download."
else
    print_status "Downloading MASt3R checkpoint (~1GB, please wait)..."

    # Use wget with progress bar, or curl as fallback
    if command -v wget &> /dev/null; then
        wget -c "$CHECKPOINT_URL" -O "$CHECKPOINT_PATH"
    elif command -v curl &> /dev/null; then
        curl -L -o "$CHECKPOINT_PATH" "$CHECKPOINT_URL"
    else
        print_error "Neither wget nor curl is available. Please install one of them."
        print_error "Or manually download from: $CHECKPOINT_URL"
        exit 1
    fi

    print_status "Checkpoint downloaded successfully!"
fi

echo ""

# ============================================================
# Step 3: Install Python dependencies
# ============================================================
print_status "Installing Python dependencies..."

# Install MASt3R dependencies
if [ -f "third_party/mast3r/requirements.txt" ]; then
    print_status "Installing MASt3R requirements..."
    pip install -r third_party/mast3r/requirements.txt
fi

# Install DUSt3R dependencies (MASt3R is built on DUSt3R)
if [ -f "third_party/mast3r/dust3r/requirements.txt" ]; then
    print_status "Installing DUSt3R requirements..."
    pip install -r third_party/mast3r/dust3r/requirements.txt
fi

# Install our project's requirements
if [ -f "requirements.txt" ]; then
    print_status "Installing project requirements..."
    pip install -r requirements.txt
fi

print_status "Python dependencies installed!"
echo ""

# ============================================================
# Step 4: (Optional) Compile RoPE CUDA kernels
# ============================================================
# RoPE (Rotary Position Embedding) CUDA kernels speed up inference
# This step is optional and may fail on some systems

print_status "Attempting to compile CUDA kernels (optional)..."

CUROPE_DIR="third_party/mast3r/dust3r/croco/models/curope"

if [ -d "$CUROPE_DIR" ]; then
    cd "$CUROPE_DIR"

    # Try to compile, but don't fail if it doesn't work
    if python setup.py build_ext --inplace 2>/dev/null; then
        print_status "CUDA kernels compiled successfully!"
    else
        print_warning "CUDA kernel compilation failed (non-critical)."
        print_warning "MASt3R will still work, but may be slightly slower."
    fi

    # Return to project root
    cd - > /dev/null
else
    print_warning "CuRoPE directory not found. Skipping kernel compilation."
fi

echo ""

# ============================================================
# Step 5: Verify installation
# ============================================================
print_status "Verifying installation..."

# Try to import MASt3R
python3 -c "
import sys
sys.path.insert(0, 'third_party/mast3r')
sys.path.insert(0, 'third_party/mast3r/dust3r')
from mast3r.model import AsymmetricMASt3R
print('MASt3R module imported successfully!')
" 2>/dev/null

if [ $? -eq 0 ]; then
    print_status "MASt3R import verification passed!"
else
    print_warning "MASt3R import verification failed."
    print_warning "You may need to add the paths to your PYTHONPATH."
fi

echo ""

# ============================================================
# Setup Complete!
# ============================================================
echo "============================================================"
print_status "MASt3R setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run the smoke test to verify everything works:"
echo "     python scripts/test_inference.py"
echo ""
echo "  2. If using Conda, activate your environment first:"
echo "     conda activate face-auth"
echo ""
echo "  3. For GPU usage, ensure CUDA is properly installed."
echo ""
echo "Troubleshooting:"
echo "  - If imports fail, check that paths are correct"
echo "  - For memory errors, try setting force_fp16=True in config.yaml"
echo "  - See README.md for more detailed instructions"
echo ""
