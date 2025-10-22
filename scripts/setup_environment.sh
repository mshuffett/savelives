#!/bin/bash
# Setup script for RunPod environment
# Installs all dependencies needed for medical ASR training

set -e  # Exit on error

echo "=== Medical ASR Training Environment Setup ==="
echo ""

# Update system packages
echo "Updating system packages..."
apt-get update -qq
apt-get install -y -qq git wget curl ffmpeg libsndfile1 > /dev/null 2>&1
echo "✓ System packages updated"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --quiet --upgrade pip setuptools wheel

# Install PyTorch (ensure CUDA compatibility)
echo "Installing PyTorch with CUDA support..."
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "✓ PyTorch installed"

# Install NeMo from source (latest features)
echo "Installing NVIDIA NeMo..."
pip install --quiet "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"
echo "✓ NeMo installed"

# Install remaining requirements
if [ -f "requirements.txt" ]; then
    echo "Installing project requirements..."
    pip install --quiet -r requirements.txt
    echo "✓ Requirements installed"
fi

# Verify installations
echo ""
echo "=== Verifying Installation ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
python -c "import nemo; print(f'NeMo: {nemo.__version__}')"
python -c "import wandb; print(f'Wandb: {wandb.__version__}')"

echo ""
echo "✓ Environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Set environment variables in .env"
echo "  2. Download datasets: python scripts/download_data.py"
echo "  3. Start training: python src/train.py"
