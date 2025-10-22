#!/bin/bash
# Complete deployment script for RunPod
# This script runs ON the RunPod pod after SSH access is available

set -e  # Exit on error

echo "=========================================="
echo "Medical ASR Training - RunPod Deployment"
echo "=========================================="
echo ""

# Environment setup
export DEBIAN_FRONTEND=noninteractive
export WORKSPACE=/workspace
cd $WORKSPACE

# 1. Update system and install dependencies
echo "📦 Step 1/8: Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq git wget curl ffmpeg libsndfile1 sox > /dev/null 2>&1
echo "✓ System packages installed"

# 2. Clone repository
echo ""
echo "📥 Step 2/8: Cloning repository..."
if [ ! -d "savelives" ]; then
    git clone https://github.com/mshuffett/savelives.git
    cd savelives
else
    cd savelives
    git pull
fi
echo "✓ Repository ready"

# 3. Install Python dependencies
echo ""
echo "🐍 Step 3/8: Installing Python dependencies..."
pip install --quiet --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo "  Installing PyTorch..."
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install NeMo from source
echo "  Installing NeMo (this may take a few minutes)..."
pip install --quiet "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"

# Install project requirements
if [ -f "requirements.txt" ]; then
    echo "  Installing project requirements..."
    pip install --quiet -r requirements.txt
fi

echo "✓ Python environment ready"

# 4. Verify installations
echo ""
echo "🔍 Step 4/8: Verifying installations..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  CUDA Version: {torch.version.cuda}')"
python -c "import torch; print(f'  GPU Count: {torch.cuda.device_count()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'  GPU 0: {torch.cuda.get_device_name(0)}')"
fi
python -c "import nemo; print(f'  NeMo: {nemo.__version__}')"
python -c "import wandb; print(f'  Wandb: {wandb.__version__}')"
echo "✓ All dependencies verified"

# 5. Set up environment variables
echo ""
echo "🔑 Step 5/8: Setting up environment variables..."
cat > .env << EOF
# RunPod Environment
RUNPOD_API_KEY=${RUNPOD_API_KEY}
WANDB_API_KEY=${WANDB_API_KEY}
WANDB_PROJECT=medical-asr-poc-michael
CHECKPOINT_DIR=/workspace/checkpoints
DATA_DIR=/workspace/data
LOG_DIR=/workspace/logs
EOF

# Load environment
export $(cat .env | xargs)
echo "✓ Environment configured"

# 6. Download and prepare dataset
echo ""
echo "📊 Step 6/8: Downloading medical speech dataset..."
python scripts/download_data.py --dataset hani89 --max-samples 2000
echo "✓ Dataset ready (2000 samples)"

# 7. Initialize W&B
echo ""
echo "📈 Step 7/8: Initializing Weights & Biases..."
wandb login "$WANDB_API_KEY"
echo "✓ W&B authenticated"

# 8. Create directory structure
echo ""
echo "📁 Step 8/8: Creating directory structure..."
mkdir -p /workspace/checkpoints/canary-qwen-medical-lora
mkdir -p /workspace/logs/wandb
mkdir -p /workspace/logs/training
echo "✓ Directories created"

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Start training:"
echo "     python src/train.py --config config/salm_lora.yaml"
echo ""
echo "  2. Monitor training:"
echo "     tail -f logs/training/train.log"
echo ""
echo "  3. View W&B dashboard:"
echo "     https://wandb.ai/<your-username>/medical-asr-poc-michael"
echo ""
