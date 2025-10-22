# RunPod Training Pod - Status & Instructions

## 🟢 Pod Status: RUNNING

**Pod Details:**
- **ID**: `p24f468c33bbsg`
- **Name**: medical-asr-poc
- **GPU**: NVIDIA A100 80GB PCIe
- **Memory**: 188 GB RAM
- **vCPUs**: 16
- **Storage**: 50 GB
- **Cost**: $1.64/hour
- **Image**: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

**Created**: Wed Oct 22 2025 22:56:06 GMT+0000

---

## 🚀 Quick Start - Run Training

### Option 1: Web Terminal (Recommended)

1. **Access RunPod Console**:
   - Go to: https://www.runpod.io/console/pods
   - Find pod: `medical-asr-poc`
   - Click "Connect" → "Start Web Terminal"

2. **Paste and Run Init Script**:
   ```bash
   curl -sSL https://raw.githubusercontent.com/mshuffett/savelives/main/scripts/runpod_init_autonomous.sh | bash
   ```

   Or manually:
   ```bash
   # Copy the contents of scripts/runpod_init_autonomous.sh
   # Paste into terminal and press Enter
   ```

3. **Monitor Progress**:
   - Script will output progress in real-time
   - W&B dashboard: https://wandb.ai/<your-username>/medical-asr-poc-michael
   - Logs saved to: `/workspace/autonomous_setup.log`

### Option 2: Jupyter Notebook

1. **Access Jupyter**:
   - Internal URL: http://100.65.15.28:60590
   - Access via RunPod's "Connect" → "Jupyter Lab"

2. **Open Terminal in Jupyter**:
   - File → New → Terminal

3. **Run Init Script** (same as Option 1)

---

## 📊 What the Script Does

The autonomous init script (`runpod_init_autonomous.sh`) will:

1. ✅ **Install system dependencies** (ffmpeg, sox, etc.)
2. ✅ **Clone the repository** from GitHub
3. ✅ **Install PyTorch** with CUDA 11.8 support
4. ✅ **Install NVIDIA NeMo** from source (~5-8 minutes)
5. ✅ **Install all Python dependencies**
6. ✅ **Verify GPU access** (A100 80GB)
7. ✅ **Authenticate with Weights & Biases**
8. ✅ **Download dataset** (2000 medical speech samples)
9. ✅ **Create checkpoint directories**
10. ✅ **Start training** (POC demonstration)

**Total Setup Time**: ~15-20 minutes

---

## 📈 Monitoring Training

### Weights & Biases
- **Project**: medical-asr-poc-michael
- **Dashboard**: https://wandb.ai/<your-username>/medical-asr-poc-michael
- **Metrics Logged**:
  - Dataset statistics
  - GPU utilization
  - Training status
  - Setup completion

### Log Files
- **Setup log**: `/workspace/autonomous_setup.log`
- **Training log**: `/workspace/logs/training/training.log`
- **W&B logs**: `/workspace/logs/wandb/`

### Check Status from Terminal
```bash
# View setup progress
tail -f /workspace/autonomous_setup.log

# View training progress
tail -f /workspace/logs/training/training.log

# Check GPU usage
nvidia-smi

# Check W&B runs
wandb status
```

---

## 🛠 Manual Commands (If Needed)

### Navigate to Project
```bash
cd /workspace/savelives
```

### Run Individual Steps
```bash
# Download dataset only
python scripts/download_data.py --dataset hani89 --max-samples 2000

# Run training wrapper
python run_training.py

# Check GPU
nvidia-smi

# View W&B runs
wandb runs medical-asr-poc-michael
```

### Troubleshooting
```bash
# Check Python environment
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import nemo; print(f'NeMo: {nemo.__version__}')"

# Reinstall dependencies
pip install -r requirements.txt

# Clear cache and retry
rm -rf /workspace/savelives
# Then re-run init script
```

---

## 💰 Cost Management

**Current Rate**: $1.64/hour

**Estimated Costs**:
- Setup (20 min): ~$0.55
- Training POC (1 hour): ~$1.64
- **Total for complete POC**: ~$2-3

### Stop Pod When Done
```bash
# From local machine:
export RUNPOD_API_KEY="rpa_VVQLJLKLAUIDLJ6PPLYBBGQIWDLN70J5BCP5L36V1dht52"
python scripts/runpod_manager.py stop p24f468c33bbsg

# Or via web:
https://www.runpod.io/console/pods → Stop pod
```

### Terminate Pod (Permanent)
```bash
# From local machine:
python scripts/runpod_manager.py terminate p24f468c33bbsg

# Or via web:
https://www.runpod.io/console/pods → Terminate pod
```

---

## 📝 Next Steps After POC

1. **Review W&B Metrics**: Verify setup completion
2. **Implement Full NeMo Training**: Replace simplified wrapper with actual SALM training
3. **Scale Dataset**: Increase from 2000 to full dataset
4. **Hyperparameter Tuning**: Optimize LoRA rank, learning rate
5. **Production Deployment**: Package for inference

---

## 🔗 Important Links

- **RunPod Console**: https://www.runpod.io/console/pods
- **GitHub Repo**: https://github.com/mshuffett/savelives
- **W&B Project**: https://wandb.ai/<your-username>/medical-asr-poc-michael
- **NeMo Docs**: https://docs.nvidia.com/nemo-framework/
- **Model Card**: https://huggingface.co/nvidia/canary-qwen-2.5b

---

**Status**: Pod is RUNNING and ready for training initialization
**Last Updated**: 2025-10-22
**Continuous Runner Active**: Yes (PID 40791, check-ins every 3 min)
