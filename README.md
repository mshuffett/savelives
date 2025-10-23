# Medical ASR Fine-Tuning POC

Proof-of-concept for fine-tuning **NVIDIA Canary-Qwen-2.5B** on medical dictation and conversational doctor audio using LoRA/PEFT.

## 🎯 Project Overview

This POC demonstrates domain adaptation of a state-of-the-art speech foundation model to clinical dictation using:
- **Model**: nvidia/canary-qwen-2.5b
- **Method**: LoRA (Low-Rank Adaptation) via NeMo PEFT
- **Infrastructure**: RunPod GPU pods (A100 40GB/80GB)
- **Observability**: Weights & Biases

**Goal**: Show WER improvement on medical vocabulary with minimal computational cost.

## 📁 Directory Structure

```
savelives/
├── PROJECT_BRIEF.md          # Detailed project requirements
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .env.example               # Template for secrets
│
├── config/
│   └── salm_lora.yaml        # NeMo training configuration
│
├── scripts/
│   ├── runpod_manager.py     # RunPod API wrapper
│   ├── download_data.py      # Dataset download & preprocessing
│   └── setup_environment.sh  # Environment setup for RunPod
│
├── src/
│   ├── train.py              # Main training entry point
│   ├── data/                 # Dataset utilities
│   ├── model/                # Model configuration
│   └── evaluation/           # WER evaluation
│
├── data/                      # .gitignored
│   ├── raw/                  # Downloaded datasets
│   └── processed/            # NeMo manifests
│
├── checkpoints/               # .gitignored
│   └── canary-qwen-medical-lora/
│
└── logs/                      # .gitignored
    ├── wandb/
    └── training/
```

## 🚀 Quick Start

### 1. Local Setup

```bash
# Clone repository
git clone https://github.com/mshuffett/savelives.git
cd savelives

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies locally
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
#   - RUNPOD_API_KEY
#   - WANDB_API_KEY
```

### 2. Download Dataset

```bash
# Download Hani89 synthetic medical speech (2000 samples for POC)
python scripts/download_data.py --dataset hani89 --max-samples 2000

# This creates:
#   data/raw/hani89/audio/*.wav
#   data/processed/train_manifest.json
#   data/processed/val_manifest.json
```

### 3. Launch RunPod Training

```bash
# Create GPU pod
python scripts/runpod_manager.py create medical-asr-training \
    --gpu "NVIDIA A100 80GB" \
    --disk 50

# SSH into pod (get command from output)
ssh root@<pod-ip> -p <port>

# On pod: Setup environment
cd /workspace
git clone https://github.com/mshuffett/savelives.git
cd savelives
bash scripts/setup_environment.sh

# Copy .env file with secrets
nano .env  # Paste your keys

# Download data on pod
python scripts/download_data.py --dataset hani89 --max-samples 2000

# Start training
python src/train.py --config config/salm_lora.yaml
```

### 4. Monitor Training

- **Weights & Biases**: https://wandb.ai/<your-username>/medical-asr-poc-michael
- **Checkpoints**: `/workspace/checkpoints/canary-qwen-medical-lora/`
- **Logs**: `/workspace/logs/`

### 5. RunPod Continuous Command (auto-resume)

For unattended training with auto-resume and backoff‑retry, use the loop script:

```bash
# On the pod inside the repo root
bash runpod/start.sh
```

This will:
- Ensure dependencies are installed
- Download the 2k‑sample subset if missing
- Train or resume from the latest checkpoint in `/workspace/checkpoints/canary-qwen-medical-lora`
- Run a quick WER eval on ~200 validation samples
- On failure, back off and retry without exiting

## 🔧 Configuration

### LoRA Parameters (`config/salm_lora.yaml`)

```yaml
model:
  restore_from_path: nvidia/canary-qwen-2.5b
  lora:
    task_type: CAUSAL_LM
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1

trainer:
  devices: 1
  accelerator: gpu
  precision: bf16
  max_epochs: 5
  gradient_clip_val: 1.0
  accumulate_grad_batches: 4

data:
  train_ds:
    manifest_filepath: /workspace/data/processed/train_manifest.json
    sample_rate: 16000
    batch_size: 4
  validation_ds:
    manifest_filepath: /workspace/data/processed/val_manifest.json
    sample_rate: 16000
    batch_size: 4
```

### Estimated Costs

- **GPU**: A100 40GB @ ~$1.50/hr
- **Training time**: 8-12 hours for 2000 samples
- **Total**: ~$12-18 for complete POC

## 📊 Dataset Information

**Primary**: [Hani89/synthetic-medical-speech-dataset](https://huggingface.co/datasets/Hani89/synthetic-medical-speech-dataset)
- 10K-100K synthetic medical speech samples
- Generated via TTS from medical text
- Privacy-safe (no real patient data)
- **Selected for POC**: Largest available, good medical vocabulary coverage

**Secondary** (validation): [yfyeung/medical](https://huggingface.co/datasets/yfyeung/medical)
- 272 simulated patient-doctor conversations
- Real conversational structure
- Too small for primary training

See `docs/dataset_research.md` for detailed comparison.

## 🎓 Technical References

- **Model card**: https://huggingface.co/nvidia/canary-qwen-2.5b
- **NeMo examples**: https://github.com/NVIDIA/NeMo/tree/main/examples/speechlm2
- **NeMo docs**: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/speechlm2/models.html

## 📝 Development Notes

### Training Workflow

1. **Freeze LLM weights** - Only train speech encoder and LoRA adapters
2. **LoRA rank 16** - Balance between expressiveness and efficiency
3. **Batch size 4** - Fits in A100 40GB with gradient accumulation
4. **Checkpoint every 500 steps** - Enables recovery from failures
5. **Auto-resume** - Continue from latest checkpoint on pod restart

### Key Files

- `config/salm_lora.yaml` - All hyperparameters and paths
- `scripts/runpod_manager.py` - Pod lifecycle management
- `scripts/download_data.py` - Dataset → NeMo manifest conversion
- `src/train.py` - Training entry point (wraps NeMo)

## 🔄 Next Steps

After POC validation:

1. **Scale up dataset** - Use full Hani89 dataset (~10K+ samples)
2. **Add real data** - Incorporate actual medical dictations (with proper consent/de-identification)
3. **Multi-GPU training** - Scale to 4-8 GPUs for faster iteration
4. **Hyperparameter sweep** - Optimize LoRA rank, learning rate, batch size
5. **Production deployment** - Package model for inference API

## 🐛 Troubleshooting

### Out of Memory

```yaml
# Reduce batch size in config/salm_lora.yaml
train_ds:
  batch_size: 2  # down from 4
trainer:
  accumulate_grad_batches: 8  # up from 4
```

### NeMo Installation Issues

```bash
# Uninstall and reinstall from source
pip uninstall nemo_toolkit
pip install "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"
```

### Wandb Login Failures

```bash
# Re-authenticate
wandb login --relogin <your-api-key>
```

## 📄 License

See LICENSE file.

## 🤝 Contributing

This is a POC project. For production use, please consult with the team.

---

**Project Status**: Initial Setup Complete
**Last Updated**: 2025-10-22
