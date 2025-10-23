#!/usr/bin/env python3
"""
Modal-based nvidia/canary-qwen-2.5b fine-tuning script using NVIDIA NeMo SALM.
Canary model comes with LoRA adapters pre-configured (r=128, alpha=256).
"""

import modal
import os

# Create Modal app
app = modal.App("medical-asr-canary-finetune")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "build-essential", "libsndfile1", "ffmpeg",
        "libopenmpi-dev", "sox", "libsox-dev",
    )
    .pip_install(
        "torch==2.2.0",  # PyTorch (NeMo compatible version)
        "wandb==0.18.0",
        "soundfile==0.12.1",
        "numpy==1.26.4",
    )
    .run_commands(
        # Install NeMo from main branch (has SALM support)
        # SALM requires multimodal dependencies (includes megatron.core)
        "pip install Cython packaging",
        "pip install 'nemo_toolkit[asr,multimodal]@git+https://github.com/NVIDIA/NeMo.git@main'",
    )
)


# Mount for dataset (will upload TTS-generated data)
dataset_volume = modal.Volume.from_name("medical-asr-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # or "A100" for faster training
    timeout=7200,  # 2 hour timeout
    volumes={"/data": dataset_volume},
    secrets=[
        modal.Secret.from_dict({
            "WANDB_API_KEY": "ad198476a9d880adf9d176444ff8f2a42616d8bc",
        })
    ],
)
def finetune_canary():
    """Fine-tune Canary-Qwen-2.5b on medical speech dataset using NeMo."""
    import torch
    import wandb
    import json
    from pathlib import Path

    print("\n" + "="*60)
    print("Medical ASR - Canary-Qwen-2.5b Fine-Tuning (NeMo SALM)")
    print("="*60 + "\n")

    # Check GPU
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

    # Initialize W&B
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init(
        project="medical-asr-poc-michael",
        name="canary-qwen-2.5b-medical-finetune",
        config={
            "model": "nvidia/canary-qwen-2.5b",
            "dataset": "tts-medical-200",
            "framework": "NeMo SALM",
            "lora_rank": 128,
            "lora_alpha": 256,
            "epochs": 3,
            "batch_size": 4,
            "platform": "modal",
        }
    )

    # Import NeMo SALM
    print("Importing NeMo SALM...")
    try:
        from nemo.collections.speechlm2.models import SALM
        print("✓ NeMo SALM imported successfully\n")
    except Exception as e:
        print(f"✗ Failed to import NeMo SALM: {e}\n")
        wandb.log({"status": "nemo_import_failed", "error": str(e)})
        wandb.finish()
        return {"status": "failed", "error": "NeMo import failed"}

    # Load Canary model
    print("Loading nvidia/canary-qwen-2.5b from HuggingFace...")
    print("This may take several minutes to download...\n")

    try:
        model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
        print("✓ Model loaded successfully!\n")

        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model parameters:")
        print(f"  Total: {num_params:,}")
        print(f"  Trainable (LoRA): {num_trainable:,}")
        print(f"  Frozen: {num_params - num_trainable:,}\n")

        wandb.log({
            "status": "model_loaded",
            "total_params": num_params,
            "trainable_params": num_trainable,
        })

    except Exception as e:
        print(f"✗ Failed to load model: {e}\n")
        wandb.log({
            "status": "model_load_failed",
            "error": str(e),
        })
        wandb.finish()
        return {"status": "failed", "model_loaded": False, "error": str(e)}

    # Load training data
    print("Loading training data from manifests...")

    # Check if manifests exist in mounted volume
    data_dir = Path("/data")
    train_manifest = data_dir / "train_manifest_tts.json"
    val_manifest = data_dir / "val_manifest_tts.json"

    if not train_manifest.exists():
        print(f"✗ Training manifest not found at {train_manifest}")
        print("Please upload training data to Modal volume first")
        wandb.log({"status": "data_missing"})
        wandb.finish()
        return {"status": "failed", "error": "Training data not found"}

    # Count samples in manifests
    with open(train_manifest) as f:
        train_samples = sum(1 for _ in f)
    with open(val_manifest) as f:
        val_samples = sum(1 for _ in f)

    print(f"✓ Found training data:")
    print(f"  Train: {train_samples} samples")
    print(f"  Val: {val_samples} samples\n")

    # TODO: Implement NeMo fine-tuning
    # NeMo SALM uses a different training API than HuggingFace Transformers
    # Need to:
    # 1. Create NeMo data loader from manifests
    # 2. Configure NeMo trainer
    # 3. Run fine-tuning with LoRA adapters

    print("=" * 60)
    print("NOTE: NeMo SALM fine-tuning API implementation in progress")
    print("=" * 60)
    print("\nCurrent status:")
    print("✓ Model loading: WORKING")
    print("✓ LoRA adapters: PRE-CONFIGURED (r=128, alpha=256)")
    print("✓ Data manifests: READY")
    print("⏳ Training loop: TODO\n")

    print("Next steps:")
    print("1. Implement NeMo data loader for audio manifests")
    print("2. Configure NeMo Trainer with appropriate settings")
    print("3. Run fine-tuning with gradient checkpointing")
    print("4. Evaluate on validation set")
    print("5. Save fine-tuned adapter weights\n")

    wandb.log({
        "status": "implementation_in_progress",
        "train_samples": train_samples,
        "val_samples": val_samples,
    })

    wandb.finish()

    return {
        "status": "partial",
        "model_loaded": True,
        "data_loaded": True,
        "training_implemented": False,
        "train_samples": train_samples,
        "val_samples": val_samples,
    }


@app.local_entrypoint()
def main():
    """Local entrypoint to trigger fine-tuning."""
    result = finetune_canary.remote()
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
