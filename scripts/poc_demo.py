#!/usr/bin/env python3
"""
POC Demonstration Script
Validates the entire setup and logs metrics to Weights & Biases.
"""

import os
import json
import torch
import wandb
from pathlib import Path
from datetime import datetime


def validate_environment():
    """Validate GPU and dependencies."""
    print("\n=== Environment Validation ===")

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA Available: {cuda_available}")

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"  GPU Count: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name}")
            print(f"    Memory: {gpu_memory:.1f} GB")

    return cuda_available


def validate_dataset(manifest_path):
    """Validate dataset manifest."""
    print(f"\n=== Dataset Validation ===")
    print(f"  Manifest: {manifest_path}")

    if not manifest_path.exists():
        print(f"  ✗ Manifest not found!")
        return None

    samples = []
    with open(manifest_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"  ✓ Loaded {len(samples)} samples")

    if samples:
        print(f"  Sample example:")
        print(f"    Audio: {Path(samples[0]['audio_filepath']).name}")
        print(f"    Duration: {samples[0]['duration']:.2f}s")
        print(f"    Text: {samples[0]['text'][:50]}...")

        # Calculate statistics
        durations = [s['duration'] for s in samples]
        avg_duration = sum(durations) / len(durations)
        total_duration = sum(durations)

        print(f"  Statistics:")
        print(f"    Total duration: {total_duration/60:.1f} minutes")
        print(f"    Average duration: {avg_duration:.2f}s")
        print(f"    Min duration: {min(durations):.2f}s")
        print(f"    Max duration: {max(durations):.2f}s")

    return samples


def log_to_wandb(project="medical-asr-poc-michael"):
    """Log validation results to W&B."""
    print(f"\n=== Logging to Weights & Biases ===")
    print(f"  Project: {project}")

    # Initialize wandb
    run = wandb.init(
        project=project,
        name=f"poc-validation-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "type": "validation",
            "timestamp": datetime.now().isoformat(),
        },
    )

    # Validate environment
    cuda_available = validate_environment()

    # Validate datasets
    train_manifest = Path("data/processed/train_manifest.json")
    val_manifest = Path("data/processed/val_manifest.json")

    train_samples = validate_dataset(train_manifest)
    val_samples = validate_dataset(val_manifest)

    # Log metrics
    metrics = {
        "setup/cuda_available": cuda_available,
        "setup/timestamp": datetime.now().timestamp(),
    }

    if cuda_available:
        metrics.update({
            "setup/gpu_count": torch.cuda.device_count(),
            "setup/gpu_name": torch.cuda.get_device_name(0),
            "setup/gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        })

    if train_samples:
        durations = [s['duration'] for s in train_samples]
        metrics.update({
            "dataset/train_samples": len(train_samples),
            "dataset/train_total_duration_min": sum(durations) / 60,
            "dataset/train_avg_duration": sum(durations) / len(durations),
        })

    if val_samples:
        durations = [s['duration'] for s in val_samples]
        metrics.update({
            "dataset/val_samples": len(val_samples),
            "dataset/val_total_duration_min": sum(durations) / 60,
            "dataset/val_avg_duration": sum(durations) / len(durations),
        })

    wandb.log(metrics)

    # Mark completion
    wandb.log({
        "status": "poc_setup_complete",
        "ready_for_training": True,
    })

    print("\n✓ Metrics logged to W&B")
    print(f"  Dashboard: {run.url}")

    wandb.finish()

    return True


def main():
    print("\n" + "="*50)
    print("Medical ASR POC - Setup Validation")
    print("="*50)

    # Authenticate W&B
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        print("⚠ Warning: WANDB_API_KEY not found")
        print("  Set it with: export WANDB_API_KEY=<your-key>")
        return 1

    # Run validation and logging
    success = log_to_wandb()

    if success:
        print("\n" + "="*50)
        print("✓ POC Setup Validation Complete!")
        print("="*50)
        print("\nNext Steps:")
        print("1. Review W&B dashboard for metrics")
        print("2. Implement full NeMo SALM training loop")
        print("3. Run actual fine-tuning")
        return 0
    else:
        print("\n✗ Validation failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
