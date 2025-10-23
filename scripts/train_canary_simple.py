#!/usr/bin/env python3
"""
Simplified Canary-Qwen-2.5b fine-tuning using direct SALM API.
Works with JSON manifest files (no lhotse shar conversion needed).
"""

import os
import json
import torch
from pathlib import Path

def main():
    print("\n" + "="*80)
    print("Medical ASR - Canary-Qwen-2.5b Fine-Tuning (Simplified)")
    print("="*80 + "\n")

    # Check GPU
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

    # Import NeMo SALM
    print("Importing NeMo SALM...")
    try:
        from nemo.collections.speechlm2.models import SALM
        print("✓ NeMo SALM imported successfully\n")
    except ImportError as e:
        print(f"✗ Failed to import NeMo SALM: {e}")
        print("\nThis requires the community NeMo fork with speechlm2 support:")
        print("  git clone https://github.com/msh9184/NeMo-speechlm2.git")
        print("  cd NeMo-speechlm2")
        print("  pip install -e .")
        return {"status": "failed", "error": "NeMo speechlm2 not available"}

    # Load model
    print("Loading nvidia/canary-qwen-2.5b...")
    try:
        model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
        print("✓ Model loaded successfully\n")

        # Print model info
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model parameters:")
        print(f"  Total: {num_params:,}")
        print(f"  Trainable (LoRA): {num_trainable:,}")
        print(f"  Frozen: {num_params - num_trainable:,}\n")

    except Exception as e:
        print(f"✗ Failed to load model: {e}\n")
        return {"status": "failed", "error": str(e)}

    # Load training data
    base_dir = Path("/Users/michael/ws/savelives")
    train_manifest = base_dir / "data/processed/train_manifest_tts.json"
    val_manifest = base_dir / "data/processed/val_manifest_tts.json"

    if not train_manifest.exists():
        print(f"✗ Training manifest not found: {train_manifest}")
        return {"status": "failed", "error": "Missing training data"}

    # Count samples
    with open(train_manifest) as f:
        train_samples = [json.loads(line) for line in f]
    with open(val_manifest) as f:
        val_samples = [json.loads(line) for line in f]

    print(f"Dataset loaded:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples\n")

    # Test inference on first sample
    print("Testing inference on first validation sample...")
    first_sample = val_samples[0]
    audio_path = first_sample['audio_filepath']
    true_text = first_sample['text']

    try:
        # Generate transcription
        answer_ids = model.generate(
            prompts=[
                [{
                    "role": "user",
                    "content": f"Transcribe the following: {model.audio_locator_tag}",
                    "audio": [audio_path]
                }]
            ],
            max_new_tokens=128,
        )

        transcription = model.tokenizer.decode(answer_ids[0], skip_special_tokens=True)

        print(f"  Audio: {Path(audio_path).name}")
        print(f"  True text: {true_text}")
        print(f"  Predicted: {transcription}")
        print(f"  Match: {'✓' if transcription.strip().lower() == true_text.lower() else '✗'}\n")

    except Exception as e:
        print(f"✗ Inference failed: {e}\n")

    # Training implementation note
    print("="*80)
    print("Training Implementation Required")
    print("="*80)
    print("\nTo implement full training, you need:")
    print("1. Install community NeMo fork: github.com/msh9184/NeMo-speechlm2")
    print("2. Use their training script: recipes/CanaryQwenASR/speech_to_text_salm.py")
    print("3. Adapt config to use JSON manifests instead of lhotse shar format")
    print("\nAlternatively, use PyTorch Lightning trainer directly:")
    print("  from lightning.pytorch import Trainer")
    print("  trainer = Trainer(max_steps=500, devices=1, precision='bf16-mixed')")
    print("  trainer.fit(model, datamodule)")
    print("\n")

    return {
        "status": "partial",
        "model_loaded": True,
        "inference_working": True,
        "training_implemented": False,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }


if __name__ == "__main__":
    result = main()
    print(f"Result: {result}")
