#!/usr/bin/env python3
"""
Medical ASR - Canary-Qwen-2.5b Fine-Tuning using Official NeMo SALM API
Based on: https://github.com/NVIDIA/NeMo/tree/main/examples/speechlm2

This script adapts the official NeMo SALM training approach for our medical dataset
using JSON manifests instead of lhotse shar format.
"""

import os
import json
import torch
from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def main():
    print("\n" + "="*80)
    print("Medical ASR - Canary-Qwen-2.5b Fine-Tuning (Official NeMo SALM)")
    print("="*80 + "\n")

    # Check GPU
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

    # Import NeMo SALM components
    print("Importing NeMo SALM components...")
    try:
        from nemo.collections.speechlm2.models import SALM
        from nemo.collections.speechlm2.data.salm_dataset import SALMDataset
        from nemo.collections.speechlm2.data.datamodule import DataModule
        print("✓ NeMo SALM components imported successfully\n")
    except ImportError as e:
        print(f"✗ Failed to import NeMo SALM: {e}")
        print("\nThis requires official NeMo with speechlm2 support:")
        print("  pip install 'nemo_toolkit[asr,multimodal]@git+https://github.com/NVIDIA/NeMo.git@main'")
        return {"status": "failed", "error": "NeMo speechlm2 not available"}

    # Configuration paths
    base_dir = Path("/Users/michael/ws/savelives")
    train_manifest = base_dir / "data/processed/train_manifest_tts.json"
    val_manifest = base_dir / "data/processed/val_manifest_tts.json"
    config_path = base_dir / "configs/salm_medical_asr.yaml"

    if not train_manifest.exists():
        print(f"✗ Training manifest not found: {train_manifest}")
        return {"status": "failed", "error": "Missing training data"}

    # Load configuration
    print(f"Loading configuration from {config_path}...")
    if not config_path.exists():
        print(f"✗ Configuration file not found: {config_path}")
        print("Creating default configuration...")

        # Create default config based on official NeMo SALM
        config = OmegaConf.create({
            "name": "Medical-ASR-Canary-Qwen",
            "model": {
                "pretrained_weights": "nvidia/canary-qwen-2.5b",
                "audio_locator_tag": "<|audio|>",
                "prompt_format": "canary_qwen",
                "lora": {
                    "task_type": "CAUSAL_LM",
                    "r": 128,
                    "lora_alpha": 256,
                    "lora_dropout": 0.01,
                    "target_modules": ["q_proj", "v_proj"],
                    "inference_mode": False,
                },
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "lr": 5e-4,
                    "betas": [0.9, 0.98],
                    "weight_decay": 1e-3,
                },
                "lr_scheduler": {
                    "_target_": "nemo.core.optim.lr_scheduler.CosineAnnealing",
                    "warmup_steps": 50,
                    "min_lr": 1e-6,
                    "max_steps": 500,
                },
            },
            "trainer": {
                "devices": 1,
                "accelerator": "gpu",
                "precision": "bf16-mixed",
                "max_steps": 500,
                "val_check_interval": 50,
                "log_every_n_steps": 10,
                "gradient_clip_val": 1.0,
                "accumulate_grad_batches": 4,
            },
            "data": {
                "train_ds": {
                    "manifest_filepath": str(train_manifest),
                    "sample_rate": 16000,
                    "batch_size": 4,
                    "shuffle": True,
                    "num_workers": 4,
                    "prompt_format": "canary_qwen",
                    "asr_context_prompt": "Transcribe the following: ",
                },
                "validation_ds": {
                    "manifest_filepath": str(val_manifest),
                    "sample_rate": 16000,
                    "batch_size": 1,
                    "shuffle": False,
                    "num_workers": 4,
                    "prompt_format": "canary_qwen",
                    "asr_context_prompt": "Transcribe the following: ",
                },
            },
            "exp_manager": {
                "exp_dir": str(base_dir / "experiments"),
                "name": "Medical-ASR-Canary-Qwen",
                "create_tensorboard_logger": True,
                "create_checkpoint_callback": True,
                "create_wandb_logger": True,
                "wandb_logger_kwargs": {
                    "project": "medical-asr-poc-michael",
                    "name": "canary-qwen-finetune",
                },
            },
        })
    else:
        config = OmegaConf.load(config_path)

    print("✓ Configuration loaded\n")

    # Count training samples
    with open(train_manifest) as f:
        train_samples = [json.loads(line) for line in f]
    with open(val_manifest) as f:
        val_samples = [json.loads(line) for line in f]

    print(f"Dataset:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples\n")

    # Initialize model
    print("Loading SALM model...")
    try:
        # Load pretrained model with LoRA adapters
        model = SALM.from_pretrained(
            config.model.pretrained_weights,
            override_config_path=None,  # Use default model config
        )

        # Configure LoRA if specified
        if hasattr(config.model, 'lora') and config.model.lora:
            print("  Configuring LoRA adapters...")
            # LoRA is already built into canary-qwen-2.5b, just verify
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable parameters: {num_trainable:,} ({num_trainable / 1e6:.1f}M)")

        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}\n")
        return {"status": "failed", "error": str(e)}

    # Create dataset
    print("Creating SALMDataset...")
    try:
        dataset = SALMDataset(
            tokenizer=model.tokenizer,
            prompt_format=config.model.prompt_format,
        )
        print("✓ Dataset created\n")
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        print("\nNote: SALMDataset may require lhotse shar format.")
        print("For POC, using simple inference approach instead of full training pipeline.")
        print("See train_canary_simple.py for inference-only approach.\n")
        return {"status": "partial", "error": "Dataset creation not supported with JSON manifests"}

    # Create DataModule
    print("Creating DataModule...")
    try:
        datamodule = DataModule(
            cfg=config.data,
            tokenizer=model.tokenizer,
            dataset=dataset,
        )
        print("✓ DataModule created\n")
    except Exception as e:
        print(f"✗ Failed to create DataModule: {e}\n")
        return {"status": "failed", "error": str(e)}

    # Setup W&B logging
    wandb_logger = None
    if config.exp_manager.create_wandb_logger:
        print("Initializing Weights & Biases...")
        wandb_logger = WandbLogger(
            name=config.exp_manager.wandb_logger_kwargs.name,
            project=config.exp_manager.wandb_logger_kwargs.project,
            save_dir=config.exp_manager.exp_dir,
        )
        print("✓ W&B logger initialized\n")

    # Setup checkpointing
    checkpoint_callback = None
    if config.exp_manager.create_checkpoint_callback:
        checkpoint_dir = Path(config.exp_manager.exp_dir) / config.exp_manager.name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="{epoch}-{step}-{val_wer:.4f}",
            monitor="val_wer",
            mode="min",
            save_top_k=3,
            save_last=True,
        )
        print(f"✓ Checkpointing to {checkpoint_dir}\n")

    # Create trainer
    print("Creating PyTorch Lightning Trainer...")
    trainer = Trainer(
        devices=config.trainer.devices,
        accelerator=config.trainer.accelerator,
        precision=config.trainer.precision,
        max_steps=config.trainer.max_steps,
        val_check_interval=config.trainer.val_check_interval,
        log_every_n_steps=config.trainer.log_every_n_steps,
        gradient_clip_val=config.trainer.gradient_clip_val,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=[checkpoint_callback] if checkpoint_callback else [],
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    print("✓ Trainer created\n")

    # Start training
    print("="*80)
    print("Starting Training...")
    print("="*80 + "\n")

    try:
        trainer.fit(model, datamodule)
        print("\n" + "="*80)
        print("✓ Training Complete!")
        print("="*80 + "\n")

        return {
            "status": "success",
            "final_step": trainer.global_step,
            "best_checkpoint": checkpoint_callback.best_model_path if checkpoint_callback else None,
        }

    except Exception as e:
        print(f"\n✗ Training failed: {e}\n")
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    result = main()
    print(f"\nFinal result: {result}")
