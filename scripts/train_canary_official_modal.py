#!/usr/bin/env python3
"""
Medical ASR - Canary-Qwen-2.5b Fine-Tuning on Modal (Official NeMo SALM)
Based on: https://github.com/NVIDIA/NeMo/tree/main/examples/speechlm2

This script deploys the official NeMo SALM training approach to Modal GPU infrastructure.
"""

import os
import modal

# Create Modal app
app = modal.App("medical-asr-canary-official")

# Create Modal image with official NeMo dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "build-essential",
        "libsndfile1",
        "ffmpeg",
        "libopenmpi-dev",
        "sox",
        "libsox-dev",
    )
    .pip_install(
        "torch==2.2.0",
        "wandb==0.18.0",
        "soundfile==0.12.1",
        "numpy==1.26.4",
        "pytorch-lightning==2.1.0",
        "omegaconf==2.3.0",
    )
    .run_commands(
        # Install Cython and packaging first
        "pip install Cython packaging",
        # Install official NeMo with speechlm2 support
        "pip install 'nemo_toolkit[asr,multimodal]@git+https://github.com/NVIDIA/NeMo.git@main'",
    )
)

# Create Modal volume for dataset and checkpoints
volume = modal.Volume.from_name("medical-asr-data", create_if_missing=True)

# Training configuration
MEDICAL_PHRASES = [
    "patient presents with chest pain",
    "blood pressure is one twenty over eighty",
    "heart rate is seventy two beats per minute",
    "patient has a history of diabetes",
    "prescribed medication for hypertension",
    "follow up in two weeks",
    "patient reports shortness of breath",
    "temperature is ninety eight point six degrees",
    "no known drug allergies",
    "family history of heart disease",
    "patient complains of headache and dizziness",
    "respiratory rate is sixteen breaths per minute",
    "oxygen saturation is ninety eight percent",
    "patient is alert and oriented",
    "administered aspirin eighty one milligrams",
    "ordered complete blood count",
    "chest x ray shows no abnormalities",
    "patient has swelling in lower extremities",
    "history of coronary artery disease",
    "patient is taking lisinopril daily",
]


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,  # 2 hours
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_canary_salm():
    """
    Fine-tune nvidia/canary-qwen-2.5b using official NeMo SALM training pipeline.
    """
    import json
    import torch
    from pathlib import Path
    from omegaconf import OmegaConf
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    print("\n" + "="*80)
    print("Medical ASR - Canary-Qwen-2.5b Fine-Tuning (Official NeMo SALM + Modal)")
    print("="*80 + "\n")

    # Initialize W&B
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        import wandb
        wandb.login(key=wandb_api_key)
        print("✓ Weights & Biases initialized\n")

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
        return {"status": "failed", "error": "NeMo speechlm2 not available"}

    # Data paths (Modal volume)
    data_dir = Path("/data")
    train_manifest = data_dir / "train_manifest.json"
    val_manifest = data_dir / "val_manifest.json"
    checkpoint_dir = data_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Check if manifests exist, if not create dummy ones for testing
    if not train_manifest.exists():
        print("⚠ Training manifest not found, creating dummy data for testing...")
        # This is just a placeholder - in production you'd upload real data
        train_data = [
            {"audio_filepath": "/tmp/sample.wav", "duration": 3.0, "text": phrase}
            for phrase in MEDICAL_PHRASES[:10]
        ]
        val_data = [
            {"audio_filepath": "/tmp/sample.wav", "duration": 3.0, "text": phrase}
            for phrase in MEDICAL_PHRASES[10:13]
        ]

        with open(train_manifest, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')

        with open(val_manifest, 'w') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')

        print(f"✓ Created dummy manifests:\n  Train: {train_manifest}\n  Val: {val_manifest}\n")

    # Configuration
    config = OmegaConf.create({
        "name": "Medical-ASR-Canary-Qwen-Modal",
        "model": {
            "pretrained_weights": "nvidia/canary-qwen-2.5b",
            "audio_locator_tag": "<|audio|>",
            "prompt_format": "canary_qwen",
            "log_prediction_train": True,
            "log_prediction_train_samples": 5,
            "log_prediction_train_interval": 50,
            "log_prediction_valid": True,
            "log_prediction_valid_samples": 20,
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
                "foreach": True,
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
                "token_equivalent_duration": 0.08,
                "min_duration": 0.3,
                "max_duration": 40.0,
            },
            "validation_ds": {
                "manifest_filepath": str(val_manifest),
                "sample_rate": 16000,
                "batch_size": 1,
                "shuffle": False,
                "num_workers": 4,
                "prompt_format": "canary_qwen",
                "asr_context_prompt": "Transcribe the following: ",
                "token_equivalent_duration": 0.08,
            },
        },
    })

    print("Configuration:")
    print(f"  Model: {config.model.pretrained_weights}")
    print(f"  LoRA: r={config.model.lora.r}, alpha={config.model.lora.lora_alpha}")
    print(f"  Training: {config.trainer.max_steps} steps")
    print(f"  Batch size: {config.data.train_ds.batch_size}")
    print(f"  GPU precision: {config.trainer.precision}\n")

    # Count training samples
    with open(train_manifest) as f:
        train_samples = sum(1 for _ in f)
    with open(val_manifest) as f:
        val_samples = sum(1 for _ in f)

    print(f"Dataset:")
    print(f"  Train: {train_samples} samples")
    print(f"  Val: {val_samples} samples\n")

    # Initialize model
    print("Loading SALM model...")
    try:
        model = SALM.from_pretrained(config.model.pretrained_weights)

        # Verify LoRA parameters
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {num_trainable:,} ({num_trainable / 1e6:.1f}M)")
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}\n")
        return {"status": "failed", "error": str(e)}

    # Create dataset and datamodule
    print("Creating SALMDataset...")
    try:
        dataset = SALMDataset(
            tokenizer=model.tokenizer,
            prompt_format=config.model.prompt_format,
        )
        print("✓ Dataset created\n")

        print("Creating DataModule...")
        datamodule = DataModule(
            cfg=config.data,
            tokenizer=model.tokenizer,
            dataset=dataset,
        )
        print("✓ DataModule created\n")
    except Exception as e:
        print(f"✗ Failed to create dataset/datamodule: {e}")
        print("\nNote: SALMDataset may require lhotse shar format.")
        print("Falling back to simple inference test...\n")

        # Try simple inference test instead
        try:
            # Create dummy audio for testing
            import soundfile as sf
            import numpy as np

            # Generate 3 seconds of silence
            dummy_audio = np.zeros(16000 * 3, dtype=np.float32)
            sf.write("/tmp/sample.wav", dummy_audio, 16000)

            print("Testing inference on dummy audio...")
            answer_ids = model.generate(
                prompts=[
                    [{
                        "role": "user",
                        "content": f"Transcribe the following: {model.audio_locator_tag}",
                        "audio": ["/tmp/sample.wav"]
                    }]
                ],
                max_new_tokens=128,
            )

            transcription = model.tokenizer.decode(answer_ids[0], skip_special_tokens=True)
            print(f"  Predicted: {transcription}")
            print("\n✓ Inference working\n")

            return {
                "status": "partial",
                "inference_working": True,
                "training_implemented": False,
                "error": "Dataset/DataModule creation requires lhotse shar format",
            }
        except Exception as e2:
            print(f"✗ Inference test also failed: {e2}\n")
            return {"status": "failed", "error": str(e2)}

    # Setup W&B logging
    wandb_logger = WandbLogger(
        name="canary-qwen-medical-modal",
        project="medical-asr-poc-michael",
        save_dir=str(data_dir),
    )
    print("✓ W&B logger initialized\n")

    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="medical-asr-{epoch}-{step}-{val_wer:.4f}",
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
        callbacks=[checkpoint_callback],
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

        # Commit volume to persist checkpoints
        volume.commit()

        return {
            "status": "success",
            "final_step": trainer.global_step,
            "best_checkpoint": checkpoint_callback.best_model_path,
            "wandb_url": wandb_logger.experiment.url if wandb_logger else None,
        }

    except Exception as e:
        print(f"\n✗ Training failed: {e}\n")
        return {"status": "failed", "error": str(e)}


@app.local_entrypoint()
def main():
    """
    Local entrypoint to launch training on Modal.
    """
    print("Launching Canary-Qwen training on Modal A10G GPU...")
    result = train_canary_salm.remote()
    print(f"\nTraining result: {result}")
    return result
