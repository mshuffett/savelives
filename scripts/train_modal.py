#!/usr/bin/env python3
"""
Modal-based Whisper fine-tuning script.
Much simpler than RunPod - no SSH, no manual pod management.
"""

import modal
import os

# Create Modal app
app = modal.App("medical-asr-whisper")

# Define the container image with all dependencies
# Fix numpy/wandb/pyarrow compatibility
# datasets 2.14.0 requires pyarrow<14.0.0 (PyExtensionType was removed in pyarrow 14+)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy==1.26.4",  # Pin numpy to 1.x for wandb compatibility
        "pyarrow==13.0.0",  # Pin to version with PyExtensionType for datasets 2.14.0
        "torch==2.1.0",
        "transformers==4.35.0",
        "datasets==2.14.0",
        "soundfile==0.12.1",
        "librosa==0.10.1",
        "wandb==0.18.0",  # Newer wandb compatible with numpy 1.x
        "accelerate==0.24.0",
    )
    .apt_install("libsndfile1", "ffmpeg")
)


@app.function(
    image=image,
    gpu="A10G",  # or "A100" for faster training
    timeout=3600,  # 1 hour timeout
    secrets=[
        modal.Secret.from_dict({
            "WANDB_API_KEY": "ad198476a9d880adf9d176444ff8f2a42616d8bc"
        })
    ],
)
def train_whisper():
    """Train Whisper model on medical dataset."""
    import json
    import torch
    import wandb
    from pathlib import Path
    from datasets import Dataset, Audio
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    print("\n" + "="*60)
    print("Medical ASR - Whisper Fine-Tuning (Modal)")
    print("="*60 + "\n")

    # Check GPU
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Initialize W&B
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project="medical-asr-poc-michael",
        name="whisper-tiny-medical-modal",
        config={
            "model": "openai/whisper-tiny",
            "dataset": "synthetic-medical-200",
            "epochs": 3,
            "batch_size": 8,
            "platform": "modal",
        }
    )

    # Data collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    # Load manifest dataset
    def load_manifest_dataset(manifest_data: str):
        samples = []
        for line in manifest_data.strip().split('\n'):
            if line:
                sample = json.loads(line)
                # Note: In Modal, we'll need to download audio files separately
                # For now, using placeholder
                samples.append({
                    'audio': sample['audio_filepath'],
                    'text': sample['text'],
                })
        return Dataset.from_list(samples)

    # For now, create a minimal synthetic dataset inline
    # In production, you'd download from Google Drive
    print("Creating minimal synthetic dataset for testing...")

    import numpy as np
    import soundfile as sf
    import tempfile

    temp_dir = tempfile.mkdtemp()
    samples = []

    medical_phrases = [
        "patient presents with chest pain",
        "blood pressure is one twenty over eighty",
        "heart rate is seventy two beats per minute",
        "patient has a history of diabetes",
        "prescribed medication for hypertension",
    ]

    for i in range(20):  # Small dataset for quick test
        # Generate simple sine wave
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Save to temp file
        audio_path = f"{temp_dir}/sample_{i:03d}.wav"
        sf.write(audio_path, audio.astype(np.float32), sample_rate)

        samples.append({
            'audio': audio_path,
            'text': medical_phrases[i % len(medical_phrases)],
        })

    # Create dataset
    dataset = Dataset.from_list(samples).cast_column("audio", Audio(sampling_rate=16000))

    # Split train/val
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split['train']
    val_dataset = split['test']

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Load model
    print("\nLoading Whisper-tiny model...")
    model_name = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Prepare dataset
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    print("Preparing datasets...")
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names)

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="/tmp/whisper-medical",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=10,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        push_to_hub=False,
        fp16=True,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    # Train!
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60 + "\n")

    trainer.train()

    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)

    wandb.finish()

    return "Training completed successfully!"


@app.local_entrypoint()
def main():
    """Local entrypoint to trigger training."""
    result = train_whisper.remote()
    print(f"\n{result}")


if __name__ == "__main__":
    main()
