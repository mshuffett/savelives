#!/usr/bin/env python3
"""
Simple Whisper fine-tuning script for medical ASR POC.
Uses Whisper-tiny model with our synthetic dataset.
"""

import os
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


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text models."""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def load_manifest_dataset(manifest_path: str):
    """Load dataset from NeMo manifest format."""
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            samples.append({
                'audio': sample['audio_filepath'],
                'text': sample['text'],
            })

    return Dataset.from_list(samples).cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch, processor):
    """Prepare batch for training."""
    # Load and process audio
    audio = batch["audio"]

    # Compute log-Mel input features
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids

    return batch


def main():
    print("\n" + "="*60)
    print("Medical ASR POC - Whisper Fine-Tuning")
    print("="*60 + "\n")

    # Initialize W&B
    wandb_key = os.getenv("WANDB_API_KEY", "ad198476a9d880adf9d176444ff8f2a42616d8bc")
    wandb.login(key=wandb_key)

    wandb.init(
        project="medical-asr-poc-michael",
        name="whisper-tiny-medical-poc",
        config={
            "model": "openai/whisper-tiny",
            "dataset": "synthetic-medical-200",
            "epochs": 3,
            "batch_size": 8,
        }
    )

    # Load model and processor
    print("Loading Whisper-tiny model...")
    model_name = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Load datasets
    print("Loading training dataset...")
    train_dataset = load_manifest_dataset("data/processed/train_manifest.json")
    print(f"  Train samples: {len(train_dataset)}")

    print("Loading validation dataset...")
    val_dataset = load_manifest_dataset("data/processed/val_manifest.json")
    print(f"  Val samples: {len(val_dataset)}")

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=train_dataset.column_names
    )

    val_dataset = val_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=val_dataset.column_names
    )

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints/whisper-tiny-medical",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=50,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        push_to_hub=False,
        fp16=True,  # Use mixed precision
    )

    # Initialize trainer
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

    # Save final model
    print("\nSaving final model...")
    trainer.save_model("./checkpoints/whisper-tiny-medical/final")

    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    print(f"\nCheckpoint saved to: ./checkpoints/whisper-tiny-medical/final")
    print(f"W&B Dashboard: https://wandb.ai/compose/medical-asr-poc-michael")

    wandb.finish()


if __name__ == "__main__":
    main()
