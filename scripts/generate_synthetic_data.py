#!/usr/bin/env python3
"""
Generate synthetic audio data for POC demonstration.
Creates simple audio samples with transcriptions for testing the training pipeline.
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import argparse


def generate_sine_wave_audio(duration=3.0, sample_rate=16000, frequency=440):
    """Generate a simple sine wave audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    return audio.astype(np.float32)


def generate_synthetic_dataset(output_dir: Path, num_samples=100):
    """
    Generate synthetic audio dataset for POC.

    Args:
        output_dir: Output directory for audio files
        num_samples: Number of samples to generate
    """
    print(f"\nGenerating {num_samples} synthetic audio samples...")

    # Sample medical transcriptions
    transcriptions = [
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
    ]

    # Create output directories
    audio_dir = output_dir / "synthetic" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = output_dir.parent / "processed"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Generate audio (varying frequencies for diversity)
        frequency = 400 + (i % 10) * 50
        duration = 2.0 + (i % 5) * 0.5
        audio = generate_sine_wave_audio(duration=duration, frequency=frequency)

        # Save audio file
        audio_path = audio_dir / f"sample_{i:06d}.wav"
        sf.write(audio_path, audio, 16000)

        # Get transcription (cycle through available ones)
        text = transcriptions[i % len(transcriptions)]

        samples.append({
            "audio_filepath": str(audio_path.absolute()),
            "duration": duration,
            "text": text,
        })

    # Split into train/val (80/20)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Create manifests
    train_manifest = manifest_dir / "train_manifest.json"
    val_manifest = manifest_dir / "val_manifest.json"

    with open(train_manifest, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")

    with open(val_manifest, "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\n✓ Synthetic dataset created:")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")
    print(f"  Audio dir: {audio_dir}")
    print(f"  Train manifest: {train_manifest}")
    print(f"  Val manifest: {val_manifest}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic audio dataset for POC")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for raw data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate",
    )

    args = parser.parse_args()

    print(f"Output directory: {args.output_dir.absolute()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generate_synthetic_dataset(args.output_dir, args.num_samples)

    print("\n✓ Dataset generation complete!")


if __name__ == "__main__":
    main()
