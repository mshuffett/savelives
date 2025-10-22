#!/usr/bin/env python3
"""
Dataset Download and Preprocessing
Downloads medical speech datasets and converts them to NeMo manifest format.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def create_nemo_manifest(samples: List[Dict], output_path: Path):
    """
    Create NeMo manifest file from samples.

    NeMo manifest format (JSON lines):
    {
        "audio_filepath": "/path/to/audio.wav",
        "duration": 3.14,
        "text": "transcription text"
    }
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(samples, desc="Writing manifest"):
            manifest_entry = {
                "audio_filepath": sample["audio_filepath"],
                "duration": sample["duration"],
                "text": sample["text"],
            }
            f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

    print(f"✓ Created manifest: {output_path} ({len(samples)} samples)")


def download_hani89_dataset(output_dir: Path, max_samples: int = 2000):
    """
    Download and process Hani89/synthetic-medical-speech-dataset.

    This dataset contains thousands of synthetic medical speech samples
    generated from medical text using TTS.
    """
    print("\n=== Downloading Hani89 Synthetic Medical Speech Dataset ===")

    try:
        # Load dataset from Hugging Face
        dataset = load_dataset("Hani89/synthetic-medical-speech-dataset", split="train")
        print(f"Loaded {len(dataset)} samples from Hugging Face")

        # Limit samples for POC
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            print(f"Limited to {max_samples} samples for POC")

        # Create output directories
        audio_dir = output_dir / "hani89" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        for idx, item in enumerate(tqdm(dataset, desc="Processing samples")):
            # Extract audio array and text
            audio = item["audio"]
            text = item["transcription"]

            # Save audio file
            audio_path = audio_dir / f"sample_{idx:06d}.wav"
            sf.write(audio_path, audio["array"], audio["sampling_rate"])

            # Get duration
            duration = len(audio["array"]) / audio["sampling_rate"]

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
        manifest_dir = output_dir.parent / "processed"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        create_nemo_manifest(train_samples, manifest_dir / "train_manifest.json")
        create_nemo_manifest(val_samples, manifest_dir / "val_manifest.json")

        print(f"\n✓ Dataset ready:")
        print(f"  Train samples: {len(train_samples)}")
        print(f"  Val samples: {len(val_samples)}")
        print(f"  Audio dir: {audio_dir}")
        print(f"  Manifests: {manifest_dir}")

        return True

    except Exception as e:
        print(f"✗ Error downloading Hani89 dataset: {e}")
        return False


def download_yfyeung_dataset(output_dir: Path):
    """
    Download and process yfyeung/medical (Simulated OSCE) dataset.

    This dataset contains 272 MP3 files of simulated patient-physician conversations.
    Good for validation but too small for primary training.
    """
    print("\n=== Downloading yfyeung Medical OSCE Dataset ===")

    try:
        # Load dataset from Hugging Face
        dataset = load_dataset("yfyeung/medical", split="train")
        print(f"Loaded {len(dataset)} samples from Hugging Face")

        # Create output directories
        audio_dir = output_dir / "yfyeung" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        for idx, item in enumerate(tqdm(dataset, desc="Processing samples")):
            # Extract audio and text
            audio = item["audio"]
            text = item["text"]

            # Save audio file (convert MP3 to WAV for consistency)
            audio_path = audio_dir / f"sample_{idx:03d}.wav"
            sf.write(audio_path, audio["array"], audio["sampling_rate"])

            # Get duration
            duration = len(audio["array"]) / audio["sampling_rate"]

            samples.append({
                "audio_filepath": str(audio_path.absolute()),
                "duration": duration,
                "text": text,
            })

        # Create manifest (use all as validation set)
        manifest_dir = output_dir.parent / "processed"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        create_nemo_manifest(samples, manifest_dir / "yfyeung_manifest.json")

        print(f"\n✓ Dataset ready:")
        print(f"  Total samples: {len(samples)}")
        print(f"  Audio dir: {audio_dir}")
        print(f"  Manifest: {manifest_dir}/yfyeung_manifest.json")

        return True

    except Exception as e:
        print(f"✗ Error downloading yfyeung dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and prepare medical speech datasets")
    parser.add_argument(
        "--dataset",
        choices=["hani89", "yfyeung", "all"],
        default="hani89",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for raw data",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum samples for POC (applies to hani89)",
    )

    args = parser.parse_args()

    print(f"Output directory: {args.output_dir.absolute()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ["hani89", "all"]:
        download_hani89_dataset(args.output_dir, args.max_samples)

    if args.dataset in ["yfyeung", "all"]:
        download_yfyeung_dataset(args.output_dir)

    print("\n✓ All datasets downloaded successfully!")


if __name__ == "__main__":
    main()
