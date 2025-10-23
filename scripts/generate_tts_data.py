#!/usr/bin/env python3
"""
Generate medical speech dataset using Hume TTS (#2 on TTS Arena).
Creates realistic training data to replace sine wave placeholders.
Uses dynamic voice generation for variety.
"""

import os
import json
import subprocess
from pathlib import Path
import soundfile as sf
import numpy as np

# Medical phrases for training data
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


def generate_tts_audio(text: str, output_path: str, voice_description: str = "professional medical voice"):
    """
    Generate speech audio from text using Hume TTS with dynamic voice generation.

    Args:
        text: Text to synthesize
        output_path: Path to save audio file
        voice_description: Voice style description for Hume's dynamic generation
    """
    try:
        # Use hume-tts command with voice description
        result = subprocess.run(
            ["hume-tts", text, "-d", voice_description, "-o", output_path],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating TTS: {e.stderr}")
        return False


def convert_mp3_to_wav(mp3_path: str, wav_path: str, sample_rate: int = 16000):
    """
    Convert MP3 to WAV with specified sample rate for ASR training.

    Args:
        mp3_path: Path to MP3 file
        wav_path: Path to output WAV file
        sample_rate: Target sample rate (16kHz for ASR)
    """
    try:
        # Use ffmpeg to convert and resample
        subprocess.run(
            [
                "ffmpeg", "-i", mp3_path,
                "-ar", str(sample_rate),
                "-ac", "1",  # Mono
                "-y",  # Overwrite
                wav_path
            ],
            capture_output=True,
            check=True
        )
        # Remove original MP3
        os.remove(mp3_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting MP3 to WAV: {e}")
        return False


def add_silence_padding(audio_path: str, padding_seconds: float = 0.5):
    """
    Add silence padding to beginning and end of audio file.

    Args:
        audio_path: Path to audio file
        padding_seconds: Seconds of silence to add on each side
    """
    # Read audio
    audio, sample_rate = sf.read(audio_path)

    # Create silence
    silence_samples = int(sample_rate * padding_seconds)
    silence = np.zeros(silence_samples, dtype=audio.dtype)

    # Add padding
    padded_audio = np.concatenate([silence, audio, silence])

    # Overwrite original file
    sf.write(audio_path, padded_audio, sample_rate)

    return len(padded_audio) / sample_rate  # Return duration


def main():
    """Generate medical speech dataset with Hume TTS."""
    print("\n" + "="*60)
    print("Medical ASR - Hume TTS Dataset Generation")
    print("="*60 + "\n")

    # Setup paths
    base_dir = Path("/Users/michael/ws/savelives")
    audio_dir = base_dir / "data" / "raw" / "tts_medical" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = base_dir / "data" / "processed"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Voice style descriptions for diversity
    voice_descriptions = [
        "professional female medical voice",
        "professional male medical voice",
        "calm female doctor voice",
        "confident male doctor voice",
        "friendly female nurse voice",
        "clear male medical professional",
        "warm female healthcare provider",
        "authoritative male physician",
    ]

    # Generate samples
    num_samples = 200  # Start with 200 samples for testing
    samples_per_phrase = num_samples // len(MEDICAL_PHRASES)

    print(f"Generating {num_samples} speech samples...")
    print(f"Using {len(voice_descriptions)} different voice styles for variety")
    print("Service: Hume TTS (#2 on TTS Arena, ELO 1669)\n")

    train_manifest = []
    val_manifest = []

    sample_idx = 0
    total_chars = 0
    successful = 0
    failed = 0

    for phrase_idx, phrase in enumerate(MEDICAL_PHRASES):
        for rep in range(samples_per_phrase):
            # Cycle through voice descriptions for variety
            voice_desc = voice_descriptions[(phrase_idx * samples_per_phrase + rep) % len(voice_descriptions)]

            # Generate audio file
            audio_filename = f"sample_{sample_idx:06d}.wav"
            temp_mp3 = audio_dir / f"sample_{sample_idx:06d}.mp3"
            audio_path = audio_dir / audio_filename

            try:
                # Generate TTS (outputs MP3)
                if not generate_tts_audio(phrase, str(temp_mp3), voice_description=voice_desc):
                    failed += 1
                    sample_idx += 1
                    continue

                # Convert MP3 to WAV at 16kHz for ASR
                if not convert_mp3_to_wav(str(temp_mp3), str(audio_path)):
                    failed += 1
                    sample_idx += 1
                    continue

                # Add silence padding
                duration = add_silence_padding(str(audio_path), padding_seconds=0.5)

                # Create manifest entry
                manifest_entry = {
                    "audio_filepath": str(audio_path),
                    "duration": round(duration, 2),
                    "text": phrase,
                }

                # Split 80/20 train/val
                if sample_idx % 5 == 0:
                    val_manifest.append(manifest_entry)
                else:
                    train_manifest.append(manifest_entry)

                total_chars += len(phrase)
                successful += 1

                if (sample_idx + 1) % 10 == 0:
                    print(f"Generated {sample_idx + 1}/{num_samples} samples... ({successful} successful, {failed} failed)")

            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                failed += 1

            sample_idx += 1

    # Write manifests
    train_manifest_path = manifest_dir / "train_manifest_tts.json"
    val_manifest_path = manifest_dir / "val_manifest_tts.json"

    with open(train_manifest_path, 'w') as f:
        for entry in train_manifest:
            f.write(json.dumps(entry) + '\n')

    with open(val_manifest_path, 'w') as f:
        for entry in val_manifest:
            f.write(json.dumps(entry) + '\n')

    # Summary
    print("\n" + "="*60)
    print("✓ Dataset Generation Complete!")
    print("="*60)
    print(f"\nGenerated {successful} successful samples ({failed} failed)")
    print(f"  Train: {len(train_manifest)} samples")
    print(f"  Val: {len(val_manifest)} samples")
    print(f"\nTotal characters synthesized: {total_chars:,}")
    print(f"TTS Service: Hume TTS (#2 on TTS Arena, ELO 1669)")
    print(f"\nManifests:")
    print(f"  Train: {train_manifest_path}")
    print(f"  Val: {val_manifest_path}")
    print(f"\nAudio files: {audio_dir}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
