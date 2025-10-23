#!/usr/bin/env python3
"""
Modal-based nvidia/canary-qwen-2.5b inference test using NVIDIA NeMo.
This validates we can load and use the model before attempting fine-tuning.
"""

import modal
import os

# Create Modal app
app = modal.App("medical-asr-canary-test")

# Define the container image with all dependencies
# Note: NeMo installation is complex and takes 10-15 minutes
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


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[
        modal.Secret.from_dict({
            "WANDB_API_KEY": "ad198476a9d880adf9d176444ff8f2a42616d8bc",
        })
    ],
)
def test_canary_loading():
    """Test loading and basic inference with Canary-Qwen-2.5b via NeMo SALM."""
    import torch
    import wandb
    import numpy as np
    import soundfile as sf
    import tempfile

    print("\n" + "="*60)
    print("Medical ASR - Canary-Qwen-2.5b Model Loading Test (NeMo)")
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
        name="canary-qwen-2.5b-nemo-test",
        config={
            "model": "nvidia/canary-qwen-2.5b",
            "framework": "NeMo",
            "test_type": "model_loading",
        }
    )

    # Import NeMo SALM
    print("Importing NeMo SALM...")
    try:
        from nemo.collections.speechlm2.models import SALM
        print("✓ NeMo SALM imported successfully\n")
    except Exception as e:
        print(f"✗ Failed to import NeMo SALM: {e}")
        print("This likely means NeMo installation didn't complete properly\n")
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
        print(f"  Trainable: {num_trainable:,}")
        print(f"  Frozen: {num_params - num_trainable:,}\n")

        wandb.log({
            "status": "model_loaded",
            "total_params": num_params,
            "trainable_params": num_trainable,
        })

        # Test inference with a simple audio file
        print("Testing inference on synthetic audio...")

        # Generate test audio (sine wave)
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Save to temp file
        temp_dir = tempfile.mkdtemp()
        audio_path = f"{temp_dir}/test_audio.wav"
        sf.write(audio_path, audio.astype(np.float32), sample_rate)

        # Run inference using NeMo SALM API
        print(f"Running transcription on {audio_path}...")

        try:
            answer_ids = model.generate(
                prompts=[
                    [{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}", "audio": [audio_path]}]
                ],
                max_new_tokens=128,
            )
            transcription = model.tokenizer.ids_to_text(answer_ids[0].cpu())

            print(f"\n✓ Transcription result: '{transcription}'\n")

            wandb.log({
                "status": "inference_success",
                "transcription": transcription,
            })

            result = {
                "status": "success",
                "model_loaded": True,
                "inference_working": True,
                "transcription": transcription,
            }

        except Exception as e:
            print(f"✗ Inference failed: {e}\n")
            wandb.log({
                "status": "inference_failed",
                "error": str(e),
            })
            result = {
                "status": "partial_success",
                "model_loaded": True,
                "inference_working": False,
                "error": str(e),
            }

    except Exception as e:
        print(f"✗ Failed to load model: {e}\n")
        wandb.log({
            "status": "model_load_failed",
            "error": str(e),
        })
        result = {
            "status": "failed",
            "model_loaded": False,
            "error": str(e),
        }

    print("="*60)
    print(f"Test Complete - Status: {result['status']}")
    print("="*60)

    wandb.finish()
    return result


@app.local_entrypoint()
def main():
    """Local entrypoint to trigger model loading test."""
    result = test_canary_loading.remote()
    print(f"\nFinal Result: {result}")


if __name__ == "__main__":
    main()
