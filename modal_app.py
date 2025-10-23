import os
import subprocess
import modal

app = modal.App("savelives-medical-asr")

# Volumes for persistence
data_vol = modal.Volume.from_name("savelives-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("savelives-checkpoints", create_if_missing=True)
logs_vol = modal.Volume.from_name("savelives-logs", create_if_missing=True)

# Base image with CUDA and Python
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "ffmpeg", "libsndfile1", "curl",
        "build-essential", "python3-dev"
    )
    .pip_install(
        # Build tooling + possible C-extension helpers
        "setuptools>=69", "wheel", "cython<3",
        # Core frameworks (align with bitsandbytes and CUDA 12.x wheels)
        "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
        # Trainer + ecosystem
        "pytorch-lightning>=2.2.0,<3",
        # NeMo from Git (asr-only to reduce optional deps)
        "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git",
        # Project deps
        "transformers>=4.44.0", "peft>=0.7.0", "accelerate>=0.25.0", "bitsandbytes>=0.41.0",
        "datasets>=2.14.0", "soundfile>=0.12.0", "librosa>=0.10.0", "pydub>=0.25.1",
        "numpy>=1.24.0", "scipy>=1.11.0", "wandb>=0.15.0", "python-dotenv>=1.0.0",
        "omegaconf>=2.3.0", "hydra-core>=1.3.0", "tqdm>=4.66.0", "jiwer<4",
    )
    .env(
        {
            # Respect optional envs if provided at runtime
            "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "medical-asr-poc-michael"),
            "CHECKPOINT_DIR": "/workspace/checkpoints",
            "DATA_DIR": "/workspace/data",
            "LOG_DIR": "/workspace/logs",
        }
    )
    .add_local_dir(".", "/workspace/savelives")
)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 24,
    volumes={
        "/workspace/data": data_vol,
        "/workspace/checkpoints": ckpt_vol,
        "/workspace/logs": logs_vol,
    },
    secrets=[modal.Secret.from_name("wandb-secret", required=False)],
)
def train():
    cwd = "/workspace/savelives"
    env = os.environ.copy()
    print("CUDA available?", subprocess.run(["python", "-c", "import torch;print(torch.cuda.is_available())"], cwd=cwd, env=env, check=False, text=True, capture_output=True).stdout.strip())

    # Download data subset
    subprocess.run(
        ["python", "scripts/download_data.py", "--dataset", "hani89", "--max-samples", "2000"],
        cwd=cwd,
        env=env,
        check=True,
    )

    # Kick off training (resume-safe)
    subprocess.run(
        ["python", "src/train.py", "--config", "config/salm_lora.yaml", "--resume"],
        cwd=cwd,
        env=env,
        check=True,
    )

    # Quick eval on ~200 samples
    subprocess.run(
        ["python", "src/evaluation/evaluate.py", "--limit", "200"],
        cwd=cwd,
        env=env,
        check=True,
    )
    # Persist volume writes
    data_vol.commit()
    ckpt_vol.commit()
    logs_vol.commit()


@app.local_entrypoint()
def main():
    # Invoke the remote train function
    train.remote()
