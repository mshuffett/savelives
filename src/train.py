#!/usr/bin/env python3
"""
Main training script for Canary‑Qwen medical ASR fine‑tuning.

Implements a thin wrapper around NVIDIA NeMo SpeechLM2 (SALM) training using
PyTorch Lightning. It:
  - Loads a YAML config (trainer, model, data, exp_manager)
  - Initializes W&B via NeMo exp_manager
  - Loads `nvidia/canary-qwen-2.5b` via `SALM.from_pretrained` when requested
  - Applies LoRA when specified in the config (delegated to NeMo if available)
  - Trains and auto‑resumes from latest checkpoint if `--resume` is set

Notes:
  - Requires `nemo_toolkit` (installed from Git) and PyTorch Lightning 2.x
  - Data manifests should be in NeMo JSONL format
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_wandb() -> None:
    """Initialize Weights & Biases login if a key is present.

    exp_manager will instantiate the logger; this only ensures the CLI is authenticated.
    """
    try:
        import wandb

        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            wandb.login(key=wandb_key)
            print("✓ Weights & Biases login OK")
        else:
            print("⚠ WANDB_API_KEY not set; proceeding without explicit login")
    except Exception as e:
        print(f"⚠ Wandb init skipped: {e}")


def train(config_path: str, resume: bool = False):
    """
    Run NeMo training with the specified config.

    Args:
        config_path: Path to NeMo YAML configuration file
    """
    try:
        # Import NeMo modules
        from omegaconf import OmegaConf, DictConfig
        from pytorch_lightning import Trainer
        import nemo.collections.speechlm2 as slm
        from nemo.collections.speechlm2.data import SALMDataset, DataModule
        from nemo.utils.exp_manager import exp_manager
        from nemo.utils import logging

        print(f"\n=== Starting Canary-Qwen Medical LoRA Training ===")
        print(f"Config: {config_path}")

        # Load configuration
        cfg: DictConfig = OmegaConf.load(config_path)

        # Setup logging
        logging.info(f"Training configuration:\n{OmegaConf.to_yaml(cfg)}")

        # Initialize wandb
        setup_wandb()

        # Resolve checkpoint directory for resume
        ckpt_dir = Path(
            os.getenv("CHECKPOINT_DIR", "/workspace/checkpoints/canary-qwen-medical-lora")
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Build PyTorch Lightning trainer from cfg.trainer
        # Use NeMo exp_manager to attach loggers, callbacks, and checkpointing
        # Convert trainer config to kwargs and drop logger/callbacks (handled by exp_manager)
        _trainer_cfg = OmegaConf.to_container(cfg.get("trainer", {}), resolve=True) or {}
        if isinstance(_trainer_cfg, dict):
            _trainer_cfg.pop("logger", None)
            _trainer_cfg.pop("callbacks", None)
        trainer = Trainer(**_trainer_cfg)
        exp_manager(trainer, cfg.get("exp_manager", None))

        # Instantiate model
        model = None
        restore_from = None

        # If a HF/NeMo pretrained name is provided, prefer loading from it
        if hasattr(cfg, "model"):
            restore_from = cfg.model.get("restore_from_path") if isinstance(cfg.model, DictConfig) else None

        if restore_from:
            print(f"Loading SALM from pretrained: {restore_from}")
            model = slm.models.SALM.from_pretrained(restore_from, map_location="auto")
        else:
            print("Instantiating SALM from cfg.model")
            model = slm.models.SALM(OmegaConf.to_container(cfg.model, resolve=True))

        # Construct DataModule
        # Prefer cfg.data; fall back to model.train_ds/validation_ds compatibility shim
        data_cfg = cfg.get("data")
        if data_cfg is None:
            # Backfill a minimal data config from model.* if needed
            data_cfg = OmegaConf.create(
                {
                    "train_ds": cfg.model.get("train_ds", {}),
                    "validation_ds": cfg.model.get("validation_ds", {}),
                }
            )
            logging.warning("No cfg.data provided; using cfg.model.train_ds/validation_ds as fallback")

        dataset = SALMDataset(tokenizer=model.tokenizer)
        datamodule = DataModule(data_cfg, tokenizer=model.tokenizer, dataset=dataset)

        # Find latest checkpoint if resume is requested
        ckpt_path = None
        if resume:
            # Prefer last.ckpt if present
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists():
                ckpt_path = str(last_ckpt)
            else:
                # Fallback: pick most recent .ckpt in directory
                ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
                ckpt_path = str(ckpts[0]) if ckpts else None
            if ckpt_path:
                print(f"Resuming from checkpoint: {ckpt_path}")

        # Fit
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        print("\n✓ Training finished")
        return 0

    except ImportError as e:
        print(f"✗ Error: Missing NeMo installation")
        print(f"  {e}")
        print("\nPlease install NeMo with:")
        print('  pip install "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"')
        return 1

    except Exception as e:
        print(f"✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Canary-Qwen medical ASR model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/salm_lora.yaml",
        help="Path to NeMo configuration file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )

    args = parser.parse_args()

    # Validate config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"✗ Error: Config file not found: {config_path}")
        return 1

    # Run training
    return train(str(config_path), resume=bool(args.resume))


if __name__ == "__main__":
    sys.exit(main())
