#!/usr/bin/env python3
"""
Main training script for Canary-Qwen medical ASR fine-tuning.
Uses NeMo's SALM training pipeline with LoRA for efficient adaptation.
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


def setup_wandb():
    """Initialize Weights & Biases logging."""
    import wandb

    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
        print("✓ Weights & Biases initialized")
    else:
        print("⚠ Warning: WANDB_API_KEY not found, logging may fail")


def train(config_path: str):
    """
    Run NeMo training with the specified config.

    Args:
        config_path: Path to NeMo YAML configuration file
    """
    try:
        # Import NeMo modules
        from nemo.core.config import hydra_runner
        from nemo.utils import logging
        from omegaconf import OmegaConf

        print(f"\n=== Starting Canary-Qwen Medical LoRA Training ===")
        print(f"Config: {config_path}")

        # Load configuration
        cfg = OmegaConf.load(config_path)

        # Setup logging
        logging.info(f"Training configuration:\n{OmegaConf.to_yaml(cfg)}")

        # Initialize wandb
        setup_wandb()

        # Import and run NeMo training
        # Note: The actual training implementation depends on NeMo's SALM module
        # This is a placeholder that should be updated based on the actual
        # NeMo speechlm2 training script structure

        print("\n⚠ Note: This script requires the full NeMo SALM training implementation")
        print("Please refer to NeMo examples/speechlm2/salm_train.py for the complete training loop")
        print("\nFor now, this serves as a template and configuration manager.")

        # TODO: Import and call the actual NeMo SALM training function
        # from nemo.collections.speechlm2 import SALMTrainer
        # trainer = SALMTrainer(cfg)
        # trainer.fit()

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
    return train(str(config_path))


if __name__ == "__main__":
    sys.exit(main())
