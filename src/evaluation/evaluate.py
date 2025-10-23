#!/usr/bin/env python3
"""
Minimal WER evaluation for the Canary‑Qwen SALM fine‑tune.

Reads a NeMo manifest (JSONL: audio_filepath, duration, text), runs SALM
generation for each audio sample, and computes WER with `jiwer`.

Outputs metrics JSON to logs/eval/metrics.json and optionally logs to Weights & Biases.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torchaudio
from dotenv import load_dotenv
from jiwer import wer

import nemo.collections.speechlm2 as slm


@dataclass
class Sample:
    audio_filepath: str
    text: str


def load_manifest(path: Path, limit: int | None = None) -> List[Sample]:
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            samples.append(Sample(audio_filepath=obj["audio_filepath"], text=obj["text"]))
    return samples


def evaluate(
    model: slm.models.SALM,
    manifest: Path,
    batch_size: int = 4,
    limit: int | None = 200,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    samples = load_manifest(manifest, limit=limit)
    preds: List[str] = []
    refs: List[str] = []

    def _prep(wav_path: str):
        audio, sr = torchaudio.load(wav_path)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        return audio.squeeze(0)

    # Simple batching over samples
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        # Prepare tensors
        audios = [
            _prep(s.audio_filepath).to(device) for s in batch
        ]  # list of [T]
        audio_lens = torch.tensor([a.shape[-1] for a in audios], device=device)
        # Pad to longest
        max_len = int(max(audio_lens).item())
        padded = torch.stack([torch.nn.functional.pad(a, (0, max_len - a.shape[-1])) for a in audios])

        prompts = [[{"role": "user", "content": f"{model.audio_locator_tag}"}]] * len(batch)

        with torch.no_grad():
            outputs = model.generate(
                prompts=prompts,
                audios=padded,
                audio_lens=audio_lens,
                generation_config=None,
            )

        # Decode (ids_to_text or tokenizer decode depending on interface)
        # Try common decode APIs safely
        for j in range(len(batch)):
            out = outputs[j]
            text = None
            if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "ids_to_text"):
                try:
                    text = model.tokenizer.ids_to_text(out)
                except Exception:
                    pass
            if text is None:
                # Fallback if outputs are already text
                text = out if isinstance(out, str) else str(out)

            # Record
            preds.append(text)
            refs.append(batch[j].text)

    metric = wer(refs, preds)
    return {"wer": metric, "num_samples": len(samples)}


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate WER on validation manifest")
    parser.add_argument("--checkpoint", type=str, default="nvidia/canary-qwen-2.5b")
    parser.add_argument(
        "--manifest", type=str, default="data/processed/val_manifest.json"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading SALM: {args.checkpoint}")
    model = slm.models.SALM.from_pretrained(args.checkpoint, map_location="auto")
    metrics = evaluate(model, Path(args.manifest), batch_size=args.batch_size, limit=args.limit)

    out_dir = Path(os.getenv("LOG_DIR", "logs")) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ WER: {metrics['wer']:.4f} on {metrics['num_samples']} samples")
    print(f"Saved: {out_path}")

    # Optional: log to W&B if available
    try:
        import wandb

        if os.getenv("WANDB_API_KEY"):
            wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project=os.getenv("WANDB_PROJECT", "medical-asr-poc-michael"), name="eval")
        wandb.log(metrics)
        wandb.finish()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

