# Medical ASR Fine‑Tuning POC — Project Structure & Tooling Spec

Date: 2025-10-22
Owner: Michael
Scope: Structure and tooling plan for the POC defined in `PROJECT_BRIEF.md`.

## Goals
- Keep the POC simple, reproducible, and resume‑safe on RunPod.
- Favor NeMo SpeechLM2 + LoRA/PEFT over full fine‑tuning to minimize VRAM.
- Produce one trained LoRA checkpoint and minimal eval (WER on ~200 val samples).

## Repository Layout
- `runpod/`
  - `start.sh` — Continuous Command main loop: idempotent setup → train/resume → healthcheck → backoff/retry.
  - `setup.sh` — CUDA/PyTorch/NeMo/PEFT/bitsandbytes install; cache warmup; environment probes.
  - `healthcheck.sh` — sanity checks for GPU, disk, network, latest checkpoint presence, W&B auth.
- `configs/`
  - `base.yaml` — project name, seeds, logging, paths, W&B settings.
  - `model/canary_qwen.yaml` — NeMo SpeechLM2 + LoRA params; freeze map (LLM frozen; encoder + projection + LoRA trainable).
  - `train/lora_small.yaml` — batch size, grad accum, LR, scheduler, precision, ckpt cadence, max steps.
  - `data/<dataset>.yaml` — HF dataset id, splits, audio/text keys, resampling, subset size (≤ 2k for POC).
- `src/medical_asr/`
  - `data.py` — load/clean Hugging Face datasets, create manifests compatible with NeMo; enforce subset size; deterministic split.
  - `training.py` — assemble model per config, attach LoRA adapters (PEFT or NeMo hooks), resume from latest, checkpointing.
  - `evaluation.py` — quick WER/CER with `jiwer` on validation split; export metrics JSON + log to W&B.
  - `logging.py` — W&B init, structured logging helpers.
  - `checkpoints.py` — periodic save, rotation, optional background S3 sync.
  - `nemo_hooks.py` — freeze/unfreeze policy and adapter injection helpers.
- `scripts/`
  - `train.py` — CLI/Hydra entrypoint that wires configs → `training.py`.
  - `eval.py` — offline evaluation against a provided checkpoint.
  - `prepare_subset.py` — pick dataset and produce manifests for NeMo from HF datasets.
- `requirements.in` — human‑edited dependency list.
- `requirements.txt` — pinned, generated for reproducibility (compiled from `.in`).
- `README.md` — POC summary, datasets considered/selected, commands to run on RunPod, W&B link.
- `docs/`
  - `datasets_review.md` — short notes on candidate datasets and selection rationale.
  - (this file) `medical-asr-poc-structure-and-tooling.md`.
- `experiments/` — optional: run artifacts like `params.yaml`, `metrics.json`, `WER.csv` for quick inspection.
- `tests/` — smoke tests only: config loads, dataset split sizes, resume logic sanity.
- `.env.example` — documented variables; `.env.local` (gitignored) for project‑specific secrets; `.envrc` for direnv.

## Core Tools
- Python 3.10+, CUDA 11+, PyTorch 2.1+.
- NVIDIA NeMo SpeechLM2 (per `examples/speechlm2`) with LoRA; PEFT as needed.
- Config: Hydra/OmegaConf for composable YAMLs and CLI overrides.
- Observability: Weights & Biases (`project: medical-asr-poc-michael`), step metrics, model checkpoints as artifacts optional.
- Metrics: `jiwer` for WER/CER; log to W&B.
- Packaging: `pip-tools` (or `uv pip compile`) to generate pinned `requirements.txt` from `requirements.in`.
- Lint/format (Python): `ruff` (lint+format) and `mypy` (selective) for fast feedback.
- Tests: `pytest` smoke tests; keep under 1 minute locally.

## RunPod Integration
- Use RunPod Continuous Command via `runpod/start.sh`:
  1) `setup.sh` ensures correct drivers/libs and installs Python deps idempotently.
  2) Restore latest checkpoint (atomic `latest` pointer) if present.
  3) Launch `python -m scripts.train +config=...`.
  4) On error: capture logs → exponential backoff → retry; do not exit unless unrecoverable.
- Checkpoints on fast local NVMe; optional background sync to `S3_BUCKET` if set.
- Ensure auto‑resume on pod restart and consistent W&B run reattachment.

## Configs & Reproducibility
- Single source of truth in `configs/`, invoked as:
  `python -m scripts.train model=canary_qwen train=lora_small data=yfyeung_medical`
- Freeze policy explicit in config: LLM frozen; encoder + projection + LoRA trainable.
- Seeded, deterministic data split and capped subset size (≤ 2,000) for the POC.
- Pinned `requirements.txt` and recorded CLI overrides in W&B config.

## Data Handling
- Pull candidates from Hugging Face:
  - `Hani89/synthetic-medical-speech-dataset`
  - `United-Syn-Med/synthetic_medical_speech`
  - `yfyeung/medical`
- Document pros/cons and chosen subset in `docs/datasets_review.md`.
- `prepare_subset.py` writes NeMo‑compatible manifest(s), validates fields, and resamples audio if needed.

## CI (Lightweight)
- GitHub Actions: cache pip; run `ruff check`, `mypy` (targeted modules), and `pytest -q` smoke tests.
- Optional: Nightly sanity on tiny subset to ensure scripts and configs remain runnable.

## Secrets & Environment
- Prefer existing global secrets in `~/.env.zsh` for RUNPOD/W&B.
- Project‑specific secrets in `.env.local` (gitignored); `.env.example` lists required keys.
- `direnv` auto‑loads `.env.local` via `.envrc` when entering the project.

## Quality Gates for the POC
- Configs load and validate.
- Dataset subset size and split are deterministic and within the 2k cap.
- Training can resume from the latest checkpoint after a forced restart.
- W&B logs (loss, lr, WER) appear and link in `README.md`.
- Minimal evaluation runs on ~200 val samples and emits WER.

## Open Questions / Next Steps
1) Dataset choice and exact subset spec (speaker balance, duration cap).
2) Checkpoint cadence vs. storage budget on RunPod NVMe.
3) Whether to sync checkpoints to S3 or keep local for this POC.
4) Preference for NeMo native LoRA vs. PEFT wrappers.

