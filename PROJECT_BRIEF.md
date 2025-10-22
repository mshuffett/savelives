# Medical ASR Fine-Tuning POC - Project Brief

**Task:**
Set up and run a **proof-of-concept fine-tuning pipeline** for **`nvidia/canary-qwen-2.5b`** to adapt it for **medical dictation and conversational doctor audio**, following the training guidelines from both **Hugging Face** and the **NVIDIA NeMo** `examples/speechlm2` folder.

---

## 🎯 Objective / Business Context

We're creating a **minimum-viable proof of concept** for a next-generation medical speech recognition system that can handle doctor dictations and conversational audio.
This POC's goal is **to demonstrate domain adaptation feasibility and WER improvement** — not to reach final production quality yet.

Key business outcomes:

- Show that a small LoRA/PEFT fine-tune can improve transcription accuracy on medical vocabulary.
- Produce a reproducible cloud training setup that we can later expand into large-scale fine-tuning and evaluation.
- Integrate with common observability tools (Weights & Biases, Drive or S3 checkpoints).
- Run continuously on **RunPod** infrastructure using the **Run Continuous Command** workflow.

---

## ⚙️ Requirements for the Agent

**Environment / Infrastructure**

- Use **RunPod** to provision a GPU pod (A100 40 GB or higher).
- My **RunPod API key is already available in the environment**, so don't prompt for it.
- Use the continuous command so that you don't stop and when prompted reflect on whether you accomplished the full project goal and don't give up till you have.
- Use **Python 3.10+**, **CUDA 11+**, **PyTorch 2.1+**, **NeMo**, **Transformers ≥ 4.44**, **PEFT**, **bitsandbytes**, and **wandb**.

---

## 📦 Core Technical Tasks

1. **Read and summarize**:
   - The Hugging Face model card: https://huggingface.co/nvidia/canary-qwen-2.5b
   - The NeMo training folder: https://github.com/NVIDIA-NeMo/NeMo/tree/main/examples/speechlm2

   Verify:
   - Required NeMo installation line (`nemo_toolkit[asr,tts]`).
   - Default config file (`salm.yaml` or equivalent) and LoRA parameters.
   - Recommended freeze/unfreeze pattern (LLM frozen, encoder + projection + LoRA trainable).

2. **Set up project repo and structure**
   - Create a lightweight workspace (e.g., `/workspace/medical-asr-finetune`).
   - Include `train.py`, `config.yaml`, and a README summarizing setup and goals.
   - Add a `.env` loader for RunPod and wandb secrets.

3. **Dataset review and selection**
   Evaluate and document in the README a few candidate open datasets for the initial experiment:
   - [Hani89/synthetic-medical-speech-dataset](https://huggingface.co/datasets/Hani89/synthetic-medical-speech-dataset)
   - [United-Syn-Med](https://huggingface.co/datasets/United-Syn-Med/synthetic_medical_speech)
   - [yfyeung/medical (Simulated OSCE dataset)](https://huggingface.co/datasets/yfyeung/medical)
   - Optional: ACI-Bench or other open simulated patient-doctor conversations.

   Choose one small subset (≤ 2,000 samples) for the POC.

   **Note:** These are examples but you should dig a bit and do some research as to which dataset to use and also document which ones you looked at and why you chose the one you did.

4. **Training setup**
   - Implement LoRA fine-tuning (using **PEFT** or NeMo's native LoRA hooks).
   - Configure W&B logging (`project="medical-asr-poc-[yourname]"`) and model checkpointing every N steps.
   - Auto-resume from latest checkpoint on pod restart.
   - Save checkpoints to local disk and optionally sync to an S3 bucket or Drive if desired.
   - For reproducibility, include a `requirements.txt` and `run.sh` that installs deps and launches the job.

5. **Run**
   - Launch fine-tuning on RunPod so the job auto-resumes.
   - Keep sleeping and checking the status and if it fails restart it.
   - Monitor W&B dashboard for progress and loss.
   - Produce one fine-tuned checkpoint and minimal evaluation (on a 200-sample validation split).

---

## 📊 Deliverables

- A working RunPod Continuous Command script that runs end-to-end fine-tuning.
- A brief `README.md` summarizing:
  - Purpose of the POC
  - Dataset(s) used
  - Expected runtime / cost
  - Links to W&B dashboard and Hugging Face model card
- Output: one trained model folder (`canary-qwen-medical-lora/`) with LoRA weights.

---

## 🔗 Reference Links for the Agent

- Model card: https://huggingface.co/nvidia/canary-qwen-2.5b
- NeMo examples: https://github.com/NVIDIA-NeMo/NeMo/tree/main/examples/speechlm2
- Docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/speechlm2/models.html
- Dataset candidates:
  - https://huggingface.co/datasets/Hani89/synthetic-medical-speech-dataset
  - https://huggingface.co/datasets/United-Syn-Med/synthetic_medical_speech
  - https://huggingface.co/datasets/yfyeung/medical

---

## 🧭 Guidance for Agent Behavior

- Treat this as a **prototype build**, not a production deployment.
- Favor simplicity and reproducibility (bash + Python scripts).
- Use LoRA or PEFT to minimize GPU memory footprint.
- Ensure wandb runs under the existing environment token.
- Assume internet access for pulling models/datasets.

---

## 💬 Summary for Business Stakeholders

> This proof-of-concept demonstrates how quickly we can adapt a state-of-the-art speech foundation model (Canary-Qwen-2.5B) to a clinical dictation domain using open data and commodity GPUs.
> It establishes a reproducible RunPod + NeMo pipeline with cost tracking (via W&B) and modular training code, laying the foundation for future expansion to multi-speaker and real-world audio data.
