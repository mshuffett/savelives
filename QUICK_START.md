# 🚀 Medical ASR POC - Quick Start Guide

## Current Status: Ready to Deploy

Your RunPod GPU pod is **running** and ready for training initialization.

---

## ⚡ Start Training in 3 Steps

### Step 1: Access RunPod Pod

Go to: **https://www.runpod.io/console/pods**

Find your pod:
- **Name**: `medical-asr-poc`
- **ID**: `p24f468c33bbsg`
- **GPU**: A100 80GB PCIe
- **Status**: 🟢 RUNNING

Click: **"Connect" → "Start Web Terminal"**

### Step 2: Run Autonomous Init Script

In the web terminal, paste this **one command**:

```bash
curl -sSL https://raw.githubusercontent.com/mshuffett/savelives/main/scripts/runpod_init_autonomous.sh | bash
```

### Step 3: Monitor Progress

**Weights & Biases Dashboard**:
https://wandb.ai/<your-username>/medical-asr-poc-michael

**Or watch logs in real-time**:
```bash
tail -f /workspace/autonomous_setup.log
```

---

## ⏱️ What to Expect

The script will run for approximately **20-30 minutes** and will:

1. ✅ Install system dependencies (2 min)
2. ✅ Clone repository (1 min)
3. ✅ Install PyTorch + NeMo (10-15 min)
4. ✅ Download medical speech dataset - 2000 samples (5-10 min)
5. ✅ Set up W&B logging (1 min)
6. ✅ Create directories and verify GPU (1 min)
7. ✅ Run POC training demonstration (1-2 min)

**Total**: ~20-30 minutes

---

## 📊 Monitoring

### Real-Time Progress
```bash
# In the RunPod terminal
tail -f /workspace/autonomous_setup.log
```

### Weights & Biases
- Project: medical-asr-poc-michael
- Dashboard: https://wandb.ai/<username>/medical-asr-poc-michael
- Metrics: Dataset stats, GPU usage, setup status

### GPU Check
```bash
nvidia-smi
```

---

## ✅ Success Indicators

You'll know it worked when you see:

```
========================================
✅ POC Setup Complete!
========================================

Next: Implement full NeMo SALM training loop
```

And in W&B, you'll see:
- ✓ Dataset loaded: 1600 training samples
- ✓ W&B logging active
- ✓ Status: poc_setup_complete
- ✓ ready_for_training: True

---

## 💰 Cost

- **Current cost**: ~$0.30 (running for ~11 minutes)
- **Projected total**: $2-3 for complete POC
- **Hourly rate**: $1.64/hr

---

## 🛑 When Done

### Stop Pod (Pause, can resume later)
```bash
# From your local machine:
export RUNPOD_API_KEY="rpa_VVQLJLKLAUIDLJ6PPLYBBGQIWDLN70J5BCP5L36V1dht52"
python scripts/runpod_manager.py stop p24f468c33bbsg
```

### Terminate Pod (Permanent deletion)
```bash
python scripts/runpod_manager.py terminate p24f468c33bbsg
```

Or use the RunPod web console: https://www.runpod.io/console/pods

---

## 🔧 Troubleshooting

### Script Fails or Stops
```bash
# Check what happened
cat /workspace/autonomous_setup.log

# Restart from the beginning
rm -rf /workspace/savelives
curl -sSL https://raw.githubusercontent.com/mshuffett/savelives/main/scripts/runpod_init_autonomous.sh | bash
```

### Can't Access Web Terminal
- Try Jupyter instead: Click "Connect" → "Jupyter Lab"
- Open terminal in Jupyter: File → New → Terminal
- Run the same curl command

### Want to Run Steps Manually
See: `RUNPOD_STATUS.md` for detailed manual commands

---

## 📚 Full Documentation

- **Project Brief**: `PROJECT_BRIEF.md`
- **Complete README**: `README.md`
- **RunPod Details**: `RUNPOD_STATUS.md`
- **Full Status**: `STATUS_SUMMARY.md`
- **Dataset Research**: `docs/dataset_research.md`

---

## 🎯 What This POC Demonstrates

1. ✅ **Infrastructure**: Complete project structure and automation
2. ✅ **GPU Provisioning**: RunPod pod management via API
3. ✅ **Dataset Pipeline**: Download → Convert to NeMo format
4. ✅ **Monitoring**: W&B integration and logging
5. ✅ **Automation**: One-command autonomous setup
6. ⏳ **Training Framework**: Ready for full NeMo SALM integration

---

## 🚀 Next Phase (After POC)

1. Implement full NeMo SALM training loop
2. Scale to complete dataset (10K+ samples)
3. Hyperparameter tuning
4. Add real medical dictation data
5. Production deployment

---

**Ready?** Just run that one curl command in the RunPod terminal!

**Questions?** Check `STATUS_SUMMARY.md` for complete details.

**Stuck?** All scripts are in the `scripts/` directory and fully documented.
