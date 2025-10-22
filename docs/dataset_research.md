# Dataset Research & Selection

## Overview

This document captures the research and rationale for dataset selection for the medical ASR fine-tuning POC.

## Evaluation Criteria

For the POC, we prioritized:
1. **Medical domain relevance** - Contains medical vocabulary and context
2. **Size** - Sufficient samples for fine-tuning (1000+ preferred)
3. **Quality** - Clear audio, accurate transcriptions
4. **Privacy** - Synthetic or properly de-identified data
5. **Accessibility** - Publicly available, no complex licensing

## Candidate Datasets

### 1. Hani89/synthetic-medical-speech-dataset ✅ SELECTED

**Source**: https://huggingface.co/datasets/Hani89/synthetic-medical-speech-dataset

**Stats**:
- **Size**: 10K-100K samples
- **Format**: Parquet (audio arrays + transcriptions)
- **Generation**: TTS-generated from medical text
- **Language**: English
- **Privacy**: Fully synthetic (no real patient data)

**Pros**:
- ✅ Largest available dataset
- ✅ Good medical vocabulary coverage
- ✅ Privacy-safe for POC
- ✅ Easy to download via HuggingFace
- ✅ Consistent quality (TTS-generated)

**Cons**:
- ❌ Synthetic voices may not match real clinical audio
- ❌ No background noise or realistic recording conditions
- ❌ Limited prosody variation

**Decision**: **Selected as primary training dataset** due to size and medical domain coverage. POC will use 2000 samples.

---

### 2. yfyeung/medical (Simulated OSCE)

**Source**: https://huggingface.co/datasets/yfyeung/medical

**Stats**:
- **Size**: 272 samples
- **Format**: MP3 audio + text transcripts
- **Content**: Simulated patient-physician conversations (OSCE exams)
- **Domains**: Respiratory, GI, Cardiovascular, MSK, Dermatology
- **Language**: English

**Pros**:
- ✅ Conversational structure (realistic dialogue)
- ✅ Multiple medical domains
- ✅ Research-backed (Nature publication)
- ✅ Good for testing conversational understanding

**Cons**:
- ❌ Too small for primary training (272 samples)
- ❌ Limited to OSCE scenarios (may not represent actual clinical dictation)

**Decision**: **Good for validation/testing**, but insufficient size for primary training. Could be used as a held-out evaluation set.

---

### 3. United-Syn-Med/synthetic_medical_speech

**Source**: https://huggingface.co/datasets/United-Syn-Med/synthetic_medical_speech

**Status**: ❌ Dataset appears to be private or inaccessible (401 error during research)

**Decision**: Skipped due to access issues.

---

### 4. Other Datasets Considered

**ACI-Bench** (Mentioned in brief)
- Not publicly available at time of research
- Would need further investigation

**LibriSpeech Medical Subsets**
- General speech, not medical-specific
- Rejected due to domain mismatch

---

## Final Selection Rationale

**Primary Training**: Hani89/synthetic-medical-speech-dataset
- Size advantage (10K+ samples vs. 272)
- Medical vocabulary coverage
- Sufficient for POC demonstration
- Synthetic nature is acceptable for initial validation

**Validation/Testing**: yfyeung/medical
- Conversational structure tests different capability
- Good held-out test set
- Real-world dialogue patterns

---

## POC Configuration

```yaml
Training Data:
  Dataset: Hani89/synthetic-medical-speech-dataset
  Samples: 2000 (for POC)
  Split: 80/20 train/val (1600 train, 400 val)

Validation Data:
  Dataset: Same (from 80/20 split)
  Purpose: Monitor training progress

Future Expansion:
  - Scale to full Hani89 dataset (~10K+ samples)
  - Add yfyeung/medical as held-out test set
  - Incorporate real clinical dictations (when available)
```

---

## Next Steps

1. **Validate WER improvement** on Hani89 synthetic data
2. **Test generalization** on yfyeung conversational data
3. **Identify real data sources** for production:
   - Partner hospitals/clinics (with proper consent)
   - Medical transcription services
   - Clinical trial audio (de-identified)

---

**Research Date**: 2025-10-22
**Researcher**: Claude Code (automated)
**Status**: Initial selection complete, ready for POC training
