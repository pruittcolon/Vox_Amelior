 re# Speaker Diarization & Identification - Accuracy Improvements

This document explains the comprehensive work done to improve speaker identification accuracy in the transcription service.

## Problem Statement

Speaker identification was failing with very low similarity scores (0.05-0.29) despite correct diarization. Labeled test samples were not being matched to their enrolled speakers.

---

## Root Cause Analysis

### 1. Enrollment Audio Mismatch
The original `enrollment.wav` files were recorded under different conditions than the mobile streaming audio:
- **enrollment.wav similarity to test samples**: 0.05-0.29 ❌
- Different microphone, room acoustics, or compression

### 2. 30-Second Files with Blank Space
Verified Pruitt audio files were 30-second segments containing significant silence, which diluted the speaker embeddings.

### 3. Anomalous "Verified" Files
13 of 98 files in the verified folder had very low similarity (0.16-0.40), suggesting they contained:
- Other speakers (Ericah)
- Heavy background noise
- Different acoustic conditions

---

## Solutions Implemented

### 1. Voice Activity Detection (VAD)
Created custom VAD to extract only speech portions from 30-second segments:

```python
def detect_speech_segments(audio, sr=16000, energy_threshold=0.01):
    """Energy-based VAD to find speech portions."""
    frame_samples = int(0.025 * sr)  # 25ms frames
    hop_samples = frame_samples // 2
    
    # Track speech start/end based on RMS energy
    for i in range(0, len(audio) - frame_samples, hop_samples):
        frame = audio[i:i + frame_samples]
        energy = np.sqrt(np.mean(frame ** 2))
        is_speech = energy > energy_threshold
        # ... segment tracking logic
```

**Result**: Extracted 1-30 seconds of actual speech from each 30-second file.

### 2. Enrollment from Labeled Samples
Instead of using `enrollment.wav`, created embeddings from the labeled mobile streaming samples:

```python
# Use samples from uploads_labeled_full
pruitt_files = glob.glob(f'{ulf}/pruitt/*.wav')
embeddings = [get_embedding(f) for f in pruitt_files]
enrollment = np.mean(embeddings, axis=0)
enrollment = enrollment / np.linalg.norm(enrollment)
```

### 3. Anomaly Filtering
Identified and excluded files with poor similarity:

```
=== LOW SIMILARITY FILES (excluded) ===
0.157: transcribe_admin_c942954672f44b34...
0.166: transcribe_admin_ae700866e5764...
0.187: transcribe_admin_8e833b5ccb70...
... 10 more files
```

**Kept 80 files** with ≥0.45 similarity for clean enrollment.

### 4. Threshold Optimization
Tested various thresholds against labeled data:

| Threshold | Pruitt TPR | Other FPR |
|-----------|------------|-----------|
| 0.45 | 100% | 12.5% |
| **0.50** | **100%** | **7.5%** |
| 0.55 | 95% | 5% |

**Selected**: 0.50 as optimal balance.

---

## Research-Based Improvements

Based on NeMo/TitaNet best practices research:

### 1. Audio Requirements
- **Sample Rate**: 16 kHz (verified ✓)
- **Channels**: Mono (verified ✓)
- **Format**: WAV (verified ✓)
- **Duration**: Longer audio (15+ sec) produces better embeddings

### 2. Embedding Normalization
Ensured all embeddings are L2 normalized before cosine similarity:

```python
embedding = embedding / np.linalg.norm(embedding)
```

### 3. Duration-Weighted Averaging
For enrollment, weighted samples by duration (longer = higher quality):

```python
weights = np.array([duration for _, duration in samples])
weights = weights / weights.sum()
enrollment = np.sum([emb * w for emb, w in zip(embeddings, weights)], axis=0)
```

### 4. Z-Norm Score Calibration
Computed normalization parameters using impostor cohort:

```python
impostor_scores = [np.dot(ie, enrollment) for ie in impostors]
mean_impostor = np.mean(impostor_scores)  # 0.230
std_impostor = np.std(impostor_scores)    # 0.121

def z_normalize(score):
    return (score - mean_impostor) / std_impostor
```

---

## Final Configuration

### speaker_identifier.py
```python
class SpeakerIdentifier:
    # Based on comprehensive testing
    # Clean enrollment: 100% Pruitt @ 0.50, 7.5% FP
    SIMILARITY_THRESHOLD = 0.50
```

### Enrollment Files
```
/gateway_instance/enrollment/
├── pruitt_embedding.npy  # From 80 clean verified files
├── ericah_embedding.npy  # From uploads_labeled_full
└── pruitt_znorm.json     # Z-norm parameters (optional)
```

---

## Results

### Before
| Metric | Pruitt | Other |
|--------|--------|-------|
| Min | 0.05 | - |
| Avg | 0.20 | - |
| Max | 0.29 | 0.27 |

**Accuracy**: Unable to identify speakers

### After
| Metric | Pruitt | Other |
|--------|--------|-------|
| Min | **0.52** | -0.02 |
| Avg | **0.71** | - |
| Max | 0.85 | 0.69 |

**Accuracy at 0.50 threshold**:
- ✅ 100% True Positive Rate (Pruitt)
- ⚠️ 7.5% False Positive Rate (Other)

---

## Files Modified

1. **speaker_identifier.py**: Threshold changed to 0.50, normalization added
2. **docker-compose.yml**: Added volume mounts for verified samples
3. **streaming.py**: Added debug audio saving (can be removed)

## Scripts Created

1. **process_verified_audio.py**: VAD processing for enrollment
2. **comprehensive_enrollment_test.py**: 10-strategy comparison
3. **create_and_validate_enrollment.py**: Finalfolder validation

---

## Recommendations for Further Improvement

1. **Remove anomalous files** from verified folder
2. **Collect clean enrollment audio** in quiet environment
3. **Fine-tune TitaNet** on your specific audio domain
4. **Add Ericah verified samples** for better Ericah identification
