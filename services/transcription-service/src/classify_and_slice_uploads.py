#!/usr/bin/env python3
"""
Comprehensive Audio Classification & Slicing Pipeline
Scans all uploads, classifies speakers, slices to speech-only, and organizes into folders.
"""

import glob
import os
import sys
import tempfile
import warnings

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/app")

# Configuration
UPLOADS_PATH = "/gateway_instance/uploads"
OUTPUT_PRUITT = "/gateway_instance/decemberpruitt"
OUTPUT_ERICAH = "/gateway_instance/decemberericah"
MODEL_PATH = "/gateway_instance/enrollment/speaker_classifier.pth"
CONFIDENCE_THRESHOLD = 0.8


# --- VAD Utils ---
def detect_speech_segments(
    audio: np.ndarray, sr: int = 16000, frame_size: float = 0.025, energy_threshold: float = 0.02
):
    frame_samples = int(frame_size * sr)
    hop_samples = frame_samples // 2
    segments = []
    speech_start = None
    min_speech_duration = int(0.3 * sr)
    min_silence_duration = int(0.3 * sr)
    silence_count = 0

    for i in range(0, len(audio) - frame_samples, hop_samples):
        frame = audio[i : i + frame_samples]
        energy = np.sqrt(np.mean(frame**2))
        if energy > energy_threshold:
            silence_count = 0
            if speech_start is None:
                speech_start = i
        else:
            if speech_start is not None:
                silence_count += hop_samples
                if silence_count >= min_silence_duration:
                    end_sample = i - silence_count + hop_samples
                    if end_sample - speech_start >= min_speech_duration:
                        segments.append((speech_start, end_sample))
                    speech_start = None
                    silence_count = 0
    if speech_start is not None:
        end_sample = len(audio)
        if end_sample - speech_start >= min_speech_duration:
            segments.append((speech_start, end_sample))
    return segments


def extract_speech_audio(audio: np.ndarray, sr: int = 16000):
    """Extract only speech portions. Returns (speech_audio, duration) or (None, 0)."""
    try:
        segments = detect_speech_segments(audio, sr)
        if not segments:
            return None, 0
        speech = np.concatenate([audio[s:e] for s, e in segments])
        duration = len(speech) / sr
        if duration < 1.0:  # Minimum 1 second of speech
            return None, 0
        return speech, duration
    except:
        return None, 0


# --- Models ---
titanet_model = None


def load_titanet():
    global titanet_model
    if titanet_model is None:
        print("[System] Loading TitaNet Model...", flush=True)
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        titanet_model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        titanet_model = titanet_model.to("cpu").eval()
    return titanet_model


def get_embedding(speech_audio: np.ndarray, sr: int = 16000):
    """Get TitaNet embedding from speech audio array."""
    model = load_titanet()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, speech_audio, sr)
            tmp_name = tmp.name

        emb = model.get_embedding(tmp_name)
        os.unlink(tmp_name)

        if hasattr(emb, "cpu"):
            emb = emb.cpu().numpy()
        emb = emb.flatten()
        return emb / np.linalg.norm(emb)
    except:
        return None


class SpeakerClassifier(nn.Module):
    def __init__(self):
        super(SpeakerClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 0=Other, 1=Pruitt, 2=Ericah
        )

    def forward(self, x):
        return self.network(x)


def main():
    print("=" * 110)
    print("COMPREHENSIVE AUDIO CLASSIFICATION & SLICING PIPELINE")
    print("=" * 110)

    # 1. Create output directories
    os.makedirs(OUTPUT_PRUITT, exist_ok=True)
    os.makedirs(OUTPUT_ERICAH, exist_ok=True)
    print("[System] Output folders created:")
    print(f"   - {OUTPUT_PRUITT}")
    print(f"   - {OUTPUT_ERICAH}")

    # 2. Load Neural Classifier
    device = torch.device("cpu")
    classifier = SpeakerClassifier().to(device)
    try:
        classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        classifier.eval()
        print("[System] Neural Classifier Loaded.", flush=True)
    except Exception as e:
        print(f"[Error] Could not load classifier: {e}")
        return

    # 3. Get all WAV files (exclude the output directories)
    all_files = glob.glob(os.path.join(UPLOADS_PATH, "*.wav"))
    print(f"[System] Found {len(all_files)} files to process.", flush=True)
    print("-" * 110)
    print(f"{'#':<5} | {'Filename':<40} | {'Duration':<8} | {'P(Pruitt)':<10} | {'P(Ericah)':<10} | {'Action'}")
    print("-" * 110)

    # Counters
    pruitt_count = 0
    ericah_count = 0
    silence_count = 0
    other_count = 0

    for i, fpath in enumerate(all_files, 1):
        fname = os.path.basename(fpath)
        fname_display = fname[:37] + "..." if len(fname) > 40 else fname

        # Skip files already in output directories
        if "/decemberpruitt/" in fpath or "/decemberericah/" in fpath:
            continue

        # Read audio
        try:
            audio, sr = sf.read(fpath)
        except Exception:
            print(f"{i:<5} | {fname_display:<40} | {'ERROR':<8} | {'---':<10} | {'---':<10} | Read Error", flush=True)
            continue

        # Extract speech
        speech, speech_dur = extract_speech_audio(audio, sr)

        if speech is None:
            print(
                f"{i:<5} | {fname_display:<40} | {'0.0s':<8} | {'---':<10} | {'---':<10} | SKIP (Silence)", flush=True
            )
            silence_count += 1
            continue

        # Get embedding
        emb = get_embedding(speech, sr)
        if emb is None:
            print(
                f"{i:<5} | {fname_display:<40} | {speech_dur:.1f}s     | {'---':<10} | {'---':<10} | SKIP (Embed Error)",
                flush=True,
            )
            silence_count += 1
            continue

        # Classify
        with torch.no_grad():
            t_emb = torch.FloatTensor(emb).unsqueeze(0).to(device)
            logits = classifier(t_emb)
            probs = F.softmax(logits, dim=1).numpy()[0]

        p_pruitt = probs[1]
        p_ericah = probs[2]

        action = "SKIP (Low Confidence)"

        if p_pruitt >= CONFIDENCE_THRESHOLD:
            # Save sliced audio to Pruitt folder
            out_path = os.path.join(OUTPUT_PRUITT, fname)
            sf.write(out_path, speech, sr)
            action = "SAVED -> decemberpruitt/"
            pruitt_count += 1
        elif p_ericah >= CONFIDENCE_THRESHOLD:
            # Save sliced audio to Ericah folder
            out_path = os.path.join(OUTPUT_ERICAH, fname)
            sf.write(out_path, speech, sr)
            action = "SAVED -> decemberericah/"
            ericah_count += 1
        else:
            other_count += 1

        print(
            f"{i:<5} | {fname_display:<40} | {speech_dur:.1f}s     | {p_pruitt:.3f}      | {p_ericah:.3f}      | {action}",
            flush=True,
        )

    # 4. Summary
    print("-" * 110)
    print("SUMMARY")
    print("-" * 110)
    print(f"   Total Files Processed: {len(all_files)}")
    print(f"   Silence/Noise Skipped: {silence_count}")
    print(f"   Other (Low Confidence): {other_count}")
    print(f"   Saved to decemberpruitt/: {pruitt_count}")
    print(f"   Saved to decemberericah/: {ericah_count}")
    print("=" * 110)


if __name__ == "__main__":
    main()
