#!/usr/bin/env python3
"""
False Positive Stress Test
Tests 100 random unlabeled files from `uploads` to check for false detections.
Output is printed in real-time.
"""

import glob
import os
import random
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
PRUITT_VERIFIED = "/gateway_instance/pruitt_verified"
ERICAH_VERIFIED = "/gateway_instance/ericah_verified"
MODEL_PATH = "/gateway_instance/enrollment/speaker_classifier.pth"


# Reuse Strict VAD
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


def extract_speech_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    try:
        segments = detect_speech_segments(audio, sr)
        if not segments:
            return None
        speech = np.concatenate([audio[s:e] for s, e in segments])
        if len(speech) < 1.0 * sr:
            return None  # Strict 1s min
        return speech
    except:
        return None


# Models
titanet_model = None


def get_embedding(audio_path):
    global titanet_model
    if titanet_model is None:
        print("[System] Loading TitaNet Model...", flush=True)
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        titanet_model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        titanet_model = titanet_model.to("cpu").eval()

    try:
        audio, sr = sf.read(audio_path)
        speech = extract_speech_audio(audio, sr)
        if speech is None:
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, speech, sr)
            tmp_name = tmp.name

        emb = titanet_model.get_embedding(tmp_name)
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
            nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)


def main():
    print("=" * 105)
    print("FALSE POSITIVE STRESS TEST (100 Unknown Files)")
    print("=" * 105)

    # 1. Load Classifier
    device = torch.device("cpu")
    model = SpeakerClassifier().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("[System] Neural Classifier Loaded.", flush=True)
    except Exception as e:
        print(f"[Error] Could not load classifier: {e}")
        return

    # 2. Select Files
    # Exclude known verified files
    p_files = set([os.path.basename(f) for f in glob.glob(os.path.join(PRUITT_VERIFIED, "*.wav"))])
    e_files = set([os.path.basename(f) for f in glob.glob(os.path.join(ERICAH_VERIFIED, "*.wav"))])
    known_set = p_files.union(e_files)

    all_uploads = glob.glob(os.path.join(UPLOADS_PATH, "*.wav"))
    candidates = [f for f in all_uploads if os.path.basename(f) not in known_set]

    if len(candidates) < 100:
        print(f"[Warning] Only found {len(candidates)} unknown files. Testing all of them.")
        test_files = candidates
    else:
        random.seed(999)  # New seed
        test_files = random.sample(candidates, 100)

    print(f"[System] Testing {len(test_files)} files...", flush=True)
    print("-" * 105)
    print(f"{'#':<3} | {'Filename':<35} | {'P(Pruitt)':<10} | {'P(Ericah)':<10} | {'Decision':<10} | {'Status'}")
    print("-" * 105)

    detections = 0
    silence = 0
    other = 0

    for i, fpath in enumerate(test_files, 1):
        fname = os.path.basename(fpath)[:32]
        emb = get_embedding(fpath)

        if emb is None:
            print(f"{i:<3} | {fname:<35} | {'---':<10} | {'---':<10} | {'SILENCE':<10} | Rejected by VAD", flush=True)
            silence += 1
            continue

        with torch.no_grad():
            t_emb = torch.FloatTensor(emb).unsqueeze(0).to(device)
            logits = model(t_emb)
            probs = F.softmax(logits, dim=1).numpy()[0]

        p_pruitt = probs[1]
        p_ericah = probs[2]

        color = ""
        decision = "Other"
        status = "OK (Negative)"

        if p_pruitt > 0.8:
            decision = "PRUITT"
            status = "POTENTIAL DETECTION"
            detections += 1
        elif p_ericah > 0.8:
            decision = "ERICAH"
            status = "POTENTIAL DETECTION"
            detections += 1
        else:
            other += 1

        print(
            f"{i:<3} | {fname:<35} | {p_pruitt:.3f}      | {p_ericah:.3f}      | {decision:<10} | {status}", flush=True
        )
        # time.sleep(0.1) # readable stream

    print("-" * 105)
    print("Summary:")
    print(f"   Total Tested: {len(test_files)}")
    print(f"   Silence/Noise: {silence}")
    print(f"   Confirmed Other: {other}")
    print(f"   Positive Detections: {detections}")
    print(
        f"   Estimated Specificity: {100 * (silence + other) / len(test_files):.1f}% (Assuming all are truly negative)"
    )
    print("=" * 105)


if __name__ == "__main__":
    main()
