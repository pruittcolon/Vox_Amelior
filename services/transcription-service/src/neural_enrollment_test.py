#!/usr/bin/env python3
"""
Neural Enrollment Test Bench
Tests the trained Neural Speaker Classifier on the 50-file dataset.
Reports direct Confidence Probability instead of Cosine Similarity.
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
PRUITT_VERIFIED_PATH = "/gateway_instance/pruitt_verified"
ERICAH_VERIFIED_PATH = "/gateway_instance/ericah_verified"
UPLOADS_PATH = "/gateway_instance/uploads"
MODEL_PATH = "/gateway_instance/enrollment/speaker_classifier.pth"


# Reuse Strict VAD Logic from training
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
            return None
        return speech
    except:
        return None


# --- Models ---
# 1. TitaNet
titanet_model = None


def get_embedding(audio_path):
    global titanet_model
    if titanet_model is None:
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


# 2. Classifier
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
    print("NEURAL CLASSIFIER VERIFICATION (Probabilistic Output)")
    print("=" * 105)

    # Load Classifier
    device = torch.device("cpu")
    net = SpeakerClassifier().to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()
    print("Loaded Neural Classifier from disk.")

    # Select Test Files
    random.seed(42)
    p_files = sorted(glob.glob(os.path.join(PRUITT_VERIFIED_PATH, "*.wav")))
    test_p = random.sample(p_files, min(10, len(p_files)))

    e_files = sorted(glob.glob(os.path.join(ERICAH_VERIFIED_PATH, "*.wav")))
    test_e = random.sample(e_files, min(10, len(e_files)))

    verified_names = set([os.path.basename(f) for f in p_files + e_files])
    all_uploads = glob.glob(os.path.join(UPLOADS_PATH, "*.wav"))
    other_candidates = [f for f in all_uploads if os.path.basename(f) not in verified_names]
    test_o = random.sample(other_candidates, min(30, len(other_candidates)))

    all_tests = []
    for f in test_p:
        all_tests.append(("Pruitt", f))
    for f in test_e:
        all_tests.append(("Ericah", f))
    for f in test_o:
        all_tests.append(("Other", f))

    print(
        f"{'#':<3} | {'Type':<10} | {'Filename':<35} | {'P(Pruitt)':<10} | {'P(Ericah)':<10} | {'P(Other)':<10} | {'Result':<10}"
    )
    print("-" * 105)

    correct_count = 0
    valid_count = 0

    for i, (label, fpath) in enumerate(all_tests, 1):
        emb = get_embedding(fpath)

        if emb is None:
            # If rejected by VAD, it's implicitly "Other/Silence"
            print(
                f"{i:<3} | {label:<10} | {os.path.basename(fpath)[:32]:<35} | {'---':<10} | {'---':<10} | {'1.000':<10} | {'SILENCE':<10}"
            )
            if label == "Other":
                correct_count += 1
            valid_count += 1
            continue

        # Inference
        with torch.no_grad():
            t_emb = torch.FloatTensor(emb).unsqueeze(0).to(device)
            logits = net(t_emb)
            probs = F.softmax(logits, dim=1).numpy()[0]

        # 0=Other, 1=Pruitt, 2=Ericah
        p_other = probs[0]
        p_pruitt = probs[1]
        p_ericah = probs[2]

        # Decision
        identified = "Other"
        if p_pruitt > 0.8:
            identified = "Pruitt"
        elif p_ericah > 0.8:
            identified = "Ericah"

        # Check
        is_correct = False
        if (
            label == "Pruitt"
            and identified == "Pruitt"
            or label == "Ericah"
            and identified == "Ericah"
            or label == "Other"
            and identified == "Other"
        ):
            is_correct = True

        mark = "OK" if is_correct else "FAIL"

        print(
            f"{i:<3} | {label:<10} | {os.path.basename(fpath)[:32]:<35} | {p_pruitt:.3f}      | {p_ericah:.3f}      | {p_other:.3f}      | {identified:<7} {mark}"
        )

        if is_correct:
            correct_count += 1
        valid_count += 1

    print("-" * 105)
    acc = (correct_count / valid_count) * 100
    print(f"Total Verified: {valid_count}/{len(all_tests)}")
    print(f"Final Accuracy: {acc:.1f}%")
    print("=" * 105)


if __name__ == "__main__":
    main()
