#!/usr/bin/env python3
"""
Test Retrained Classifier V2
Runs the new classifier on sample files and compares to expected labels.
"""

import glob
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/app")

# Configuration
PRUITT_PATH = "/gateway_instance/decemberpruitt"
ERICAH_PATH = "/gateway_instance/decemberericah"
UPLOADS_PATH = "/gateway_instance/uploads"
MODEL_PATH = "/gateway_instance/enrollment/speaker_classifier.pth"

# --- Models ---
titanet_model = None


def get_embedding(audio_path):
    global titanet_model
    if titanet_model is None:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        titanet_model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        titanet_model = titanet_model.to("cpu").eval()

    try:
        emb = titanet_model.get_embedding(audio_path)
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
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.network(x)


def main():
    print("=" * 100)
    print("TESTING RETRAINED CLASSIFIER V2")
    print("=" * 100)

    # Load Classifier
    device = torch.device("cpu")
    model = SpeakerClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[System] Loaded Classifier V2", flush=True)

    # Select test samples (different from training set)
    random.seed(999)  # Different seed from training

    pruitt_files = sorted(glob.glob(os.path.join(PRUITT_PATH, "*.wav")))
    ericah_files = sorted(glob.glob(os.path.join(ERICAH_PATH, "*.wav")))

    # Test on 20 of each
    test_pruitt = random.sample(pruitt_files, min(20, len(pruitt_files)))
    test_ericah = random.sample(ericah_files, min(20, len(ericah_files)))

    # Random Other files
    known_names = set([os.path.basename(f) for f in pruitt_files + ericah_files])
    all_uploads = glob.glob(os.path.join(UPLOADS_PATH, "*.wav"))
    other_candidates = [f for f in all_uploads if os.path.basename(f) not in known_names]
    test_other = random.sample(other_candidates, min(20, len(other_candidates)))

    all_tests = []
    for f in test_pruitt:
        all_tests.append(("Pruitt", f))
    for f in test_ericah:
        all_tests.append(("Ericah", f))
    for f in test_other:
        all_tests.append(("Other", f))

    print(f"Testing {len(all_tests)} files...\n")
    print(
        f"{'#':<3} | {'Expected':<8} | {'P(Pruitt)':<10} | {'P(Ericah)':<10} | {'P(Other)':<10} | {'Decision':<10} | {'Status'}"
    )
    print("-" * 85)

    correct = 0
    total = 0

    for i, (label, fpath) in enumerate(all_tests, 1):
        emb = get_embedding(fpath)

        if emb is None:
            print(f"{i:<3} | {label:<8} | {'---':<10} | {'---':<10} | {'---':<10} | {'ERROR':<10} | Embed fail")
            continue

        with torch.no_grad():
            t_emb = torch.FloatTensor(emb).unsqueeze(0).to(device)
            logits = model(t_emb)
            probs = F.softmax(logits, dim=1).numpy()[0]

        p_other = probs[0]
        p_pruitt = probs[1]
        p_ericah = probs[2]

        # Decision
        decision = "Other"
        if p_pruitt > 0.8:
            decision = "Pruitt"
        elif p_ericah > 0.8:
            decision = "Ericah"

        # Check
        is_correct = label == decision
        mark = "OK" if is_correct else "FAIL"

        if is_correct:
            correct += 1
        total += 1

        print(
            f"{i:<3} | {label:<8} | {p_pruitt:.3f}      | {p_ericah:.3f}      | {p_other:.3f}      | {decision:<10} | {mark}"
        )

    print("-" * 85)
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"Total: {total} files")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.1f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()
