#!/usr/bin/env python3
"""
Verify Best Enrollment - Detailed Report
Generates a file-by-file report of speaker identification using the saved best embeddings.
"""

import glob
import os
import random
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/app")

# Configuration
PRUITT_VERIFIED_PATH = "/gateway_instance/pruitt_verified"
ERICAH_VERIFIED_PATH = "/gateway_instance/ericah_verified"
UPLOADS_PATH = "/gateway_instance/uploads"
ENROLLMENT_DIR = "/gateway_instance/enrollment"

# Model
model = None


def get_embedding(audio_path: str) -> np.ndarray:
    global model
    if model is None:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        model = model.to("cpu").eval()

    try:
        emb = model.get_embedding(audio_path)
        if hasattr(emb, "cpu"):
            emb = emb.cpu().numpy()
        emb = emb.flatten()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
    except:
        return np.zeros(192)


def main():
    print("=" * 100)
    print("DETAILED VERIFICATION REPORT (50 FILES)")
    print("=" * 100)

    # 1. Load Embeddings
    try:
        p_emb = np.load(os.path.join(ENROLLMENT_DIR, "pruitt_embedding.npy"))
        e_emb = np.load(os.path.join(ENROLLMENT_DIR, "ericah_embedding.npy"))
        print("Loaded enrollment embeddings.")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    # 2. Select Test Files (Same seed as before for consistency)
    random.seed(42)

    # Pruitt
    p_files = sorted(glob.glob(os.path.join(PRUITT_VERIFIED_PATH, "*.wav")))
    test_p = random.sample(p_files, min(10, len(p_files)))

    # Ericah
    e_files = sorted(glob.glob(os.path.join(ERICAH_VERIFIED_PATH, "*.wav")))
    test_e = random.sample(e_files, min(10, len(e_files)))

    # Other
    verified_names = set([os.path.basename(f) for f in p_files + e_files])
    all_uploads = glob.glob(os.path.join(UPLOADS_PATH, "*.wav"))
    other_cands = [f for f in all_uploads if os.path.basename(f) not in verified_names]
    test_o = random.sample(other_cands, min(30, len(other_cands)))

    all_tests = []
    for f in test_p:
        all_tests.append(("Pruitt (Verified)", f))
    for f in test_e:
        all_tests.append(("Ericah (Verified)", f))
    for f in test_o:
        all_tests.append(("Other (Unverified)", f))

    # 3. Run Test
    print(
        f"{'#':<3} | {'Type':<18} | {'Filename':<35} | {'Pruitt':<6} | {'Ericah':<6} | {'Result':<10} | {'Correct?':<8}"
    )
    print("-" * 105)

    correct_count = 0

    for i, (label, fpath) in enumerate(all_tests, 1):
        fname = os.path.basename(fpath)
        emb = get_embedding(fpath)

        sim_p = np.dot(emb, p_emb)
        sim_e = np.dot(emb, e_emb)

        # Identification Logic
        # Match if > 0.50 and highest score matches
        identified = "Unknown"
        max_sim = max(sim_p, sim_e)

        if max_sim > 0.50:
            if sim_p > sim_e:
                identified = "Pruitt"
            else:
                identified = "Ericah"

        # Check Correctness
        is_correct = False
        if (
            "Pruitt" in label
            and identified == "Pruitt"
            or "Ericah" in label
            and identified == "Ericah"
            or "Other" in label
            and identified == "Unknown"
        ):
            is_correct = True

        if is_correct:
            correct_count += 1

        mark = "OK" if is_correct else "FAIL"

        # Truncate filename
        fname_disp = (fname[:32] + "..") if len(fname) > 34 else fname

        print(f"{i:<3} | {label:<18} | {fname_disp:<35} | {sim_p:.3f}  | {sim_e:.3f}  | {identified:<10} | {mark:<8}")

    print("-" * 105)
    acc = (correct_count / len(all_tests)) * 100
    print(f"Total Computation: {len(all_tests)} files")
    print(f"Final Validation Accuracy: {acc:.1f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()
