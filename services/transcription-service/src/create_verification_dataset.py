#!/usr/bin/env python3
"""
Create Verification Dataset (200 random files)
Classifies files and renames them with confidence scores for manual review.
"""

import glob
import os
import random
import shutil
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
OUTPUT_BASE = "/gateway_instance/verification_review_200"
MODEL_PATH = "/gateway_instance/enrollment/speaker_classifier_v2.pth"
THRESHOLD = 0.7  # User requested 0.7

# --- Models ---
titanet_model = None


def get_embedding(audio_path):
    global titanet_model
    if titanet_model is None:
        print("[System] Loading TitaNet...", flush=True)
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        titanet_model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        titanet_model = titanet_model.to("cpu").eval()

    try:
        # Note: For verification, we want to classify the file *as is* (or with VAD).
        # Let's use VAD to be consistent with the classifier's training.
        audio, sr = sf.read(audio_path)

        # Simple energy VAD to extract speech
        frame_len = int(sr * 0.025)
        hop = int(frame_len / 2)
        energy_thresh = 0.02

        speech_segments = []
        for i in range(0, len(audio) - frame_len, hop):
            frame = audio[i : i + frame_len]
            if np.sqrt(np.mean(frame**2)) > energy_thresh:
                speech_segments.append(frame)

        if len(speech_segments) == 0:
            return None  # Silence

        speech = np.concatenate(speech_segments)
        if len(speech) < 1.0 * sr:
            return None  # Too short

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
    print("CREATING VERIFICATION DATASET (200 random files)")
    print("=" * 100)

    # 1. Load Model
    device = torch.device("cpu")
    model = SpeakerClassifier().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("[System] Loaded Classifier V2")
    except:
        print("[Error] Could not load V2 model. Falling back to V1 or failing.")
        model.load_state_dict(torch.load("/gateway_instance/enrollment/speaker_classifier.pth", map_location=device))

    # 2. Select 200 Random Files
    all_uploads = glob.glob(os.path.join(UPLOADS_PATH, "*.wav"))
    random.seed(12345)  # Fixed seed for reproducibility
    selected_files = random.sample(all_uploads, min(200, len(all_uploads)))

    print(f"[System] Selected {len(selected_files)} files.")
    print("-" * 100)
    print(f"{'Filename':<30} | {'P(Pruitt)':<10} | {'P(Ericah)':<10} | {'Decision'}")
    print("-" * 100)

    counts = {"Pruitt": 0, "Ericah": 0, "Other": 0, "Silence": 0}

    for i, fpath in enumerate(selected_files):
        fname = os.path.basename(fpath)
        emb = get_embedding(fpath)

        p_pruitt = 0.0
        p_ericah = 0.0
        category = "Other"

        if emb is None:
            category = "Silence"
            counts["Silence"] += 1
            # Still copy silence files to 'Other' folder for review?
            # Or make a 'silence' folder? User said 'other'.
            # Let's put them in 'other' but marked as silence.
            p_pruitt = 0.0
            p_ericah = 0.0
        else:
            with torch.no_grad():
                t_emb = torch.FloatTensor(emb).unsqueeze(0).to(device)
                logits = model(t_emb)
                probs = F.softmax(logits, dim=1).numpy()[0]
                p_pruitt = probs[1]
                p_ericah = probs[2]

            if p_pruitt >= THRESHOLD:
                category = "Pruitt"
                counts["Pruitt"] += 1
            elif p_ericah >= THRESHOLD:
                category = "Ericah"
                counts["Ericah"] += 1
            else:
                counts["Other"] += 1

        # Rename logic: p[score]_e[score]_[name]
        # example: p0.98_e0.00_file.wav
        new_name = f"pruitt{p_pruitt:.2f}_ericah{p_ericah:.2f}_{fname}"

        # Determine subfolder
        subfolder = category.lower()
        if subfolder == "silence":
            subfolder = "other"  # Group silence with other

        dest_path = os.path.join(OUTPUT_BASE, subfolder, new_name)

        # Copy file (COPY, do not move)
        shutil.copy2(fpath, dest_path)

        print(f"{fname[:28]:<30} | {p_pruitt:.3f}      | {p_ericah:.3f}      | {category}")

    print("-" * 100)
    print("SUMMARY")
    print(f"   Pruitt: {counts['Pruitt']}")
    print(f"   Ericah: {counts['Ericah']}")
    print(f"   Other:  {counts['Other']} (includes {counts['Silence']} silence)")
    print(f"   Total:  {len(selected_files)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
