#!/usr/bin/env python3
"""
Retrain Neural Speaker Classifier with High-Quality Data
Uses the newly classified files from decemberpruitt/ and decemberericah/
"""

import glob
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/app")

# Configuration - Use NEW high-quality data
PRUITT_PATH = "/gateway_instance/decemberpruitt"  # 144 pre-sliced files
ERICAH_PATH = "/gateway_instance/decemberericah"  # 84 pre-sliced files
UPLOADS_PATH = "/gateway_instance/uploads"  # For 'Other' class
TV_PATH = "/gateway_instance/tv_sliced"  # TV audio slices for 'Other' class
OUTPUT_MODEL_PATH = "/gateway_instance/enrollment/speaker_classifier_v2.pth"

# Training Config - More epochs for better convergence
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

# --- TitaNet Model ---
titanet_model = None


def load_titanet():
    global titanet_model
    if titanet_model is None:
        print("[System] Loading TitaNet Model...", flush=True)
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        titanet_model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        titanet_model = titanet_model.to("cpu").eval()
    return titanet_model


def get_embedding(audio_path):
    """Get TitaNet embedding from pre-sliced audio file (no VAD needed)."""
    model = load_titanet()
    try:
        emb = model.get_embedding(audio_path)
        if hasattr(emb, "cpu"):
            emb = emb.cpu().numpy()
        emb = emb.flatten()
        return emb / np.linalg.norm(emb)
    except Exception:
        return None


# --- Neural Network ---
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
            nn.Linear(64, 3),  # 0: Other, 1: Pruitt, 2: Ericah
        )

    def forward(self, x):
        return self.network(x)


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def main():
    print("=" * 80)
    print("RETRAINING NEURAL SPEAKER CLASSIFIER (V2)")
    print("Using High-Quality December Data")
    print("=" * 80)

    # 1. Gather Data
    print("\n[1/4] Gathering Data...")

    pruitt_files = sorted(glob.glob(os.path.join(PRUITT_PATH, "*.wav")))
    ericah_files = sorted(glob.glob(os.path.join(ERICAH_PATH, "*.wav")))

    # Other: Sample 150 random uploads (excluding known files) + ALL TV slices
    known_names = set([os.path.basename(f) for f in pruitt_files + ericah_files])
    all_uploads = glob.glob(os.path.join(UPLOADS_PATH, "*.wav"))
    other_candidates = [f for f in all_uploads if os.path.basename(f) not in known_names]
    random.seed(42)
    other_files = random.sample(other_candidates, min(150, len(other_candidates)))

    # Add TV slices to Other class (critical for filtering TV audio)
    tv_files = sorted(glob.glob(os.path.join(TV_PATH, "*.wav")))
    other_files.extend(tv_files)

    print(f"   Pruitt (December): {len(pruitt_files)}")
    print(f"   Ericah (December): {len(ericah_files)}")
    print(f"   Other (Random):    {len(other_files) - len(tv_files)}")
    print(f"   TV Slices:         {len(tv_files)}")
    print(f"   Total Other:       {len(other_files)}")

    # 2. Extract Embeddings
    print("\n[2/4] Extracting Embeddings...")

    X = []
    y = []  # 0=Other, 1=Pruitt, 2=Ericah

    def process_files(files, label, label_name):
        count = 0
        for i, f in enumerate(files):
            if i % 20 == 0:
                print(f"   {label_name}: {i}/{len(files)}", flush=True)
            emb = get_embedding(f)
            if emb is not None:
                X.append(emb)
                y.append(label)
                count += 1
        print(f"   {label_name}: {count} embeddings extracted")
        return count

    n_p = process_files(pruitt_files, 1, "Pruitt")
    n_e = process_files(ericah_files, 2, "Ericah")
    n_o = process_files(other_files, 0, "Other")

    # 3. Data Augmentation (light noise for robustness)
    print("\n[3/4] Data Augmentation...")

    # Add slight perturbations to create more training samples
    augmented_X = []
    augmented_y = []

    for emb, label in zip(X, y):
        augmented_X.append(emb)
        augmented_y.append(label)

        # Add 2 augmented versions with slight noise
        for _ in range(2):
            noisy = emb + np.random.normal(0, 0.02, 192)
            noisy = noisy / np.linalg.norm(noisy)
            augmented_X.append(noisy)
            augmented_y.append(label)

    print(f"   Original samples: {len(X)}")
    print(f"   Augmented samples: {len(augmented_X)}")

    # 4. Training
    print("\n[4/4] Training Model...")

    dataset = EmbeddingDataset(augmented_X, augmented_y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SpeakerClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"   Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss / len(dataloader):.4f} - Acc: {100 * correct / total:.1f}%"
            )

    # 5. Save
    print("\n[5/5] Saving Model...")
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"   Saved to {OUTPUT_MODEL_PATH}")

    # Also save as the primary classifier
    primary_path = "/gateway_instance/enrollment/speaker_classifier.pth"
    torch.save(model.state_dict(), primary_path)
    print(f"   Also saved to {primary_path}")

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
