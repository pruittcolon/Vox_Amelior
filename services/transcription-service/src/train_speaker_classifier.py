#!/usr/bin/env python3
"""
Train Neural Speaker Classifier (MLP)
Trains a PyTorch MLP on TitaNet embeddings to classify Pruitt vs Ericah vs Other.
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/app")

# Configuration
PRUITT_PATH = "/gateway_instance/pruitt_verified"
ERICAH_PATH = "/gateway_instance/ericah_verified"
UPLOADS_PATH = "/gateway_instance/uploads"
OUTPUT_MODEL_PATH = "/gateway_instance/enrollment/speaker_classifier.pth"

# Training Config
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001


# --- VAD Utils ---
# --- VAD Utils ---
def detect_speech_segments(
    audio: np.ndarray, sr: int = 16000, frame_size: float = 0.025, energy_threshold: float = 0.02
):  # Increased threshold from 0.01 to 0.02
    frame_samples = int(frame_size * sr)
    hop_samples = frame_samples // 2

    # Calculate global energy profile to help adaptive thresholding
    # (Simple heuristic: if constant noise, RMS will be high everywhere)
    # For now, just using stricter constant threshold + min duration

    segments = []
    speech_start = None
    min_speech_duration = int(0.3 * sr)
    min_silence_duration = int(0.3 * sr)  # Increased silence tolerance to merge chopping words
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
        # Fallback only if totally empty, but be careful of noise
        if not segments:
            # Only fallback if file is verified, otherwise it's just silence
            return None

        # Concatenate speech
        speech = np.concatenate([audio[s:e] for s, e in segments])

        # FINAL CHECK: Must have at least 1.0s of speech
        if len(speech) < 1.0 * sr:
            return None

        return speech
    except:
        return None


# --- TitaNet Model ---
titanet_model = None


def get_embedding(audio_path, label_name="Unknown"):
    global titanet_model
    if titanet_model is None:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        titanet_model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        titanet_model = titanet_model.to("cpu").eval()

    try:
        audio, sr = sf.read(audio_path)
        original_dur = len(audio) / sr

        # Pre-process with VAD
        speech = extract_speech_audio(audio, sr)

        if speech is None:
            # print(f"   [Skipped] {os.path.basename(audio_path)}: No valid speech detected (len={original_dur:.1f}s)")
            return None

        speech_dur = len(speech) / sr
        # print(f"   [Processed] {os.path.basename(audio_path)}: {speech_dur:.1f}s speech (from {original_dur:.1f}s)")

        # Save temp for NeMo
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, speech, sr)
            tmp_name = tmp.name

        emb = titanet_model.get_embedding(tmp_name)
        os.unlink(tmp_name)

        if hasattr(emb, "cpu"):
            emb = emb.cpu().numpy()
        emb = emb.flatten()
        return emb / np.linalg.norm(emb)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
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
            nn.Linear(64, 3),  # 0: Other, 1: Pruitt, 2: Ericah
            # Note: Typically standard is usually mapping 0,1,2.
            # Let's define: 0=Other, 1=Pruitt, 2=Ericah
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
    print("TRAINING NEURAL SPEAKER CLASSIFIER")
    print("=" * 80)

    # 1. Gather Data
    print("\n[1/4] Gathering Data...")

    pruitt_files = glob.glob(os.path.join(PRUITT_PATH, "*.wav"))
    ericah_files = glob.glob(os.path.join(ERICAH_PATH, "*.wav"))

    # Other: Sample 100 random uploads that are NOT Verified
    verified_names = set([os.path.basename(f) for f in pruitt_files + ericah_files])
    all_uploads = glob.glob(os.path.join(UPLOADS_PATH, "*.wav"))
    other_candidates = [f for f in all_uploads if os.path.basename(f) not in verified_names]
    other_files = random.sample(other_candidates, min(100, len(other_candidates)))

    print(f"   Pruitt: {len(pruitt_files)}")
    print(f"   Ericah: {len(ericah_files)}")
    print(f"   Other:  {len(other_files)}")

    # 2. Extract Embeddings
    print("\n[2/4] Extracting Embeddings (this may take a minute)...")

    X = []
    y = []  # 0=Other, 1=Pruitt, 2=Ericah

    def process_files(files, label, label_name):
        count = 0
        for f in files:
            emb = get_embedding(f)
            if emb is not None:
                X.append(emb)
                y.append(label)
                count += 1
        print(f"   Processed {count} {label_name} embeddings")
        return count

    n_p = process_files(pruitt_files, 1, "Pruitt")
    n_e = process_files(ericah_files, 2, "Ericah")
    n_o = process_files(other_files, 0, "Other")

    # 3. Balancing (Simple Oversampling)
    # Target ~100 per class
    target_count = 100

    # Balance Ericah (Iterate existing e_indices)
    e_indices = [i for i, label in enumerate(y) if label == 2]
    if e_indices:
        needed = target_count - n_e
        if needed > 0:
            print(f"   Oversampling Ericah by {needed} samples...")
            for _ in range(needed):
                idx = random.choice(e_indices)
                # Add slight noise to prevent exact duplicates (Mixup-lite)
                emb = X[idx] + np.random.normal(0, 0.01, 192)
                emb = emb / np.linalg.norm(emb)
                X.append(emb)
                y.append(2)

    # 4. Training
    print("\n[3/4] Training Model...")

    dataset = EmbeddingDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SpeakerClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

        if (epoch + 1) % 10 == 0:
            print(
                f"   Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss / len(dataloader):.4f} - Acc: {100 * correct / total:.1f}%"
            )

    # 5. Save
    print("\n[4/4] Saving Model...")
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"   Saved to {OUTPUT_MODEL_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
