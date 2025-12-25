#!/usr/bin/env python3
"""
Deep diagnostic: Test embedding with short audio segments like streaming does.
"""

import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, "/app")


def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def main():
    print("=" * 60)
    print("DEEP SPEAKER IDENTIFICATION DIAGNOSTIC")
    print("=" * 60)

    # Load TitaNet
    print("\n1. Loading TitaNet model...")
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    model = model.to("cpu")
    model.eval()
    print("   TitaNet loaded!")

    # Load full enrollment audio
    enrollment_path = "/gateway_instance/enrollment/pruitt/enrollment.wav"
    print(f"\n2. Loading enrollment audio: {enrollment_path}")
    audio_full, sr = sf.read(enrollment_path)
    print(f"   Duration: {len(audio_full) / sr:.1f}s, Sample rate: {sr}, Shape: {audio_full.shape}")

    # Extract embedding from FULL audio
    print("\n3. Embedding from FULL 120s enrollment audio:")
    emb_full = model.get_embedding(enrollment_path)
    if hasattr(emb_full, "cpu"):
        emb_full = emb_full.cpu().numpy().flatten()
    norm_full = np.linalg.norm(emb_full)
    emb_full_normalized = emb_full / norm_full if norm_full > 0 else emb_full
    print(f"   Norm: {norm_full:.4f}")

    # Test with SHORT segments (like streaming does)
    print("\n4. Testing with SHORT audio segments (simulating streaming):")
    durations = [2.0, 5.0, 10.0, 15.0, 30.0]

    for dur in durations:
        # Extract a segment from middle of audio
        start_sample = int(30 * sr)  # Start at 30 seconds
        end_sample = start_sample + int(dur * sr)
        segment = audio_full[start_sample:end_sample]

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, segment, sr)
            tmp_path = tmp.name

        try:
            emb_seg = model.get_embedding(tmp_path)
            if hasattr(emb_seg, "cpu"):
                emb_seg = emb_seg.cpu().numpy().flatten()
            norm_seg = np.linalg.norm(emb_seg)
            emb_seg_normalized = emb_seg / norm_seg if norm_seg > 0 else emb_seg

            sim = cosine_similarity(emb_full_normalized, emb_seg_normalized)
            status = "✅" if sim >= 0.6 else "⚠️" if sim >= 0.3 else "❌"
            print(f"   {status} {dur:5.1f}s segment: similarity = {sim:.4f}")
        except Exception as e:
            print(f"   ❌ {dur}s ERROR: {e}")
        finally:
            os.unlink(tmp_path)

    # Test with DIFFERENT part of audio
    print("\n5. Testing different parts of enrollment audio:")
    for i, start_sec in enumerate([0, 40, 80]):
        start_sample = int(start_sec * sr)
        end_sample = start_sample + int(10 * sr)
        segment = audio_full[start_sample:end_sample]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, segment, sr)
            tmp_path = tmp.name

        try:
            emb_seg = model.get_embedding(tmp_path)
            if hasattr(emb_seg, "cpu"):
                emb_seg = emb_seg.cpu().numpy().flatten()
            norm_seg = np.linalg.norm(emb_seg)
            emb_seg_normalized = emb_seg / norm_seg if norm_seg > 0 else emb_seg

            sim = cosine_similarity(emb_full_normalized, emb_seg_normalized)
            print(f"   Part {i + 1} ({start_sec}-{start_sec + 10}s): {sim:.4f}")
        finally:
            os.unlink(tmp_path)

    # Test DIFFERENT audio quality (simulating mobile streaming)
    print("\n6. Testing with quality variations:")

    # Test with noise
    segment = audio_full[int(30 * sr) : int(40 * sr)]
    noise = np.random.randn(len(segment)) * 0.01
    segment_noisy = segment + noise
    segment_noisy = np.clip(segment_noisy, -1.0, 1.0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, segment_noisy, sr)
        tmp_path = tmp.name

    emb_noisy = model.get_embedding(tmp_path)
    if hasattr(emb_noisy, "cpu"):
        emb_noisy = emb_noisy.cpu().numpy().flatten()
    emb_noisy = emb_noisy / np.linalg.norm(emb_noisy)
    sim_noisy = cosine_similarity(emb_full_normalized, emb_noisy)
    print(f"   With 1% noise: {sim_noisy:.4f}")
    os.unlink(tmp_path)

    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("If short segments have low similarity to full audio,")
    print("the enrollment was created from specific audio that")
    print("doesn't match current speech patterns.")
    print("=" * 60)


if __name__ == "__main__":
    main()
