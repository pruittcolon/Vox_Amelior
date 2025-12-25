#!/usr/bin/env python3
"""
Save live streaming audio to diagnose format differences.
"""

import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, "/app")


def main():
    print("=" * 60)
    print("DIAGNOSE LIVE AUDIO VS VERIFIED SAMPLES")
    print("=" * 60)

    # Load TitaNet
    print("\n1. Loading TitaNet model...")
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    model = model.to("cpu")
    model.eval()
    print("   TitaNet loaded!")

    # Load and analyze verified sample
    verified_sample = "/gateway_instance/pruitt_verified/pruitt_test_snippet.wav"
    print(f"\n2. Analyzing verified sample: {verified_sample}")

    verified_audio, verified_sr = sf.read(verified_sample)
    print(f"   Sample rate: {verified_sr}")
    print(f"   Shape: {verified_audio.shape}")
    print(f"   Dtype: {verified_audio.dtype}")
    print(f"   Range: [{verified_audio.min():.4f}, {verified_audio.max():.4f}]")
    print(f"   Mean: {verified_audio.mean():.6f}")
    print(f"   Std: {verified_audio.std():.4f}")

    # Extract embedding from verified sample
    verified_emb = model.get_embedding(verified_sample)
    if hasattr(verified_emb, "cpu"):
        verified_emb = verified_emb.cpu().numpy().flatten()
    verified_norm = np.linalg.norm(verified_emb)
    verified_emb_normalized = verified_emb / verified_norm if verified_norm > 0 else verified_emb
    print(f"   Embedding norm: {verified_norm:.4f}")

    # Load current enrollment
    enrollment_path = "/gateway_instance/enrollment/pruitt_embedding.npy"
    enrollment_emb = np.load(enrollment_path)
    enrollment_norm = np.linalg.norm(enrollment_emb)
    enrollment_emb_normalized = enrollment_emb / enrollment_norm if enrollment_norm > 0 else enrollment_emb

    # Cosine similarity
    sim = np.dot(verified_emb_normalized, enrollment_emb_normalized)
    print(f"   Similarity to enrollment: {sim:.4f}")

    # Now simulate what streaming does with audio
    print("\n3. Simulating streaming audio processing:")

    # Read as int16 (like mobile app sends)
    with open(verified_sample, "rb") as f:
        raw_bytes = f.read()

    # Skip WAV header (44 bytes typically)
    audio_bytes = raw_bytes[44:]

    # Convert like streaming does: int16 PCM -> float32
    audio_int16 = np.frombuffer(audio_bytes[: len(audio_bytes) // 2 * 2], dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    print(f"   Int16 range: [{audio_int16.min()}, {audio_int16.max()}]")
    print(f"   Float32 range: [{audio_float32.min():.4f}, {audio_float32.max():.4f}]")
    print(f"   Float32 mean: {audio_float32.mean():.6f}")
    print(f"   Float32 std: {audio_float32.std():.4f}")

    # Save and test
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_float32, verified_sr)
        tmp_path = tmp.name

    simulated_emb = model.get_embedding(tmp_path)
    if hasattr(simulated_emb, "cpu"):
        simulated_emb = simulated_emb.cpu().numpy().flatten()
    simulated_norm = np.linalg.norm(simulated_emb)
    simulated_emb_normalized = simulated_emb / simulated_norm if simulated_norm > 0 else simulated_emb

    sim_to_verified = np.dot(simulated_emb_normalized, verified_emb_normalized)
    sim_to_enrollment = np.dot(simulated_emb_normalized, enrollment_emb_normalized)

    print(f"   Simulated embedding norm: {simulated_norm:.4f}")
    print(f"   Similarity to direct read: {sim_to_verified:.4f}")
    print(f"   Similarity to enrollment: {sim_to_enrollment:.4f}")

    os.unlink(tmp_path)

    # Check if issue is sample rate
    print("\n4. Testing sample rate impact:")
    for target_sr in [8000, 16000, 44100, 48000]:
        if target_sr != verified_sr:
            # Resample
            from scipy import signal

            resampled = signal.resample(verified_audio, int(len(verified_audio) * target_sr / verified_sr))
        else:
            resampled = verified_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, resampled.astype(np.float32), target_sr)
            tmp_path = tmp.name

        try:
            sr_emb = model.get_embedding(tmp_path)
            if hasattr(sr_emb, "cpu"):
                sr_emb = sr_emb.cpu().numpy().flatten()
            sr_emb_normalized = sr_emb / np.linalg.norm(sr_emb)

            sim = np.dot(sr_emb_normalized, enrollment_emb_normalized)
            print(f"   {target_sr}Hz: similarity = {sim:.4f}")
        except Exception as e:
            print(f"   {target_sr}Hz: ERROR - {e}")
        finally:
            os.unlink(tmp_path)

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("If simulated streaming audio has low similarity,")
    print("the issue is audio format/conversion. Check mobile app PCM format.")
    print("=" * 60)


if __name__ == "__main__":
    main()
