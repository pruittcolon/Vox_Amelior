#!/usr/bin/env python3
"""
Prepare TV Training Data

Slices the 4-minute TV audio file into multiple training segments.
These segments will be added to the "Other" class when retraining the speaker classifier.
"""

import os

import numpy as np
import soundfile as sf

# Configuration
TV_SOURCE_DIR = "/gateway_instance/enrollment_disabled/television"
TV_OUTPUT_DIR = "/gateway_instance/tv_sliced"
SEGMENT_DURATION_SEC = 8  # 8 seconds per slice
OVERLAP_SEC = 2  # 2 second overlap for more samples
SAMPLE_RATE = 16000


def slice_audio(audio: np.ndarray, sr: int, segment_duration: float, overlap: float):
    """Slice audio into overlapping segments."""
    segment_samples = int(segment_duration * sr)
    hop_samples = int((segment_duration - overlap) * sr)

    segments = []
    for start in range(0, len(audio) - segment_samples, hop_samples):
        segment = audio[start : start + segment_samples]
        segments.append(segment)

    # Include final segment if there's remaining audio
    if len(audio) > segment_samples:
        segments.append(audio[-segment_samples:])

    return segments


def main():
    print("=" * 60)
    print("PREPARING TV TRAINING DATA")
    print("=" * 60)

    # Create output directory
    os.makedirs(TV_OUTPUT_DIR, exist_ok=True)
    print(f"\n[1/3] Output directory: {TV_OUTPUT_DIR}")

    # Find TV audio files
    tv_files = []
    for f in os.listdir(TV_SOURCE_DIR):
        if f.endswith(".wav"):
            tv_files.append(os.path.join(TV_SOURCE_DIR, f))

    print(f"[2/3] Found {len(tv_files)} TV audio files")

    # Process each file
    total_slices = 0
    for tv_file in tv_files:
        print(f"\n   Processing: {os.path.basename(tv_file)}")

        try:
            audio, sr = sf.read(tv_file)
            duration = len(audio) / sr
            print(f"      Duration: {duration:.1f}s, Sample rate: {sr}Hz")

            # Resample if needed
            if sr != SAMPLE_RATE:
                # Simple resample by ratio (for quick implementation)
                ratio = SAMPLE_RATE / sr
                new_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
                audio = audio[indices]
                sr = SAMPLE_RATE
                print(f"      Resampled to {sr}Hz")

            # Ensure mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Slice into segments
            segments = slice_audio(audio, sr, SEGMENT_DURATION_SEC, OVERLAP_SEC)
            print(f"      Created {len(segments)} segments")

            # Save segments
            base_name = os.path.splitext(os.path.basename(tv_file))[0]
            for i, segment in enumerate(segments):
                output_path = os.path.join(TV_OUTPUT_DIR, f"{base_name}_slice_{i:03d}.wav")
                sf.write(output_path, segment, SAMPLE_RATE)
                total_slices += 1

        except Exception as e:
            print(f"      ERROR: {e}")

    print(f"\n[3/3] Created {total_slices} total TV audio slices")
    print(f"      Location: {TV_OUTPUT_DIR}/")
    print("=" * 60)
    print("DONE! Now run retrain_classifier_v2.py to include TV in training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
