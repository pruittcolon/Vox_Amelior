#!/usr/bin/env python3
"""
Process verified Pruitt audio using VAD to extract speech segments,
then create enrollment embeddings.
"""

import glob
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, "/app")


def detect_speech_segments(
    audio: np.ndarray, sr: int = 16000, frame_size: float = 0.025, energy_threshold: float = 0.01
) -> list[tuple[int, int]]:
    """
    Simple energy-based VAD to detect speech segments.
    Returns list of (start_sample, end_sample) tuples.
    """
    frame_samples = int(frame_size * sr)
    hop_samples = frame_samples // 2

    # Calculate RMS energy per frame
    segments = []
    speech_start = None
    min_speech_duration = int(0.3 * sr)  # Minimum 300ms speech
    min_silence_duration = int(0.2 * sr)  # 200ms silence to end segment

    silence_count = 0

    for i in range(0, len(audio) - frame_samples, hop_samples):
        frame = audio[i : i + frame_samples]
        energy = np.sqrt(np.mean(frame**2))

        is_speech = energy > energy_threshold

        if is_speech:
            silence_count = 0
            if speech_start is None:
                speech_start = i
        else:
            if speech_start is not None:
                silence_count += hop_samples
                if silence_count >= min_silence_duration:
                    # End of speech segment
                    end_sample = i - silence_count + hop_samples
                    if end_sample - speech_start >= min_speech_duration:
                        segments.append((speech_start, end_sample))
                    speech_start = None
                    silence_count = 0

    # Handle trailing speech
    if speech_start is not None:
        end_sample = len(audio)
        if end_sample - speech_start >= min_speech_duration:
            segments.append((speech_start, end_sample))

    return segments


def extract_speech_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract only speech portions from audio using VAD."""
    segments = detect_speech_segments(audio, sr)

    if not segments:
        # If no speech detected, try lower threshold
        segments = detect_speech_segments(audio, sr, energy_threshold=0.005)

    if not segments:
        return audio  # Return original if still no speech

    # Concatenate all speech segments
    speech_parts = []
    for start, end in segments:
        speech_parts.append(audio[start:end])

    return np.concatenate(speech_parts)


def main():
    print("=" * 60)
    print("PROCESS VERIFIED PRUITT AUDIO WITH VAD")
    print("=" * 60)

    # Load TitaNet
    print("\n1. Loading TitaNet model...", flush=True)
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    model.to("cpu").eval()
    print("   Done!", flush=True)

    def get_emb(audio_path: str = None, audio: np.ndarray = None, sr: int = 16000) -> np.ndarray:
        if audio_path:
            emb = model.get_embedding(audio_path)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, sr)
                emb = model.get_embedding(tmp.name)
                os.unlink(tmp.name)
        if hasattr(emb, "cpu"):
            emb = emb.cpu().numpy()
        emb = emb.flatten()
        return emb / np.linalg.norm(emb)

    # Find verified Pruitt files
    verified_dir = "/gateway_instance/pruitt_verified_backup"
    wav_files = sorted(glob.glob(f"{verified_dir}/*.wav"))
    print(f"\n2. Found {len(wav_files)} WAV files", flush=True)

    # Process each file with VAD
    print("\n3. Processing files with VAD...", flush=True)
    all_embeddings = []
    processed_count = 0

    for i, wav_file in enumerate(wav_files):
        fname = os.path.basename(wav_file)
        try:
            audio, sr = sf.read(wav_file)
            original_duration = len(audio) / sr

            # Extract speech using VAD
            speech_audio = extract_speech_audio(audio, sr)
            speech_duration = len(speech_audio) / sr

            if speech_duration < 0.5:
                print(f"   [{i + 1}/{len(wav_files)}] {fname[:30]}: No speech detected, skipping")
                continue

            # Extract embedding from speech portion
            emb = get_emb(audio=speech_audio, sr=sr)
            all_embeddings.append(emb)
            processed_count += 1

            print(
                f"   [{i + 1}/{len(wav_files)}] {fname[:30]}: {speech_duration:.1f}s speech / {original_duration:.1f}s total",
                flush=True,
            )

        except Exception as e:
            print(f"   [{i + 1}/{len(wav_files)}] {fname[:30]}: ERROR - {e}")

    print(f"\n4. Processed {processed_count} files successfully", flush=True)

    if not all_embeddings:
        print("   ERROR: No embeddings extracted!")
        return

    # Create averaged embedding
    print("\n5. Creating Pruitt enrollment embedding...", flush=True)
    avg_embedding = np.mean(all_embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    # Save embedding
    output_path = "/gateway_instance/enrollment/pruitt_embedding.npy"
    np.save(output_path, avg_embedding.astype(np.float32))
    print(f"   Saved to: {output_path}")

    # Test against some files
    print("\n6. Validation test...", flush=True)
    sims = []
    for wav_file in wav_files[:10]:
        try:
            audio, sr = sf.read(wav_file)
            speech_audio = extract_speech_audio(audio, sr)
            if len(speech_audio) / sr >= 0.5:
                emb = get_emb(audio=speech_audio, sr=sr)
                sim = np.dot(emb, avg_embedding)
                sims.append(sim)
                print(f"   {os.path.basename(wav_file)[:30]}: {sim:.3f}")
        except:
            pass

    if sims:
        print(f"\n   Similarity: min={min(sims):.3f}, avg={np.mean(sims):.3f}, max={max(sims):.3f}")

    # Test against Other samples
    print("\n7. Testing against Other samples...", flush=True)
    other_files = glob.glob("/gateway_instance/uploads_labeled_full/other/*.wav")
    other_sims = []
    for f in other_files[:20]:
        try:
            emb = get_emb(audio_path=f)
            sim = np.dot(emb, avg_embedding)
            other_sims.append(sim)
        except:
            pass

    if other_sims:
        print(f"   Other max similarity: {max(other_sims):.3f}")
        if max(other_sims) < min(sims):
            margin = min(sims) - max(other_sims)
            print(f"   Good separation margin: {margin:.3f}")
        else:
            print("   WARNING: Overlap between Pruitt and Other!")

    print("\n" + "=" * 60)
    print("DONE! Restart transcription-service to use new embedding.")
    print("=" * 60)


if __name__ == "__main__":
    main()
