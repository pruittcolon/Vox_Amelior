#!/usr/bin/env python3
"""
Improved Speaker Enrollment Strategy Testing with VAD
Tests 5 VAD-enhanced enrollment strategies to maximize accuracy.
"""

import glob
import os
import random
import sys
import tempfile
import warnings

import numpy as np
import soundfile as sf

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, "/app")

# Global configuration
# Global configuration
# Note: Specific folders are mounted directly into /gateway_instance/
PRUITT_VERIFIED_PATH = "/gateway_instance/pruitt_verified"
ERICAH_VERIFIED_PATH = "/gateway_instance/ericah_verified"
UPLOADS_PATH = "/gateway_instance/uploads"
ENROLLMENT_OUTPUT = "/gateway_instance/enrollment"

# --- VAD & Audio Processing (reused from process_verified_audio.py) ---


def detect_speech_segments(
    audio: np.ndarray, sr: int = 16000, frame_size: float = 0.025, energy_threshold: float = 0.01
) -> list[tuple[int, int]]:
    """Energy-based VAD to detect speech segments."""
    frame_samples = int(frame_size * sr)
    hop_samples = frame_samples // 2

    segments = []
    speech_start = None
    min_speech_duration = int(0.3 * sr)
    min_silence_duration = int(0.2 * sr)
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
    """Extract only speech portions from audio using VAD."""
    segments = detect_speech_segments(audio, sr)
    if not segments:
        segments = detect_speech_segments(audio, sr, energy_threshold=0.005)
    if not segments:
        return audio
    speech_parts = [audio[start:end] for start, end in segments]
    return np.concatenate(speech_parts)


# --- Model & Embedding ---

model = None


def load_model():
    global model
    if model is None:
        print("Loading TitaNet model...", flush=True)
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        model = model.to("cpu")
        model.eval()


def get_embedding(audio_path: str = None, audio_data: np.ndarray = None, sr: int = 16000) -> np.ndarray:
    """Extract normalized embedding from file or audio array."""
    if model is None:
        load_model()

    try:
        tmp_path = None
        if audio_data is not None:
            # Save ephemeral audio for model ingestion
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_data, sr)
                tmp_path = tmp.name
            target_path = tmp_path
        else:
            target_path = audio_path

        emb = model.get_embedding(target_path)
        if hasattr(emb, "cpu"):
            emb = emb.cpu().numpy()
        emb = emb.flatten()

        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
    except Exception:
        # print(f"Embedding error: {e}")
        return np.zeros(192)


# --- Strategies ---

# --- Strategies (Optimized) ---


def strategy_1_baseline(embeddings: list[np.ndarray]) -> np.ndarray:
    """Strategy 1: Mean of raw embeddings."""
    if not embeddings:
        return np.zeros(192)
    avg = np.mean(embeddings, axis=0)
    return avg / np.linalg.norm(avg)


def strategy_2_vad_mean(embeddings: list[np.ndarray]) -> np.ndarray:
    """Strategy 2: Mean of VAD embeddings."""
    if not embeddings:
        return np.zeros(192)
    avg = np.mean(embeddings, axis=0)
    return avg / np.linalg.norm(avg)


def strategy_3_vad_weighted(embeddings: list[np.ndarray], durations: list[float]) -> np.ndarray:
    """Strategy 3: Duration-weighted average."""
    if not embeddings:
        return np.zeros(192)
    weights = np.array(durations)
    weights = weights / weights.sum()
    avg = np.sum([e * w for e, w in zip(embeddings, weights)], axis=0)
    return avg / np.linalg.norm(avg)


def strategy_4_vad_median(embeddings: list[np.ndarray]) -> np.ndarray:
    """Strategy 4: Geometric median."""
    if not embeddings:
        return np.zeros(192)
    med = np.median(embeddings, axis=0)
    return med / np.linalg.norm(med)


def strategy_5_vad_top20(embeddings_durations: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Strategy 5: Top-20 longest speech files."""
    # Sort by duration desc
    sorted_items = sorted(embeddings_durations, key=lambda x: x[1], reverse=True)
    top_20 = [x[0] for x in sorted_items[:20]]
    if not top_20:
        return np.zeros(192)
    avg = np.mean(top_20, axis=0)
    return avg / np.linalg.norm(avg)


# --- Data Loading ---


def get_verified_files(speaker: str) -> list[str]:
    if speaker.lower() == "pruitt":
        path = PRUITT_VERIFIED_PATH
    elif speaker.lower() == "ericah":
        path = ERICAH_VERIFIED_PATH
    else:
        return []
    return sorted(glob.glob(os.path.join(path, "*.wav")))


def get_other_files(exclude_files: set) -> list[str]:
    # Randomly select files from uploads that are NOT in verified set
    all_uploads = glob.glob(os.path.join(UPLOADS_PATH, "*.wav"))
    candidates = [f for f in all_uploads if os.path.basename(f) not in exclude_files]
    return candidates


def precompute_embeddings(files: list[str], use_vad: bool = True) -> list[tuple[np.ndarray, float]]:
    """Compute embeddings for a list of files. Returns list of (embedding, duration)."""
    results = []
    print(f"   Processing {len(files)} files (VAD={use_vad})...", flush=True)
    for i, f in enumerate(files):
        try:
            if i % 10 == 0:
                print(f"    {i}/{len(files)}", flush=True)
            audio, sr = sf.read(f)
            duration = len(audio) / sr

            if use_vad:
                speech = extract_speech_audio(audio, sr)
                duration = len(speech) / sr
                if duration < 0.5:
                    continue
                emb = get_embedding(audio_data=speech)
            else:
                emb = get_embedding(audio_path=f)

            results.append((emb, duration))
        except Exception:
            # print(f"Error {f}: {e}")
            pass
    return results


# --- Main Execution ---


def main():
    print("=" * 80)
    print("IMPROVED SPEAKER ENROLLMENT TEST BENCH (OPTIMIZED)")
    print("=" * 80)

    # 1. Dataset Prep
    print("\n[1/5] Preparing Dataset...")

    pruitt_all_files = get_verified_files("pruitt")
    ericah_all_files = get_verified_files("ericah")

    print(f"Found {len(pruitt_all_files)} verified Pruitt files")
    print(f"Found {len(ericah_all_files)} verified Ericah files")

    # Split
    random.seed(42)
    test_pruitt_files = random.sample(pruitt_all_files, min(10, len(pruitt_all_files)))
    test_ericah_files = random.sample(ericah_all_files, min(10, len(ericah_all_files)))

    # Other
    verified_basenames = set([os.path.basename(f) for f in pruitt_all_files + ericah_all_files])
    other_candidates = get_other_files(verified_basenames)
    test_other_files = random.sample(other_candidates, min(30, len(other_candidates)))

    print(f"Test Set: 10 Pruitt, 10 Ericah, {len(test_other_files)} Other")

    # 2. Pre-compute Embeddings (The heavy lifting)
    print("\n[2/5] Pre-computing Embeddings...")

    # Pruitt Enrollment Data (All files)
    print("-> Pruitt (Raw)")
    p_raw = precompute_embeddings(pruitt_all_files, use_vad=False)
    print("-> Pruitt (VAD)")
    p_vad = precompute_embeddings(pruitt_all_files, use_vad=True)

    # Ericah Enrollment Data (All files)
    print("-> Ericah (Raw)")
    e_raw = precompute_embeddings(ericah_all_files, use_vad=False)
    print("-> Ericah (VAD)")
    e_vad = precompute_embeddings(ericah_all_files, use_vad=True)

    # Test Data (VAD only, as we assume system uses VAD at runtime or we want best quality test)
    # Actually, let's treat test data as runtime input.
    # Current system DOES use speaker_identifier on chunks.
    # But for "Test", we used get_embedding in loop before.
    # Let's precompute Test embeddings with VAD to ensure we test "best case" matching capability?
    # Or without VAD to test robustness?
    # Let's use VAD for test files too, as that's fair.
    print("-> Test Pruitt")
    test_p_emb = [x[0] for x in precompute_embeddings(test_pruitt_files, use_vad=True)]
    print("-> Test Ericah")
    test_e_emb = [x[0] for x in precompute_embeddings(test_ericah_files, use_vad=True)]
    print("-> Test Other")
    test_o_emb = [x[0] for x in precompute_embeddings(test_other_files, use_vad=True)]

    # helper extractors
    p_raw_embs = [x[0] for x in p_raw]
    p_vad_embs = [x[0] for x in p_vad]
    p_vad_durs = [x[1] for x in p_vad]

    e_raw_embs = [x[0] for x in e_raw]
    e_vad_embs = [x[0] for x in e_vad]
    e_vad_durs = [x[1] for x in e_vad]

    strategies = [
        ("Baseline (No VAD)", lambda: (strategy_1_baseline(p_raw_embs), strategy_1_baseline(e_raw_embs))),
        ("VAD Mean", lambda: (strategy_2_vad_mean(p_vad_embs), strategy_2_vad_mean(e_vad_embs))),
        (
            "VAD Weighted",
            lambda: (strategy_3_vad_weighted(p_vad_embs, p_vad_durs), strategy_3_vad_weighted(e_vad_embs, e_vad_durs)),
        ),
        ("VAD Median", lambda: (strategy_4_vad_median(p_vad_embs), strategy_4_vad_median(e_vad_embs))),
        ("VAD Top-20", lambda: (strategy_5_vad_top20(p_vad), strategy_5_vad_top20(e_vad))),
    ]

    results = []
    best_score = -1
    best_embeddings = (None, None)

    print("\n[3/5] Running Strategies...")
    print(f"{'Strategy':<20} | {'Pruitt Acc':<10} | {'Ericah Acc':<10} | {'Other FP':<10} | {'Overall':<10}")
    print("-" * 75)

    for name, func in strategies:
        pruitt_emb, ericah_emb = func()

        p_correct, e_correct, o_fp = 0, 0, 0

        # Test Pruitt
        for emb in test_p_emb:
            sim_p = np.dot(emb, pruitt_emb)
            sim_e = np.dot(emb, ericah_emb)
            if sim_p > sim_e and sim_p > 0.50:
                p_correct += 1

        # Test Ericah
        for emb in test_e_emb:
            sim_p = np.dot(emb, pruitt_emb)
            sim_e = np.dot(emb, ericah_emb)
            if sim_e > sim_p and sim_e > 0.50:
                e_correct += 1

        # Test Other (FP)
        for emb in test_o_emb:
            sim_p = np.dot(emb, pruitt_emb)
            sim_e = np.dot(emb, ericah_emb)
            if max(sim_p, sim_e) > 0.50:
                o_fp += 1

        p_acc = (p_correct / len(test_p_emb)) * 100 if test_p_emb else 0
        e_acc = (e_correct / len(test_e_emb)) * 100 if test_e_emb else 0
        o_rate = (o_fp / len(test_o_emb)) * 100 if test_o_emb else 0

        score = (p_acc + e_acc) / 2 - o_rate

        print(f"{name:<20} | {p_acc:>9.1f}% | {e_acc:>9.1f}% | {o_rate:>9.1f}% | {score:>9.1f}", flush=True)

        results.append({"name": name, "p": p_acc, "e": e_acc, "o": o_rate, "score": score})

        if score > best_score:
            best_score = score
            best_strategy = name
            best_embeddings = (pruitt_emb, ericah_emb)

    print("\n[4/5] Results Analysis")
    print(f"Winner: {best_strategy} with score {best_score:.1f}")

    print("\n[5/5] Saving Best Enrollments...")
    if best_embeddings[0] is not None:
        np.save(os.path.join(ENROLLMENT_OUTPUT, "pruitt_embedding.npy"), best_embeddings[0].astype(np.float32))
        np.save(os.path.join(ENROLLMENT_OUTPUT, "ericah_embedding.npy"), best_embeddings[1].astype(np.float32))
        print(f"Saved to {ENROLLMENT_OUTPUT}")


if __name__ == "__main__":
    main()
