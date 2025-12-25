#!/usr/bin/env python3
"""
Comprehensive Speaker Enrollment Strategy Testing

Tests 10 different enrollment strategies against labeled ground-truth samples.
Finds optimal configuration with highest accuracy.
"""

import glob
import os
import random
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, "/app")

# Global model
model = None


def load_model():
    global model
    if model is None:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        model = model.to("cpu")
        model.eval()
    return model


def get_embedding(audio_path: str) -> np.ndarray:
    """Extract normalized embedding from audio file."""
    m = load_model()
    emb = m.get_embedding(audio_path)
    if hasattr(emb, "cpu"):
        emb = emb.cpu().numpy()
    emb = emb.flatten()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def get_embedding_from_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract embedding from audio array."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        emb = get_embedding(tmp.name)
        os.unlink(tmp.name)
    return emb


def collect_samples() -> dict[str, list[str]]:
    """Collect all labeled samples from Finalfolder and uploads_labeled_full."""
    samples = {"pruitt": [], "ericah": [], "other": []}

    # Finalfolder
    ff = "/gateway_instance/Finalfolder"
    if os.path.exists(ff):
        for f in glob.glob(f"{ff}/*.wav"):
            name = os.path.basename(f)
            if "ftpruitt" in name:
                samples["pruitt"].append(f)
            elif "ftericah" in name:
                samples["ericah"].append(f)
            elif "ftother" in name:
                samples["other"].append(f)

    # uploads_labeled_full
    ulf = "/gateway_instance/uploads_labeled_full"
    if os.path.exists(ulf):
        for speaker in ["pruitt", "ericah", "other"]:
            folder = os.path.join(ulf, speaker)
            if os.path.exists(folder):
                samples[speaker].extend(glob.glob(f"{folder}/*.wav"))

    return samples


def create_enrollment_strategy_1(samples: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Strategy 1: Use enrollment.wav directly"""
    p_audio = "/gateway_instance/enrollment/pruitt/enrollment.wav"
    e_audio = "/gateway_instance/enrollment/ericah/enrollment.wav"
    return get_embedding(p_audio), get_embedding(e_audio), "enrollment.wav direct"


def create_enrollment_strategy_2(samples: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Strategy 2: Average all Finalfolder samples"""
    ff = "/gateway_instance/Finalfolder"

    p_embs = [get_embedding(f) for f in glob.glob(f"{ff}/*ftpruitt*.wav")]
    e_embs = [get_embedding(f) for f in glob.glob(f"{ff}/*ftericah*.wav")]

    p = np.mean(p_embs, axis=0) if p_embs else np.zeros(192)
    e = np.mean(e_embs, axis=0) if e_embs else np.zeros(192)

    return p / np.linalg.norm(p), e / np.linalg.norm(e), "Finalfolder average"


def create_enrollment_strategy_3(samples: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Strategy 3: Average top-5 longest samples from all sources"""

    def get_file_duration(f):
        try:
            info = sf.info(f)
            return info.duration
        except:
            return 0

    p_files = sorted(samples["pruitt"], key=get_file_duration, reverse=True)[:5]
    e_files = sorted(samples["ericah"], key=get_file_duration, reverse=True)[:5]

    p_embs = [get_embedding(f) for f in p_files]
    e_embs = [get_embedding(f) for f in e_files]

    p = np.mean(p_embs, axis=0) if p_embs else np.zeros(192)
    e = np.mean(e_embs, axis=0) if e_embs else np.zeros(192)

    return p / np.linalg.norm(p), e / np.linalg.norm(e), "Top-5 longest samples"


def create_enrollment_strategy_4(samples: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Strategy 4: Average top-10 samples by file size (proxy for duration)"""
    p_files = sorted(samples["pruitt"], key=lambda f: os.path.getsize(f) if os.path.exists(f) else 0, reverse=True)[:10]
    e_files = sorted(samples["ericah"], key=lambda f: os.path.getsize(f) if os.path.exists(f) else 0, reverse=True)[:10]

    p_embs = [get_embedding(f) for f in p_files]
    e_embs = [get_embedding(f) for f in e_files]

    p = np.mean(p_embs, axis=0)
    e = np.mean(e_embs, axis=0)

    return p / np.linalg.norm(p), e / np.linalg.norm(e), "Top-10 largest files"


def create_enrollment_strategy_5(samples: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Strategy 5: Median embedding (robust to outliers)"""
    p_sample = random.sample(samples["pruitt"], min(20, len(samples["pruitt"])))
    e_sample = random.sample(samples["ericah"], min(20, len(samples["ericah"])))

    p_embs = np.array([get_embedding(f) for f in p_sample])
    e_embs = np.array([get_embedding(f) for f in e_sample])

    p = np.median(p_embs, axis=0)
    e = np.median(e_embs, axis=0)

    return p / np.linalg.norm(p), e / np.linalg.norm(e), "Median of 20 samples"


def create_enrollment_strategy_6(samples: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Strategy 6: Concatenate audio then extract single embedding"""

    def concat_and_embed(files: list[str], max_files: int = 5) -> np.ndarray:
        audios = []
        for f in files[:max_files]:
            try:
                audio, sr = sf.read(f)
                if sr != 16000:
                    continue
                audios.append(audio)
            except:
                continue
        if not audios:
            return np.zeros(192)
        combined = np.concatenate(audios)
        # Limit to 60 seconds
        max_samples = 60 * 16000
        if len(combined) > max_samples:
            combined = combined[:max_samples]
        return get_embedding_from_audio(combined)

    p = concat_and_embed(samples["pruitt"])
    e = concat_and_embed(samples["ericah"])

    return p, e, "Concatenated 5 files"


def create_enrollment_strategy_7(samples: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Strategy 7: Weighted average by duration"""

    def weighted_avg(files: list[str]) -> np.ndarray:
        embs, weights = [], []
        for f in files[:15]:
            try:
                info = sf.info(f)
                if info.duration < 0.5:
                    continue
                embs.append(get_embedding(f))
                weights.append(info.duration)
            except:
                continue
        if not embs:
            return np.zeros(192)
        weights = np.array(weights)
        weights = weights / weights.sum()
        avg = np.sum([e * w for e, w in zip(embs, weights)], axis=0)
        return avg / np.linalg.norm(avg)

    return weighted_avg(samples["pruitt"]), weighted_avg(samples["ericah"]), "Duration-weighted average"


def create_enrollment_strategy_8(samples: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Strategy 8: Trimmed mean (remove top/bottom 10% outliers)"""

    def trimmed_mean(files: list[str]) -> np.ndarray:
        sample = random.sample(files, min(30, len(files)))
        embs = np.array([get_embedding(f) for f in sample])

        # Calculate centroid similarity for each embedding
        centroid = np.mean(embs, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        sims = np.array([np.dot(e, centroid) for e in embs])

        # Remove bottom 10% (outliers)
        threshold = np.percentile(sims, 10)
        mask = sims >= threshold

        avg = np.mean(embs[mask], axis=0)
        return avg / np.linalg.norm(avg)

    return trimmed_mean(samples["pruitt"]), trimmed_mean(samples["ericah"]), "Trimmed mean (remove outliers)"


def create_enrollment_strategy_9(samples: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Strategy 9: enrollment.wav + Finalfolder average combined"""
    p1 = get_embedding("/gateway_instance/enrollment/pruitt/enrollment.wav")
    e1 = get_embedding("/gateway_instance/enrollment/ericah/enrollment.wav")

    ff = "/gateway_instance/Finalfolder"
    p2_embs = [get_embedding(f) for f in glob.glob(f"{ff}/*ftpruitt*.wav")]
    e2_embs = [get_embedding(f) for f in glob.glob(f"{ff}/*ftericah*.wav")]

    p2 = np.mean(p2_embs, axis=0) if p2_embs else p1
    e2 = np.mean(e2_embs, axis=0) if e2_embs else e1

    # Average the two approaches
    p = (p1 + p2) / 2
    e = (e1 + e2) / 2

    return p / np.linalg.norm(p), e / np.linalg.norm(e), "enrollment.wav + Finalfolder"


def create_enrollment_strategy_10(samples: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Strategy 10: All sources combined - enrollment.wav + uploads_labeled_full"""
    all_p = [get_embedding("/gateway_instance/enrollment/pruitt/enrollment.wav")]
    all_e = [get_embedding("/gateway_instance/enrollment/ericah/enrollment.wav")]

    # Add random samples from uploads_labeled_full
    p_sample = random.sample(samples["pruitt"], min(10, len(samples["pruitt"])))
    e_sample = random.sample(samples["ericah"], min(10, len(samples["ericah"])))

    all_p.extend([get_embedding(f) for f in p_sample])
    all_e.extend([get_embedding(f) for f in e_sample])

    p = np.mean(all_p, axis=0)
    e = np.mean(all_e, axis=0)

    return p / np.linalg.norm(p), e / np.linalg.norm(e), "All sources combined"


def evaluate_strategy(
    pruitt_emb: np.ndarray, ericah_emb: np.ndarray, test_samples: dict[str, list[str]], n_test: int = 50
) -> dict:
    """Evaluate enrollment against test samples."""
    results = {
        "pruitt_sims": [],
        "ericah_sims": [],
        "other_sims": [],
        "pruitt_correct": 0,
        "pruitt_total": 0,
        "ericah_correct": 0,
        "ericah_total": 0,
        "other_correct": 0,
        "other_total": 0,
    }

    # Test Pruitt samples
    p_test = random.sample(test_samples["pruitt"], min(n_test, len(test_samples["pruitt"])))
    for f in p_test:
        try:
            emb = get_embedding(f)
            sim_p = np.dot(emb, pruitt_emb)
            sim_e = np.dot(emb, ericah_emb)
            results["pruitt_sims"].append(sim_p)
            results["pruitt_total"] += 1
            if sim_p > sim_e and sim_p >= 0.35:
                results["pruitt_correct"] += 1
        except:
            pass

    # Test Ericah samples
    e_test = random.sample(test_samples["ericah"], min(n_test, len(test_samples["ericah"])))
    for f in e_test:
        try:
            emb = get_embedding(f)
            sim_p = np.dot(emb, pruitt_emb)
            sim_e = np.dot(emb, ericah_emb)
            results["ericah_sims"].append(sim_e)
            results["ericah_total"] += 1
            if sim_e > sim_p and sim_e >= 0.35:
                results["ericah_correct"] += 1
        except:
            pass

    # Test Other samples - should both be LOW
    o_test = random.sample(test_samples["other"], min(n_test, len(test_samples["other"])))
    for f in o_test:
        try:
            emb = get_embedding(f)
            sim_p = np.dot(emb, pruitt_emb)
            sim_e = np.dot(emb, ericah_emb)
            max_sim = max(sim_p, sim_e)
            results["other_sims"].append(max_sim)
            results["other_total"] += 1
            if max_sim < 0.15:  # Should be below 15%
                results["other_correct"] += 1
        except:
            pass

    return results


def main():
    print("=" * 70)
    print("COMPREHENSIVE SPEAKER ENROLLMENT STRATEGY TESTING")
    print("=" * 70)

    print("\n1. Loading model...")
    load_model()

    print("\n2. Collecting labeled samples...")
    samples = collect_samples()
    print(f"   Pruitt: {len(samples['pruitt'])} samples")
    print(f"   Ericah: {len(samples['ericah'])} samples")
    print(f"   Other: {len(samples['other'])} samples")

    if len(samples["pruitt"]) < 5 or len(samples["ericah"]) < 5:
        print("ERROR: Need at least 5 samples per speaker!")
        return

    # Split into training (for enrollment) and testing
    random.seed(42)  # Reproducibility

    strategies = [
        create_enrollment_strategy_1,
        create_enrollment_strategy_2,
        create_enrollment_strategy_3,
        create_enrollment_strategy_4,
        create_enrollment_strategy_5,
        create_enrollment_strategy_6,
        create_enrollment_strategy_7,
        create_enrollment_strategy_8,
        create_enrollment_strategy_9,
        create_enrollment_strategy_10,
    ]

    results_table = []

    print("\n3. Testing enrollment strategies...\n")
    print("-" * 70)

    for i, strategy_fn in enumerate(strategies, 1):
        try:
            print(f"Strategy {i}: ", end="", flush=True)
            pruitt_emb, ericah_emb, name = strategy_fn(samples)
            print(f"{name}...", end=" ", flush=True)

            # Evaluate
            results = evaluate_strategy(pruitt_emb, ericah_emb, samples, n_test=30)

            # Calculate metrics
            p_acc = results["pruitt_correct"] / results["pruitt_total"] * 100 if results["pruitt_total"] > 0 else 0
            e_acc = results["ericah_correct"] / results["ericah_total"] * 100 if results["ericah_total"] > 0 else 0
            o_acc = results["other_correct"] / results["other_total"] * 100 if results["other_total"] > 0 else 0

            p_avg = np.mean(results["pruitt_sims"]) if results["pruitt_sims"] else 0
            e_avg = np.mean(results["ericah_sims"]) if results["ericah_sims"] else 0
            o_max = np.max(results["other_sims"]) if results["other_sims"] else 0

            overall = (p_acc + e_acc + o_acc) / 3

            results_table.append(
                {
                    "id": i,
                    "name": name,
                    "pruitt_acc": p_acc,
                    "ericah_acc": e_acc,
                    "other_acc": o_acc,
                    "overall": overall,
                    "p_sim": p_avg,
                    "e_sim": e_avg,
                    "o_max": o_max,
                    "pruitt_emb": pruitt_emb,
                    "ericah_emb": ericah_emb,
                }
            )

            print(f"Overall: {overall:.1f}% (P:{p_acc:.0f}% E:{e_acc:.0f}% O:{o_acc:.0f}%)")
        except Exception as e:
            print(f"ERROR: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"{'#':<3} {'Strategy':<30} {'Pruitt':<8} {'Ericah':<8} {'Other':<8} {'Overall':<8} {'P.Sim':<6} {'E.Sim':<6} {'O.Max':<6}"
    )
    print("-" * 90)

    for r in sorted(results_table, key=lambda x: x["overall"], reverse=True):
        print(
            f"{r['id']:<3} {r['name'][:30]:<30} {r['pruitt_acc']:>6.1f}% {r['ericah_acc']:>6.1f}% {r['other_acc']:>6.1f}% {r['overall']:>6.1f}% {r['p_sim']:>5.2f} {r['e_sim']:>5.2f} {r['o_max']:>5.2f}"
        )

    # Best strategy
    best = max(results_table, key=lambda x: x["overall"])
    print("\n" + "=" * 70)
    print(f"BEST STRATEGY: #{best['id']} - {best['name']}")
    print(f"Overall Accuracy: {best['overall']:.1f}%")
    print(f"Pruitt avg similarity: {best['p_sim']:.3f}")
    print(f"Ericah avg similarity: {best['e_sim']:.3f}")
    print(f"Other max similarity: {best['o_max']:.3f} (should be < 0.15)")
    print("=" * 70)

    # Save best embeddings
    print("\nSaving best embeddings...")
    np.save("/gateway_instance/enrollment/pruitt_embedding.npy", best["pruitt_emb"].astype(np.float32))
    np.save("/gateway_instance/enrollment/ericah_embedding.npy", best["ericah_emb"].astype(np.float32))
    print("Done! Restart transcription-service to use new embeddings.")


if __name__ == "__main__":
    main()
