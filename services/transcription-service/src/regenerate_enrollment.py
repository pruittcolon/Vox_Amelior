#!/usr/bin/env python3
"""
Test verified Pruitt samples against current enrollment,
then regenerate enrollment from verified samples.
"""

import os
import sys

import numpy as np

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
    print("TEST AND REGENERATE PRUITT ENROLLMENT")
    print("=" * 60)

    # Load TitaNet
    print("\n1. Loading TitaNet model...")
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    model = model.to("cpu")
    model.eval()
    print("   TitaNet loaded!")

    # Load current enrollment embedding
    current_emb_path = "/gateway_instance/enrollment/pruitt_embedding.npy"
    print(f"\n2. Loading current enrollment: {current_emb_path}")
    current_emb = np.load(current_emb_path)
    current_emb_norm = np.linalg.norm(current_emb)
    current_emb_normalized = current_emb / current_emb_norm if current_emb_norm > 0 else current_emb
    print(f"   Shape: {current_emb.shape}, Norm: {current_emb_norm:.4f}")

    # Test verified samples against current enrollment
    verified_dir = "/gateway_instance/pruitt_verified"
    print("\n3. Testing verified samples against current enrollment:")

    wav_files = sorted([f for f in os.listdir(verified_dir) if f.endswith(".wav")])
    verified_embeddings = []

    for wav_file in wav_files[:10]:  # Test up to 10 files
        wav_path = os.path.join(verified_dir, wav_file)

        try:
            emb = model.get_embedding(wav_path)
            if hasattr(emb, "cpu"):
                emb = emb.cpu().numpy().flatten()

            emb_norm = np.linalg.norm(emb)
            emb_normalized = emb / emb_norm if emb_norm > 0 else emb

            verified_embeddings.append(emb_normalized)

            # Compare to current enrollment
            sim = cosine_similarity(current_emb_normalized, emb_normalized)
            status = "✅" if sim >= 0.6 else "⚠️" if sim >= 0.3 else "❌"
            print(f"   {status} {wav_file[:40]}: {sim:.4f}")
        except Exception as e:
            print(f"   ❌ {wav_file}: ERROR - {e}")

    print(f"\n4. Verified samples found: {len(verified_embeddings)}")

    if not verified_embeddings:
        print("   ERROR: No embeddings extracted from verified samples!")
        return

    # Create NEW enrollment by averaging verified samples
    print("\n5. Creating NEW enrollment from verified samples:")
    avg_embedding = np.mean(verified_embeddings, axis=0)
    avg_norm = np.linalg.norm(avg_embedding)
    new_embedding = avg_embedding / avg_norm if avg_norm > 0 else avg_embedding
    print(f"   New embedding norm: {np.linalg.norm(new_embedding):.4f}")

    # Test new embedding against all verified samples
    print("\n6. Verifying new enrollment against all samples:")
    all_sims = []
    for i, emb in enumerate(verified_embeddings):
        sim = cosine_similarity(new_embedding, emb)
        all_sims.append(sim)
    avg_sim = np.mean(all_sims)
    min_sim = np.min(all_sims)
    max_sim = np.max(all_sims)
    print(f"   Similarity: min={min_sim:.4f}, avg={avg_sim:.4f}, max={max_sim:.4f}")

    # Save new enrollment
    print(f"\n7. Saving new enrollment to: {current_emb_path}")

    # Backup
    backup_path = current_emb_path + ".backup_" + str(int(os.path.getmtime(current_emb_path)))
    os.rename(current_emb_path, backup_path)
    print(f"   Backed up old to: {backup_path}")

    np.save(current_emb_path, new_embedding.astype(np.float32))
    print("   ✅ New embedding saved!")

    # Verify saved embedding
    loaded = np.load(current_emb_path)
    sim_to_new = cosine_similarity(loaded, new_embedding)
    print(f"   Verification similarity: {sim_to_new:.4f}")

    print("\n" + "=" * 60)
    print("DONE! New enrollment created from verified mobile samples.")
    print("Restart transcription-service to use new embedding:")
    print("  docker compose restart transcription-service")
    print("=" * 60)


if __name__ == "__main__":
    main()
