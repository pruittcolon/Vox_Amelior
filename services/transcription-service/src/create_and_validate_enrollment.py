#!/usr/bin/env python3
"""
Create enrollment from Finalfolder ground-truth samples and validate.
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/app")


def main():
    print("=" * 60)
    print("CREATE ENROLLMENT FROM FINALFOLDER + VALIDATE")
    print("=" * 60)

    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    model = model.to("cpu")
    model.eval()
    print("TitaNet loaded!")

    finalfolder = "/gateway_instance/Finalfolder"

    # Step 1: Collect all ftpruitt and ftericah samples
    pruitt_files = sorted(glob.glob(f"{finalfolder}/*ftpruitt*.wav"))
    ericah_files = sorted(glob.glob(f"{finalfolder}/*ftericah*.wav"))
    other_files = sorted(glob.glob(f"{finalfolder}/*ftother*.wav"))

    print(f"\nFound: {len(pruitt_files)} Pruitt, {len(ericah_files)} Ericah, {len(other_files)} Other samples")

    # Step 2: Extract embeddings from ftpruitt samples
    print("\n=== Creating Pruitt Embedding ===")
    pruitt_embeddings = []
    for f in pruitt_files[:20]:  # Use up to 20 samples
        try:
            emb = model.get_embedding(f)
            if hasattr(emb, "cpu"):
                emb = emb.cpu().numpy().flatten()
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                emb = emb / emb_norm
            pruitt_embeddings.append(emb)
        except Exception as e:
            print(f"  Skip {os.path.basename(f)}: {e}")

    if pruitt_embeddings:
        pruitt_avg = np.mean(pruitt_embeddings, axis=0)
        pruitt_avg = pruitt_avg / np.linalg.norm(pruitt_avg)
        np.save("/gateway_instance/enrollment/pruitt_embedding.npy", pruitt_avg.astype(np.float32))
        print(f"  Created Pruitt embedding from {len(pruitt_embeddings)} samples")
    else:
        print("  ERROR: No Pruitt embeddings extracted!")
        return

    # Step 3: Extract embeddings from ftericah samples
    print("\n=== Creating Ericah Embedding ===")
    ericah_embeddings = []
    for f in ericah_files[:20]:
        try:
            emb = model.get_embedding(f)
            if hasattr(emb, "cpu"):
                emb = emb.cpu().numpy().flatten()
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                emb = emb / emb_norm
            ericah_embeddings.append(emb)
        except Exception as e:
            print(f"  Skip {os.path.basename(f)}: {e}")

    if ericah_embeddings:
        ericah_avg = np.mean(ericah_embeddings, axis=0)
        ericah_avg = ericah_avg / np.linalg.norm(ericah_avg)
        np.save("/gateway_instance/enrollment/ericah_embedding.npy", ericah_avg.astype(np.float32))
        print(f"  Created Ericah embedding from {len(ericah_embeddings)} samples")
    else:
        print("  ERROR: No Ericah embeddings extracted!")
        return

    # Step 4: Validate ALL samples against new embeddings
    print("\n=== Validating ===")

    # Load new embeddings
    pruitt_emb = np.load("/gateway_instance/enrollment/pruitt_embedding.npy")
    ericah_emb = np.load("/gateway_instance/enrollment/ericah_embedding.npy")

    results = {"correct": 0, "wrong": 0, "details": []}

    all_files = pruitt_files + ericah_files + other_files
    for f in all_files:
        fname = os.path.basename(f)
        # Determine expected speaker
        if "ftpruitt" in fname:
            expected = "Pruitt"
        elif "ftericah" in fname:
            expected = "Ericah"
        else:
            expected = "Other"

        try:
            emb = model.get_embedding(f)
            if hasattr(emb, "cpu"):
                emb = emb.cpu().numpy().flatten()
            emb = emb / np.linalg.norm(emb)

            sim_pruitt = np.dot(emb, pruitt_emb)
            sim_ericah = np.dot(emb, ericah_emb)

            if sim_pruitt > sim_ericah and sim_pruitt >= 0.5:
                predicted = "Pruitt"
            elif sim_ericah > sim_pruitt and sim_ericah >= 0.5:
                predicted = "Ericah"
            else:
                predicted = "Unknown"

            correct = (expected == predicted) or (expected == "Other" and predicted == "Unknown")
            if correct:
                results["correct"] += 1
                status = "✅"
            else:
                results["wrong"] += 1
                status = "❌"

            results["details"].append(
                f"{status} {fname[:40]}: P:{sim_pruitt:.2f} E:{sim_ericah:.2f} -> {predicted} (expected {expected})"
            )
        except Exception as e:
            results["details"].append(f"❌ {fname[:40]}: ERROR {e}")

    # Print results
    print(
        f"\nResults: {results['correct']}/{len(all_files)} correct ({100 * results['correct'] / len(all_files):.1f}%)"
    )
    print()
    for detail in results["details"][:30]:  # Show first 30
        print(detail)
    if len(results["details"]) > 30:
        print(f"... and {len(results['details']) - 30} more")

    print("\n" + "=" * 60)
    print("DONE! Restart transcription-service to use new embeddings:")
    print("  docker compose restart transcription-service")
    print("=" * 60)


if __name__ == "__main__":
    main()
