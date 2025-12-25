#!/usr/bin/env python3
"""
Generate Cohort Embeddings for AS-Norm
Selects 200 random 'Other' files and saves their embeddings to a single .npy file.
"""

import glob
import os
import random
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/app")

UPLOADS_PATH = "/gateway_instance/uploads"
OUTPUT_PATH = "/gateway_instance/enrollment/cohort_embeddings.npy"
COHORT_SIZE = 200


def main():
    print("=" * 80)
    print("GENERATING COHORT EMBEDDINGS")
    print("=" * 80)

    # 1. Load TitaNet
    print("[System] Loading TitaNet...")
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    titanet_model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    titanet_model = titanet_model.to("cpu").eval()

    # 2. Select Files
    all_uploads = glob.glob(os.path.join(UPLOADS_PATH, "*.wav"))
    random.seed(42)
    selected = random.sample(all_uploads, min(COHORT_SIZE, len(all_uploads)))
    print(f"[System] Selected {len(selected)} random files for cohort.")

    # 3. Extract Embeddings
    embeddings = []

    for i, fpath in enumerate(selected):
        if i % 20 == 0:
            print(f"   Processing {i}/{len(selected)}...", flush=True)
        try:
            # Note: For cohort, we handle raw audio logic (simple load)
            emb = None
            if hasattr(titanet_model, "get_embedding"):
                emb = titanet_model.get_embedding(fpath)
            else:
                emb, _ = titanet_model.infer_file(fpath)
                emb = emb[0]

            if hasattr(emb, "cpu"):
                emb = emb.cpu().numpy()
            emb = emb.flatten()

            # Normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
                embeddings.append(emb)
        except:
            pass

    # 4. Save
    final_embeddings = np.array(embeddings)
    np.save(OUTPUT_PATH, final_embeddings)
    print(f"\n[Success] Saved {len(final_embeddings)} embeddings to {OUTPUT_PATH}")
    print(f"Shape: {final_embeddings.shape}")
    print("=" * 80)


if __name__ == "__main__":
    main()
