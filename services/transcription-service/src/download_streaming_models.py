#!/usr/bin/env python3
"""
Pre-download Parakeet RNNT models for streaming comparison.
Downloads models to the HuggingFace cache so they're ready for use.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODELS_TO_DOWNLOAD = [
    # Streaming-capable RNNT models
    "nvidia/parakeet-rnnt-0.6b",
    "nvidia/parakeet-rnnt-1.1b",
    # Also get the dedicated streaming FastConformer
    "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
]


def main():
    print("=" * 80)
    print("DOWNLOADING PARAKEET STREAMING MODELS")
    print("=" * 80)

    import nemo.collections.asr as nemo_asr

    for model_name in MODELS_TO_DOWNLOAD:
        print(f"\n[Downloading] {model_name}...")
        try:
            model = nemo_asr.models.ASRModel.from_pretrained(model_name)
            params = sum(p.numel() for p in model.parameters())
            print(f"   ✅ Success! Parameters: {params:,}")
            # Free memory
            del model
            import torch

            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    print("\n" + "=" * 80)
    print("All downloads complete. Models are cached in /app/models/hub/")
    print("=" * 80)


if __name__ == "__main__":
    main()
