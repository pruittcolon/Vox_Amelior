#!/usr/bin/env python3
"""
Model Setup Script

Downloads and configures AI models for Nemo Server.
Run this script to set up all required models.

Usage:
    python scripts/model_setup.py --all       # Download all models
    python scripts/model_setup.py --granite   # Download Granite 1B only
    python scripts/model_setup.py --verify    # Verify existing models

Author: Nemo Server Team
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Add shared modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.model_config import LLM_MODELS, ASR_MODELS, ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Default models directory
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models1"))


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: Download URL
        dest_path: Destination file path
        chunk_size: Download chunk size in bytes
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest_path.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"✅ Downloaded: {dest_path}")
        return True
        
    except requests.RequestException as e:
        logger.error(f"❌ Download failed: {e}")
        return False


def verify_model(model: ModelConfig) -> dict[str, any]:
    """
    Verify a model file exists and is valid.
    
    Returns:
        Dictionary with verification results
    """
    path = Path(model.path.replace("/app/models", str(MODELS_DIR)))
    
    result = {
        "name": model.name,
        "path": str(path),
        "exists": path.exists(),
        "size_mb": 0,
        "valid": False,
    }
    
    if path.exists():
        size = path.stat().st_size
        result["size_mb"] = round(size / 1024 / 1024, 1)
        # Basic validation: file is not empty and is a reasonable size
        result["valid"] = size > 1024 * 1024  # At least 1MB
        
    return result


def download_granite_model() -> bool:
    """Download IBM Granite 4.0 1B model."""
    model = LLM_MODELS.get("granite-1b")
    if not model:
        logger.error("Granite model not found in registry")
        return False
    
    if not model.download_url:
        logger.error("No download URL configured for Granite model")
        return False
    
    dest_path = MODELS_DIR / "granite-4.0-1b-Q8_0.gguf"
    
    if dest_path.exists():
        logger.info(f"Model already exists: {dest_path}")
        return True
    
    logger.info(f"Downloading Granite 1B model from HuggingFace...")
    logger.info(f"URL: {model.download_url}")
    logger.info(f"Destination: {dest_path}")
    
    return download_file(model.download_url, dest_path)


def verify_all_models() -> None:
    """Verify all configured models."""
    print("\n" + "=" * 60)
    print("MODEL VERIFICATION REPORT")
    print("=" * 60)
    
    print("\n📦 LLM Models:")
    print("-" * 40)
    for name, model in LLM_MODELS.items():
        result = verify_model(model)
        status = "✅" if result["valid"] else "❌"
        size_str = f"{result['size_mb']} MB" if result["size_mb"] else "N/A"
        print(f"  {status} {name}: {size_str}")
        if not result["exists"]:
            print(f"      Path: {result['path']}")
            if model.download_url:
                print(f"      Download: python scripts/model_setup.py --granite")
    
    print("\n🎤 ASR Models:")
    print("-" * 40)
    for name, model in ASR_MODELS.items():
        # ASR models are downloaded at runtime via HuggingFace hub
        print(f"  ℹ️  {name}: {model.model_id}")
        print(f"      (Downloaded automatically on first use)")
    
    print("\n" + "=" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and configure AI models for Nemo Server"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all missing models"
    )
    parser.add_argument(
        "--granite",
        action="store_true",
        help="Download IBM Granite 1B model"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing models"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models1",
        help="Models directory path"
    )
    
    args = parser.parse_args()
    
    global MODELS_DIR
    MODELS_DIR = Path(args.models_dir)
    
    if args.verify or (not args.all and not args.granite):
        verify_all_models()
        return 0
    
    success = True
    
    if args.all or args.granite:
        if not download_granite_model():
            success = False
    
    if success:
        print("\n✅ All requested models downloaded successfully!")
        verify_all_models()
        return 0
    else:
        print("\n❌ Some downloads failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
