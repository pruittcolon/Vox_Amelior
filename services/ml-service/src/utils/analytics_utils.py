"""
Analytics Utilities

Shared utility functions for analytics routes:
- secure_file_path: Path traversal protection
- convert_to_native: JSON serialization for numpy types
- AnalyticsGemmaClient: Simple Gemma service client
- load_dataset: Load CSV/Excel files
- sample_for_analytics: Sample large datasets
"""

import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
MAX_ANALYTICS_ROWS = 10000


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from file path.
    
    Supports CSV, Excel, and Parquet formats.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def sample_for_analytics(
    df: pd.DataFrame, max_rows: int = MAX_ANALYTICS_ROWS
) -> pd.DataFrame:
    """Sample large datasets for faster analytics."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def secure_file_path(filename: str, base_dir: str | None = None) -> str:
    """
    SECURITY: Safely construct a file path from untrusted filename.
    Prevents path traversal attacks (e.g., '../../../etc/passwd').

    Args:
        filename: Untrusted filename from user input
        base_dir: Base directory (defaults to UPLOAD_DIR)

    Returns:
        Safe absolute path within base_dir

    Raises:
        HTTPException: If path traversal is detected
    """
    if base_dir is None:
        base_dir = UPLOAD_DIR

    # Normalize the base directory
    base_dir = os.path.abspath(base_dir)

    # Remove any path separators and dangerous sequences
    safe_filename = os.path.basename(filename)

    # Block null bytes (C string terminator attack)
    if "\x00" in safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename: null byte detected")

    # Block empty filename
    if not safe_filename or safe_filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Construct the full path
    full_path = os.path.abspath(os.path.join(base_dir, safe_filename))

    # CRITICAL: Verify the path is within base_dir (prevents traversal)
    if not full_path.startswith(base_dir + os.sep) and full_path != base_dir:
        logger.warning(f"[SECURITY] Path traversal attempt blocked: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename: path traversal detected")

    return full_path


def convert_to_native(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    Handles DataFrames, Series, and various numpy types.
    """
    # Handle None first
    if obj is None:
        return None

    # Handle pandas DataFrames and Series
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()

    # Handle dicts and lists recursively
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]

    # Handle numpy types
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        # Handle infinity and NaN for JSON compliance
        if np.isinf(val) or np.isnan(val):
            return None
        return val
    elif isinstance(obj, float):
        # Handle Python floats too
        if np.isinf(obj) or np.isnan(obj):
            return None
        return obj
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_native(obj.tolist())
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()

    # Check for scalar NA values (not DataFrames/Series)
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        # pd.isna fails on complex objects, that's fine
        pass

    return obj


class AnalyticsGemmaClient:
    """Simple client for Gemma service access from analytics routes."""

    def __init__(self):
        """Initialize with Gemma service URL from environment."""
        self.base_url = os.getenv("GEMMA_SERVICE_URL", "http://gemma-service:8001")

    def __call__(
        self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3
    ) -> dict[str, Any] | None:
        """
        Call Gemma service with prompt.

        Args:
            prompt: Text prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Response dict or None on failure
        """
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=30,
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Gemma request failed: {e}")
        return None


# Re-export for convenience
__all__ = [
    "secure_file_path",
    "convert_to_native",
    "AnalyticsGemmaClient",
    "load_dataset",
    "sample_for_analytics",
    "UPLOAD_DIR",
]
