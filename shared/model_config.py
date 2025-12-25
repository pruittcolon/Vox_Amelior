"""
Model Configuration Registry

Centralized configuration for all AI models used in Nemo Server.
Supports dynamic model switching and cache quantization.

Author: Nemo Server Team
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ModelType(Enum):
    """Supported model backend types."""
    LLAMA_CPP = "llama-cpp"
    NEMO = "nemo"
    HUGGINGFACE = "huggingface"


class CacheQuantization(Enum):
    """KV cache quantization levels."""
    FP16 = "fp16"      # Full precision (~835MB for 8k ctx)
    Q8_0 = "q8_0"      # 8-bit quantized (~418MB for 8k ctx)
    Q4_0 = "q4_0"      # 4-bit quantized (~209MB for 8k ctx)


@dataclass
class ModelConfig:
    """Configuration for a single AI model."""
    
    name: str
    path: str
    model_type: ModelType
    context_size: int = 8192
    default_cache: CacheQuantization = CacheQuantization.Q8_0
    gpu_layers: int = -1  # -1 = all layers on GPU
    description: str = ""
    download_url: str | None = None
    size_bytes: int = 0
    
    def exists(self) -> bool:
        """Check if model file exists."""
        return Path(self.path).exists()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "path": self.path,
            "type": self.model_type.value,
            "context_size": self.context_size,
            "default_cache": self.default_cache.value,
            "gpu_layers": self.gpu_layers,
            "description": self.description,
            "exists": self.exists(),
            "size_mb": round(self.size_bytes / 1024 / 1024, 1) if self.size_bytes else 0,
        }


@dataclass
class ASRModelConfig:
    """Configuration for ASR/transcription models."""
    
    name: str
    model_id: str
    model_type: ModelType = ModelType.NEMO
    description: str = ""
    supports_streaming: bool = True
    supports_cache_aware: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "type": self.model_type.value,
            "description": self.description,
            "supports_streaming": self.supports_streaming,
            "supports_cache_aware": self.supports_cache_aware,
        }


# ==============================================================================
# MODEL REGISTRY
# ==============================================================================

MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")

LLM_MODELS: dict[str, ModelConfig] = {
    "gemma-3-4b": ModelConfig(
        name="gemma-3-4b",
        path=f"{MODELS_DIR}/gemma-3-4b-it-UD-Q4_K_XL.gguf",
        model_type=ModelType.LLAMA_CPP,
        context_size=8192,
        default_cache=CacheQuantization.Q8_0,
        gpu_layers=-1,
        description="Gemma 3 4B Instruct - High quality, larger model",
        size_bytes=2544288896,  # ~2.4GB
    ),
    "granite-1b": ModelConfig(
        name="granite-1b",
        path=f"{MODELS_DIR}/granite-4.0-1b-Q8_0.gguf",
        model_type=ModelType.LLAMA_CPP,
        context_size=4096,
        default_cache=CacheQuantization.Q8_0,
        gpu_layers=-1,
        description="IBM Granite 4.0 1B - Fast, lightweight model",
        download_url="https://huggingface.co/NikolayKozloff/granite-4.0-1b-Q8_0-GGUF/resolve/main/granite-4.0-1b-q8_0.gguf",
        size_bytes=1200000000,  # ~1.1GB
    ),
}

ASR_MODELS: dict[str, ASRModelConfig] = {
    "parakeet-tdt-0.6b": ASRModelConfig(
        name="parakeet-tdt-0.6b",
        model_id="nvidia/parakeet-tdt-0.6b-v2",
        model_type=ModelType.NEMO,
        description="Parakeet TDT 0.6B - Default high-accuracy ASR",
        supports_streaming=True,
        supports_cache_aware=True,
    ),
    "parakeet-rnnt-0.6b": ASRModelConfig(
        name="parakeet-rnnt-0.6b",
        model_id="nvidia/parakeet-rnnt-0.6b",
        model_type=ModelType.NEMO,
        description="Parakeet RNNT 0.6B - Alternative streaming ASR",
        supports_streaming=True,
        supports_cache_aware=True,
    ),
}


def get_llm_model(name: str) -> ModelConfig | None:
    """Get LLM model configuration by name."""
    return LLM_MODELS.get(name)


def get_asr_model(name: str) -> ASRModelConfig | None:
    """Get ASR model configuration by name."""
    return ASR_MODELS.get(name)


def list_llm_models() -> list[dict[str, Any]]:
    """List all available LLM models."""
    return [model.to_dict() for model in LLM_MODELS.values()]


def list_asr_models() -> list[dict[str, Any]]:
    """List all available ASR models."""
    return [model.to_dict() for model in ASR_MODELS.values()]


def get_cache_type_ggml(cache_type: CacheQuantization | str) -> int:
    """
    Get GGML type constant for KV cache quantization.
    
    Returns the llama.cpp GGML type constant for the specified cache type.
    Must be imported from llama_cpp at runtime.
    """
    if isinstance(cache_type, str):
        cache_type = CacheQuantization(cache_type.lower())
    
    # These values correspond to llama_cpp constants
    # GGML_TYPE_F16 = 1, GGML_TYPE_Q8_0 = 8, GGML_TYPE_Q4_0 = 2
    CACHE_TYPE_MAP = {
        CacheQuantization.FP16: 1,   # GGML_TYPE_F16
        CacheQuantization.Q8_0: 8,   # GGML_TYPE_Q8_0
        CacheQuantization.Q4_0: 2,   # GGML_TYPE_Q4_0
    }
    return CACHE_TYPE_MAP.get(cache_type, 8)  # Default to Q8_0


def estimate_kv_cache_size(
    context_size: int,
    cache_type: CacheQuantization = CacheQuantization.Q8_0,
    model_name: str = "gemma-3-4b",
) -> int:
    """
    Estimate KV cache size in bytes for a given model and context size.
    
    Formula: 2 * n_layers * n_heads * head_dim * context * bytes_per_element
    """
    # Model architecture parameters
    MODEL_PARAMS = {
        "gemma-3-4b": {"layers": 26, "kv_heads": 8, "head_dim": 256},
        "granite-1b": {"layers": 16, "kv_heads": 8, "head_dim": 128},
    }
    
    BYTES_PER_ELEMENT = {
        CacheQuantization.FP16: 2,
        CacheQuantization.Q8_0: 1,
        CacheQuantization.Q4_0: 0.5,
    }
    
    params = MODEL_PARAMS.get(model_name, MODEL_PARAMS["gemma-3-4b"])
    bytes_per = BYTES_PER_ELEMENT.get(cache_type, 1)
    
    # KV cache for both keys and values
    cache_size = (
        2 *  # K and V
        params["layers"] *
        params["kv_heads"] *
        params["head_dim"] *
        context_size *
        bytes_per
    )
    return int(cache_size)
