"""
Models Router - Model Registry API.

Provides endpoints for:
- Model registry listing
- Model metrics
- Dynamic model routing
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["models"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    name: str
    version: str
    provider: str
    type: str  # chat, embedding, classification
    status: str
    context_length: int
    capabilities: list[str]
    created_at: str


class ModelMetrics(BaseModel):
    """Model performance metrics."""

    model_id: str
    requests_total: int
    requests_24h: int
    avg_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    tokens_processed: int
    cost_usd: float


class ModelRegistryResponse(BaseModel):
    """Model registry response."""

    models: list[ModelInfo]
    total: int


class RouteRequest(BaseModel):
    """Model routing request."""

    task: str  # chat, embedding, summarization
    max_latency_ms: Optional[int] = None
    max_cost_per_token: Optional[float] = None
    required_capabilities: list[str] = Field(default_factory=list)


class RouteResponse(BaseModel):
    """Model routing response."""

    recommended_model: str
    reason: str
    fallback_models: list[str]


# =============================================================================
# MODEL REGISTRY DATA (In-memory for now)
# =============================================================================

MODEL_REGISTRY = {
    "gemma-2b": {
        "id": "gemma-2b",
        "name": "Gemma 2B",
        "version": "1.0",
        "provider": "local",
        "type": "chat",
        "status": "available",
        "context_length": 8192,
        "capabilities": ["chat", "summarization", "analysis"],
    },
    "whisper-base": {
        "id": "whisper-base",
        "name": "Whisper Base",
        "version": "base.en",
        "provider": "local",
        "type": "transcription",
        "status": "available",
        "context_length": 0,
        "capabilities": ["transcription", "speech-to-text"],
    },
    "parakeet": {
        "id": "parakeet",
        "name": "NeMo Parakeet",
        "version": "1.x",
        "provider": "nvidia",
        "type": "transcription",
        "status": "available",
        "context_length": 0,
        "capabilities": ["transcription", "streaming"],
    },
}

MODEL_METRICS = {
    "gemma-2b": {
        "requests_total": 1250,
        "requests_24h": 45,
        "avg_latency_ms": 850.0,
        "p95_latency_ms": 2100.0,
        "error_rate": 0.02,
        "tokens_processed": 125000,
        "cost_usd": 0.0,  # Local model
    },
    "whisper-base": {
        "requests_total": 890,
        "requests_24h": 22,
        "avg_latency_ms": 1200.0,
        "p95_latency_ms": 3500.0,
        "error_rate": 0.01,
        "tokens_processed": 0,
        "cost_usd": 0.0,
    },
}


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("/registry", response_model=ModelRegistryResponse)
async def list_models(
    type: Optional[str] = Query(None, description="Filter by model type"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    status: Optional[str] = Query(None, description="Filter by status"),
):
    """
    List all registered models.

    Args:
        type: Filter by model type (chat, embedding, transcription)
        provider: Filter by provider (local, openai, nvidia)
        status: Filter by status (available, unavailable)

    Returns:
        List of registered models
    """
    models = list(MODEL_REGISTRY.values())

    if type:
        models = [m for m in models if m["type"] == type]
    if provider:
        models = [m for m in models if m["provider"] == provider]
    if status:
        models = [m for m in models if m["status"] == status]

    model_infos = [
        ModelInfo(
            id=m["id"],
            name=m["name"],
            version=m["version"],
            provider=m["provider"],
            type=m["type"],
            status=m["status"],
            context_length=m["context_length"],
            capabilities=m["capabilities"],
            created_at=datetime.utcnow().isoformat(),
        )
        for m in models
    ]

    return ModelRegistryResponse(models=model_infos, total=len(model_infos))


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """
    Get model by ID.

    Args:
        model_id: Model identifier

    Returns:
        Model information
    """
    model = MODEL_REGISTRY.get(model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelInfo(
        id=model["id"],
        name=model["name"],
        version=model["version"],
        provider=model["provider"],
        type=model["type"],
        status=model["status"],
        context_length=model["context_length"],
        capabilities=model["capabilities"],
        created_at=datetime.utcnow().isoformat(),
    )


@router.get("/{model_id}/metrics", response_model=ModelMetrics)
async def get_model_metrics(model_id: str):
    """
    Get model performance metrics.

    Args:
        model_id: Model identifier

    Returns:
        Model metrics
    """
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Model not found")

    metrics = MODEL_METRICS.get(model_id, {
        "requests_total": 0,
        "requests_24h": 0,
        "avg_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "error_rate": 0.0,
        "tokens_processed": 0,
        "cost_usd": 0.0,
    })

    return ModelMetrics(model_id=model_id, **metrics)


@router.post("/route", response_model=RouteResponse)
async def route_request(request: RouteRequest):
    """
    Route a request to the best model.

    Args:
        request: Routing request with constraints

    Returns:
        Recommended model and fallbacks
    """
    candidates = []

    for model_id, model in MODEL_REGISTRY.items():
        # Check capabilities
        if request.required_capabilities:
            if not all(cap in model["capabilities"] for cap in request.required_capabilities):
                continue

        # Check status
        if model["status"] != "available":
            continue

        # Check latency constraint
        metrics = MODEL_METRICS.get(model_id, {})
        if request.max_latency_ms:
            avg_latency = metrics.get("avg_latency_ms", float("inf"))
            if avg_latency > request.max_latency_ms:
                continue

        candidates.append(model_id)

    if not candidates:
        raise HTTPException(status_code=404, detail="No suitable model found")

    # Sort by error rate (lower is better)
    candidates.sort(key=lambda m: MODEL_METRICS.get(m, {}).get("error_rate", 1.0))

    return RouteResponse(
        recommended_model=candidates[0],
        reason="Lowest error rate among available models",
        fallback_models=candidates[1:3],
    )


logger.info("✅ Models Router initialized with registry endpoints")
