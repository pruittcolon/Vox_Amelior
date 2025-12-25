"""
Gemma AI Service - GPU Coordinator Integration
Handles LLM inference with dynamic GPU access via coordinator
"""

import inspect
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gemma_analyzer import GemmaAnalyzer
from llama_cpp import Llama

# Try to import GGML types for quantized KV cache
try:
    from llama_cpp import GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0
    GGML_TYPES_AVAILABLE = True
except ImportError:
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q4_0 = 2
    GGML_TYPES_AVAILABLE = False
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# Add shared modules to path
sys.path.insert(0, "/app")

# Configure logging (stdout)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("gemma")

# Configuration
GEMMA_MODEL_PATH = os.getenv("GEMMA_MODEL_PATH", "/app/models/gemma-3-4b-it-UD-Q4_K_XL.gguf")
GEMMA_GPU_LAYERS = int(os.getenv("GEMMA_GPU_LAYERS", "25"))  # Increased for 64k context
GEMMA_CONTEXT_SIZE = int(os.getenv("GEMMA_CONTEXT_SIZE", "16384"))  # Default 16k context with Q8 cache
GEMMA_BATCH_SIZE = int(os.getenv("GEMMA_BATCH_SIZE", "512"))
JWT_ONLY = os.getenv("JWT_ONLY", "false").lower() in {"1", "true", "yes"}
GPU_COORDINATOR_URL = os.getenv("GPU_COORDINATOR_URL", "http://gpu-coordinator:8002")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8004")
DEFAULT_REPEAT_PENALTY = float(os.getenv("GEMMA_REPEAT_PENALTY", "1.2"))
DEFAULT_REPEAT_LAST_N = int(os.getenv("GEMMA_REPEAT_LAST_N", "256"))
DEFAULT_PENALIZE_NL = os.getenv("GEMMA_PENALIZE_NL", "true").lower() in {"1", "true", "yes"}
DEFAULT_MIROSTAT = int(os.getenv("GEMMA_MIROSTAT", "2"))
DEFAULT_MIROSTAT_TAU = float(os.getenv("GEMMA_MIROSTAT_TAU", "5.0"))
DEFAULT_MIROSTAT_ETA = float(os.getenv("GEMMA_MIROSTAT_ETA", "0.1"))
DEFAULT_STOP_SEQUENCES = [token.strip() for token in os.getenv("GEMMA_DEFAULT_STOP", "").split(",") if token.strip()]

# KV Cache quantization setting (q8_0, q4_0, or fp16)
GEMMA_CACHE_TYPE = os.getenv("GEMMA_CACHE_TYPE", "q8_0").lower()
CACHE_TYPE_MAP = {
    "fp16": GGML_TYPE_F16,
    "q8_0": GGML_TYPE_Q8_0,
    "q4_0": GGML_TYPE_Q4_0,
}
current_cache_type: str = GEMMA_CACHE_TYPE

try:
    _LLAMA_CALL_KWARGS = set(inspect.signature(Llama.__call__).parameters.keys())
except Exception:
    _LLAMA_CALL_KWARGS = set()


def _llama_supports_kwarg(name: str) -> bool:
    return not _LLAMA_CALL_KWARGS or name in _LLAMA_CALL_KWARGS


# Global model instance
gemma_model: Llama | None = None
model_on_gpu: bool = False
service_auth = None
current_model_name: str = "gemma-3-4b"  # Track which model is loaded
gemma_analyzer: GemmaAnalyzer | None = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def merge_stop_sequences(requested: list[str] | None) -> list[str]:
    stops: list[str] = list(DEFAULT_STOP_SEQUENCES)
    if requested:
        for token in requested:
            token = (token or "").strip()
            if token and token not in stops:
                stops.append(token)
    return stops


def decoding_kwargs_from_request(
    *,
    repeat_penalty: float | None = None,
    repeat_last_n: int | None = None,
    penalize_nl: bool | None = None,
    mirostat: int | None = None,
    mirostat_tau: float | None = None,
    mirostat_eta: float | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    rp = DEFAULT_REPEAT_PENALTY if repeat_penalty is None else repeat_penalty
    if rp is not None and _llama_supports_kwarg("repeat_penalty"):
        kwargs["repeat_penalty"] = rp

    rl = DEFAULT_REPEAT_LAST_N if repeat_last_n is None else repeat_last_n
    if rl and rl > 0 and _llama_supports_kwarg("repeat_last_n"):
        kwargs["repeat_last_n"] = rl

    penalize_newlines = DEFAULT_PENALIZE_NL if penalize_nl is None else penalize_nl
    if _llama_supports_kwarg("penalize_nl"):
        kwargs["penalize_nl"] = penalize_newlines

    mirostat_mode = DEFAULT_MIROSTAT if mirostat is None else mirostat
    if mirostat_mode and _llama_supports_kwarg("mirostat"):
        kwargs["mirostat"] = mirostat_mode
        if _llama_supports_kwarg("mirostat_tau"):
            kwargs["mirostat_tau"] = DEFAULT_MIROSTAT_TAU if mirostat_tau is None else mirostat_tau
        if _llama_supports_kwarg("mirostat_eta"):
            kwargs["mirostat_eta"] = DEFAULT_MIROSTAT_ETA if mirostat_eta is None else mirostat_eta

    return kwargs


def get_service_headers(expires_in: int = 60) -> dict[str, str]:
    """Get service authentication headers (JWT only)."""
    if not service_auth:
        raise RuntimeError("Service authentication not initialized")
    try:
        token = service_auth.create_token(expires_in=expires_in)
        return {"X-Service-Token": token}
    except Exception as exc:
        logger.error(f"[GEMMA] Failed to create service JWT: {exc}")
        raise


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================


def load_model_cpu():
    """Load model on CPU/RAM - ready for GPU transfer"""
    global gemma_model, model_on_gpu

    logger.info(f"[GEMMA] Loading model from: {GEMMA_MODEL_PATH}")
    logger.info("[GEMMA] Mode: CPU (n_gpu_layers=0)")
    logger.info(f"[GEMMA] Context size: {GEMMA_CONTEXT_SIZE} tokens")

    if not Path(GEMMA_MODEL_PATH).exists():
        raise RuntimeError(f"Model file not found: {GEMMA_MODEL_PATH}")

    try:
        gemma_model = Llama(
            model_path=GEMMA_MODEL_PATH,
            n_ctx=GEMMA_CONTEXT_SIZE,
            n_batch=128,  # Smaller batch for stability
            n_gpu_layers=0,  # CPU mode
            n_threads=6,
            use_mlock=False,  # Don't lock memory
            use_mmap=True,  # Use memory mapping
            verbose=False,
        )
        model_on_gpu = False
        logger.info("[GEMMA] Model loaded on CPU/RAM successfully!")

    except Exception as e:
        logger.error(f"[GEMMA] ERROR loading model: {e}")
        raise


def load_model_gpu():
    """Load model directly on GPU at startup"""
    global gemma_model, model_on_gpu

    logger.info(f"[GEMMA] Loading model from: {GEMMA_MODEL_PATH}")
    logger.info(f"[GEMMA] Mode: GPU (n_gpu_layers={GEMMA_GPU_LAYERS})")
    logger.info(f"[GEMMA] Context size: {GEMMA_CONTEXT_SIZE} tokens")

    if not Path(GEMMA_MODEL_PATH).exists():
        raise RuntimeError(f"Model file not found: {GEMMA_MODEL_PATH}")

    # Get KV cache quantization type
    cache_ggml_type = CACHE_TYPE_MAP.get(current_cache_type, GGML_TYPE_Q8_0)
    logger.info(f"[GEMMA] KV cache type: {current_cache_type} (GGML type: {cache_ggml_type})")
    
    # Build kwargs for model loading
    model_kwargs = {
        "model_path": GEMMA_MODEL_PATH,
        "n_ctx": GEMMA_CONTEXT_SIZE,
        "n_batch": 512,
        "n_gpu_layers": -1,
        "n_threads": 4,
        "use_mlock": False,
        "use_mmap": True,
        "flash_attn": True,
        "verbose": True,
    }
    
    # Add quantized KV cache if supported
    if GGML_TYPES_AVAILABLE:
        model_kwargs["type_k"] = cache_ggml_type
        model_kwargs["type_v"] = cache_ggml_type
        logger.info(f"[GEMMA] Using quantized KV cache: {current_cache_type}")
    else:
        logger.warning("[GEMMA] GGML types not available, using default FP16 cache")
    
    gemma_model = Llama(**model_kwargs)
    model_on_gpu = True
    logger.info("[GEMMA] Model loaded on GPU successfully!")

    # Log VRAM usage
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            vram_used = result.stdout.strip().split("\n")[0]
            logger.info(f"[GEMMA] VRAM used: {vram_used} MB")
    except Exception as e:
        logger.warning(f"[GEMMA] Could not query VRAM: {e}")


def move_model_to_gpu():
    """Move model from CPU to GPU for fast inference"""
    global gemma_model, model_on_gpu

    if model_on_gpu:
        logger.info("[GEMMA] Model already on GPU")
        return

    logger.info("[GEMMA] 🚀 Moving model from CPU to GPU for fast inference...")

    # Save reference to old model in case we need to restore
    old_model = gemma_model

    try:
        # Delete CPU model to free RAM
        del gemma_model
        gemma_model = None

        # Reload on GPU
        load_model_gpu()

        logger.info("[GEMMA] ✅ Model moved to GPU successfully")

    except Exception as e:
        logger.error(f"[GEMMA] ❌ ERROR moving to GPU: {e}")
        logger.info("[GEMMA] Falling back to CPU mode...")
        # Restore old model or reload on CPU
        if old_model is not None:
            gemma_model = old_model
            model_on_gpu = False
            logger.info("[GEMMA] ✅ Restored existing CPU model")
        else:
            load_model_cpu()
            logger.info("[GEMMA] ✅ Reloaded model on CPU")
        # Don't re-raise - CPU fallback is acceptable


def move_model_to_cpu():
    """Move model from GPU back to CPU to free VRAM"""
    global gemma_model, model_on_gpu

    if not model_on_gpu:
        logger.info("[GEMMA] Model already on CPU")
        return

    logger.info("[GEMMA] 💾 Moving model from GPU back to CPU to free VRAM...")

    try:
        # Delete GPU model to free VRAM
        del gemma_model
        gemma_model = None

        # Force garbage collection and clear CUDA cache
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("[GEMMA] CUDA cache cleared")
        except ImportError:
            pass

        # Reload on CPU (background mode)
        load_model_cpu()

        logger.info("[GEMMA] ✅ Model moved back to CPU successfully")

    except Exception as e:
        logger.error(f"[GEMMA] ❌ ERROR moving to CPU: {e}")
        raise


class ServiceAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT service authentication (Phase 3: Enforce JWT-only + replay)."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health checks (internal utility)
        if request.url.path in ["/health", "/settings"]:
            return await call_next(request)
        
        # Allow localhost/container-internal requests for stress testing
        # This enables docker exec curl commands and internal health probes
        client_host = request.client.host if request.client else None
        if client_host in ["127.0.0.1", "localhost", "::1"]:
            logger.debug(f"[AUTH] Localhost bypass for {request.url.path}")
            return await call_next(request)

        jwt_token = request.headers.get("X-Service-Token")
        if not jwt_token or not service_auth:
            logger.error(f"❌ Missing JWT for {request.url.path}")
            return JSONResponse(status_code=401, content={"detail": "Missing service token"})

        try:
            # Allowed callers: gateway, transcription-service (signals), ml-service (scoring)
            allowed = ["gateway", "transcription-service", "ml-service"]
            payload = service_auth.verify_token(jwt_token, allowed_services=allowed, expected_aud="internal")

            # Note: Replay protection is already handled inside verify_token()
            # No need for additional check here

            rid_short = str(payload.get("request_id", ""))[:8]
            logger.info(f"✅ JWT OK s={payload.get('service_id')} aud=internal rid={rid_short} path={request.url.path}")
            return await call_next(request)
        except Exception as e:
            logger.error(f"❌ JWT rejected: {e} path={request.url.path}")
            return JSONResponse(status_code=401, content={"detail": "Invalid service token"})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Adds request_id, analysis_id, and structured access logs with latency."""

    async def dispatch(self, request: Request, call_next):
        import time

        req_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
        analysis_id = request.headers.get("X-Analysis-Id") or request.query_params.get("analysis_id")
        start = time.monotonic()
        path = request.url.path
        method = request.method
        try:
            response = await call_next(request)
            duration_ms = int((time.monotonic() - start) * 1000)
            try:
                response.headers["X-Request-Id"] = req_id
            except Exception:
                pass
            logger.info(
                f"[ACCESS] {method} {path} {response.status_code} rid={req_id} analysis_id={analysis_id or '-'} {duration_ms}ms"
            )
            return response
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                f"[ERROR] {method} {path} rid={req_id} analysis_id={analysis_id or '-'} {duration_ms}ms err={exc}"
            )
            raise


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown"""
    # Startup
    logger.info("Starting Gemma AI Service...")
    logger.info("=" * 80)

    # ISO 27002: Fail-closed security check
    from shared.security.startup_checks import assert_secure_mode

    assert_secure_mode()  # Blocks startup if SECURE_MODE=true with unsafe flags

    # Initialize service auth (Phase 3)
    global service_auth, gemma_analyzer
    try:
        from shared.security.service_auth import get_service_auth, load_service_jwt_keys

        jwt_keys = load_service_jwt_keys("gemma-service")
        service_auth = get_service_auth(service_id="gemma-service", service_secret=jwt_keys)
        logger.info(
            "✅ JWT service auth initialized (enforcing JWT-only, aud=internal, replay protected, keys=%s)",
            len(jwt_keys),
        )
    except Exception as e:
        logger.error(f"❌ JWT service auth initialization failed: {e}")
        raise

    # START ON GPU DIRECTLY (transcription now starts on CPU)
    # With START_ON_CPU=true for transcription, Gemma can claim GPU at startup
    logger.info("[GEMMA] 🚀 Loading model DIRECTLY on GPU (transcription on CPU)...")
    try:
        load_model_gpu()
        gemma_analyzer = GemmaAnalyzer(rag_url=RAG_SERVICE_URL)
        logger.info("[GEMMA] ✅ Loaded on GPU successfully with full 25k context!")
        logger.info("Gemma AI Service started successfully (GPU mode)")
    except Exception as e:
        logger.error(f"[GEMMA] ❌ Failed to load on GPU: {e}")
        logger.info("[GEMMA] ⚠️ Falling back to CPU mode...")
        load_model_cpu()
        gemma_analyzer = GemmaAnalyzer(rag_url=RAG_SERVICE_URL)
        logger.info("[GEMMA] ✅ Loaded on CPU successfully!")
        logger.info("Gemma AI Service started (CPU fallback mode)")
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("Shutting down Gemma AI Service...")
    global gemma_model
    if gemma_model:
        del gemma_model
    logger.info("Gemma AI Service shutdown complete")


# Initialize FastAPI
app = FastAPI(
    title="Gemma AI Service",
    description="LLM inference service with dynamic GPU access",
    version="1.0.0",
    lifespan=lifespan,
)

# Add JWT middleware (Phase 2: Permissive)
app.add_middleware(ServiceAuthMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# ISO 27002 8.9 / OWASP API7:2023: CORS - No wildcards for internal services
# Gemma is internal-only, accessed via api-gateway proxy
CORS_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://localhost",
    "https://127.0.0.1",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST MODELS
# ============================================================================


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: list[str] | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    penalize_nl: bool | None = None
    mirostat: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    context: list[dict[str, Any]] | None = None
    stop: list[str] | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    penalize_nl: bool | None = None
    mirostat: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None


class GemmaAnalyzeRequest(BaseModel):
    filters: dict[str, Any] = Field(default_factory=dict)
    custom_prompt: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.3


class RAGChatRequest(BaseModel):
    """RAG-enhanced chat request"""

    query: str
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    session_id: str | None = None
    top_k_results: int = 8
    start_date: str | None = None
    end_date: str | None = None
    last_n_transcripts: int | None = None
    speakers: list[str] | None = None
    bias_emotions: list[str] | None = None
    use_memories: bool = False
    include_memories_top_k: int = 3
    stop: list[str] | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    penalize_nl: bool | None = None
    mirostat: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None


# ============================================================================
# ROUTES
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy" if gemma_model is not None else "unhealthy",
        "model_loaded": gemma_model is not None,
        "model_on_gpu": model_on_gpu,
        "model_name": current_model_name,
        "model_path": GEMMA_MODEL_PATH,
        "gpu_layers": GEMMA_GPU_LAYERS,
        "context_size": GEMMA_CONTEXT_SIZE,
        "cache_type": current_cache_type,
    }

    # Try to get VRAM usage
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            vram_info = result.stdout.strip().split("\n")[0].split(",")
            status["vram_used_mb"] = int(vram_info[0].strip())
            status["vram_total_mb"] = int(vram_info[1].strip())
    except Exception:
        pass

    return status


@app.get("/models/list")
async def list_models():
    """List available LLM models and current settings"""
    return {
        "models": [
            {
                "name": "gemma-3-4b",
                "description": "Gemma 3 4B Instruct - High quality, larger model",
                "context_size": 8192,
                "size_mb": 2430,
            },
            {
                "name": "granite-1b",
                "description": "IBM Granite 4.0 1B - Fast, lightweight model",
                "context_size": 4096,
                "size_mb": 1100,
            },
        ],
        "current_model": current_model_name,
        "cache_types": ["fp16", "q8_0", "q4_0"],
        "current_cache": current_cache_type,
    }


@app.get("/settings")
async def get_settings():
    """Get current model settings"""
    return {
        "model_name": current_model_name,
        "model_path": GEMMA_MODEL_PATH,
        "context_size": GEMMA_CONTEXT_SIZE,
        "cache_type": current_cache_type,
        "model_on_gpu": model_on_gpu,
        "ggml_types_available": GGML_TYPES_AVAILABLE,
        "available_cache_types": ["fp16", "q8_0", "q4_0"],
        "available_context_sizes": [1024, 2048, 4096, 8192, 16384, 32768],
    }


@app.post("/settings/update")
async def update_settings(request: Request):
    """
    Update model settings dynamically.
    
    This will reload the model with new settings.
    Warning: This operation takes 10-30 seconds.
    
    Body:
        - cache_type: "fp16", "q8_0", or "q4_0"
        - context_size: int (e.g., 2048, 4096, 8192, 16384)
    """
    global current_cache_type, GEMMA_CONTEXT_SIZE, gemma_model, model_on_gpu
    
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    new_cache = body.get("cache_type", current_cache_type)
    new_context = body.get("context_size", GEMMA_CONTEXT_SIZE)
    
    # Validate cache type
    if new_cache not in CACHE_TYPE_MAP:
        return {"status": "error", "message": f"Invalid cache_type. Must be one of: {list(CACHE_TYPE_MAP.keys())}"}
    
    # Validate context size
    valid_contexts = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    if new_context not in valid_contexts:
        return {"status": "error", "message": f"Invalid context_size. Must be one of: {valid_contexts}"}
    
    # Check if anything changed
    if new_cache == current_cache_type and new_context == GEMMA_CONTEXT_SIZE:
        return {
            "status": "no_change",
            "message": "Settings unchanged",
            "cache_type": current_cache_type,
            "context_size": GEMMA_CONTEXT_SIZE,
        }
    
    logger.info(f"[GEMMA] 🔄 Updating settings: cache={new_cache}, context={new_context}")
    
    # Store old values for rollback
    old_cache = current_cache_type
    old_context = GEMMA_CONTEXT_SIZE
    
    try:
        # Update global settings
        current_cache_type = new_cache
        GEMMA_CONTEXT_SIZE = new_context
        
        # Clear current model
        if gemma_model is not None:
            try:
                gemma_model.reset()
            except Exception:
                pass
            gemma_model = None
            model_on_gpu = False
            
            # Force CUDA memory release
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
        
        # Reload model with new settings
        load_model_gpu()
        
        # Get new VRAM usage
        vram_used = 0
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                vram_used = int(result.stdout.strip().split("\n")[0])
        except Exception:
            pass
        
        logger.info(f"[GEMMA] ✅ Settings updated: cache={new_cache}, context={new_context}, VRAM={vram_used}MB")
        
        return {
            "status": "success",
            "message": "Model reloaded with new settings",
            "cache_type": current_cache_type,
            "context_size": GEMMA_CONTEXT_SIZE,
            "model_on_gpu": model_on_gpu,
            "vram_used_mb": vram_used,
        }
        
    except Exception as e:
        logger.error(f"[GEMMA] ❌ Failed to update settings: {e}")
        # Rollback
        current_cache_type = old_cache
        GEMMA_CONTEXT_SIZE = old_context
        try:
            load_model_gpu()
        except Exception:
            pass
        
        return {
            "status": "error",
            "message": f"Failed to update settings: {str(e)}",
            "cache_type": current_cache_type,
            "context_size": GEMMA_CONTEXT_SIZE,
        }



@app.post("/move-to-cpu")
async def move_to_cpu_endpoint():
    """Move model from GPU to CPU (called when other services start)"""
    global model_on_gpu

    if not model_on_gpu:
        return {"status": "already_on_cpu", "message": "Model is already on CPU"}

    logger.info("[GEMMA] 📡 Received signal to move to CPU (other services starting)")

    try:
        move_model_to_cpu()
        return {
            "status": "success",
            "message": "Model moved to CPU, GPU freed for transcription",
            "model_on_gpu": False,
        }
    except Exception as e:
        logger.error(f"[GEMMA] Failed to move to CPU: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/release-session")
async def release_session():
    """
    Release GPU session - moves model to CPU and signals transcription to resume.
    Called by frontend when Gemma page closes (via beforeunload).
    """
    global model_on_gpu

    if not model_on_gpu:
        logger.info("[GEMMA] 📡 Session release requested but model already on CPU")
        return {"status": "already_on_cpu", "message": "Model already on CPU"}

    logger.info("[GEMMA] 📡 Session release requested - moving to CPU for transcription")

    try:
        move_model_to_cpu()

        # Signal GPU coordinator that Gemma session ended
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{GPU_COORDINATOR_URL}/gemma/release/session-end",
                    json={"result": {"status": "session_closed"}},
                    headers=get_service_headers(),
                )
            logger.info("[GEMMA] ✅ GPU coordinator notified of session end")
        except Exception as coord_error:
            logger.warning(f"[GEMMA] Could not notify coordinator: {coord_error}")

        return {"status": "released", "message": "GPU released for transcription", "model_on_gpu": False}
    except Exception as e:
        logger.error(f"[GEMMA] Failed to release session: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/warmup")
async def warmup_gpu():
    """
    Warmup endpoint - moves model to GPU and waits until ready
    Call this BEFORE sending your actual generate request
    """
    global model_on_gpu, gemma_model

    if gemma_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info("[GEMMA] 🔥 Warmup requested - preparing GPU...")
    task_id = f"gemma-warmup-{uuid.uuid4().hex[:8]}"
    gpu_acquired = False
    response_payload: dict[str, Any] | None = None
    release_payload: dict[str, Any] | None = None

    try:
        logger.info(f"📡 [WARMUP] Requesting GPU slot from coordinator (task={task_id})...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            coord_response = await client.post(
                f"{GPU_COORDINATOR_URL}/gemma/request",
                json={
                    "task_id": task_id,
                    "messages": [{"role": "system", "content": "warmup"}],
                    "max_tokens": 1,
                    "temperature": 0.7,
                },
                headers=get_service_headers(),
            )
            if coord_response.status_code != 200:
                logger.error(f"❌ [WARMUP] Failed to acquire GPU slot (status {coord_response.status_code})")
                raise HTTPException(status_code=503, detail="Failed to acquire GPU slot")
        gpu_acquired = True

        if not model_on_gpu:
            logger.info("[GEMMA] 🚀 Attempting to move model to GPU for warmup...")
            try:
                move_model_to_gpu()
                logger.info("[GEMMA] ✅ Model ready on GPU!")
                response_payload = {
                    "status": "ready",
                    "message": "Model warmed up and ready on GPU",
                    "model_on_gpu": True,
                    "task_id": task_id,
                }
                release_payload = {"result": {"status": "warmup", "model_on_gpu": True}}
            except Exception as gpu_error:
                logger.warning(f"[GEMMA] ⚠️ Could not move to GPU: {gpu_error}")
                logger.info("[GEMMA] 💾 GPU load failed - continuing on CPU")
                response_payload = {
                    "status": "ready_cpu",
                    "message": "Model ready on CPU (GPU unavailable)",
                    "model_on_gpu": False,
                    "task_id": task_id,
                    "warning": "Using CPU - inference will be slower",
                }
                release_payload = {"result": {"status": "warmup_cpu", "warning": str(gpu_error)}}
        else:
            logger.info("[GEMMA] ✅ Model already on GPU!")
            response_payload = {
                "status": "ready",
                "message": "Model already on GPU",
                "model_on_gpu": True,
                "task_id": task_id,
            }
            release_payload = {"result": {"status": "warmup_cached", "model_on_gpu": True}}
    except HTTPException:
        release_payload = {"result": {"error": "warmup_failed_before_acquire"}}
        raise
    except Exception as e:
        release_payload = {"result": {"error": str(e)}}
        raise
    finally:
        # Only release GPU slot if warmup FAILED
        # If successful, keep GPU for subsequent /chat calls in this session
        # User must call /release-session when done with Gemma
        if gpu_acquired and not model_on_gpu:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        f"{GPU_COORDINATOR_URL}/gemma/release/{task_id}",
                        json=release_payload or {"result": {"status": "warmup_failed"}},
                        headers=get_service_headers(),
                    )
                logger.info("[GEMMA] ✅ Released GPU slot after failed warmup")
            except Exception as release_error:
                logger.error(f"[GEMMA] ❌ Failed to release GPU slot: {release_error}")
        elif model_on_gpu:
            logger.info("[GEMMA] 🔒 Keeping GPU slot for session - call /release-session when done")

    return response_payload


async def run_llm_task(
    task_prefix: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    *,
    top_p: float = 0.9,
    stop: list[str] | None = None,
    decoding_overrides: dict[str, Any] | None = None,
):
    if gemma_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    task_id = f"{task_prefix}-{uuid.uuid4().hex[:12]}"
    logger.info("=" * 80)
    logger.info(f"🔄 [TASK {task_id}] Prompt preview: {prompt[:100]}...")
    logger.info(f"🔄 [TASK {task_id}] Max tokens: {max_tokens}, Temp: {temperature}")

    gpu_slot_acquired = False
    using_cpu_fallback = False
    try:
        logger.info(f"📡 [TASK {task_id}] Requesting GPU slot from coordinator...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            coord_response = await client.post(
                f"{GPU_COORDINATOR_URL}/gemma/request",
                json={
                    "task_id": task_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                headers=get_service_headers(),
            )
            if coord_response.status_code != 200:
                logger.error(f"❌ [TASK {task_id}] Failed to acquire GPU slot: {coord_response.status_code}")
                raise HTTPException(status_code=503, detail="Failed to acquire GPU slot from coordinator")
        gpu_slot_acquired = True

        if not model_on_gpu:
            logger.warning(f"⚠️ [TASK {task_id}] Model not on GPU! Attempting warmup...")
            try:
                move_model_to_gpu()
                logger.info(f"🚀 [TASK {task_id}] Moved to GPU, executing inference...")
            except Exception as gpu_error:
                logger.warning(f"⚠️ [TASK {task_id}] GPU load failed, using CPU: {gpu_error}")
                using_cpu_fallback = True
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        await client.post(
                            f"{GPU_COORDINATOR_URL}/gemma/release/{task_id}",
                            json={"error": "GPU unavailable - using CPU"},
                            headers=get_service_headers(),
                        )
                    gpu_slot_acquired = False
                except Exception as release_error:
                    logger.error(f"❌ Failed to release GPU slot: {release_error}")
        else:
            logger.info(f"🚀 [TASK {task_id}] Executing inference on GPU (already warmed up)...")

        stop_sequences = merge_stop_sequences(stop)
        llama_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop_sequences,
            "echo": False,
        }
        if decoding_overrides:
            llama_kwargs.update(
                decoding_kwargs_from_request(
                    repeat_penalty=decoding_overrides.get("repeat_penalty"),
                    repeat_last_n=decoding_overrides.get("repeat_last_n"),
                    penalize_nl=decoding_overrides.get("penalize_nl"),
                    mirostat=decoding_overrides.get("mirostat"),
                    mirostat_tau=decoding_overrides.get("mirostat_tau"),
                    mirostat_eta=decoding_overrides.get("mirostat_eta"),
                )
            )
        else:
            llama_kwargs.update(decoding_kwargs_from_request())

        # CRITICAL: Reset model state before each inference to prevent KV cache buildup
        try:
            gemma_model.reset()
            logger.debug(f"[TASK {task_id}] Model state reset before generation")
        except Exception as e:
            logger.warning(f"[TASK {task_id}] Could not reset model state: {e}")

        # GPU READINESS: Ensure CUDA operations are complete before inference
        if model_on_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for any pending GPU transfers
                    logger.debug(f"[TASK {task_id}] CUDA synchronized, GPU ready")
            except Exception as sync_err:
                logger.debug(f"[TASK {task_id}] CUDA sync skipped: {sync_err}")

        response = gemma_model(**llama_kwargs)

        # CRITICAL: Clear KV cache after inference to prevent VRAM buildup
        try:
            gemma_model.reset()  # Clear KV cache
            import gc

            gc.collect()
            if model_on_gpu:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            logger.debug(f"[TASK {task_id}] KV cache cleared after generation")
        except Exception as cache_err:
            logger.warning(f"[TASK {task_id}] Could not clear cache: {cache_err}")

        usage = response.get("usage", {})
        model_name = response.get("model", "gemma-3-4b-it")
        result = {
            "text": response["choices"][0]["text"],
            "tokens_generated": usage.get("completion_tokens"),
            "task_id": task_id,
            "mode": "cpu" if using_cpu_fallback else "gpu",
            "usage": usage,
            "model": model_name,
        }

        # SESSION-BASED GPU RETENTION: Model stays on GPU for fast inference during session
        # GPU is released when page closes via /release-session endpoint
        # This prevents rapid GPU↔CPU transitions that cause OOM errors
        logger.info(f"💾 [TASK {task_id}] Model staying on GPU for session performance")

        # DO NOT release GPU slot after each inference - keep for the session
        # This prevents transcription from reclaiming GPU between QA chunks
        if gpu_slot_acquired:
            logger.info(f"🔒 [TASK {task_id}] Keeping GPU slot (release via /release-session)")
        else:
            logger.info(f"✅ [TASK {task_id}] Complete (CPU fallback mode)")

        logger.info(f"✅ [TASK {task_id}] Inference complete - Generated {result.get('tokens_generated')} tokens")
        logger.info(f"📝 [TASK {task_id}] Response preview: {result['text'][:200]}...")
        logger.info("=" * 80)
        return result

    except Exception as e:
        logger.error(f"❌ [TASK {task_id}] Error: {e}")
        logger.error("=" * 80)
        try:
            if gpu_slot_acquired:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        f"{GPU_COORDINATOR_URL}/gemma/release/{task_id}",
                        json={"result": {"error": str(e)}},
                        headers=get_service_headers(),
                    )
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    return await run_llm_task(
        "gemma-gen",
        request.prompt,
        request.max_tokens,
        request.temperature,
        top_p=request.top_p,
        stop=request.stop,
        decoding_overrides={
            "repeat_penalty": request.repeat_penalty,
            "repeat_last_n": request.repeat_last_n,
            "penalize_nl": request.penalize_nl,
            "mirostat": request.mirostat,
            "mirostat_tau": request.mirostat_tau,
            "mirostat_eta": request.mirostat_eta,
        },
    )


@app.post("/analyze")
async def gemma_quick_analyze(request: GemmaAnalyzeRequest):
    if gemma_analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer unavailable")

    filters = request.filters or {}
    try:
        service_headers = get_service_headers(expires_in=120)
    except Exception as exc:
        logger.warning(f"[ANALYZE] Unable to create service headers: {exc}")
        service_headers = None

    async def _llm(prompt: str, max_tokens: int, temperature: float):
        return await run_llm_task(
            "gemma-ana",
            prompt,
            max_tokens,
            temperature,
        )

    try:
        return await gemma_analyzer.run_analysis(
            llm_callable=_llm,
            filters=filters,
            custom_prompt=request.custom_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            service_headers=service_headers,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[ANALYZE] Failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat with conversation history
    Uses GPU coordinator for exclusive GPU access
    """
    if gemma_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    task_id = f"gemma-chat-{uuid.uuid4().hex[:12]}"

    logger.info("=" * 80)
    logger.info(f"💬 [CHAT] Task {task_id} started")
    logger.info(f"💬 [CHAT] Messages: {len(request.messages)}, Max tokens: {request.max_tokens}")

    try:
        # Request GPU from coordinator
        logger.info(f"📡 [TASK {task_id}] Requesting GPU slot from coordinator...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            coord_response = await client.post(
                f"{GPU_COORDINATOR_URL}/gemma/request",
                json={
                    "task_id": task_id,
                    "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "context": request.context or [],
                },
                headers=get_service_headers(),
            )

            if coord_response.status_code != 200:
                logger.error(f"❌ [TASK {task_id}] Failed to acquire GPU slot: {coord_response.status_code}")
                raise HTTPException(status_code=503, detail="Failed to acquire GPU slot from coordinator")

            logger.info(f"✅ [TASK {task_id}] GPU slot acquired")
        gpu_slot_acquired = True
        gpu_slot_released = False

        # Move model to GPU if it's on CPU
        if not model_on_gpu:
            logger.info(f"[TASK {task_id}] Moving model to GPU for inference...")
            move_model_to_gpu()

        logger.info(f"[TASK {task_id}] Using model in {('GPU' if model_on_gpu else 'CPU')} mode")

        # Build prompt from messages
        prompt = "You are a helpful AI assistant. Always respond in English.\n\n"

        # Add context if provided
        if request.context:
            prompt += "Context from memories:\n"
            for ctx in request.context[:3]:  # Top 3 memories
                prompt += f"- {ctx.get('content', '')[:200]}\n"
            prompt += "\n"

        # Add conversation history
        for msg in request.messages:
            role = msg.role.upper()
            prompt += f"{role}: {msg.content}\n"

        prompt += "ASSISTANT: "

        logger.info(f"[TASK {task_id}] Executing inference...")

        # Track inference time
        import time

        start_time = time.time()

        # CRITICAL: Reset model state before each inference to prevent KV cache buildup
        # This clears the internal context from previous calls, ensuring each request
        # starts fresh and doesn't accumulate tokens that could exhaust the context window
        try:
            gemma_model.reset()
            logger.debug(f"[TASK {task_id}] Model state reset successfully")
        except Exception as e:
            logger.warning(f"[TASK {task_id}] Could not reset model state: {e}")

        stop_sequences = merge_stop_sequences(request.stop)
        llama_kwargs = {
            "prompt": prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "echo": False,
            "stop": stop_sequences,
        }
        llama_kwargs.update(
            decoding_kwargs_from_request(
                repeat_penalty=request.repeat_penalty,
                repeat_last_n=request.repeat_last_n,
                penalize_nl=request.penalize_nl,
                mirostat=request.mirostat,
                mirostat_tau=request.mirostat_tau,
                mirostat_eta=request.mirostat_eta,
            )
        )

        response = gemma_model(**llama_kwargs)

        generated_text = response["choices"][0]["text"].strip()
        tokens_generated = response["usage"].get("completion_tokens", 0)
        prompt_tokens = response["usage"].get("prompt_tokens", 0)

        # Calculate inference time and tokens/sec
        import time

        inference_end = time.time()
        inference_time = inference_end - start_time if "start_time" in locals() else 0
        tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0

        if tokens_generated == 0:
            logger.warning(f"⚠️ [TASK {task_id}] Chat generated 0 tokens. Check stop tokens/prompt.")

        # Get VRAM usage stats
        vram_stats = {}
        try:
            import subprocess

            vram_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if vram_result.returncode == 0:
                info = vram_result.stdout.strip().split("\n")[0].split(",")
                vram_stats = {
                    "vram_used_mb": int(info[0].strip()),
                    "vram_total_mb": int(info[1].strip()),
                    "gpu_utilization": int(info[2].strip()),
                }
        except Exception as e:
            logger.debug(f"Could not query VRAM: {e}")

        result = {
            "message": generated_text or "",
            "tokens_generated": tokens_generated,
            "prompt_tokens": prompt_tokens,
            "total_tokens": tokens_generated + prompt_tokens,
            "task_id": task_id,
            "mode": "gpu" if model_on_gpu else "cpu",
            "inference_time_sec": round(inference_time, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
            **vram_stats,
        }

        logger.info(f"✅ [TASK {task_id}] Inference complete - Generated {result['tokens_generated']} tokens")
        logger.info(f"💬 [TASK {task_id}] Response: {result['message'][:200]}...")

        # Release GPU slot back to coordinator
        if gpu_slot_acquired and not gpu_slot_released:
            logger.info(f"🔓 [TASK {task_id}] Releasing GPU slot...")
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{GPU_COORDINATOR_URL}/gemma/release/{task_id}",
                    json={"result": result},
                    headers=get_service_headers(),
                )
            logger.info(f"✅ [TASK {task_id}] GPU slot released, model remains on GPU")
            gpu_slot_released = True
        logger.info("=" * 80)

        return result

    except Exception as e:
        logger.error(f"❌ [TASK {task_id}] Error: {e}")
        logger.error("=" * 80)

        # Release GPU on error if still held
        try:
            if gpu_slot_acquired and not gpu_slot_released:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        f"{GPU_COORDINATOR_URL}/gemma/release/{task_id}",
                        json={"result": {"error": str(e)}},
                        headers=get_service_headers(),
                    )
        except Exception:
            pass

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/rag")
async def chat_with_rag(request: RAGChatRequest):
    """
    RAG-enhanced chat: Retrieves relevant context from transcripts, then generates answer

    Flow:
    1. Search RAG service for relevant segments
    2. Build context prompt with results
    3. Generate answer with Gemma
    4. Return answer with sources
    """
    if gemma_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    task_id = f"gemma-rag-{uuid.uuid4().hex[:12]}"
    gpu_slot_acquired = False
    gpu_slot_released = False
    used_gpu = False

    def format_timestamp(seconds: float | None) -> str | None:
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            return None
        minutes = int(value // 60)
        secs = int(value % 60)
        return f"{minutes:02d}:{secs:02d}"

    async def fetch_rag_results(doc_type: str, top_k: int) -> list[dict[str, Any]]:
        payload = {
            "query": request.query,
            "top_k": top_k,
            "doc_type": doc_type,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "last_n_transcripts": request.last_n_transcripts,
            "speakers": request.speakers,
            "bias_emotions": request.bias_emotions,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{RAG_SERVICE_URL}/search/semantic",
                json=payload,
                headers=get_service_headers(),
            )
        if response.status_code != 200:
            logger.warning(f"[TASK {task_id}] RAG search {doc_type} failed: {response.status_code}")
            return []
        data = response.json()
        results = data.get("results", [])
        logger.info(f"[TASK {task_id}] Retrieved {len(results)} {doc_type} results")
        return results

    try:
        transcript_results = await fetch_rag_results("transcript_segment", request.top_k_results)
        memory_results: list[dict[str, Any]] = []
        if request.use_memories:
            memory_results = await fetch_rag_results("memory", request.include_memories_top_k)

        if not transcript_results and not memory_results:
            logger.info(f"[TASK {task_id}] No RAG context found for query")
            return {
                "answer": "I couldn't find any conversation excerpts that answer that question within the selected scope.",
                "mode": "cpu" if not model_on_gpu else "gpu",
                "tokens_generated": 0,
                "confidence": 0.0,
                "sources": [],
                "task_id": task_id,
            }

        context_lines: list[str] = []
        sources: list[dict[str, Any]] = []
        citation_index = 1

        def append_context(result_list: list[dict[str, Any]], prefix: str = "C") -> None:
            nonlocal citation_index
            for result in result_list:
                metadata = result.get("metadata") or {}
                text = result.get("text") or ""
                label = f"{prefix}{citation_index}"

                if result.get("type") == "memory":
                    title = metadata.get("title") or "Memory"
                    context_lines.append(f'[{label}] Memory • {title}: "{text}"')
                    sources.append(
                        {
                            "label": label,
                            "type": "memory",
                            "title": title,
                            "text_preview": text[:180],
                            "score": result.get("score", 0.0),
                            "created_at": metadata.get("created_at"),
                            "memory_id": metadata.get("memory_id"),
                        }
                    )
                else:
                    speaker = metadata.get("speaker") or "Unknown speaker"
                    emotion = metadata.get("emotion") or "neutral"
                    timestamp_display = format_timestamp(metadata.get("start_time")) or "--:--"
                    context_lines.append(f'[{label}] {timestamp_display} • {speaker} ({emotion}): "{text}"')
                    sources.append(
                        {
                            "label": label,
                            "type": "transcript_segment",
                            "speaker": speaker,
                            "timestamp": timestamp_display,
                            "start_time": metadata.get("start_time"),
                            "end_time": metadata.get("end_time"),
                            "job_id": metadata.get("job_id"),
                            "transcript_id": metadata.get("transcript_id"),
                            "segment_id": metadata.get("segment_id"),
                            "emotion": emotion,
                            "score": result.get("score", 0.0),
                            "text_preview": text[:180],
                            "created_at": metadata.get("created_at"),
                        }
                    )
                citation_index += 1

        append_context(transcript_results[: request.top_k_results])
        if memory_results:
            append_context(memory_results[: request.include_memories_top_k], prefix="M")

        instruction = (
            "You are an assistant that answers questions using the provided conversation excerpts. "
            "Each excerpt has a label like [C1]. Use those labels in your answer as citations. "
            "If the excerpts do not contain the answer, say that you cannot find evidence."
        )
        context_block = "\n".join(context_lines)
        prompt = (
            f"{instruction}\n\nConversation excerpts:\n{context_block}\n\nUser question: {request.query}\n\nAnswer:"
        )

        logger.info(f"[TASK {task_id}] Prompt prepared with {len(context_lines)} excerpts")

        # Acquire GPU slot
        logger.info(f"[TASK {task_id}] Requesting GPU access…")
        async with httpx.AsyncClient(timeout=30.0) as client:
            coord_response = await client.post(
                f"{GPU_COORDINATOR_URL}/gemma/request",
                json={
                    "task_id": task_id,
                    "messages": [{"role": "user", "content": request.query}],
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                },
                headers=get_service_headers(),
            )
        if coord_response.status_code != 200:
            raise HTTPException(status_code=503, detail="Failed to acquire GPU")
        gpu_slot_acquired = True

        if not model_on_gpu:
            try:
                move_model_to_gpu()
                used_gpu = True
            except Exception as gpu_error:
                logger.warning(f"⚠️ [TASK {task_id}] Failed to move model to GPU: {gpu_error}. Falling back to CPU.")
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        f"{GPU_COORDINATOR_URL}/gemma/release/{task_id}",
                        json={"result": {"warning": str(gpu_error), "mode": "cpu"}},
                        headers=get_service_headers(),
                    )
                gpu_slot_acquired = False
                gpu_slot_released = True
        else:
            used_gpu = True

        stop_sequences = merge_stop_sequences(request.stop)
        generation_kwargs = {
            "prompt": prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "echo": False,
            "stop": stop_sequences,
        }
        generation_kwargs.update(
            decoding_kwargs_from_request(
                repeat_penalty=request.repeat_penalty,
                repeat_last_n=request.repeat_last_n,
                penalize_nl=request.penalize_nl,
                mirostat=request.mirostat,
                mirostat_tau=request.mirostat_tau,
                mirostat_eta=request.mirostat_eta,
            )
        )

        # CRITICAL: Reset model state before each inference to prevent KV cache buildup
        try:
            gemma_model.reset()
            logger.debug(f"[TASK {task_id}] Model state reset before RAG generation")
        except Exception as e:
            logger.warning(f"[TASK {task_id}] Could not reset model state: {e}")

        response = gemma_model(**generation_kwargs)
        answer_text = response["choices"][0]["text"].strip()
        tokens_generated = response["usage"].get("completion_tokens", 0)

        if tokens_generated == 0:
            logger.warning(f"⚠️ [TASK {task_id}] Generated 0 tokens; check prompt/stop settings")

        score_values = [res.get("score", 0.0) for res in transcript_results[:3] if isinstance(res, dict)]
        if score_values:
            avg_score = sum(score_values) / len(score_values)
            confidence = max(0.0, min(0.99, (avg_score + 1.0) / 2.0))
        else:
            confidence = 0.4 if answer_text else 0.0

        result = {
            "answer": answer_text or "I couldn't derive an answer from the provided excerpts.",
            "query": request.query,
            "sources": sources,
            "sources_count": len(sources),
            "tokens_generated": tokens_generated,
            "task_id": task_id,
            "has_context": bool(transcript_results or memory_results),
            "mode": "gpu" if model_on_gpu else "cpu",
            "confidence": confidence,
        }

        if gpu_slot_acquired and not gpu_slot_released:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{GPU_COORDINATOR_URL}/gemma/release/{task_id}",
                    json={"result": result},
                    headers=get_service_headers(),
                )
            gpu_slot_released = True

        if model_on_gpu and used_gpu:
            move_model_to_cpu()

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TASK {task_id}] Error: {e}")
        import traceback

        traceback.print_exc()

        try:
            if gpu_slot_acquired and not gpu_slot_released:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        f"{GPU_COORDINATOR_URL}/gemma/release/{task_id}",
                        json={"result": {"error": str(e)}},
                        headers=get_service_headers(),
                    )
        except Exception as release_error:
            logger.error(f"[TASK {task_id}] Failed to release GPU after error: {release_error}")

        try:
            if model_on_gpu and used_gpu:
                move_model_to_cpu()
        except Exception as cpu_error:
            logger.error(f"[TASK {task_id}] Failed to move model back to CPU after error: {cpu_error}")

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    stats = {
        "model_loaded": gemma_model is not None,
        "model_on_gpu": model_on_gpu,
        "gpu_layers": GEMMA_GPU_LAYERS,
        "context_size": GEMMA_CONTEXT_SIZE,
        "coordinator_url": GPU_COORDINATOR_URL,
    }

    # Get VRAM usage
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info = result.stdout.strip().split("\n")[0].split(",")
            stats["vram_used_mb"] = int(info[0].strip())
            stats["vram_total_mb"] = int(info[1].strip())
            stats["gpu_utilization"] = int(info[2].strip())
    except Exception:
        pass

    return stats


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
