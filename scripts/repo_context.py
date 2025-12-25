
"""
Repository Context Definitions
This file contains manual architectural descriptions that are injected into REPO_MAP.md
to provide high-level context beyond what automated analysis can derive.
"""

# Guide specifically meant for AI Agents reading the map
AGENTIC_GUIDE = """
# 🤖 Agentic AI Implementation Manual

> **Welcome, System 2 Agent.**
> This document is your Source of Truth for implementing features in the Nemo Server.
> **DO NOT** invent new patterns. **DO NOT** create ad-hoc files. Follow these "System 2" protocols strictly.

---

## 🏗️ Core Architecture (The "Hub-and-Spoke")

The system uses a **Centralized API Gateway** pattern.
- **Entry Point**: `services/api-gateway` (Port 8000).
- **Spokes**: `gemma-service`, `transcription-service`, `rag-service`, etc.
- **Protocol**: HTTP/REST over a `bridge` network.
- **Internal DNS**: Services address each other by container name (e.g., `http://gemma-service:8001`).

### 🔑 Critical Security Rules
1.  **Service-to-Service Auth**:
    - **NEVER** make an internal HTTP request without an `X-Service-Token` header.
    - **Use**: `shared.security.service_auth` to generate tokens.
    - **Verify**: Downstream services MUST use `ServiceAuthMiddleware` to validate tokens.
2.  **Secret Management**:
    - **NEVER** hardcode secrets.
    - **Read**: Use `services/shared/security/secrets.py:get_secret("secret_name")`.
    - **Mount**: Ensure `docker-compose.yml` mounts the secret file to `/run/secrets/`.

---

## 🛠️ Implementation Patterns

### Pattern 1: Creates a New AI Microservice (e.g., "Image Gen Service")

**Goal**: Add a new service that does heavy compute (GPU/CPU).

**Step 1: Create Directory Structure**
```
services/image-gen-service/
├── Dockerfile          # FROM python:3.10-slim
├── requirements.txt    # fastaip, uvicorn, torch, shared dependencies
└── src/
    ├── __init__.py
    └── main.py         # The FastAPI entry point
```

**Step 2: Implement `src/main.py`**
- **Must Have**:
    - `lifespan` context manager for startup/shutdown.
    - `ServiceAuthMiddleware` (copied/imported from shared).
    - `/health` endpoint (no auth required).
    - GPU Coordination (if using GPU).

**Step 3: Update `docker-compose.yml`**
- Register service under `services:`.
- Mount `shared` as read-only: `- ./shared:/app/shared:ro`.
- Mount secrets: `- jwt_secret`.
- Define network: `- nemo_network`.

---

### Pattern 2: Expose Feature via API Gateway (The "Router" Pattern)

**Goal**: Expose an endpoint from a microservice to the Frontend.

**Step 1: Create the Router**
File: `services/api-gateway/src/routers/image_gen.py`
```python
from fastapi import APIRouter, Depends
from src.auth.permissions import require_auth
from src.main import proxy_request

router = APIRouter(tags=["images"])
SERVICE_URL = "http://image-gen-service:80xx"

@router.post("/api/images/generate")
async def generate_image(payload: dict, session=Depends(require_auth)):
    return await proxy_request(f"{SERVICE_URL}/generate", "POST", json=payload)
```

**Step 2: Register in Gateway**
File: `services/api-gateway/src/main.py`
```python
from src.routers import image_gen
app.include_router(image_gen.router)
```

**Step 3: Add Environment Variable**
File: `docker-compose.yml` (under api-gateway)
```yaml
environment:
  IMAGE_GEN_URL: http://image-gen-service:80xx
```

---

### Pattern 3: GPU Coordination (The "Semaphore" Pattern)

**Goal**: Prevent Out-Of-Memory (OOM) crashes when multiple services use the single NVIDIA GPU.

**Protocol**:
1.  **Request Lock**: Call `gpu-coordinator` (`http://gpu-coordinator:8002/request`).
2.  **Wait**: The coordinator signals `transcription-service` to PAUSE (unload model).
3.  **Grant**: You receive a "Lock Acquired" response.
4.  **Execute**: Run your inference (Gemma, Stable Diffusion, etc.).
5.  **Release**: Call `gpu-coordinator` (`/release`). Coordinator signals Transcription to RESUME.

**Code Example (Python)**:
```python
# In your service's inference method
from shared.gpu import acquire_gpu_lock, release_gpu_lock

async def generate():
    await acquire_gpu_lock(service_id="my-service", relative_priority=1)
    try:
        # Run Heavy GPU Inference
        result = model.generate(...)
    finally:
        await release_gpu_lock(service_id="my-service")
```

---

### Pattern 4: Transcription & Audio Pipeline

**Architecture**:
Audio -> Split (VAD) -> ASR (Parakeet) -> Diarization (TitaNet) -> RAG Indexing

**Key Files**:
- `services/transcription-service/src/main.py`: The orchestrator.
- `services/transcription-service/src/parakeet_pipeline.py`: The ASR logic.
- `services/transcription-service/src/streaming.py`: WebSocket handler for real-time.

**Adding a New Step (e.g., Translation)**:
1.  Intercept the `segments` list in `_transcribe_with_parakeet`.
2.  Apply your transformation (e.g., `translate_text(seg['text'])`).
3.  Store metadata in the `audio_metrics` dict attached to the segment.

---

### Pattern 5: Frontend Implementation (No-Build)

**Goal**: Add a UI component for your new feature.

**Location**: `frontend/` (Served by Nginx/Gateway)

**Rules**:
1.  **NO Build Steps**: Use native ES Modules (`import ... from './utils.js'`).
2.  **Styling**: Use `style.css` variables (Glassmorphism).
3.  **API Calls**: Use relative paths (`/api/...`) which Gateway proxies.

---
"""

SECTION_CONTEXT = {
    "services": """
> **Architecture Overview**
> The backend follows a **Microservices Architecture** where each service has a distinct responsibility.
> Services communicate via HTTP (REST) and share common security/utility code from the `shared/` directory.
>
> **Key Patterns:**
> - **Service-to-Service Auth**: All internal communication is secured via JWTs signed with unique per-service secrets (see `shared/security/service_auth.py`).
> - **Fail-Closed Security**: Services refuse to start if security configurations (auth, secrets) are invalid.
> - **Centralized Gateway**: The `api-gateway` handles all external traffic, authentication (User sessions), and routing.
""",

    "services-api-gateway": """
> **API Gateway & Ingress (ENTRY POINT)**
> The `api-gateway` is the entry point for the entire Nemo Server. It is a FastAPI application that:
> 1.  **Routes Requests**: Proxies traffic to downstream services (`transcription`, `emotion`, `gemma`, etc.).
> 2.  **Manages Authentication**: Handles user login/registration and issues encrypted Session Cookies (AES-256).
> 3.  **Enforces Security**: Implements CSRF protection, Rate Limiting, and Enterprise Audit logging.
> 4.  **Static Files**: Serves the Frontend assets (if running in monolith mode).
>
> **Development Note**: When adding new capabilities, you usually need to:
> 1.  Add the logic in a downstream service (e.g. `gemma-service`).
> 2.  Expose it via a new router in `services/api-gateway/src/routers/`.
> 3.  Register the router in `services/api-gateway/src/main.py`.
""",

    "services-gemma-service": """
> **LLM & Intelligence**
> The `gemma-service` hosts the local LLM (Gemma 2 9B/3-4B).
> - **GPU Coordination**: dynamically requests GPU slots from a `gpu-coordinator` (if present) to share resources with transcription.
> - **Inference**: Uses `llama-cpp-python` for efficient local inference (GGUF).
> - **RAG Integration**: Can query `rag-service` for context augmentation.
""",

    "services-ml-service": """
> **Data Analytics & ML (System 2)**
> A "Universal ML Agent" service for structured data analysis.
> - **Ingestion**: Supports CSV, Excel, SQL (Postgres, SQLite), NoSQL (Mongo, Redis), and Cloud sources.
> - **Engines**: Contains specialized engines for Semantics, Analytics, and Vector operations.
> - **Purpose**: To provide deep insights and charts based on uploaded or connected data.
""",

    "services-emotion-service": """
> **Emotion Classification**
> A dedicated lightweight service for text-based emotion analysis.
> - **Model**: Uses `distilroberta-base` (via `shared/emotion_analyzer.py`).
> - **Function**: Classifies transcripts into 7 emotions (joy, sadness, anger, etc.) for metadata enrichment.
> - **Inputs**: Raw text strings.
> - **Outputs**: Probabilistic distribution of emotions (e.g., `{'joy': 0.8, 'neutral': 0.2}`).
""",

    "services-rag-service": """
> **Memory & Semantic Search**
> The core memory system of the server.
> - **Vector DB**: Uses FAISS + Sentence Transformers (`all-MiniLM-L6-v2`) for semantic search.
> - **Metadata Store**: SQLite database (`rag.db`) storing full transcripts, segments, and memories.
> - **Encryption**: Supports SQLCipher for encryption-at-rest.
> - **Features**: Enables asking questions about past conversations, filtering by speaker/time/emotion.
""",

    "shared": """
> **Shared Library Layer**
> This directory contains code shared across all Python microservices to ensure consistency and DRY principles.
>
> **Key Modules:**
> - `auth/`: Common authentication logic (User models, Session models).
> - `security/`: Critical security primitives (ServiceAuth JWTs, Secret Management).
> - `logging/`: Structured JSON logging configuration for ELK/Datadog integration.
> - `telemetry/`: Prometheus metrics and OpenTelemetry tracing setup.
""",
    
    "services-transcription-service": """
> **ASR & Audio Processing**
> The core hearing module of the system.
> - **ASR Engine**: Uses **NVIDIA NeMo Parakeet** (RNNT) for fast, accurate streaming transcription.
> - **Streaming**: Implements a WebSocket architecture with sliding window buffering for real-time results.
> - **Diarization**: Uses **TitaNet** for speaker embeddings and clustering to identify "who said what".
> - **VAD**: Voice Activity Detection to filter silence and reduce processing load.
""",

    "services-n8n-service": """
> **Automation Bridge**
> Connects the AI system to the outside world via n8n workflows.
> - **Voice Commands**: Matches transcript patterns (regex) to trigger smart home actions.
> - **Emotion Alerts**: Tracks consecutive emotion states to fire webhooks (e.g., "User staying sad").
> - **Webhooks**: Sends structured payloads to configured n8n endpoints.
""",

    "services-queue-service": """
> **GPU Coordinator**
> Custom distributed resource scheduler to manage VRAM contention.
> - **Problem**: Prevents OOM crashes when running LLM (16GB) and ASR (4GB) on a single 24GB GPU.
> - **Mechanism**: Uses Redis-based distributed locks with priority (Chat > Voice Command > Background).
> - **Protocol**: Signals transcription service to "Pause" (offload model) before granting lock to LLM.
""",

    "services-mobile-app": """
> **Mobile Client (Flutter)**
> The primary user interface for the smart glasses/wearable.
> - **Framework**: Dart/Flutter.
> - **Features**: Real-time audio streaming, live transcript view, authentication.
> - **Connectivity**: Uses secure WebSockets (WSS) to communicate with the API Gateway.
> - **Modes**: Includes "Google A.I. Mode" for direct Android Intent commands and "Chat Mode" for LLM interaction.
""",

    "docker": """
> **Containerization & Deployment**
> Infrastructure as Code for Docker Swarm / Compose deployments.
> - **Compose**: `docker-compose.yml` orchestrates the 11-service mesh.
> - **Images**: Custom Dockerfiles for each Python/Node.js service (`Dockerfile.transcription`, `Dockerfile.gemma`, etc.).
> - **Nginx**: Gateway configuration and SSL termination.
""",

    "k8s": """
> **Kubernetes Manifests**
> Configuration for scalable deployment on K8s (Kind/EKS).
> - **Deployments**: YAML definitions for all microservices.
> - **Ingress**: Configuration for external access.
> - **GPU Support**: specialized configurations for NVIDIA GPU pass-through.
""",

    "scripts": """
> **Utilities & CLI Tools**
> Helper scripts for maintenance, testing, and security.
> - **CLI Suite**: `nemo_cli_suite.py` for managing the system (start/stop/test).
> - **Testing**: `nemo_test_suite.py` contains rigorous integration and security tests.
> - **Security**: `verify_security.py` and `security_hardening.py` for audit compliance.
> - **Documentation**: `repo_mapper.py` (this tool) for generating architectural maps.
""",

    "services-fiserv-service": """
> **Banking Automation (Fiserv)**
> Automation layer for Fiserv Banking Hub DNA platform.
> - **Integration**: connects to 9 banking providers (DNA, Premier, etc.).
> - **Capabilities**: Account lookup, transaction analysis, anomaly detection.
> - **Security**: OAuth-based token refresh and S2S validation.
""",

    "services-insights-service": """
> **Visualization & Analytics**
> Bridges raw data (ML Service/RAG) to human understanding.
> - **Outputs**: Interactive Plotly charts, Dashboards, PDF/HTML reports.
> - **Flow**: ML Service -> Insights Service -> Frontend (JSON Config).
""",

    "frontend": """
> **Frontend (No-Build SPA)**
> High-performance, vanilla ES Modules application.
> - **Architecture**: Zero-build, standard ESM, served directly by Nginx/Gateway.
> - **Design**: Glassmorphism UI system with GPU-accelerated CSS animations.
> - **Real-time**: Uses SSE and WebSockets for live transcript and health feeds.
""",
}
