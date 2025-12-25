# API Gateway Service

The central routing layer for the Nemo Server platform. Handles authentication, request routing, and serves the frontend application.

## Overview

The API Gateway acts as a reverse proxy and authentication layer between clients and backend microservices:

- **Authentication & Authorization**: JWT-based session management with encrypted SQLite storage
- **Request Routing**: Proxies requests to appropriate backend services (Gemma, ML, RAG, etc.)
- **Frontend Serving**: Hosts static HTML/JS frontend files at `/ui/`
- **Security**: Rate limiting, CORS, security headers, replay attack protection

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API Gateway (Port 8000)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  main.py (~500 lines)                                                       │
│  ├── Middleware (CORS, Rate Limiting, Security Headers)                     │
│  ├── Lifespan (Auth Init, Service Auth)                                     │
│  ├── proxy_request() helper                                                 │
│  └── Router Registration                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  src/routers/                                                               │
│  ├── auth.py            → User authentication & sessions                    │
│  ├── gemma.py           → LLM inference (Gemma AI)                          │
│  ├── rag.py             → Semantic search & memory                          │
│  ├── ml.py              → ML analytics & predictions                        │
│  ├── banking.py         → Banking dashboard & Fiserv integration            │
│  ├── fiserv.py          → Fiserv API proxy                                  │
│  ├── salesforce/        → Salesforce CRM integration (sub-package)          │
│  ├── enterprise.py      → Enterprise features                               │
│  ├── call_intelligence.py → AI-powered call analytics & insights           │
│  ├── call_lifecycle.py  → End-to-end call management & workflow            │
│  ├── transcription.py   → Audio transcription                               │
│  ├── transcripts.py     → Transcript queries                                │
│  ├── enrollment.py      → Speaker enrollment                                │
│  ├── email_analyzer.py  → Email analysis & AI response                      │
│  ├── analysis.py        → General analysis endpoints                        │
│  ├── websocket.py       → Real-time WebSocket connections                   │
│  └── health.py          → Health check endpoints                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Backend Microservices                              │
├───────────────┬───────────────┬───────────────┬───────────────┬─────────────┤
│ gemma-service │ ml-service    │ rag-service   │ emotion-svc   │ fiserv-svc  │
│ :8001         │ :8006         │ :8004         │ :8005         │ :8015       │
└───────────────┴───────────────┴───────────────┴───────────────┴─────────────┘
```

---

## Adding a New Service Integration

This section explains how to properly add a new backend service to the API Gateway.

### Step 1: Create the Router File

Create a new file in `src/routers/` following this template:

```python
"""
{ServiceName} Router - {Brief description}

Provides {list of features}.
"""
import logging
import os
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request

# Import auth dependencies (with fallback for testing)
try:
    from src.auth.permissions import require_auth, Session
except ImportError:
    def require_auth():
        return None
    Session = None

logger = logging.getLogger(__name__)

# ==============================================================================
# Router Configuration
# ==============================================================================
router = APIRouter(
    prefix="/api/v1/myservice",  # Optional: Add prefix for all routes
    tags=["myservice"]           # OpenAPI tag for grouping
)

# Service URL from environment
MY_SERVICE_URL = os.getenv("MY_SERVICE_URL", "http://my-service:8XXX")


# ==============================================================================
# Helper: Lazy import proxy_request to avoid circular imports
# ==============================================================================
def _get_proxy_request():
    """Lazy import to avoid circular dependencies."""
    from src.main import proxy_request
    return proxy_request


# ==============================================================================
# Endpoints
# ==============================================================================
@router.get("/status")
async def get_status(session: Session = Depends(require_auth)):
    """Check service status. Requires authentication."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{MY_SERVICE_URL}/health", "GET")


@router.post("/action")
async def perform_action(
    request: Dict[str, Any],
    session: Session = Depends(require_auth)
):
    """Perform an action. Requires authentication."""
    proxy_request = _get_proxy_request()
    return await proxy_request(
        f"{MY_SERVICE_URL}/action",
        "POST",
        json=request
    )


# Public endpoint (no auth required)
@router.get("/public-info")
async def public_info():
    """Public endpoint - no authentication required."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{MY_SERVICE_URL}/info", "GET")


logger.info("✅ MyService Router initialized")
```

### Step 2: Register the Router in `main.py`

Add the import and registration in `src/main.py`:

```python
# ============================================================================
# Router Imports & Registration (around line 380)
# ============================================================================

# ... existing imports ...

# My New Service router
try:
    from src.routers.myservice import router as myservice_router
    _myservice_available = True
except ImportError as e:
    myservice_router = None
    _myservice_available = False
    logger.warning(f"MyService router unavailable: {e}")

# ... further down, in router registration section (around line 470) ...

if _myservice_available:
    app.include_router(myservice_router)
    logger.info("✅ MyService Router mounted")
```

### Step 3: Add Environment Variable

Add the service URL to `docker-compose.yml`:

```yaml
api-gateway:
  environment:
    MY_SERVICE_URL: http://my-service:8XXX
```

### Step 4: Update Configuration (Optional)

If your service needs additional config, add to `src/config/settings.py`:

```python
MY_SERVICE_URL: str = "http://my-service:8XXX"
MY_SERVICE_TIMEOUT: int = 30
```

---

## Proxy Request Helper

The `proxy_request()` function in `main.py` handles all outgoing requests to microservices:

```python
await proxy_request(
    url: str,                         # Full URL to the backend endpoint
    method: str = "POST",             # HTTP method (GET, POST)
    json: Optional[Dict] = None,      # JSON body
    params: Optional[Dict] = None,    # Query parameters
    files: Optional[Dict] = None,     # File uploads
    extra_headers: Optional[Dict] = None  # Additional headers
)
```

**Features:**
- Automatically injects `X-Service-Token` JWT for inter-service authentication
- 120-second timeout (configurable)
- Proper error handling with HTTP status codes
- Returns parsed JSON response

**Example Usage:**
```python
# GET request
result = await proxy_request(f"{SERVICE_URL}/endpoint", "GET")

# POST with JSON body
result = await proxy_request(f"{SERVICE_URL}/analyze", "POST", json={"data": payload})

# File upload
result = await proxy_request(f"{SERVICE_URL}/upload", "POST", files={"file": (name, content, mime)})
```

---

## Authentication

### For Protected Endpoints

```python
from src.auth.permissions import require_auth, Session

@router.post("/protected")
async def protected_endpoint(session: Session = Depends(require_auth)):
    user_id = session.user_id  # Access authenticated user
    role = session.role        # 'admin' or 'user'
    return {"user": user_id}
```

### For Public Endpoints

Simply omit the `Depends(require_auth)`:

```python
@router.get("/public")
async def public_endpoint():
    return {"message": "Anyone can access this"}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMMA_URL` | `http://gemma-service:8001` | Gemma AI service URL |
| `RAG_URL` | `http://rag-service:8004` | RAG service URL |
| `ML_SERVICE_URL` | `http://ml-service:8006` | ML analytics service URL |
| `EMOTION_URL` | `http://emotion-service:8005` | Emotion analysis URL |
| `TRANSCRIPTION_URL` | `http://transcription-service:8003` | Transcription service URL |
| `FISERV_URL` | `http://fiserv-service:8015` | Fiserv banking service URL |
| `ALLOWED_ORIGINS` | `http://127.0.0.1,http://localhost` | CORS allowed origins |
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `RATE_LIMIT_DEFAULT` | `120` | Requests per minute per IP |
| `SECURE_MODE` | `true` | Block unsafe dev flags in production |

### Secrets

Stored in Docker secrets or `/run/secrets/`:
- `session_key` - 32-byte base64 encryption key for session storage
- `jwt_secret` - JWT signing key for service authentication
- `users_db_key` - Encryption key for users database

---

## Directory Structure

```
services/api-gateway/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/
│   ├── main.py              # Application entry point (~500 lines)
│   ├── config/
│   │   └── settings.py      # Centralized configuration
│   ├── auth/
│   │   ├── auth_manager.py  # User/session management
│   │   └── permissions.py   # require_auth dependency
│   ├── routers/             # Domain-specific routers
│   │   ├── __init__.py
│   │   ├── auth.py          # Authentication routes
│   │   ├── gemma.py         # Gemma AI routes
│   │   ├── ml.py            # ML analytics routes
│   │   ├── banking.py       # Banking dashboard
│   │   ├── fiserv.py        # Fiserv API proxy
│   │   ├── salesforce/      # Salesforce sub-package
│   │   └── ...              # Other routers
│   └── core/
│       └── middleware.py    # Custom middleware (optional)
├── shared/                   # Shared utilities (symlink to /shared)
└── instance/                 # Runtime data (users.db, etc.)
```

---

## Development

### Run Standalone (for testing)

```bash
cd services/api-gateway
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Run in Docker

```bash
docker compose up api-gateway
```

### Rebuild After Changes

```bash
docker compose build api-gateway
docker compose up -d api-gateway --force-recreate
```

---

## API Examples

### Login
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

### Authenticated Request
```bash
curl -X POST http://localhost:8000/api/gemma/chat \
  -H "Content-Type: application/json" \
  -H "Cookie: session_id=YOUR_SESSION_COOKIE" \
  -d '{"message": "Hello, Gemma!"}'
```

### File Upload
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data.csv"
```

---

## Security Features

1. **Password Hashing**: bcrypt with salt
2. **Database Encryption**: pysqlcipher3 for encrypted SQLite
3. **Session Expiration**: Configurable TTL for sessions
4. **Rate Limiting**: Per-IP request throttling
5. **Replay Protection**: Redis-backed JTI tracking for S2S calls
6. **Fail-Closed Startup**: Blocks startup without valid secrets when `SECURE_MODE=true`
7. **Security Headers**: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection
8. **CORS**: Configurable allowed origins
9. **Input Validation**: Pydantic models for request validation

---

## Monitoring

### Health Check
```bash
curl http://localhost:8000/health
# {"status": "healthy", "service": "api-gateway", "version": "2.0.0"}
```

### OpenAPI Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Logs
Structured JSON logs when `STRUCTURED_LOGGING=true`:
- Authentication events
- Request routing
- Service communication errors
- Performance metrics

---

## Troubleshooting

### Router Not Loading
Check startup logs for import errors:
```bash
docker logs api-gateway 2>&1 | grep -i "router\|import\|error"
```

### Service Connection Errors
Verify the backend service is healthy:
```bash
curl http://localhost:8000/health  # Gateway
curl http://localhost:8006/health  # ML Service (example)
```

### 401 Unauthorized
- Ensure you're passing the session cookie
- Check if the session has expired
- Verify the endpoint requires authentication

### 503 Service Unavailable
- Backend service may be down or unhealthy
- Check Docker container status: `docker ps`
- Check service logs: `docker logs <service-name>`
