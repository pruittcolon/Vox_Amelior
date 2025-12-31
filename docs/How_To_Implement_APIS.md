# How To Implement APIs

This guide outlines the standard process for implementing and exposing APIs in the Vox Amelior microservices architecture.

## Architecture Overview

The system follows a strict **Microservices** pattern:
1.  **Frontend**: Makes requests to the **API Gateway** only.
2.  **API Gateway**: Proxies requests to the appropriate backend service (ML Service, Gemma Service, etc.).
3.  **Backend Services**: Handle the actual business logic and return data.

**Crucial Rule**: The API Gateway should **NOT** contain business logic or run engines locally. It must only route requests.

## Implementation Steps

### 1. Backend Service (e.g., ML Service)

Define your endpoint in the appropriate router within `services/ml-service/src/routers/`.

**Example:**
file: `services/ml-service/src/routers/my_new_router.py`

```python
from fastapi import APIRouter
router = APIRouter(tags=["my_feature"])

@router.post("/my-feature/analyze")
async def analyze_data(data: dict):
    return {"result": "success"}
```

**Register the Router:**
Ensure your new router is included in `services/ml-service/src/main.py`:

```python
from .routers.my_new_router import router as my_router
app.include_router(my_router, prefix="/analytics") 
# Resulting URL: http://ml-service:8006/analytics/my-feature/analyze
```

### 2. API Gateway

The Gateway automatically proxies requests under `/analytics/` to the ML Service.
- **Path**: `services/api-gateway/src/routers/ml.py`
- **Mechanism**: The wildcard route `@router.post("/analytics/{path:path}")` handles forwarding.

**Verification:**
You generally do **NOT** need to add new code to the Gateway for standard ML endpoints, as the proxy covers it.
If you need a specialized route or validational middleware, add it to `services/api-gateway/src/routers/ml.py`.

### 3. Frontend

Make requests to the **Gateway** URL. The `API_BASE` is automatically set to the origin (Gateway).

**Example:**
file: `frontend/assets/js/nexus/core/api.js`

```javascript
// Correct: User standard /analytics/ path which Gateway proxies
const endpoint = `/analytics/my-feature/analyze`; 
const response = await fetchWithTimeout(`${API_BASE}${endpoint}`, {
    method: 'POST',
    body: JSON.stringify(data)
});
```

## Legacy Code Warning
- **Do NOT** import engine classes directly into the API Gateway.
- **Do NOT** use `run_standard_engine` in the Gateway.
- Always implement logic in the Microservice (ML Service).

## GPU Access
For services requiring GPU (Gemma, Transcription):
- Use the `Coordinator` via `GPUCoordinatorClient`.
- Ensure your service handles the `X-Service-Token` for updates.
