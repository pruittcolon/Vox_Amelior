# Security Issue Report: CORS & Gateway Routing

## 🚨 The Problem

When implementing the Fiserv Banking integration, multiple security violations occurred:

### Error 1: CORS Blocked
```
Cross-Origin Request Blocked: The Same Origin Policy disallows reading the remote resource 
at http://localhost:8015/api/v1/usage. (Reason: CORS request did not succeed).
```

### Error 2: 404 Not Found
```
GET http://localhost:8000/fiserv/api/v1/usage [HTTP/1.1 404 Not Found]
```

### Error 3: 403 Forbidden
```
POST http://localhost:8000/api/v1/banking/analyze/12345 [HTTP/1.1 403 Forbidden]
```

---

## 🔍 Root Cause Analysis

### Why CORS Happened (Error 1)

The original `banking.html` had:
```javascript
const API_BASE = 'http://localhost:8015';  // Direct to fiserv-service
```

This caused the browser to make cross-origin requests:
- Browser page served from: `http://localhost:8000` (nginx/Gateway)
- API request target: `http://localhost:8015` (Fiserv service)
- **Different ports = different origins = CORS blocked**

### Why 404 Happened (Error 2)

After fixing to use `/fiserv/*` routing:
1. The Gateway's Fiserv router (`fiserv.py`) was created
2. BUT the Gateway container was running OLD code (Docker not restarted)
3. The route didn't exist in the running container

### Why 403 Happened (Error 3)

The `/api/v1/banking/analyze` endpoint requires authentication:
```python
session: Session = Depends(require_auth)
```
User was not logged in, so the request was rejected.

### Why Fiserv Service Wasn't Available

The Fiserv service was running in a **separate docker-compose** (`services/fiserv-service/docker-compose.yml`) on port 8015, completely isolated from the main stack. This violates several security principles:

1. **Not on internal network** - Fiserv was exposed on host
2. **No Gateway routing** - Browser accessed it directly
3. **No authentication** - Anyone could call Fiserv APIs
4. **No audit logging** - Requests bypassed Gateway

---

## The Complete Fix

### Fix 1: Update `banking.html` to use Gateway routing

```diff
- const API_BASE = 'http://localhost:8015';
+ // SECURITY: All requests go through Gateway
+ const API_BASE = '/fiserv';
```

**File**: `frontend/banking.html` (line ~1005)

### Fix 2: Create Gateway Fiserv Proxy Router

**New file**: `services/api-gateway/src/routers/fiserv.py`

This router:
- Receives requests at `/fiserv/*`
- Validates authentication (for sensitive endpoints)
- Proxies to `fiserv-service:8015` on internal Docker network

### Fix 3: Register Router in Gateway

**File**: `services/api-gateway/src/main.py`

```python
from src.routers.fiserv import router as fiserv_router
app.include_router(fiserv_router)
```

### Fix 4: Add Fiserv to Main Docker Compose

**File**: `docker/docker-compose.yml`

Added:
1. `FISERV_SERVICE_URL` environment variable for Gateway
2. `fiserv-service` container definition on internal network

```yaml
# In api-gateway environment:
FISERV_SERVICE_URL: http://fiserv-service:8015

# New service definition:
fiserv-service:
  build:
    context: ..
    dockerfile: services/fiserv-service/Dockerfile
  container_name: refactored_fiserv
  environment:
    FISERV_MOCK_MODE: "true"
  volumes:
    - ../services/fiserv-service/src:/app/src:ro
  networks:
    - nemo_network
```

---

## 🔄 How to Apply the Fix

```bash
cd /home/pruittcolon/Desktop/Nemo_Server/docker

# Stop all containers
docker compose down

# Rebuild with new Fiserv service
docker compose build fiserv-service api-gateway

# Start stack
docker compose up -d

# Verify Fiserv is running
docker compose ps | grep fiserv

# Test Gateway routing
curl http://localhost:8000/fiserv/api/v1/usage
```

---

## 🏗️ Correct Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BROWSER (port 8000)                       │
└───────────────────────────┬─────────────────────────────────┘
                            │ 
                            ▼ /fiserv/*, /api/v1/banking/*
┌───────────────────────────────────────────────────────────┐
│                    NGINX (TLS Termination)                 │
│                    Ports 80, 443                           │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                    API GATEWAY                             │
│  • CORS handling        • Session validation              │
│  • Rate limiting        • Audit logging                   │
│  • Fiserv proxy router  • Banking analysis router         │
└─────────┬─────────────────────────────┬───────────────────┘
          │                             │
          ▼                             ▼
┌─────────────────┐           ┌─────────────────┐
│ fiserv-service  │           │   ml-service    │
│   (port 8015)   │           │   (port 8006)   │
│  INTERNAL ONLY  │           │  INTERNAL ONLY  │
└─────────────────┘           └─────────────────┘
```

---

## 📋 Checklist for Future Service Integrations

When adding ANY new backend service:

- [ ] **Add to main `docker-compose.yml`** - Don't use separate compose files for services that frontend needs
- [ ] **Add `*_SERVICE_URL` env var** to Gateway
- [ ] **Create proxy router** in `api-gateway/src/routers/`
- [ ] **Register router** in `api-gateway/src/main.py`
- [ ] **Use relative paths** in frontend (e.g., `/fiserv/*` not `http://localhost:8015`)
- [ ] **Rebuild & restart stack** after changes
- [ ] **Test from browser** - CORS errors indicate violations

---

## 📁 Files Modified

| File | Change |
|------|--------|
| `frontend/banking.html` | `API_BASE` → `/fiserv` |
| `services/api-gateway/src/routers/fiserv.py` | **NEW** proxy router |
| `services/api-gateway/src/main.py` | Register fiserv_router |
| `docker/docker-compose.yml` | Add fiserv-service + env var |

---

## ⚠️ Lessons Learned

1. **Internal services must be on Docker network** - Never expose on host ports for browser access
2. **All browser → backend traffic MUST go through Gateway** - Authentication, logging, CORS
3. **Docker containers run old code until rebuilt/restarted** - Changes to mounted volumes may not auto-reload
4. **Test in browser, not just curl** - CORS only applies to browsers
