# Nexus Page and ML Service Analysis Report

## Summary

Nexus.html is the main data analysis page that runs **22 ML engines** sequentially on uploaded data. The page was freezing due to missing request timeouts and potential ML service bottlenecks.

---

## Architecture Overview

```
nexus.html
    |
    +-> assets/js/nexus/pages/main.js (537 lines)
    |       |
    |       +-> core/api.js - uploadFile(), runEngine(), askGemma()
    |       +-> engines/engine-runner.js - Sequential engine execution
    |       +-> engines/engine-definitions.js - 22 engine definitions
    |
    +-> /api/analytics/run-engine/{engineName} (API Gateway)
            |
            +-> ML Service (2574 lines)
                    |
                    +-> 22 analysis engines
                    +-> GPU client (optional, currently disabled)
```

---

## Root Causes of Freezing

### 1. No Request Timeouts in Frontend

**Location**: `frontend/assets/js/nexus/core/api.js`

```javascript
// Lines 72-84: runEngine() has no timeout
const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: getAuthHeaders(),
    credentials: 'include',
    body: JSON.stringify({...})
});  // <- NO TIMEOUT - will hang forever if server is slow
```

**Impact**: If any engine takes too long (e.g., waiting for GPU or processing large data), the UI freezes indefinitely.

### 2. Sequential Execution with 22 Engines

**Location**: `frontend/assets/js/nexus/engines/engine-runner.js`

```javascript
// Lines 155-239: Sequential loop
for (let i = startIndex; i < ALL_ENGINES.length; i++) {
    const data = await runEngine(engine.name, {...});  // Blocking call
    const summary = await getGemmaSummary(...);        // Another blocking call
}
```

**Impact**: 22 engines x 2 API calls each = 44 sequential requests. If each takes 5-10 seconds, total time is 4-8+ minutes of blocking.

### 3. Gemma Summary Per Engine

**Location**: `frontend/assets/js/nexus/engines/engine-runner.js` (lines 192-193)

```javascript
log(`Getting Gemma summary for ${engine.display}...`, 'info');
const summary = await getGemmaSummary(engine.name, engine.display, data);
```

This calls `/api/public/chat` **22 times** - one for each engine result. Each request:
1. Requires GPU access (if Gemma is on GPU)
2. Waits for inference (1-5 seconds per request)
3. Has no timeout

### 4. ML Service GPU Dependency (Optional)

The ML service has a GPU client (`core/gpu_client.py`) that uses the GPU coordinator. Currently disabled (`GPU_ENABLED=false` in docker-compose), but when enabled:
- ML engines compete with Gemma for GPU
- 22 engines could conflict with Gemma summary requests

---

## The 22 Engines

| Category | Engines |
|----------|---------|
| ML & Analytics (7) | Titan AutoML, K-Means, HDBSCAN, SOM, Isolation Forest, DTW, GARCH |
| Financial (12) | Revenue Forecast, ROI, LTV, Cost Optimizer, Market Analysis, Churn, Cohort, etc. |
| Advanced (3) | RAG Evaluation, Chaos Analysis, Oracle Causality |

---

## GPU Coordinator Integration Status

### ML Service GPU Client

**File**: `services/ml-service/src/core/gpu_client.py` (432 lines)

```python
# Uses legacy endpoint (still supported)
response = await client.post(
    f"{self.coordinator_url}/gemma/request",  # Line 237
    json={...}
)
```

**Status**: Uses legacy `/gemma/request` endpoint which is PRESERVED in the refactored coordinator.

### Gemma Service

**Status**: Uses `/gemma/request` - PRESERVED and working.

### Transcription Service

**Status**: Uses Redis pub/sub channels (`transcription_pause`, etc.) - PRESERVED in shared/gpu/listener.py.

---

## Implemented Fixes

### Fix 1: Request Timeouts (IMPLEMENTED)

Updated `api.js` with `fetchWithTimeout()` wrapper using AbortController:

- File uploads: 2 minute timeout
- Engine execution: 60 second timeout
- Gemma chat: 30 second timeout

```javascript
async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    // ... handles timeout gracefully
}
```

### Fix 2: GPU Health Check (IMPLEMENTED)

Added `checkGpuHealth()` and `checkGpuCoordinatorStatus()` functions that:

1. Check GPU VRAM availability before analysis starts
2. Check if GPU is currently owned by another service
3. Warn user if VRAM is critically low (<1GB)
4. Give user option to cancel or continue on CPU

### Fix 3: Pre-Analysis Warning (IMPLEMENTED)

`runFullAnalysis()` now runs GPU checks before starting:

```javascript
const [gpuHealth, coordinatorStatus] = await Promise.all([
    checkGpuHealth(),
    checkGpuCoordinatorStatus()
]);

if (gpuHealth.warning && gpuHealth.vramFreeGb < 1.0) {
    const proceed = confirm(`GPU Warning: ${gpuHealth.warning}...`);
    if (!proceed) return;
}
```

---

## Still Recommended (Future Improvements)

### Batch Gemma Summaries

Instead of 22 individual Gemma calls, batch them:

```javascript
// After all engines complete, get summaries in one call
const allResults = Object.values(session.results);
const batchPrompt = `Summarize these ${allResults.length} analyses...`;
const batchSummary = await askGemma(batchPrompt, { maxTokens: 1500 });
```

### Optional Parallel Execution

Run independent engines in parallel (with concurrency limit):

```javascript
const concurrencyLimit = 3;
const chunks = chunkArray(ALL_ENGINES, concurrencyLimit);
for (const chunk of chunks) {
    await Promise.all(chunk.map(engine => runEngine(engine.name)));
}
```

---

## Current System Health

| Component | Status |
|-----------|--------|
| GPU Coordinator | Working - new endpoints added, legacy preserved |
| ML Service | Working - 22 engines functional |
| Gemma Service | Working - GPU acquisition functional |
| Transcription | Working - pause/resume functional |
| Frontend API | Needs timeout fixes |

---

## Testing Recommendations

1. **Timeout Test**: Run analysis with one slow engine and verify timeout works
2. **Pause/Resume Test**: Use stop button mid-analysis and verify resume works
3. **GPU Monitoring**: Watch `nvidia-smi` during analysis to verify no OOM
4. **Error Handling**: Test with malformed data to verify graceful error handling
