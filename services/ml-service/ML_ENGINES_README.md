# ML Service - Engine Implementation Guide

> **Purpose:** This guide provides comprehensive documentation for the ML Service's analytics engines, enabling seamless agentic implementation and maintenance.

## Table of Contents

- [Nexus Frontend Integration](#nexus-frontend-integration)
- [Architecture Overview](#architecture-overview)
- [The 22 Nexus Engines](#the-22-nexus-engines)
- [Engine Types](#engine-types)
- [API Endpoints](#api-endpoints)
- [Adding a New Engine](#adding-a-new-engine)
- [Frontend Integration](#frontend-integration)
- [Security Configuration](#security-configuration)
- [Troubleshooting](#troubleshooting)

---

## Nexus Frontend Integration

The Nexus AI platform (`nexus.html`) provides a unified interface for running all 22 ML engines on uploaded datasets. This section documents the complete data flow.

### End-to-End Flow Diagram

```
User drops CSV/JSON file
        |
        v
[Frontend: pages/main.js]
        |
        v
POST /upload --> Backend stores file, returns {filename, columns, row_count}
        |
        v
[State: state.js] --> setUploadState(filename, columns)
        |
        v
User clicks "Start Full Analysis"
        |
        v
[Engine Runner: engine-runner.js] --> initSession(), loops through ALL_ENGINES
        |
        +---> For each of 22 engines:
        |         |
        |         v
        |     POST /analytics/run-engine/{engine_name}
        |         |
        |         v
        |     [Backend: premium_engines.py:run_single_engine()]
        |         |
        |         +--> Check ENGINE_REGISTRY
        |         |         |
        |         |         v
        |         |    Instantiate engine class
        |         |         |
        |         |         v
        |         |    engine.analyze(df, config)
        |         |         |
        |         v         v
        |     Return JSON result
        |         |
        |         v
        |     [Frontend: api.js] --> getGemmaSummary() for AI explanation
        |         |
        |         v
        |     [State: state.js] --> recordEngineResult(engine_name, result)
        |         |
        |         v
        |     [UI: engine-results.js] --> displayEngineResults() with visualization
        |
        v
All 22 engines complete --> completeSession()
```

### Frontend Module Architecture

| Module | Path | Responsibility |
|--------|------|----------------|
| **Orchestrator** | `nexus/pages/main.js` | Wires upload, callbacks, triggers analysis |
| **Engine Runner** | `nexus/engines/engine-runner.js` | Sequential engine loop with pause/resume |
| **Engine Definitions** | `nexus/engines/engine-definitions.js` | Registry of all 22 engine names, icons, categories |
| **Engine Results** | `nexus/engines/engine-results.js` | Card creation, status updates, key finding extraction |
| **API Layer** | `nexus/core/api.js` | HTTP calls to `/upload`, `/run-engine`, `/chat` |
| **State Manager** | `nexus/core/state.js` | Session, upload, column selection persistence |
| **Visualizations** | `nexus/visualizations/index.js` | Aggregates 22 engine-specific visualization modules |
| **Dashboard** | `nexus/components/dashboard.js` | ECharts/Plotly real-time performance charts |

### Request Format (Frontend to Backend)

```javascript
// nexus/core/api.js:runEngine()
POST /analytics/run-engine/{engine_name}
Content-Type: application/json

{
    "filename": "uploaded_data.csv",
    "target_column": "revenue",      // Optional - Gemma AI recommends or user selects
    "config": {},                    // Engine-specific config overrides
    "use_vectorization": false       // Enable Gemma embeddings
}
```

### Response Format (Backend to Frontend)

```json
{
    "status": "success",
    "engine_name": "clustering",
    "engine_display_name": "Clustering Analysis",
    "_engine_metadata": {
        "source": "ENGINE_REGISTRY",
        "requested_engine": "clustering",
        "filename": "uploaded_data.csv"
    },
    "n_clusters": 5,
    "cluster_labels": [0, 1, 2, 0, 1, ...],
    "cluster_centers": [[...], [...], ...],
    "silhouette_score": 0.72,
    "insights": ["Cluster 0 has highest average revenue", ...]
}
```

### Pause/Resume Capability

The frontend supports pausing and resuming analysis mid-run:

```javascript
// Pause: saves currentEngineIndex to localStorage
stopAnalysis() --> setAnalysisStopped(true) + pauseSession() + saveSessionToStorage()

// Resume: loads session and continues from saved index
resumeAnalysis(savedSession) --> restoreSession() + runEngineLoop(currentEngineIndex)
```

Session data persisted in `localStorage` under key `nemo_analysis_session`.

---

## The 22 Nexus Engines

All 22 engines defined in the frontend have dedicated backend implementations in `ENGINE_REGISTRY`:

### ML and Analytics (7 Engines)

| Frontend Name | Backend Class | Category | Description |
|--------------|---------------|----------|-------------|
| `titan` | `TitanEngine` | AUTOML | Enterprise AutoML with stability selection |
| `predictive` | `PredictiveEngine` | PREDICTIVE | Time-series forecasting |
| `clustering` | `ClusteringEngine` | CLUSTERING | K-Means, DBSCAN, Hierarchical |
| `anomaly` | `AnomalyEngine` | ANOMALY | Multi-method outlier detection |
| `statistical` | `StatisticalEngine` | STATISTICAL | Comprehensive statistics and correlations |
| `trend` | `TrendEngine` | PREDICTIVE | Trend detection and seasonality |
| `graphs` | `UniversalGraphEngine` | GRAPH | Auto-generate visualizations |

### Financial Intelligence (12 Engines)

| Frontend Name | Backend Class | Category | Description |
|--------------|---------------|----------|-------------|
| `cost` | `CostOptimizationEngine` | FINANCIAL | Cost reduction analysis |
| `roi` | `ROIPredictionEngine` | FINANCIAL | ROI prediction |
| `spend_patterns` | `SpendPatternEngine` | FINANCIAL | Spending pattern analysis |
| `budget_variance` | `BudgetVarianceEngine` | FINANCIAL | Budget vs actual |
| `profit_margins` | `ProfitMarginEngine` | FINANCIAL | Profit margin analysis |
| `revenue_forecasting` | `RevenueForecastingEngine` | FINANCIAL | Revenue prediction |
| `customer_ltv` | `CustomerLTVEngine` | FINANCIAL | Customer lifetime value |
| `cash_flow` | `CashFlowEngine` | FINANCIAL | Cash flow analysis |
| `inventory_optimization` | `InventoryOptimizationEngine` | FINANCIAL | Inventory optimization |
| `pricing_strategy` | `PricingStrategyEngine` | FINANCIAL | Price optimization |
| `market_basket` | `MarketBasketAnalysisEngine` | FINANCIAL | Product associations |
| `resource_utilization` | `ResourceUtilizationEngine` | FINANCIAL | Resource efficiency |

### Advanced AI Lab (3 Engines)

| Frontend Name | Backend Class | Category | Description |
|--------------|---------------|----------|-------------|
| `rag_evaluation` | `RAGEvaluationEngine` | ADVANCED | RAG quality evaluation |
| `chaos` | `ChaosEngine` | ADVANCED | Non-linear relationship detection |
| `oracle` | `OracleEngine` | ADVANCED | Granger causality analysis |

### Engine Resolution Priority

The `/run-engine/{engine_name}` endpoint resolves engines in this order:

1. **ENGINE_REGISTRY** (primary) - Dedicated engine classes with proper schema
2. **EXTENDED_ENGINE_MAP** (fallback) - Maps to premium engines for compatibility

```python
# premium_engines.py:run_single_engine()
if engine_name_lower in ENGINE_REGISTRY:
    # Use dedicated engine from registry
    engine_class = ENGINE_REGISTRY[engine_name_lower].engine_class
    engine = engine_class()
    result = engine.analyze(df, config)
else:
    # Fallback to premium engine mapping
    mapped_engine = EXTENDED_ENGINE_MAP.get(engine_name_lower)
    engine = PREMIUM_ENGINES[mapped_engine]()
    result = engine.analyze(df, config)
```

---

The ML Service uses a **modular router architecture** with engines organized into logical groups:

```
ml-service/src/
├── main.py                    # FastAPI app, includes routers
├── routers/
│   ├── __init__.py           # Exports all routers
│   ├── core_analytics.py     # Core statistical engines
│   ├── premium_engines.py    # Premium flagship engines (Titan, etc.)
│   ├── financial_analytics.py # Financial analysis engines
│   ├── quick_analysis.py     # Fast analysis endpoints
│   └── analysis_history.py   # Session/run history management
├── engines/
│   ├── engine_registry.py    # Central engine registry
│   ├── titan_engine.py       # Titan AutoML engine
│   ├── chaos_engine.py       # Monte Carlo simulations
│   ├── chronos_engine.py     # Time-series forecasting
│   ├── scout_engine.py       # Feature exploration
│   ├── oracle_engine.py      # Ensemble predictions
│   └── ...                   # Other engine implementations
└── utils/
    └── analytics_utils.py    # Shared utilities
```

### Router Registration (main.py)

```python
# Routers are included with /analytics prefix
app.include_router(core_router, prefix="/analytics")
app.include_router(premium_router, prefix="/analytics")
app.include_router(financial_router, prefix="/analytics")
app.include_router(quick_router, prefix="/analytics")
app.include_router(history_router, prefix="/analytics")
```

---

## Engine Types

### Premium Engines (10 Flagship)

| Engine | Class | Description |
|--------|-------|-------------|
| `titan` | `TitanEngine` | Universal AutoML with Gemma ranking |
| `chaos` | `ChaosEngine` | Monte Carlo simulations |
| `scout` | `ScoutEngine` | Feature exploration and ranking |
| `oracle` | `OracleEngine` | Ensemble forecasting |
| `newton` | `NewtonEngine` | Gradient optimization |
| `flash` | `FlashEngine` | Quick-fit pattern detection |
| `mirror` | `MirrorEngine` | Synthetic data generation |
| `chronos` | `ChronosEngine` | Advanced time series |
| `deep_feature` | `DeepFeatureEngine` | Neural feature extraction |
| `galileo` | `GalileoEngine` | Observational ML insights |

### Engine Name Mapping

The `/run-engine/{engine_name}` endpoint maps frontend names to standard engines:

```python
EXTENDED_ENGINE_MAP = {
    # Core engines
    "titan": "titan",
    "predictive": "titan",      # Alias
    "clustering": "scout",
    "anomaly": "chaos",
    "statistical": "newton",
    "trend": "chronos",
    "graphs": "galileo",
    
    # Financial engines
    "cost": "flash",
    "roi": "flash",
    "spend_patterns": "chronos",
    "budget_variance": "newton",
    "profit_margins": "flash",
    "revenue_forecasting": "chronos",
    "customer_ltv": "oracle",
    "cash_flow": "chronos",
    
    # Operations engines
    "inventory_optimization": "flash",
    "pricing_strategy": "oracle",
    "market_basket": "galileo",
    "resource_utilization": "flash",
    
    # Advanced engines
    "rag_evaluation": "scout",
    "chaos": "chaos",
    "oracle": "oracle",
}
```

---

## API Endpoints

### Dynamic Engine Execution

```http
POST /analytics/run-engine/{engine_name}
Content-Type: application/json

{
  "filename": "uploaded_file.csv",
  "target_column": "target",       // Optional
  "config": {},                    // Optional engine config
  "use_vectorization": false       // Optional
}
```

### Premium Engine Execution

```http
POST /analytics/premium/{engine_name}
Content-Type: application/json

{
  "filename": "uploaded_file.csv",
  "target_column": "target",
  "config_overrides": {}
}
```

### List Available Engines

```http
GET /analytics/premium/engines
```

### History Management

```http
# Save a run
POST /analytics/history/save-run
{
  "session_id": "uuid",
  "engine_name": "titan",
  "filename": "file.csv",
  "result": {...},
  "target_column": "target"
}

# Get all sessions
GET /analytics/history/sessions

# Get session details
GET /analytics/history/sessions/{session_id}

# Get engine runs for a session
GET /analytics/history/sessions/{session_id}/engines/{engine_name}
```

---

## Adding a New Engine

### Step 1: Create Engine Class

Create a new file in `engines/`:

```python
# engines/my_new_engine.py

import logging
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)


class MyNewEngine:
    """
    MyNew Engine - Brief description of what it does.
    
    Features:
    - Feature 1
    - Feature 2
    """
    
    def __init__(self, gemma_client=None):
        """Initialize engine with optional Gemma client for AI features."""
        self.gemma_client = gemma_client
    
    def analyze(self, df: pd.DataFrame, config: dict[str, Any] = None) -> dict:
        """
        Synchronous analysis method.
        
        Args:
            df: Input DataFrame
            config: Configuration dict with keys:
                - target_column: str (optional)
                - use_vectorization: bool (optional)
                
        Returns:
            dict with analysis results
        """
        config = config or {}
        target_column = config.get("target_column")
        
        # Your analysis logic here
        results = {
            "status": "success",
            "engine": "my_new",
            "summary": "Analysis complete",
            "insights": [],
            "charts": [],
        }
        
        return results
    
    async def analyze_async(self, df: pd.DataFrame, config: dict[str, Any] = None) -> dict:
        """Async version for GPU-accelerated analysis."""
        # Implement async logic or call sync method
        return self.analyze(df, config)
    
    def run_premium(self, df: pd.DataFrame, config: dict[str, Any] = None) -> dict:
        """
        Premium analysis with enhanced output format.
        
        Returns standardized PremiumResult structure.
        """
        base_result = self.analyze(df, config)
        
        return {
            **base_result,
            "variants": [],           # Multiple analysis variants
            "feature_importance": {}, # Feature rankings
            "methodology": {},        # How analysis was done
            "holdout_validation": {}, # Validation metrics
        }
```

### Step 2: Register in Premium Engines Router

Edit `routers/premium_engines.py`:

```python
# Add import
from ..engines.my_new_engine import MyNewEngine

# Add to PREMIUM_ENGINES dict
PREMIUM_ENGINES = {
    # ... existing engines ...
    "my_new": MyNewEngine,
}

# Add to EXTENDED_ENGINE_MAP if needed
EXTENDED_ENGINE_MAP = {
    # ... existing mappings ...
    "my_new": "my_new",
    "my_alias": "my_new",  # Optional alias
}
```

### Step 3: Add to Engine List Endpoint

```python
@router.get("/premium/engines")
async def list_premium_engines():
    engine_info = {
        # ... existing engines ...
        "my_new": {
            "name": "MyNew Engine",
            "description": "Brief description",
            "icon": "🆕"
        },
    }
```

### Step 4: Update Frontend (if needed)

In `predictions.html`, add to the engines list:

```javascript
const ENGINES = [
    // ... existing engines ...
    { name: 'my_new', display: 'MyNew Engine', icon: '🆕', category: 'advanced' },
];
```

---

## Frontend Integration

### CSRF Token Handling

The frontend must include CSRF tokens for POST requests:

```javascript
// Get CSRF token from cookies
function getCsrfToken() {
    const cookies = document.cookie.split(';');
    for (let cookie of cookies) {
        const [name, value] = cookie.trim().split('=');
        if (name === 'ws_csrf') {
            return value;
        }
    }
    return null;
}

// Include in fetch requests
const response = await fetch('/analytics/run-engine/titan', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        ...(getCsrfToken() && { 'X-CSRF-Token': getCsrfToken() })
    },
    credentials: 'include',
    body: JSON.stringify({ filename, target_column, config })
});
```

### Engine Execution Pattern

```javascript
async function runEngine(engineName, filename, options = {}) {
    const csrfToken = getCsrfToken();
    
    const response = await fetch(`${API_BASE}/analytics/run-engine/${engineName}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...(csrfToken && { 'X-CSRF-Token': csrfToken })
        },
        credentials: 'include',
        body: JSON.stringify({
            filename: filename,
            target_column: options.targetColumn || null,
            config: options.config || null,
            use_vectorization: options.useVectorization || false
        })
    });
    
    if (!response.ok) {
        throw new Error(`Engine ${engineName} failed: ${response.status}`);
    }
    
    return await response.json();
}
```

---

## Security Configuration

### CSRF Exempt Paths (api-gateway/src/main.py)

```python
class CSRFMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.exempt_paths = [
            "/api/auth/login",
            "/api/auth/register",
            "/health",
            "/favicon.ico",
            "/upload",
            "/api/upload",
            "/api/public/chat",  # Public Gemma chat
        ]
        self.exempt_prefixes = [
            "/analytics/",      # ML analytics endpoints
            "/api/analytics/",  # ML analytics with /api prefix
            "/vectorize/",      # Vectorization endpoints
        ]
```

### Content Security Policy

The gateway sets CSP headers allowing required CDNs:

```python
"script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' "
"https://unpkg.com https://cdn.jsdelivr.net https://cdn.plot.ly; "
```

### Service-to-Service Authentication

The ML service requires JWT service tokens from the API Gateway:

```python
# API Gateway sends service token when proxying
headers = {"X-Service-Token": service_auth.generate_token()}
response = await httpx.post(f"{ML_SERVICE_URL}/analytics/{path}", headers=headers)
```

---

## Troubleshooting

### Common Issues

#### 404 Not Found on Engine Endpoints

**Cause:** Engine not registered in modular router (only in legacy `analytics_routes.py`)

**Fix:** Add engine to `PREMIUM_ENGINES` and `EXTENDED_ENGINE_MAP` in `routers/premium_engines.py`

#### 403 Forbidden on Analytics Requests

**Cause:** CSRF middleware blocking requests

**Fixes:**
1. Add path to `exempt_prefixes` in `api-gateway/src/main.py`
2. Ensure frontend sends `X-CSRF-Token` header

#### "Missing service token" Error

**Cause:** Direct request to ML service without going through API Gateway

**Fix:** Always route through API Gateway which adds service authentication

#### Engine Timeout

**Cause:** Heavy engines (mirror, titan, oracle) take >60s

**Fix:** These engines have 120s timeout. For larger datasets, consider sampling.

### Verification Commands

```bash
# Check ML service logs
docker logs refactored_ml_service --tail 50

# Check API Gateway logs
docker logs refactored_gateway --tail 50

# Test endpoint directly (will return "Missing service token")
docker exec refactored_ml_service curl -s \
  http://localhost:8006/analytics/run-engine/titan \
  -X POST -H "Content-Type: application/json" \
  -d '{"filename":"test.csv"}'

# Restart services after code changes
docker restart refactored_ml_service refactored_gateway
```

---

## File Locations Summary

| Component | Path |
|-----------|------|
| ML Service Main | `services/ml-service/src/main.py` |
| Premium Engines Router | `services/ml-service/src/routers/premium_engines.py` |
| History Router | `services/ml-service/src/routers/analysis_history.py` |
| Engine Implementations | `services/ml-service/src/engines/*.py` |
| API Gateway Main | `services/api-gateway/src/main.py` |
| ML Router (Gateway) | `services/api-gateway/src/routers/ml.py` |
| Predictions Frontend | `frontend/predictions.html` |
| Databases Frontend | `frontend/databases.html` |

---

## Quick Reference: Adding Engine Checklist

- [ ] Create engine class in `engines/my_engine.py`
- [ ] Implement `analyze()`, `analyze_async()`, `run_premium()` methods
- [ ] Import and add to `PREMIUM_ENGINES` in `routers/premium_engines.py`
- [ ] Add name mapping to `EXTENDED_ENGINE_MAP`
- [ ] Add to `list_premium_engines()` response
- [ ] (Optional) Add to frontend engine list in `predictions.html`
- [ ] Restart ML service: `docker restart refactored_ml_service`
- [ ] Test endpoint via frontend or curl

---

*Last updated: December 2024*
