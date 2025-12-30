# ML Service Engines - Integration & API Map

This document serves as the central reference for the ML Service engines, their API endpoints, visualization modules, and common errors.

## 🗺️ API & Visualization Map

This map defines how 22+ engines are routed from API to Frontend Visualization. The architecture is designed for **Simplicity**: adding a new engine requires just 1 API endpoint and 1 visualization module configuration.

| Engine ID | API Endpoint (`/api/analytics/run-engine/...`) | Visualization Module (`visualizations/engines/...`) | Key Output Data |
|-----------|------------------------------------------------|-----------------------------------------------------|-----------------|
| **titan** | `titan` | `ml/titan.js` | `feature_importance` (Waterfall), `stable_features` |
| **clustering** | `clustering` | `ml/clustering.js` | `pca_3d.points` (Scatter3D), `cluster_profiles` |
| **anomaly** | `anomaly` | `ml/anomaly.js` | `anomalies` (Scatter), `reconstruction_error` |
| **statistical** | `statistical` | `ml/statistical.js` | `correlations` (Heatmap), `distributions` |
| **trend** | `trend` | `ml/trend.js` | `trend_line`, `seasonality` |
| **predictive** | `predictive` | `ml/predictive.js` | `predictions`, `confidence_intervals` |
| **graphs** | `graphs` | `ml/graphs.js` | `nodes`, `edges` (Force Directed) |
| *(Financial)* | `cost`, `roi`, `cash_flow`, ... | `financial/*.js` | `forecast_data`, `financial_metrics` |
| *(Advanced)* | `rag_evaluation`, `chaos`, `oracle` | `advanced/*.js` | `rag_scores`, `causal_graph` |

---

## 🚀 How to Add a New Engine (Simplicity Guide)

The system uses a **Modular Microservice Pattern**. To add a new engine:

1.  **Backend (Python)**: Add function to `services/ml-service/src/engines/`.
2.  **Router (Python)**: Add route to `services/ml-service/src/routers/premium_engines.py`.
    ```python
    @router.post("/run-engine/my_new_engine")
    def run_new_engine(...): ...
    ```
3.  **Frontend (JS)**:
    *   Create `frontend/assets/js/nexus/visualizations/engines/my_new_engine.js`.
    *   Export `buildSection(data)` and `render(data)`.
    *   Register in `frontend/assets/js/nexus/visualizations/index.js`.

**That's it!** `nexus.html` automatically detects the engine name and loads your visualization. No core UI code changes needed.

---

## 🐛 Error Log & Fixes

### Error 1: HTTP 404 - File Not Found
**Cause:** Uploads are prefixed with UUIDs (e.g., `d01a..._file.csv`).
**Fix:** Always fetch the *full* filename from `/api/databases` before running an engine.

### Error 2: HTTP 500 - AttributeError `icon`
**Cause:** `EngineInfo` object missing `icon` attribute.
**Fix:** Used `getattr(engine_info, 'icon', '')` for safe access.

### Error 3: HTTP 401 - Missing Service Token
**Cause:** Direct calls to ML service port 8006 bypass Auth.
**Fix:** ALWAYS route via API Gateway (Port 8000).

### Error 4: Visualization Not Rendering (Single Engine Panel)
**Cause:** `nexus.html` test panel used hardcoded HTML generation instead of the modular `NexusViz` system.
**Fix:** Refactored `createEngineResultCard` to dynamically call `window.NexusViz.buildVizSection(engine, data)`.

---
