# ML Service Engines - Integration & API Map

This document serves as the central reference for the ML Service engines, their API endpoints, visualization modules, and common errors.

## API & Visualization Map

This map defines how 22+ engines are routed from API to Frontend Visualization. The architecture is designed for **Simplicity**: adding a new engine requires just 1 API endpoint and 1 visualization module configuration.

| Engine ID | API Endpoint | Viz Module | Key Data Fields | Status |
|-----------|--------------|------------|-----------------|--------|
| **titan** | `/run-engine/titan` | `ml/titan.js` | `feature_importance`, `stable_features` | Verified |
| **clustering** | `/run-engine/clustering` | `ml/clustering.js` | `pca_3d.points`, `cluster_profiles` | Verified |
| **anomaly** | `/run-engine/anomaly` | `ml/anomaly.js` | `method_results.isolation_forest.scores` | Fixed |
| **statistical** | `/run-engine/statistical` | `ml/statistical.js` | `descriptive.numeric` | Fixed |
| **trend** | `/run-engine/trend` | `ml/trend.js` | `dates`, `values`, `trend_line` | Graceful |
| **predictive** | `/run-engine/predictive` | `ml/predictive.js` | `predictions`, `confidence_intervals` | Graceful |
| **graphs** | `/run-engine/graphs` | `ml/graphs.js` | `graphs[]` (31 items) | Verified |
| **cost** | `/run-engine/cost` | `financial/cost.js` | `cost_breakdown`, `pareto_data` | Verified |
| **roi** | `/run-engine/roi` | `financial/roi.js` | `roi_metrics`, `scenarios` | Graceful |
| **cash_flow** | `/run-engine/cash_flow` | `financial/cashflow.js` | `summary`, `flows`, `periods` | Verified |
| **budget_variance** | `/run-engine/budget_variance` | `financial/budget.js` | `variances`, `budget_data` | Ready |
| **profit_margins** | `/run-engine/profit_margins` | `financial/margin.js` | `margins`, `breakdown` | Ready |
| **revenue_forecasting** | `/run-engine/revenue_forecasting` | `financial/forecast.js` | `forecast`, `confidence` | Ready |
| **customer_ltv** | `/run-engine/customer_ltv` | `financial/ltv.js` | `segments`, `ltv_distribution` | Ready |
| **spend_patterns** | `/run-engine/spend_patterns` | `financial/spend.js` | `patterns`, `categories` | Ready |
| **inventory_optimization** | `/run-engine/inventory_optimization` | `financial/inventory.js` | `inventory_metrics` | Ready |
| **pricing_strategy** | `/run-engine/pricing_strategy` | `financial/pricing.js` | `price_tiers`, `elasticity` | Ready |
| **market_basket** | `/run-engine/market_basket` | `financial/basket.js` | `associations`, `rules` | Ready |
| **resource_utilization** | `/run-engine/resource_utilization` | `financial/resource.js` | `utilization_data` | Ready |
| **rag_evaluation** | `/run-engine/rag_evaluation` | `advanced/rag.js` | `rag_scores` | Ready |
| **chaos** | `/run-engine/chaos` | `advanced/chaos.js` | `experiments`, `results` | Ready |
| **oracle** | `/run-engine/oracle` | `advanced/oracle.js` | `causal_graph`, `predictions` | Ready |

**Implementation Guide**: For detailed instructions on adding new engines and APIs, see the [API Implementation Guide](How_To_Implement_APIS.md).

**Status Legend:**
- **Verified**: API tested, visualization renders correctly
- **Fixed**: Data extraction updated to match API structure
- **Graceful**: Visualization handles missing data gracefully
- **Ready**: Module exists, awaiting test data

---

## How to Add a New Engine (Simplicity Guide)

The system uses a **Modular Microservice Pattern**. To add a new engine:

1.  **Backend (Python)**: Add function to `services/ml-service/src/engines/`.
2.  **Router (Python)**: Add route to `services/ml-service/src/routers/premium_engines.py`.
    ```python
    @router.post("/run-engine/my_new_engine")
    def run_new_engine(...): ...
    ```
3.  **Frontend (JS)**:
    *   Create `frontend/assets/js/nexus/visualizations/engines/my_new_engine.js`.
    *   Export `buildSection(data, vizId)` and `render(data, vizId)`.
    *   Register in `frontend/assets/js/nexus/visualizations/index.js`.

**That's it!** `nexus.html` automatically detects the engine name and loads your visualization. No core UI code changes needed.

---

## Error Log & Fixes

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
**Cause:** `nexus.html` test panel used hardcoded HTML instead of modular `NexusViz` system.
**Fix:** Refactored `createEngineResultCard` to call `window.NexusViz.buildVizSection(engine, data)`.

### Error 5: Anomaly Scores Not Found
**Cause:** Viz expected `data.scores`, API returns `data.method_results.isolation_forest.scores`.
**Fix:** Updated `anomaly.js` to check nested `method_results` for scores.

### Error 6: Statistical Column Stats Not Found
**Cause:** Viz expected `data.column_stats`, API returns `data.descriptive.numeric`.
**Fix:** Updated `statistical.js` to check `descriptive.numeric` before fallback.

### Error 7: Resource Utilization Shows 0%
**Cause:** Viz expected `data.utilization`, API returns `data.summary.avg_utilization`.
**Fix:** Updated `resource.js` to check `summary.avg_utilization` and `summary.peak_utilization`.

### Error 8: Pricing Strategy Shows $0.00
**Cause:** Viz expected `data.optimal_price`, API returns `data.recommendations[]` with `action`, `elasticity`, `reason`.
**Fix:** Updated `pricing.js` to display recommendation/action and elasticity data instead of price points.

### Error 9: Inventory Shows "No Data Available"
**Cause:** Viz expected `data.items[]`, API returns `data.abc_analysis` with `classA`, `classB`, `classC` objects.
**Fix:** Updated `inventory.js` to extract items from `abc_analysis` and display summary metrics.

### Error 10: Cash Flow Health Ratio Shows 0.00
**Cause:** Viz expected `data.summary.total_inflow/total_outflow`, API returns `data.inflows/outflows` as objects.
**Fix:** Updated `cashflow.js` to sum `inflows/outflows` objects and fallback to `net_cash_flow/burn_rate`.

---

