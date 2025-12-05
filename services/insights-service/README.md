# Insights Service

The Visualization & Analytics Dashboard Service. It powers the visual representation of data processed by the ML Service and RAG system.

## Overview

The Insights Service bridges the gap between raw data and human understanding. It generates:

- **Interactive Charts**: Line, bar, scatter, and heatmap visualizations.
- **Dashboards**: Aggregated views of system performance and business metrics.
- **Reports**: PDF/HTML summaries of analysis sessions.

## Architecture

```
Client Request
    ↓
Fetch Data (from ML Service or RAG)
    ↓
Data Transformation (Pandas)
    ↓
Visualization Generation (Plotly/Matplotlib)
    ↓
Return JSON Config or Rendered HTML
```

## Key Features

- **Dynamic Charting**: Generates Plotly JSON for frontend rendering.
- **Automated Reporting**: Creates summary reports from analysis results.
- **Real-time Monitoring**: Visualizes system health and performance metrics.

## API Endpoints

### Health Check
```bash
GET /health
```

### Generate Chart
```bash
POST /visualize
Content-Type: application/json
{
  "data": [...],
  "type": "line",
  "x_axis": "date",
  "y_axis": "value"
}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `INSIGHTS_DB_PATH` | `/app/instance/rag.db` | Path to shared database |

## Development

```bash
cd services/insights-service
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8010
```
