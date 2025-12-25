"""
Quick Analysis Router

Fast analysis endpoints (<1 second execution):
- /quick-analyze - Fast layman-friendly analysis with summaries and graphs
"""

import logging
import os
import traceback
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

# Import with fallback for different import contexts
try:
    from ..schemas.analytics_models import QuickAnalysisRequest
    from ..utils.analytics_utils import (
        convert_to_native,
        load_dataset,
        secure_file_path,
        sample_for_analytics,
    )
except ImportError:
    from schemas.analytics_models import QuickAnalysisRequest
    from utils.analytics_utils import (
        convert_to_native,
        load_dataset,
        secure_file_path,
        sample_for_analytics,
    )

logger = logging.getLogger(__name__)

router = APIRouter(tags=["quick_analysis"])


def generate_layman_summary(df: pd.DataFrame, results: dict[str, Any]) -> dict[str, Any]:
    """
    Generate plain English summary of analysis results.
    Fast execution - no heavy ML operations.
    """
    n_rows, n_cols = df.shape

    # Filter out ID columns
    id_patterns = ["rownames", "id", "index", "idx", "unnamed", "pk", "key", "uuid"]
    meaningful_cols = [
        c for c in df.columns if not any(p in c.lower() for p in id_patterns)
    ]
    meaningful_cols = [
        c for c in meaningful_cols if df[c].nunique() < len(df) * 0.95
    ]

    # Size descriptions
    if n_rows < 30:
        size_desc = "very small"
        size_warning = "⚠️ Very small dataset - Results are preliminary."
    elif n_rows < 100:
        size_desc = "small"
        size_warning = "⚠️ Small dataset - Results may vary with more data."
    elif n_rows < 1000:
        size_desc = "medium-sized"
        size_warning = None
    else:
        size_desc = "large"
        size_warning = None

    # Numeric column analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    insights = []
    if numeric_cols:
        insights.append(f"📊 Contains {len(numeric_cols)} numeric columns for analysis")
    if categorical_cols:
        insights.append(f"🏷️ Contains {len(categorical_cols)} categorical columns")

    return {
        "overview": f"This {size_desc} dataset has {n_rows:,} rows and {len(meaningful_cols)} meaningful columns.",
        "size_warning": size_warning,
        "insights": insights,
        "data_quality": {
            "rows": n_rows,
            "columns": n_cols,
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
        },
    }


def generate_graph_data(df: pd.DataFrame, results: dict[str, Any]) -> dict[str, Any]:
    """
    Generate Chart.js/Plotly-ready graph data structures.
    Fast execution - pure numpy/pandas operations.
    """
    graphs = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Distribution histogram for first numeric column
    if numeric_cols:
        col = numeric_cols[0]
        values = df[col].dropna()
        if len(values) > 0:
            hist, bin_edges = np.histogram(values, bins=20)
            graphs.append({
                "type": "histogram",
                "title": f"Distribution of {col}",
                "data": {
                    "labels": [f"{bin_edges[i]:.2f}" for i in range(len(hist))],
                    "values": hist.tolist(),
                },
            })

    # Correlation heatmap if multiple numeric columns
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols[:5]].corr()  # Limit to 5 columns
        graphs.append({
            "type": "heatmap",
            "title": "Correlation Matrix",
            "data": {
                "labels": corr.columns.tolist(),
                "values": corr.values.tolist(),
            },
        })

    return {"graphs": graphs, "total_graphs": len(graphs)}


def run_quick_analysis(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """
    Run fast standalone analysis (no heavy ML, GPU optional).
    Returns results with layman summary and graph data.
    """
    # Basic statistics
    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols[:10]:  # Limit to 10 columns
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }

    # Quality score (simple heuristic)
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    quality_score = max(0, min(100, int((1 - missing_ratio) * 100)))

    results = {
        "statistics": stats,
        "quality_score": quality_score,
        "numeric_columns": numeric_cols,
        "row_count": len(df),
        "column_count": len(df.columns),
    }

    results["layman_summary"] = generate_layman_summary(df, results)
    results["graph_data"] = generate_graph_data(df, results)

    return results


@router.post("/quick-analyze")
async def quick_analyze(request: QuickAnalysisRequest):
    """
    Fast analysis endpoint returning layman-friendly summaries and graph data.

    Runs in <1 second for most datasets. No heavy ML operations.
    Returns:
    - layman_summary: Plain English insights
    - graph_data: Chart.js-ready data structures
    - quality_score, feature_importance, correlations
    """
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)
        df = sample_for_analytics(df)  # Sample for speed

        results = run_quick_analysis(df, request.target_column)

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            **results,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
