"""
Quality Insights API Router

FastAPI endpoints for the Quality Insights Engine and Business Savings Analyzer.
Provides 3D terrain data for visualization and LLM-powered recommendations.

Author: Enterprise Analytics Team
Security: JWT-authenticated endpoints
"""

import logging
import os
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

# Import security
try:
    from shared.security.service_auth import ServiceAuth, load_service_jwt_keys
    
    _router_service_auth: ServiceAuth | None = None
    try:
        keys = load_service_jwt_keys("ml-service")
        _router_service_auth = ServiceAuth(service_name="ml-service", jwt_keys=keys)
    except Exception:
        pass
except ImportError:
    _router_service_auth = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quality-insights", tags=["quality-insights"])

# Request/Response Models
class AnalyzeRequest(BaseModel):
    """Request model for quality insights analysis"""
    insights_file: str = Field(..., description="Name of the _insights.csv file")
    risk_threshold: float = Field(default=6.0, ge=1.0, le=10.0)
    top_issues_count: int = Field(default=20, ge=5, le=100)


class BusinessSavingsRequest(BaseModel):
    """Request model for business savings analysis"""
    insights_file: str = Field(..., description="Name of the _insights.csv file")
    use_llm: bool = Field(default=True, description="Use Gemma LLM for estimates")


class TerrainData(BaseModel):
    """3D terrain data for visualization"""
    row_index: int
    quality_score: float
    color_category: str
    q1: float
    q2: float
    q3: float
    q4: float
    q5: float


class AnalyzeResponse(BaseModel):
    """Response model for quality insights analysis"""
    success: bool
    terrain_data: list[dict]
    column_scores: dict[str, float]
    problem_rows: list[dict]
    problem_row_count: int
    row_count: int  # Total number of rows in the dataset
    risk_distribution: dict[str, int]
    recommendations: list[str]
    ml_readiness: float
    avg_overall: float


class SavingsResponse(BaseModel):
    """Response model for business savings analysis"""
    success: bool
    estimated_annual_savings: str
    high_impact_issues: list[dict]
    recommendations: list[dict]
    roi_timeline: dict[str, str]


# Data directory
DATA_DIR = os.environ.get("DATA_DIR", "/app/data/uploads")


def get_insights_file_path(filename: str) -> str:
    """Get full path to insights file with validation"""
    # Security: Prevent path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(DATA_DIR, safe_filename)
    
    if not file_path.startswith(DATA_DIR):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {safe_filename}")
    
    return file_path


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_quality(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze quality insights from a Gemma-scored CSV file.
    
    Returns terrain data for 3D visualization along with problem rows
    and recommendations.
    """
    try:
        # Get file path with security check
        file_path = get_insights_file_path(request.insights_file)
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Import and run engine
        from engines.quality_insights_engine import QualityInsightsEngine
        
        engine = QualityInsightsEngine()
        result = engine.analyze(
            df,
            config={
                "risk_threshold": request.risk_threshold,
                "top_issues_count": request.top_issues_count,
            },
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result.get("message", "Analysis failed"))
        
        return AnalyzeResponse(
            success=True,
            terrain_data=result.get("terrain_data", []),
            column_scores=result.get("column_scores", {}),
            problem_rows=result.get("problem_rows", []),
            problem_row_count=result.get("problem_row_count", 0),
            row_count=len(df),  # Total rows in dataset
            risk_distribution=result.get("risk_distribution", {}),
            recommendations=result.get("recommendations", []),
            ml_readiness=result.get("ml_readiness", 0.0),
            avg_overall=result.get("avg_overall", 5.0),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quality analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/business-savings", response_model=SavingsResponse)
async def analyze_business_savings(request: BusinessSavingsRequest) -> SavingsResponse:
    """
    Analyze potential business cost savings from resolving quality issues.
    
    Uses Gemma LLM when available, falls back to heuristic calculations.
    """
    try:
        # Get file path with security check
        file_path = get_insights_file_path(request.insights_file)
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # First run quality analysis
        from engines.quality_insights_engine import QualityInsightsEngine
        
        engine = QualityInsightsEngine()
        quality_result = engine.analyze(df)
        
        if "error" in quality_result:
            raise HTTPException(status_code=400, detail=quality_result.get("message", "Analysis failed"))
        
        # Run savings analysis
        from analyzers.business_savings_analyzer import BusinessSavingsAnalyzer
        
        analyzer = BusinessSavingsAnalyzer()
        
        if request.use_llm:
            # Try async LLM analysis
            import asyncio
            savings_result = await analyzer.analyze_savings(quality_result, use_llm=True)
        else:
            # Use heuristic fallback
            metrics = analyzer._extract_metrics(quality_result)
            savings_result = analyzer._analyze_with_heuristics(metrics, {})
        
        return SavingsResponse(
            success=True,
            estimated_annual_savings=savings_result.get("estimated_annual_savings", "$0"),
            high_impact_issues=savings_result.get("high_impact_issues", []),
            recommendations=savings_result.get("recommendations", []),
            roi_timeline=savings_result.get("roi_timeline", {}),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Business savings analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files")
async def list_insights_files() -> dict[str, list[str]]:
    """
    List available _insights.csv files for analysis.
    """
    try:
        if not os.path.exists(DATA_DIR):
            return {"files": []}
        
        files = [
            f for f in os.listdir(DATA_DIR)
            if f.endswith("_insights.csv")
        ]
        
        return {"files": sorted(files)}
        
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "service": "quality-insights"}
