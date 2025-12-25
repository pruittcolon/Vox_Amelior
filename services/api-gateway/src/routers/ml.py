"""
ML Agent Router - Machine Learning and Analytics endpoints.

Provides file ingestion, ML predictions, vectorization, AutoML,
and the 27 premium analytics engines.
"""

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

try:
    from src.auth.permissions import Session, require_auth
except ImportError:

    def require_auth():
        return None

    Session = None

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ml", "analytics"])

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8006")

# Blocked file extensions for security
BLOCKED_EXTENSIONS = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",  # Executables
    ".bat",
    ".cmd",
    ".sh",
    ".ps1",  # Scripts
    ".php",
    ".phtml",
    ".jsp",
    ".asp",
    ".aspx",  # Server scripts
    ".py",
    ".rb",
    ".pl",
    ".cgi",  # Interpreted scripts
    ".jar",
    ".war",
    ".ear",  # Java archives
    ".msi",
    ".deb",
    ".rpm",
    ".apk",  # Installers
}


def _get_proxy_request():
    """Lazy import to avoid circular dependencies."""
    from src.main import proxy_request

    return proxy_request


# =============================================================================
# ML Agent Endpoints
# =============================================================================


@router.post("/api/ml/ingest")
async def ml_ingest(file: UploadFile = File(...), session: Session = Depends(require_auth)):
    """Proxy file ingestion to ML service."""
    proxy_request = _get_proxy_request()
    file_content = await file.read()
    files = {"file": (file.filename, file_content, file.content_type)}
    return await proxy_request(f"{ML_SERVICE_URL}/ingest", "POST", files=files)


@router.post("/api/ml/propose-goals")
async def ml_propose_goals(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Propose ML analysis goals based on data."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{ML_SERVICE_URL}/propose-goals", "POST", json=request)


@router.post("/api/ml/execute-analysis")
async def ml_execute_analysis(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Execute ML analysis with specified goals."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{ML_SERVICE_URL}/execute-analysis", "POST", json=request)


@router.post("/api/ml/explain-finding")
async def ml_explain_finding(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Explain an ML finding in natural language."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{ML_SERVICE_URL}/explain-finding", "POST", json=request)


# =============================================================================
# File Upload & Database Listing
# =============================================================================


def _validate_file_extension(filename: str) -> None:
    """Check if file extension is allowed."""
    if not filename:
        return
    ext = Path(filename).suffix.lower()
    if ext in BLOCKED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not allowed for security reasons")


@router.post("/upload")
async def ml_upload(file: UploadFile = File(...)):
    """Proxy file upload to ML service (with file type validation)."""
    proxy_request = _get_proxy_request()
    _validate_file_extension(file.filename)
    file_content = await file.read()
    files = {"file": (file.filename, file_content, file.content_type)}
    return await proxy_request(f"{ML_SERVICE_URL}/upload", "POST", files=files)


@router.post("/api/upload")
async def api_ml_upload(file: UploadFile = File(...)):
    """Alias for /upload with /api prefix."""
    return await ml_upload(file)


@router.get("/databases")
async def list_databases():
    """Proxy to ML service to list all available databases with embedding status."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{ML_SERVICE_URL}/databases", "GET")


@router.get("/api/databases")
async def api_list_databases():
    """Alias for /databases with /api prefix."""
    return await list_databases()


# =============================================================================
# Vectorization Endpoints
# =============================================================================


@router.post("/api/vectorize/database")
async def vectorize_database_api(request: Request):
    """Start database vectorization."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{ML_SERVICE_URL}/vectorize/database", "POST", json=body)


@router.get("/api/vectorize/status/{job_id}")
async def vectorize_status_api(job_id: str):
    """Get vectorization job status."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{ML_SERVICE_URL}/vectorize/status/{job_id}", "GET")


@router.post("/vectorize/database")
async def vectorize_database_noauth(request: Request):
    """Start database vectorization (no auth - demo mode)."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{ML_SERVICE_URL}/vectorize/database", "POST", json=body)


@router.get("/vectorize/status/{job_id}")
async def vectorize_status_noauth(job_id: str):
    """Get vectorization job status (no auth - demo mode)."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{ML_SERVICE_URL}/vectorize/status/{job_id}", "GET")


@router.post("/api/vectorize/{filename}")
async def ml_vectorize(filename: str, session: Session = Depends(require_auth)):
    """Proxy vectorization request to ML service (old endpoint for compatibility)."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{ML_SERVICE_URL}/vectorize/{filename}", "POST", json={})


@router.post("/vectorize/{path:path}")
async def vectorize_path_post(path: str, request: Request, session: Session = Depends(require_auth)):
    """Path-based vectorization POST."""
    proxy_request = _get_proxy_request()
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    return await proxy_request(f"{ML_SERVICE_URL}/vectorize/{path}", "POST", json=body)


@router.get("/vectorize/{path:path}")
async def vectorize_path_get(path: str, session: Session = Depends(require_auth)):
    """Path-based vectorization GET status."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{ML_SERVICE_URL}/vectorize/{path}", "GET")


@router.delete("/vectorize/{path:path}")
async def vectorize_path_delete(path: str, session: Session = Depends(require_auth)):
    """Path-based vectorization DELETE."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{ML_SERVICE_URL}/vectorize/{path}", "DELETE")


# =============================================================================
# Embedding & Ask Endpoints
# =============================================================================


@router.post("/embed")
async def embed(request: Request, session: Session = Depends(require_auth)):
    """Generate embeddings for text."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{ML_SERVICE_URL}/embed", "POST", json=body)


@router.post("/ask")
async def ask(request: Request, session: Session = Depends(require_auth)):
    """Ask a question about uploaded data."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{ML_SERVICE_URL}/ask", "POST", json=body)


@router.post("/analyze-full/{filename}")
async def analyze_full(filename: str, request: Request, session: Session = Depends(require_auth)):
    """Run full analysis on a file."""
    proxy_request = _get_proxy_request()
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    return await proxy_request(f"{ML_SERVICE_URL}/analyze-full/{filename}", "POST", json=body)


# =============================================================================
# Analytics Proxy (Premium Engines)
# =============================================================================


@router.post("/analytics/{path:path}")
async def analytics_post(path: str, request: Request, session: Session = Depends(require_auth)):
    """Analytics POST proxy (premium engines)."""
    proxy_request = _get_proxy_request()
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    return await proxy_request(f"{ML_SERVICE_URL}/analytics/{path}", "POST", json=body)


@router.post("/api/analytics/{path:path}")
async def api_analytics_post(path: str, request: Request, session: Session = Depends(require_auth)):
    """Analytics POST proxy with /api prefix."""
    return await analytics_post(path, request, session)


@router.get("/analytics/{path:path}")
async def analytics_get(path: str, session: Session = Depends(require_auth)):
    """Analytics GET proxy."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{ML_SERVICE_URL}/analytics/{path}", "GET")


@router.get("/api/analytics/{path:path}")
async def api_analytics_get(path: str, session: Session = Depends(require_auth)):
    """Analytics GET proxy with /api prefix."""
    return await analytics_get(path, session)


# =============================================================================
# AutoML Endpoints
# =============================================================================


@router.get("/api/insights/automl/hypotheses")
async def get_automl_hypotheses(session: Session = Depends(require_auth)):
    """Get available AutoML hypotheses."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{ML_SERVICE_URL}/automl/hypotheses", "GET")


@router.post("/api/insights/automl/run")
async def run_automl_experiment(payload: dict[str, Any], session: Session = Depends(require_auth)):
    """Run AutoML experiment."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{ML_SERVICE_URL}/automl/run", "POST", json=payload)


logger.info("✅ ML Router initialized with analytics and vectorization endpoints")
