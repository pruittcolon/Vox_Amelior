"""Database Viewer Router.

Dedicated router for database content viewing functionality.
Provides paginated row data with PII masking for Excel-like viewer.

Endpoints:
    GET /databases/{filename}/rows - Paginated row data with PII masking
    GET /databases/{filename}/schema - Column info and semantic types
"""

import os
import json
import logging
from typing import Any, Optional

import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Import SemanticMapper for PII detection
# Handle multiple import scenarios:
# 1. Running as module from ml-service container
# 2. Running directly for testing
try:
    from src.engines import SemanticMapper
except ImportError:
    try:
        from engines import SemanticMapper
    except ImportError:
        from ..engines import SemanticMapper

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/databases", tags=["Database Viewer"])

# Upload directory - must match main.py
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/app/data/uploads")


# ============================================================================
# Response Models
# ============================================================================

class ColumnInfo(BaseModel):
    """Column metadata for schema endpoint."""
    name: str
    dtype: str
    semantic_type: str
    is_pii: bool
    sample_values: list[str]
    
    
class SchemaResponse(BaseModel):
    """Response for schema endpoint."""
    filename: str
    row_count: int
    columns: list[ColumnInfo]
    

class RowsResponse(BaseModel):
    """Response for paginated rows endpoint."""
    filename: str
    page: int
    page_size: int
    total_rows: int
    total_pages: int
    columns: list[str]
    rows: list[dict[str, Any]]
    pii_columns: list[str]


# ============================================================================
# Helper Functions
# ============================================================================

def _load_dataframe(filename: str) -> pd.DataFrame:
    """
    Load a file from the upload directory into a DataFrame.
    
    Supports CSV, Excel, JSON, Parquet formats.
    
    Args:
        filename: Name of the file to load
        
    Returns:
        Loaded DataFrame
        
    Raises:
        HTTPException: If file not found or cannot be loaded
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Database '{filename}' not found")
    
    # Security: Ensure path is within UPLOAD_DIR
    real_path = os.path.realpath(file_path)
    real_upload_dir = os.path.realpath(UPLOAD_DIR)
    if not real_path.startswith(real_upload_dir):
        raise HTTPException(status_code=403, detail="Invalid file path")
    
    file_ext = filename.lower().split(".")[-1]
    
    try:
        if file_ext == "csv":
            df = pd.read_csv(file_path, low_memory=False)
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(file_path)
        elif file_ext == "json":
            df = pd.read_json(file_path)
        elif file_ext == "parquet":
            df = pd.read_parquet(file_path)
        else:
            # Try CSV as fallback
            df = pd.read_csv(file_path, low_memory=False)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading file {filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Cannot read file: {str(e)}")


def _mask_pii_row(row: dict[str, Any], pii_columns: list[str]) -> dict[str, Any]:
    """
    Mask PII values in a single row.
    
    Args:
        row: Row data as dictionary
        pii_columns: List of column names containing PII
        
    Returns:
        Row with PII values masked
    """
    masked = row.copy()
    for col in pii_columns:
        if col in masked and masked[col] is not None:
            masked[col] = "[PII REDACTED]"
    return masked


def _prepare_value_for_json(value: Any) -> Any:
    """
    Prepare a value for JSON serialization.
    
    Handles NaN, infinity, numpy types, etc.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/{filename}/schema", response_model=SchemaResponse)
async def get_database_schema(filename: str):
    """
    Get schema information for a database file.
    
    Returns column names, types, semantic classifications, and sample values.
    PII columns are identified using SemanticMapper.
    
    Args:
        filename: Name of the uploaded database file
        
    Returns:
        Schema information including column metadata
    """
    df = _load_dataframe(filename)
    
    # Get semantic schema
    semantic_schema = SemanticMapper.infer_schema(df)
    pii_columns = [col for col, stype in semantic_schema.items() if stype == "PII"]
    
    columns = []
    for col in df.columns:
        is_pii = col in pii_columns
        
        # Get sample values (mask if PII)
        sample_values = df[col].dropna().head(3).astype(str).tolist()
        if is_pii:
            sample_values = ["[PII REDACTED]"] * len(sample_values)
        
        columns.append(ColumnInfo(
            name=col,
            dtype=str(df[col].dtype),
            semantic_type=semantic_schema.get(col, "UNKNOWN"),
            is_pii=is_pii,
            sample_values=sample_values
        ))
    
    return SchemaResponse(
        filename=filename,
        row_count=len(df),
        columns=columns
    )


@router.get("/{filename}/rows", response_model=RowsResponse)
async def get_database_rows(
    filename: str,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=50, ge=1, le=100, description="Rows per page (max 100)"),
    sort_by: Optional[str] = Query(default=None, description="Column to sort by"),
    sort_order: str = Query(default="asc", regex="^(asc|desc)$", description="Sort order"),
    search: Optional[str] = Query(default=None, description="Search query for filtering")
):
    """
    Get paginated row data from a database file.
    
    Supports pagination, sorting, searching, and automatically masks PII columns.
    
    Args:
        filename: Name of the uploaded database file
        page: Page number (1-indexed)
        page_size: Number of rows per page (max 100)
        sort_by: Column name to sort by
        sort_order: 'asc' or 'desc'
        search: Text to search across all columns
        
    Returns:
        Paginated rows with PII values masked
    """
    df = _load_dataframe(filename)
    
    # Get PII columns for masking
    semantic_schema = SemanticMapper.infer_schema(df)
    pii_columns = [col for col, stype in semantic_schema.items() if stype == "PII"]
    
    # Apply search filter if provided
    if search:
        search_lower = search.lower()
        mask = df.astype(str).apply(
            lambda row: row.str.lower().str.contains(search_lower, na=False).any(),
            axis=1
        )
        df = df[mask]
    
    # Apply sorting if specified
    if sort_by and sort_by in df.columns:
        ascending = sort_order == "asc"
        try:
            df = df.sort_values(by=sort_by, ascending=ascending, na_position='last')
        except TypeError:
            # Mixed types can cause sorting issues
            df = df.sort_values(by=sort_by, key=lambda x: x.astype(str), ascending=ascending)
    
    # Calculate pagination
    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    
    # Clamp page to valid range
    page = min(page, total_pages)
    
    # Get slice for current page
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_df = df.iloc[start_idx:end_idx]
    
    # Convert to list of dicts and mask PII
    rows = []
    for _, row in page_df.iterrows():
        row_dict = {k: _prepare_value_for_json(v) for k, v in row.items()}
        row_dict = _mask_pii_row(row_dict, pii_columns)
        rows.append(row_dict)
    
    return RowsResponse(
        filename=filename,
        page=page,
        page_size=page_size,
        total_rows=total_rows,
        total_pages=total_pages,
        columns=list(df.columns),
        rows=rows,
        pii_columns=pii_columns
    )


@router.get("/{filename}/download")
async def download_database(filename: str):
    """
    Download the complete database file.
    
    Returns the file as a streaming download for efficient handling of large files.
    
    Args:
        filename: Name of the uploaded database file
        
    Returns:
        FileResponse with the complete file
    """
    from fastapi.responses import FileResponse
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Database '{filename}' not found")
    
    # Security: Ensure path is within UPLOAD_DIR
    real_path = os.path.realpath(file_path)
    real_upload_dir = os.path.realpath(UPLOAD_DIR)
    if not real_path.startswith(real_upload_dir):
        raise HTTPException(status_code=403, detail="Invalid file path")
    
    # Determine media type
    file_ext = filename.lower().split(".")[-1]
    media_types = {
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel",
        "json": "application/json",
        "parquet": "application/octet-stream"
    }
    media_type = media_types.get(file_ext, "application/octet-stream")
    
    logger.info(f"[DOWNLOAD] Serving file: {filename} ({os.path.getsize(file_path)} bytes)")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )
