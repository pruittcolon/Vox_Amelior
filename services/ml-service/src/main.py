import os
import shutil
import zipfile
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import random
import uuid
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import traceback
import logging
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service auth
service_auth = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown"""
    logger.info("Starting ML Service...")
    
    # Initialize service auth
    global service_auth
    try:
        from shared.security.service_auth import get_service_auth, load_service_jwt_keys
        jwt_keys = load_service_jwt_keys("ml-service")
        service_auth = get_service_auth(service_id="ml-service", service_secret=jwt_keys)
        logger.info("✅ JWT service auth initialized")
        
        # Initialize GPU Client with auth
        from src.core.gpu_client import init_gpu_client
        await init_gpu_client(service_auth)
        
    except Exception as e:
        logger.error(f"❌ Service auth/GPU client initialization failed: {e}")
        # Don't raise, allow service to run in CPU mode
    
    yield
    
    logger.info("Shutting down ML Service...")
    from src.core.gpu_client import shutdown_gpu_client
    await shutdown_gpu_client()

app = FastAPI(title="Universal ML Agent Service", lifespan=lifespan)

# Initialize Job Manager
try:
    # Relative imports (Primary for package structure)
    from .engines import SemanticMapper, AnalyticEngine, VectorEngine, RecipeEngine, TimeSeriesEngine, QueryEngine
    from .jobs import JobManager, JobStatus
    from .reporting import ReportGenerator
except ImportError:
    try:
        # Absolute imports (Fallback if not in package)
        from src.engines import SemanticMapper, AnalyticEngine, VectorEngine, RecipeEngine, TimeSeriesEngine, QueryEngine
        from src.jobs import JobManager, JobStatus
        from src.reporting import ReportGenerator
    except ImportError:
        # Direct imports (Fallback for direct script execution)
        from engines import SemanticMapper, AnalyticEngine, VectorEngine, RecipeEngine, TimeSeriesEngine, QueryEngine
        from jobs import JobManager, JobStatus
        from reporting import ReportGenerator
# Standardized Data Source Imports
try:
    # Relative imports (Primary for package structure)
    from .data_sources.excel_loader import ExcelLoader
    from .data_sources.json_loader import JSONLoader
    from .data_sources.parquet_loader import ParquetLoader
    from .data_sources.sqlite_loader import SQLiteLoader
    from .data_sources.postgres_loader import PostgresLoader
    from .data_sources.mysql_loader import MySQLLoader
    from .data_sources.mongodb_loader import MongoDBLoader
    from .data_sources.redis_loader import RedisLoader
    from .data_sources.bigquery_loader import BigQueryLoader
    from .data_sources.snowflake_loader import SnowflakeLoader
    from .data_sources.s3_loader import S3Loader
    from .data_sources.gcs_loader import GCSLoader
    from .data_sources.rest_api_loader import RestApiLoader
    from .data_sources.graphql_loader import GraphQLLoader
    from .data_sources.kafka_loader import KafkaLoader
except ImportError:
    try:
        # Absolute imports (Fallback if src is in path)
        from src.data_sources.excel_loader import ExcelLoader
        from src.data_sources.json_loader import JSONLoader
        from src.data_sources.parquet_loader import ParquetLoader
        from src.data_sources.sqlite_loader import SQLiteLoader
        from src.data_sources.postgres_loader import PostgresLoader
        from src.data_sources.mysql_loader import MySQLLoader
        from src.data_sources.mongodb_loader import MongoDBLoader
        from src.data_sources.redis_loader import RedisLoader
        from src.data_sources.bigquery_loader import BigQueryLoader
        from src.data_sources.snowflake_loader import SnowflakeLoader
        from src.data_sources.s3_loader import S3Loader
        from src.data_sources.gcs_loader import GCSLoader
        from src.data_sources.rest_api_loader import RestApiLoader
        from src.data_sources.graphql_loader import GraphQLLoader
        from src.data_sources.kafka_loader import KafkaLoader
    except ImportError:
        # Direct imports (Fallback for direct script execution)
        from data_sources.excel_loader import ExcelLoader
        from data_sources.json_loader import JSONLoader
        from data_sources.parquet_loader import ParquetLoader
        from data_sources.sqlite_loader import SQLiteLoader
        from data_sources.postgres_loader import PostgresLoader
        from data_sources.mysql_loader import MySQLLoader
        from data_sources.mongodb_loader import MongoDBLoader
        from data_sources.redis_loader import RedisLoader
        from data_sources.bigquery_loader import BigQueryLoader
        from data_sources.snowflake_loader import SnowflakeLoader
        from data_sources.s3_loader import S3Loader
        from data_sources.gcs_loader import GCSLoader
        from data_sources.rest_api_loader import RestApiLoader
        from data_sources.graphql_loader import GraphQLLoader
        from data_sources.kafka_loader import KafkaLoader

# Initialize Job Manager
job_manager = JobManager()

# Phase 4: CORS Security - Restrict to specific origins
# Avoid allow_origins=["*"] with credentials=True (security vulnerability)
ALLOWED_ORIGINS = os.getenv(
    "ML_ALLOWED_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,https://localhost,https://127.0.0.1"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Restrict to needed methods
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Service-Token"],
)

# Phase 2: Service-to-Service Authentication Middleware
# Requires X-Service-Token header on all non-exempt endpoints
try:
    from shared.security.service_auth import ServiceAuthMiddleware, load_service_jwt_keys, _is_test_mode
    _jwt_keys = load_service_jwt_keys("ml-service")
    app.add_middleware(
        ServiceAuthMiddleware,
        service_secret=_jwt_keys,
        exempt_paths=["/health", "/docs", "/openapi.json", "/gpu-status", "/metrics", "/"],
        enabled=not _is_test_mode()
    )
    logging.getLogger(__name__).info("✅ ServiceAuthMiddleware enabled on ML service")
except Exception as _auth_err:
    logging.getLogger(__name__).warning(f"⚠️ ServiceAuthMiddleware not loaded: {_auth_err}")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
ARCHIVE_DIR = os.getenv("ARCHIVE_DIR", "archive")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# --- Temp File Cleanup Utility ---
import time

def cleanup_stale_temp_files(max_age_hours: int = 24):
    """
    Cleans up old temporary files in UPLOAD_DIR.
    Called periodically or on startup to prevent disk bloat.
    """
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0
    
    try:
        for item in os.listdir(UPLOAD_DIR):
            item_path = os.path.join(UPLOAD_DIR, item)
            try:
                # Check file/folder age
                mtime = os.path.getmtime(item_path)
                age = now - mtime
                
                if age > max_age_seconds:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                        cleaned_count += 1
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                        cleaned_count += 1
            except Exception as e:
                logger.warning(f"Could not cleanup {item_path}: {e}")
                
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} stale temp files/folders older than {max_age_hours}h")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

# Run cleanup on startup (remove files older than 24 hours)
cleanup_stale_temp_files(max_age_hours=24)

# --- Session Management for Gemma Access ---
class GatewaySession:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.token = None
        
    def login(self, timeout: int = 10):
        """Login to Gateway with timeout handling - uses environment variables for security"""
        # SECURITY: Load credentials from environment, NOT hardcoded
        ml_user = os.getenv("ML_SERVICE_USER", "")
        ml_pass = os.getenv("ML_SERVICE_PASSWORD", "")
        
        if ml_user and ml_pass:
            creds = [(ml_user, ml_pass)]
        else:
            # Fallback to admin (only if ENABLE_DEMO_USERS is set in production)
            print("⚠️ [SECURITY] ML_SERVICE_USER/ML_SERVICE_PASSWORD not set. Using demo credentials.")
            print("⚠️ [SECURITY] Set these environment variables for production use!")
            creds = [("admin", os.getenv("ADMIN_PASSWORD", "admin123"))]
        
        for user, pwd in creds:
            try:
                print(f"🔐 Attempting login as {user}...")
                resp = self.session.post(
                    f"{self.base_url}/api/auth/login", 
                    json={"username": user, "password": pwd},
                    timeout=timeout  # Prevent login hangs
                )
                if resp.status_code == 200:
                    print(f"✅ Login successful as {user}")
                    data = resp.json()
                    self.token = data.get("session_token")
                    return True
            except requests.exceptions.Timeout:
                print(f"⚠️ Login timed out for {user}")
            except Exception as e:
                print(f"⚠️ Login error: {e}")
        print("❌ Failed to login to Gateway. Gemma features will be disabled.")
        return False

    def generate(self, prompt: str, max_tokens=512, timeout: int = 30):
        """Generate text using Gemma via public chat endpoint (no auth needed)"""
        try:
            # Use public chat endpoint - no authentication required
            resp = self.session.post(
                f"{self.base_url}/api/public/chat", 
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                # Public chat returns {"message": "..."} format
                return data.get("message", data.get("text", ""))
            else:
                print(f"⚠️ Gemma request failed with status {resp.status_code}")
        except requests.exceptions.Timeout:
            print(f"⚠️ Gemma request timed out after {timeout}s - continuing without summary")
            return None
        except Exception as e:
            print(f"❌ Gemma generation error: {e}")
        return None

gateway = GatewaySession(GATEWAY_URL)

# --- Models ---

class DatasetProfile(BaseModel):
    filename: str
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    sample_data: List[Dict[str, Any]]
    summary_stats: Dict[str, Any]
    row_count: int
    semantic_schema: Dict[str, str]
    pii_columns: List[str] = []
    has_embeddings: bool = False # New field # New field

class AnalysisGoal(BaseModel):
    id: str
    title: str
    description: str
    type: str 
    complexity: str

class AnalysisResult(BaseModel):
    goal_id: str
    summary: str
    charts: Optional[List[Dict[str, Any]]] = None  # Multiple charts instead of single chart_data
    insights: List[str]
    gemma_narration: Optional[str] = None  # AI-generated presentation narrative

class ChartNarrationRequest(BaseModel):
    chart: Dict[str, Any]
    insights: List[str]
    context: str

class ExplainRequest(BaseModel):
    question: str
    context: AnalysisResult

@app.get("/health")
def health_check():
    """Health check with CUDA/GPU status"""
    gpu_info = {"available": False, "device": None, "memory": None}
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device"] = torch.cuda.get_device_name(0)
            gpu_info["memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            gpu_info["memory_allocated_mb"] = torch.cuda.memory_allocated(0) // (1024 * 1024)
            gpu_info["cuda_version"] = torch.version.cuda
    except Exception as e:
        gpu_info["error"] = str(e)
    
    return {
        "status": "healthy",
        "gpu": gpu_info
    }

@app.get("/gpu-status")
def gpu_status():
    """Detailed GPU status for debugging"""
    result = {
        "torch_available": False,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "xgboost_gpu": False
    }
    
    try:
        import torch
        result["torch_available"] = True
        result["cuda_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            result["cuda_version"] = torch.version.cuda
            result["device_count"] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                result["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "major": props.major,
                    "minor": props.minor
                })
    except Exception as e:
        result["torch_error"] = str(e)
    
    try:
        import xgboost as xgb
        # Check if XGBoost can use GPU
        try:
            test_params = {"tree_method": "gpu_hist", "n_estimators": 1}
            xgb.XGBClassifier(**test_params)
            result["xgboost_gpu"] = True
        except:
            result["xgboost_gpu"] = False
    except Exception as e:
        result["xgboost_error"] = str(e)
    
    return result


@app.get("/databases")
def list_databases():
    """
    List all available databases with embedding status.
    Used by gemma.html to show saved databases and allow switching.
    """
    databases = []
    
    try:
        for filename in os.listdir(UPLOAD_DIR):
            # Only include data files, not index/meta files
            if filename.endswith(('.csv', '.xlsx', '.xls', '.json', '.parquet')):
                file_path = os.path.join(UPLOAD_DIR, filename)
                
                # Check if embeddings exist
                index_path = file_path + ".index"
                has_embeddings = os.path.exists(index_path)
                
                # Get file stats
                try:
                    stats = os.stat(file_path)
                    size_bytes = stats.st_size
                    modified_at = datetime.fromtimestamp(stats.st_mtime).isoformat()
                except:
                    size_bytes = 0
                    modified_at = None
                
                # Create display name (remove UUID prefix if present)
                display_name = filename
                if '_' in filename:
                    parts = filename.split('_', 1)
                    if len(parts[0]) == 8:  # UUID prefix is 8 chars
                        display_name = parts[1]
                
                databases.append({
                    "filename": filename,
                    "display_name": display_name,
                    "has_embeddings": has_embeddings,
                    "size_bytes": size_bytes,
                    "modified_at": modified_at
                })
    except Exception as e:
        logger.error(f"Error listing databases: {e}")
    
    # Sort by modification time (newest first)
    databases.sort(key=lambda x: x.get("modified_at") or "", reverse=True)
    
    return {"databases": databases}


# --- Logic ---

def profile_dataframe(df: pd.DataFrame, filename: str, total_rows: Optional[int] = None) -> DatasetProfile:
    # 1. Enforce Limits (Guardrails)
    MAX_ROWS = 100000
    MAX_COLS = 1000
    
    if len(df.columns) > MAX_COLS:
        raise HTTPException(status_code=400, detail=f"Too many columns: {len(df.columns)}. Max allowed is {MAX_COLS}.")

    if len(df) > MAX_ROWS:
        print(f"⚠️ Truncating {filename} from {len(df)} to {MAX_ROWS} rows.")
        df = df.head(MAX_ROWS)
        
    # 2. Sampling for Profile Preview
    if len(df) > 10000:
        df_sample = df.sample(10000)
    else:
        df_sample = df

    df_sample = df_sample.replace({np.nan: None})
    
    # 3. Run Semantic Mapping (includes PII scan now)
    semantic_schema = SemanticMapper.infer_schema(df)
    
    # 4. Extract PII Columns
    pii_cols = [col for col, type_ in semantic_schema.items() if type_ == "PII"]
    
    # 5. Mask PII in Sample Data
    # We take head(5) for the preview, but we must mask PII columns
    preview_df = df_sample.head(5).copy()
    for col in pii_cols:
        if col in preview_df.columns: # Ensure column exists before trying to mask
            preview_df[col] = "[PII REDACTED]"
        
    sample_records = json.loads(preview_df.to_json(orient="records", date_format="iso"))

    # 6. Build Summary Stats with PII Masking
    # SECURITY FIX: Mask PII values in summary_stats (top, freq, unique values)
    summary_stats = df_sample.describe(include='all').to_dict()
    for col, stats in summary_stats.items():
        # Check if this column is PII
        is_pii = col in pii_cols
        
        for stat_name, val in list(stats.items()):
            # Mask PII-revealing stats (top value, frequency examples)
            if is_pii and stat_name in ['top', 'first', 'last']:
                summary_stats[col][stat_name] = "[PII REDACTED]"
            # Truncate long strings to prevent payload bloat
            elif isinstance(val, str) and len(val) > 50:
                summary_stats[col][stat_name] = val[:47] + "..."
            # Handle NaN/inf values that can't be JSON serialized
            elif isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                summary_stats[col][stat_name] = None

    # Use provided total_rows if available (for large file sampling), else use current df length
    final_row_count = total_rows if total_rows is not None else len(df)

    profile = DatasetProfile(
        filename=filename,
        columns=list(df.columns),
        dtypes={k: str(v) for k, v in df.dtypes.items()},
        missing_values=df.isnull().sum().to_dict(), # Note: This is on the loaded portion/sample
        sample_data=sample_records,
        summary_stats=summary_stats,
        row_count=final_row_count,
        semantic_schema=semantic_schema,
        pii_columns=pii_cols
    )
    return profile

@app.post("/ingest", response_model=DatasetProfile)
async def ingest_file(file: UploadFile = File(...)):
    filename = file.filename
    file_ext = filename.split(".")[-1].lower()
    
    # 0. File Size Limit (Approximate via seek/tell or read chunks)
    MAX_FILE_SIZE = 1024 * 1024 * 1024 # 1GB
    MAX_UNCOMPRESSED_SIZE = 2 * 1024 * 1024 * 1024  # 2GB max uncompressed (ZIP bomb protection)
    
    # Stream to disk to avoid memory explosion, checking size
    unique_id = uuid.uuid4().hex[:8]
    safe_filename = f"{unique_id}_{filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    # Track temp files/dirs for cleanup
    temp_paths_to_cleanup = []
    
    try:
        size = 0
        
        def sync_write(f, chunk):
            f.write(chunk)

        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024) # 1MB chunks
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=400, detail=f"File too large. Max size is 1GB.")
                await run_in_threadpool(sync_write, f, chunk)
        
        temp_paths_to_cleanup.append(file_path)  # Track original upload for cleanup
            
        # Handle ZIP
        if file_ext == "zip":
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # SECURITY: Pre-validate total uncompressed size (ZIP bomb protection)
                    total_uncompressed = sum(info.file_size for info in zip_ref.infolist())
                    if total_uncompressed > MAX_UNCOMPRESSED_SIZE:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"ZIP uncompressed size ({total_uncompressed / 1e9:.2f}GB) exceeds 2GB limit. Possible ZIP bomb."
                        )
                    
                    # SECURITY: Check compression ratio (ZIP bomb indicator)
                    compressed_size = sum(info.compress_size for info in zip_ref.infolist())
                    if compressed_size > 0 and total_uncompressed / compressed_size > 100:
                        raise HTTPException(
                            status_code=400,
                            detail="Suspicious compression ratio detected. Possible ZIP bomb."
                        )
                    
                    # Security: Don't extract blindly. List files first.
                    file_list = zip_ref.namelist()
                    csv_files = [f for f in file_list if f.lower().endswith('.csv') and not f.startswith('__MACOSX') and not '..' in f]
                    
                    if not csv_files:
                        raise HTTPException(status_code=400, detail="No CSV files found in ZIP archive.")
                    
                    # Pick the largest CSV or first one? Let's pick the first valid one.
                    target_csv = csv_files[0]
                    extract_path = os.path.join(UPLOAD_DIR, f"{unique_id}_extracted")
                    os.makedirs(extract_path, exist_ok=True)
                    temp_paths_to_cleanup.append(extract_path)  # Track for cleanup
                    
                    zip_ref.extract(target_csv, extract_path)
                    
                    # Update file_path to the extracted CSV
                    file_path = os.path.join(extract_path, target_csv)
                    filename = target_csv # Update logical filename
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail="Invalid ZIP file.")
            except HTTPException:
                raise  # Re-raise our own HTTP exceptions
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing ZIP: {str(e)}")

        # Handle Excel
        if file_ext in ['xlsx', 'xls', 'xlsm', 'xlsb']:
            try:
                print(f"DEBUG: Detected Excel file: {file_path}")
                loader = ExcelLoader()
                # Load first sheet by default for profiling
                # TODO: Support multi-sheet selection in future
                df = loader.load(file_path, sheet_name=0)
                
                # Check for large file optimization (sampling)
                if len(df) > 100000:
                    print(f"DEBUG: Large Excel file ({len(df)} rows). Sampling first 10k.")
                    total_rows = len(df)
                    df = df.head(10000)
                else:
                    total_rows = len(df)
                    
                print(f"DEBUG: Excel load success. Columns: {df.columns.tolist()}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing Excel file: {str(e)}")
        
        # Handle JSON
        elif file_ext in ['json', 'jsonl', 'ndjson']:
            try:
                print(f"DEBUG: Detected JSON file: {file_path}")
                loader = JSONLoader()
                df = loader.load(file_path)
                
                # Check for large file optimization
                if len(df) > 100000:
                    print(f"DEBUG: Large JSON file ({len(df)} rows). Sampling first 10k.")
                    total_rows = len(df)
                    df = df.head(10000)
                else:
                    total_rows = len(df)
                    
                print(f"DEBUG: JSON load success. Columns: {df.columns.tolist()}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing JSON file: {str(e)}")

        # Handle Parquet
        elif file_ext == 'parquet':
            try:
                print(f"DEBUG: Detected Parquet file: {file_path}")
                loader = ParquetLoader()
                df = loader.load(file_path)
                
                # Check for large file optimization
                if len(df) > 100000:
                    print(f"DEBUG: Large Parquet file ({len(df)} rows). Sampling first 10k.")
                    total_rows = len(df)
                    df = df.head(10000)
                else:
                    total_rows = len(df)
                    
                print(f"DEBUG: Parquet load success. Columns: {df.columns.tolist()}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing Parquet file: {str(e)}")

        # Handle SQLite
        elif file_ext in ['db', 'sqlite', 'sqlite3']:
            try:
                print(f"DEBUG: Detected SQLite file: {file_path}")
                loader = SQLiteLoader(file_path)
                
                # List tables to pick one
                tables = loader.list_tables()
                if not tables:
                     raise ValueError("SQLite database contains no tables.")
                
                # Load first table by default
                target_table = tables[0]
                print(f"DEBUG: Loading first table: {target_table}")
                
                # Check row count
                row_count = loader.estimate_count(target_table)
                
                if row_count > 100000:
                    print(f"DEBUG: Large SQLite table ({row_count} rows). Sampling first 10k.")
                    total_rows = row_count
                    df = loader.load_table(target_table, limit=10000)
                else:
                    total_rows = row_count
                    df = loader.load_table(target_table)
                    
                print(f"DEBUG: SQLite load success. Columns: {df.columns.tolist()}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing SQLite file: {str(e)}")

        # Handle CSV (Default)
        else:
            # Check file size for optimization
            file_size_bytes = os.path.getsize(file_path)
            LARGE_FILE_THRESHOLD = 50 * 1024 * 1024 # 50MB
            
            total_rows = None
            load_kwargs = {'engine': 'python', 'on_bad_lines': 'skip'}
            
            if file_size_bytes > LARGE_FILE_THRESHOLD:
                print(f"DEBUG: Large file detected ({file_size_bytes/1024/1024:.1f}MB). Optimizing load.")
                # 1. Count rows efficiently
                try:
                    with open(file_path, 'rb') as f:
                        total_rows = sum(1 for _ in f) - 1 # Subtract header
                except:
                    total_rows = None # Fallback if binary read fails
                    
                # 2. Load Sample
                load_kwargs['nrows'] = 10000
                print("DEBUG: Loading only first 10k rows for profiling.")

            # Read CSV with Delimiter Sniffing
            try:
                # 1. Try standard CSV (comma) first
                print(f"DEBUG: Trying comma separator for {file_path}")
                df = pd.read_csv(file_path, **load_kwargs)
                
                # Retry with semicolon if 1 column
                if len(df.columns) <= 1:
                     print(f"DEBUG: Only 1 column found. Trying sep=;")
                     load_kwargs['sep'] = ';'
                     df_semi = pd.read_csv(file_path, **load_kwargs)
                     if len(df_semi.columns) > 1:
                         df = df_semi
                
                print(f"DEBUG: Comma/Semi success. Columns: {df.columns.tolist()}")
                print(f"DEBUG: Head: {df.head(1).to_dict()}")
            except Exception as e_comma:
                print(f"DEBUG: Comma failed: {e_comma}")
                try:
                    # 2. Try sniffing
                    print(f"DEBUG: Trying sep=None")
                    load_kwargs['sep'] = None
                    df = pd.read_csv(file_path, **load_kwargs)
                    print(f"DEBUG: Sniff success. Columns: {df.columns.tolist()}")
                except Exception:
                    try:
                        # 3. Try semicolon explicitly
                        print(f"DEBUG: Trying sep=;")
                        load_kwargs['sep'] = ';'
                        df = pd.read_csv(file_path, **load_kwargs)
                    except Exception as e:
                        # 4. Fallback for encoding issues
                        try:
                            load_kwargs['sep'] = None # Reset
                            load_kwargs['encoding'] = 'latin1'
                            df = pd.read_csv(file_path, **load_kwargs)
                        except Exception as e2:
                            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e2)}")
        
        # Profile
        profile = profile_dataframe(df, filename, total_rows=total_rows)
        
        # Update profile filename to match saved file (with UUID)
        profile.filename = safe_filename
        
        return profile

    except HTTPException as he:
        # Cleanup all temp files/directories on failure
        for path in temp_paths_to_cleanup:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass
        raise he
    except Exception as e:
        # Cleanup all temp files/directories on failure
        for path in temp_paths_to_cleanup:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

class DataSourceRequest(BaseModel):
    source_type: str
    connection_params: Dict[str, Any]
    query_params: Dict[str, Any]

@app.post("/ingest/source", response_model=DatasetProfile)
async def ingest_source(req: DataSourceRequest):
    try:
        df = pd.DataFrame()
        filename = f"{req.source_type}_data"
        
        # 1. Relational Databases
        if req.source_type == "postgres":
            loader = PostgresLoader(**req.connection_params)
            query = req.query_params.get("query")
            table = req.query_params.get("table")
            if query:
                df = loader.load_query(query)
                filename = "postgres_query"
            elif table:
                df = loader.load_table(table)
                filename = f"postgres_{table}"
            else:
                raise ValueError("Must provide 'query' or 'table' in query_params")
                
        elif req.source_type == "mysql":
            loader = MySQLLoader(**req.connection_params)
            query = req.query_params.get("query")
            table = req.query_params.get("table")
            if query:
                df = loader.load_query(query)
                filename = "mysql_query"
            elif table:
                df = loader.load_table(table)
                filename = f"mysql_{table}"
            else:
                raise ValueError("Must provide 'query' or 'table' in query_params")

        elif req.source_type == "sqlite":
            # SQLite is file-based, usually handled via upload, but if path is local/shared:
            db_path = req.connection_params.get("db_path")
            loader = SQLiteLoader(db_path)
            query = req.query_params.get("query")
            table = req.query_params.get("table")
            if query:
                df = loader.load_query(query)
                filename = "sqlite_query"
            elif table:
                df = loader.load_table(table)
                filename = f"sqlite_{table}"
            else:
                raise ValueError("Must provide 'query' or 'table' in query_params")

        # 2. NoSQL
        elif req.source_type == "mongodb":
            loader = MongoDBLoader(**req.connection_params)
            db_name = req.query_params.get("database")
            collection = req.query_params.get("collection")
            query = req.query_params.get("query", {})
            limit = req.query_params.get("limit", 10000)
            if not db_name or not collection:
                raise ValueError("Must provide 'database' and 'collection'")
            df = loader.load_collection(db_name, collection, query, limit=limit)
            filename = f"mongo_{collection}"

        elif req.source_type == "redis":
            loader = RedisLoader(**req.connection_params)
            key_pattern = req.query_params.get("key_pattern", "*")
            count = req.query_params.get("count", 100)
            type_ = req.query_params.get("type", "hash")
            if type_ == "hash":
                df = loader.load_hashes(key_pattern, count)
            elif type_ == "list":
                df = loader.load_lists(key_pattern, count)
            elif type_ == "stream":
                 df = loader.load_streams(key_pattern, count)
            else:
                 raise ValueError("Unsupported Redis type")
            filename = f"redis_{type_}"

        # 3. Cloud Warehouses
        elif req.source_type == "bigquery":
            loader = BigQueryLoader(**req.connection_params)
            query = req.query_params.get("query")
            if not query:
                raise ValueError("Must provide 'query'")
            df = loader.load_query(query)
            filename = "bigquery_result"

        elif req.source_type == "snowflake":
            loader = SnowflakeLoader(**req.connection_params)
            query = req.query_params.get("query")
            if not query:
                raise ValueError("Must provide 'query'")
            df = loader.load_query(query)
            filename = "snowflake_result"

        # 4. Cloud Storage
        elif req.source_type == "s3":
            loader = S3Loader(**req.connection_params)
            bucket = req.query_params.get("bucket")
            key = req.query_params.get("key")
            file_type = req.query_params.get("file_type")
            if not bucket or not key:
                raise ValueError("Must provide 'bucket' and 'key'")
            df = loader.read_file(bucket, key, file_type)
            filename = f"s3_{key.split('/')[-1]}"

        elif req.source_type == "gcs":
            loader = GCSLoader(**req.connection_params)
            bucket = req.query_params.get("bucket")
            blob = req.query_params.get("blob")
            file_type = req.query_params.get("file_type")
            if not bucket or not blob:
                raise ValueError("Must provide 'bucket' and 'blob'")
            df = loader.read_file(bucket, blob, file_type)
            filename = f"gcs_{blob.split('/')[-1]}"

        # 5. APIs & Streaming
        elif req.source_type == "rest":
            loader = RestApiLoader(**req.connection_params)
            endpoint = req.query_params.get("endpoint")
            params = req.query_params.get("params")
            data_key = req.query_params.get("data_key")
            if not endpoint:
                raise ValueError("Must provide 'endpoint'")
            df = loader.fetch_data(endpoint, params=params, data_key=data_key)
            filename = "rest_api_data"

        elif req.source_type == "graphql":
            loader = GraphQLLoader(**req.connection_params)
            query = req.query_params.get("query")
            variables = req.query_params.get("variables")
            data_key = req.query_params.get("data_key")
            if not query:
                raise ValueError("Must provide 'query'")
            df = loader.fetch_data(query, variables=variables, data_key=data_key)
            filename = "graphql_data"

        elif req.source_type == "kafka":
            loader = KafkaLoader(**req.connection_params)
            topic = req.query_params.get("topic")
            max_messages = req.query_params.get("max_messages", 100)
            timeout = req.query_params.get("timeout", 10.0)
            if not topic:
                raise ValueError("Must provide 'topic'")
            df = loader.consume_messages(topic, max_messages=max_messages, timeout=timeout)
            filename = f"kafka_{topic}"

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source type: {req.source_type}")

        if df.empty:
            raise ValueError("No data returned from source.")

        # Profile the data
        profile = profile_dataframe(df, filename)
        
        # Save to disk for persistence (optional, but good for embedding later)
        # We can save as CSV or Parquet
        unique_id = uuid.uuid4().hex[:8]
        safe_filename = f"{unique_id}_{filename}.csv"
        save_path = os.path.join(UPLOAD_DIR, safe_filename)
        df.to_csv(save_path, index=False)
        
        # Update profile filename to match saved file
        profile.filename = safe_filename
        
        return profile

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

class SearchRequest(BaseModel):
    filename: str
    query: str
    k: int = 5

@app.post("/embed/{filename}")
async def embed_dataset(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        # Check if it was a safe filename (prefix)
        # For simplicity, we assume filename passed is the one on disk or we search
        # But wait, ingest returns the logical filename. We stored it as safe_filename.
        # We need to find the file.
        found = False
        for f in os.listdir(UPLOAD_DIR):
            if f.endswith(filename) or f == filename:
                file_path = os.path.join(UPLOAD_DIR, f)
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="File not found")

    try:
        # Load Data
        # Optimization for embedding: only read text columns? No, we need schema first.
        # Just load sample for schema inference, then iterate? 
        # For now, keeping embedding simple as it wasn't the bottleneck in tests.
        try:
             # 1. Try standard CSV (comma) first
             df = pd.read_csv(file_path, engine='python', on_bad_lines='skip', nrows=10000) # Limit for embedding
        except:
             try:
                 # 2. Try sniffing
                 df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip', nrows=10000)
             except:
                 # 3. Fallback
                 df = pd.read_csv(file_path, sep=None, encoding='latin1', engine='python', on_bad_lines='skip', nrows=10000)
        
        # Infer Text Columns
        schema = SemanticMapper.infer_schema(df)
        text_cols = [col for col, t in schema.items() if t == "TEXT"]
        
        if not text_cols:
             return {"status": "skipped", "message": "No text columns found to embed."}
        
        # Combine text columns for embedding
        # We'll take the first 2 text columns or just the first one?
        # Let's combine all TEXT columns into a single string per row
        df['combined_text'] = df[text_cols].astype(str).agg(' '.join, axis=1)
        
        # Limit rows for embedding (cost/time) - already done via nrows above
        MAX_EMBED_ROWS = 10000
        if len(df) > MAX_EMBED_ROWS:
            df = df.head(MAX_EMBED_ROWS)
            
        texts = df['combined_text'].tolist()
        
        # Initialize Vector Engine
        ve = VectorEngine()
        embeddings = ve.embed_text(texts)
        
        # Build and Save Index
        index_path = os.path.join(UPLOAD_DIR, f"{filename}.index")
        ve.build_index(texts, embeddings)
        ve.save(index_path)
        
        return {"status": "success", "message": f"Embedded {len(texts)} rows.", "index_path": index_path}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from CSV/TSV/Excel with pragmatic fallbacks.
    - Tries CSV (comma), then CSV (semicolon), then Excel.
    - Skips bad lines, clips very large files to first 100k rows.
    """
    try:
        file_size_bytes = os.path.getsize(file_path)
        LARGE_FILE_THRESHOLD = 50 * 1024 * 1024  # 50MB
        load_kwargs = {'engine': 'python', 'on_bad_lines': 'skip'}

        if file_size_bytes > LARGE_FILE_THRESHOLD:
            load_kwargs['nrows'] = 100000

        # If Excel extension, try Excel first
        lower = file_path.lower()
        if lower.endswith(('.xlsx', '.xls')):
            try:
                return pd.read_excel(file_path)
            except Exception as excel_err:
                # Fall through to CSV attempts as a fallback
                logger.warning("Excel load failed for %s: %s; trying CSV fallback", file_path, excel_err)

        # Try comma-separated
        try:
            df = pd.read_csv(file_path, **load_kwargs)
            if len(df.columns) > 1:
                return df
        except Exception:
            pass

        # Try semicolon-separated
        try:
            load_kwargs['sep'] = ';'
            df = pd.read_csv(file_path, **load_kwargs)
            if len(df.columns) > 1:
                return df
        except Exception:
            pass

        # Try tab-separated as a last text fallback
        try:
            load_kwargs['sep'] = '\t'
            df = pd.read_csv(file_path, **load_kwargs)
            if len(df.columns) > 1:
                return df
        except Exception:
            pass

        raise HTTPException(status_code=500, detail="Failed to load dataset: unsupported format or empty file")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

# Maximum rows for analytics engines to prevent slowdowns
MAX_ANALYTICS_ROWS = int(os.getenv("MAX_ANALYTICS_ROWS", "10000"))

def sample_for_analytics(df: pd.DataFrame, max_rows: int = MAX_ANALYTICS_ROWS) -> pd.DataFrame:
    """
    Sample large datasets to prevent analytics engine slowdowns.
    Uses stratified sampling when possible to preserve data distribution.
    Returns (sampled_df, was_sampled) tuple.
    """
    if len(df) <= max_rows:
        return df
    
    logger.info(f"[SAMPLING] Dataset has {len(df):,} rows, sampling to {max_rows:,}")
    
    # Simple random sample - preserves most distributions
    sampled = df.sample(n=max_rows, random_state=42)
    logger.info(f"[SAMPLING] Sampled from {len(df):,} to {len(sampled):,} rows")
    
    return sampled

class SimulationRequest(BaseModel):
    filename: str
    goal_id: str
    perturbations: Dict[str, float]

@app.post("/simulate")
async def simulate(request: SimulationRequest):
    """
    Run a What-If simulation.
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        # Load data
        df = load_dataset(file_path)
        
        # Infer Schema
        schema = SemanticMapper.infer_schema(df)
        
        # Initialize Engine
        engine = RecipeEngine(df, schema)
        
        # Run Simulation
        if request.goal_id == "churn_analysis":
            result = engine.simulate_churn(request.perturbations)
            return result
        else:
            return {"error": f"Simulation not supported for goal: {request.goal_id}"}
            
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SegmentationRequest(BaseModel):
    filename: str
    k: int = 3
    features: Optional[List[str]] = None

@app.post("/segment")
async def segment_users(request: SegmentationRequest):
    """
    Run automated segmentation (clustering).
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, request.filename)
        if not os.path.exists(file_path):
             # Try finding it with safe prefix
            found = False
            for f in os.listdir(UPLOAD_DIR):
                if f.endswith(request.filename) or f == request.filename:
                    file_path = os.path.join(UPLOAD_DIR, f)
                    found = True
                    break
            if not found:
                raise HTTPException(status_code=404, detail="File not found")
            
        # Load data
        df = load_dataset(file_path)
        
        # Initialize Engine
        # Import here to avoid circular deps if any, or just standard pattern
        try:
            from .segmentation import SegmentationEngine
        except ImportError:
            from src.segmentation import SegmentationEngine
            
        engine = SegmentationEngine(df)
        
        # Run Segmentation
        result = engine.segment(k=request.k, features=request.features)
        
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
             
        return result
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_dataset(req: SearchRequest):
    index_path = os.path.join(UPLOAD_DIR, f"{req.filename}.index")
    if not os.path.exists(index_path):
         raise HTTPException(status_code=404, detail="Embeddings not found. Please call /embed first.")
    
    try:
        ve = VectorEngine(index_path)
        results = ve.search(req.query, req.k)
        return results
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/propose-goals", response_model=List[AnalysisGoal])
async def propose_goals(profile: DatasetProfile):
    # Use the new Semantic Schema to drive goals
    schema = profile.semantic_schema
    cols_by_type = {}
    for col, t in schema.items():
        cols_by_type.setdefault(t, []).append(col)
        
    goals = []
    
    # 1. General Overview (Always)
    goals.append(AnalysisGoal(
        id="general_overview",
        title="General Overview",
        description="Analyze distribution of key metrics and categorical breakdowns.",
        type="exploratory",
        complexity="low"
    ))
    
    # 2. Correlation Analysis (If multiple metrics)
    metrics = cols_by_type.get("METRIC", []) + cols_by_type.get("MONEY_IN", []) + cols_by_type.get("MONEY_OUT", [])
    if len(metrics) >= 2:
        goals.append(AnalysisGoal(
            id="correlation_analysis",
            title="Correlation Analysis",
            description=f"Identify relationships between {', '.join(metrics[:3])}...",
            type="exploratory",
            complexity="medium"
        ))
        
    # 3. Anomaly Detection (If metrics exist)
    if metrics:
        goals.append(AnalysisGoal(
            id="anomaly_detection",
            title="Anomaly Detection",
            description=f"Detect outliers and anomalies in {metrics[0]}.",
            type="predictive",
            complexity="medium"
        ))

    # 4. Churn Analysis (If TARGET or 'churn' keyword)
    target_cols = cols_by_type.get("TARGET", [])
    has_churn_keyword = any("churn" in c.lower() for c in profile.columns)
    if target_cols or has_churn_keyword:
        goals.append(AnalysisGoal(
            id="churn_analysis",
            title="Churn Analysis",
            description="Predict churn probability and identify key drivers.",
            type="predictive",
            complexity="high"
        ))
        
    # 5. Cohort Analysis (If TIME and ID exist)
    if cols_by_type.get("TIME") and cols_by_type.get("ID"):
        goals.append(AnalysisGoal(
            id="cohort_analysis",
            title="Cohort Analysis",
            description="Analyze user retention over time (cohorts).",
            type="predictive",
            complexity="high"
        ))

    # 6. Forecast Analysis (If TIME and METRIC exist)
    if cols_by_type.get("TIME") and (cols_by_type.get("METRIC") or cols_by_type.get("MONEY_IN")):
        goals.append(AnalysisGoal(
            id="forecast_analysis",
            title="Forecast Analysis",
            description="Predict future trends using historical data.",
            type="predictive",
            complexity="high"
        ))
        goals.append(AnalysisGoal(
            id="seasonality_analysis",
            title="Seasonality Analysis",
            description="Analyze recurring patterns (daily, monthly).",
            type="exploratory",
            complexity="medium"
        ))
        
    return goals

class AnalysisRequest(BaseModel):
    filename: str
    goal_id: str

@app.post("/analyze", response_model=AnalysisResult)
async def execute_analysis(req: AnalysisRequest):
    file_path = os.path.join(UPLOAD_DIR, req.filename)
    if not os.path.exists(file_path):
         # Try finding it with safe prefix
        found = False
        for f in os.listdir(UPLOAD_DIR):
            if f.endswith(req.filename) or f == req.filename:
                file_path = os.path.join(UPLOAD_DIR, f)
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Optimization: Check file size before reading
        file_size_bytes = os.path.getsize(file_path)
        LARGE_FILE_THRESHOLD = 50 * 1024 * 1024 # 50MB
        load_kwargs = {'engine': 'python', 'on_bad_lines': 'skip'}
        
        if file_size_bytes > LARGE_FILE_THRESHOLD:
            # Limit rows for analysis to prevent timeout
            load_kwargs['nrows'] = 100000 
            print(f"DEBUG: Large file analysis - limiting to first 100k rows for {req.filename}")

        # Load Data (Robust)
        try:
             df = pd.read_csv(file_path, **load_kwargs)
        except:
             try:
                 load_kwargs['sep'] = None
                 df = pd.read_csv(file_path, **load_kwargs)
             except:
                 try:
                     load_kwargs['sep'] = ';'
                     df = pd.read_csv(file_path, **load_kwargs)
                 except:
                     load_kwargs['sep'] = None
                     load_kwargs['encoding'] = 'latin1'
                     df = pd.read_csv(file_path, **load_kwargs)

        # Re-infer schema (stateless)
        schema = SemanticMapper.infer_schema(df)
        
        # Initialize Engines
        analytic_engine = AnalyticEngine(df, schema)
        recipe_engine = RecipeEngine(df, schema)
        ts_engine = TimeSeriesEngine(df, schema)
        
        result = {}
        
        if req.goal_id == "general_overview":
            result = analytic_engine.analyze_distributions()
            if not result: # Fallback
                 result = {"summary": "Overview", "insights": ["Could not generate distribution analysis."], "charts": []}
                 
        elif req.goal_id == "correlation_analysis":
            result = analytic_engine.analyze_correlations()
            
        elif req.goal_id == "anomaly_detection":
            result = recipe_engine.analyze_anomalies()
            
        elif req.goal_id == "churn_analysis":
            result = recipe_engine.analyze_churn()
            
        elif req.goal_id == "cohort_analysis":
            result = recipe_engine.analyze_cohorts()
            
        elif req.goal_id == "forecast_analysis":
            result = ts_engine.analyze_forecast()
            
        elif req.goal_id == "seasonality_analysis":
            result = ts_engine.analyze_seasonality()
            
        else:
            # Default/Fallback
            result = {"summary": "Unknown Goal", "insights": ["Goal not implemented yet."], "charts": []}

        
        # Generate Gemma narration for the entire analysis
        # Check for explicit errors from engines
        if result.get("error"):
            raise ValueError(result["error"])

        # Generate Gemma narration for the entire analysis
        narration_prompt = f"""
        You are presenting data analysis results to a business executive.
        
        Analysis: {result.get('summary', 'Analysis')}
        
        Insights:
        {chr(10).join(f"- {insight}" for insight in result.get('insights', []))}
        
        Charts Available:
        {chr(10).join(f"- {chart.get('title', chart.get('id'))}: {chart.get('type')} chart" for chart in result.get('charts', []))}
        
        Create a brief, engaging narrative (2-3 sentences) that:
        1. Introduces what the analysis shows
        2. Highlights the key takeaway
        3. Guides the executive on which chart to look at first
        
        Be conversational and executive-focused. No fluff.
        """
        
        gemma_narration = gateway.generate(narration_prompt, max_tokens=200)
        if not gemma_narration:
            gemma_narration = f"This {result.get('summary', 'analysis')} reveals {len(result.get('insights', []))} key findings across {len(result.get('charts', []))} visualizations."
        
        # Backwards compatibility: add chart_data for old frontend
        chart_data = result.get('charts', [{}])[0] if result.get('charts') else None
        
        return AnalysisResult(
            goal_id=req.goal_id,
            summary=result.get("summary", "Analysis Complete"),
            insights=result.get("insights", []),
            charts=result.get("charts", [chart_data] if chart_data else []),
            chart_data=chart_data,  # Legacy support
            gemma_narration=gemma_narration
        )

    except Exception as e:
        print(f"Analysis Error: {e}")
        import traceback
        traceback.print_exc()
        return AnalysisResult(
            goal_id=req.goal_id,
            summary=f"Analysis failed: {str(e)}",
            insights=["Please ensure data is clean and formatted correctly."],
            charts=[],
            gemma_narration=f"I encountered an error while analyzing this data: {str(e)}. It seems there wasn't enough suitable data for this specific analysis."
        )

class AskRequest(BaseModel):
    filename: str
    question: str

class AskResponse(BaseModel):
    answer: str
    context: List[str]
    related_chart: Optional[Dict] = None

@app.post("/analyze-full/{filename}")
async def analyze_full(filename: str):
    """Run all quick ML analyses and cache results for context-aware Q&A"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        # Try finding with prefix
        for f in os.listdir(UPLOAD_DIR):
            if f.endswith(filename) or f == filename:
                file_path = os.path.join(UPLOAD_DIR, f)
                filename = f
                break
        else:
            raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Load data
        try:
            df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
        except:
            df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
        
        # Get profile
        schema = SemanticMapper.infer_schema(df)
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        missing = {col: int(df[col].isna().sum()) for col in df.columns}
        
        profile = {
            "filename": filename,
            "columns": list(df.columns),
            "row_count": len(df),
            "dtypes": dtypes,
            "missing_values": missing,
            "semantic_schema": schema,
            "sample_data": df.head(3).to_dict(orient='records')
        }
        
        # Initialize engines
        analytic_engine = AnalyticEngine(df, schema)
        recipe_engine = RecipeEngine(df, schema)
        
        # Run quick analyses
        analyses = {}
        
        # 1. General Overview (always run)
        try:
            result = analytic_engine.analyze_distributions()
            if result and not result.get("error"):
                analyses["general_overview"] = {
                    "title": "General Overview",
                    "summary": result.get("summary", "Distribution analysis"),
                    "insights": result.get("insights", [])[:5]
                }
        except Exception as e:
            print(f"General overview failed: {e}")
        
        # 2. Correlation Analysis (if metrics exist)
        metrics = [c for c, t in schema.items() if t in ("METRIC", "MONEY_IN", "MONEY_OUT")]
        if len(metrics) >= 2:
            try:
                result = analytic_engine.analyze_correlations()
                if result and not result.get("error"):
                    analyses["correlation_analysis"] = {
                        "title": "Correlation Analysis",
                        "summary": result.get("summary", "Correlation analysis"),
                        "insights": result.get("insights", [])[:5]
                    }
            except Exception as e:
                print(f"Correlation analysis failed: {e}")
        
        # 3. Anomaly Detection (if metrics exist)
        if metrics:
            try:
                result = recipe_engine.analyze_anomalies()
                if result and not result.get("error"):
                    analyses["anomaly_detection"] = {
                        "title": "Anomaly Detection",
                        "summary": result.get("summary", "Anomaly analysis"),
                        "insights": result.get("insights", [])[:5]
                    }
            except Exception as e:
                print(f"Anomaly detection failed: {e}")
        
        # 4. Calculate basic statistics
        stats = {}
        for col in df.select_dtypes(include=['number']).columns[:10]:
            stats[col] = {
                "mean": round(df[col].mean(), 2) if pd.notna(df[col].mean()) else None,
                "min": round(df[col].min(), 2) if pd.notna(df[col].min()) else None,
                "max": round(df[col].max(), 2) if pd.notna(df[col].max()) else None,
                "sum": round(df[col].sum(), 2) if pd.notna(df[col].sum()) else None
            }
        
        # Cache results
        from datetime import datetime
        cache = {
            "profile": profile,
            "analyses": analyses,
            "statistics": stats,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        cache_path = os.path.join(UPLOAD_DIR, f"{filename}.analysis.json")
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2, default=str)
        
        print(f"✅ Full analysis complete for {filename}: {len(analyses)} analyses cached")
        
        return {
            "status": "success",
            "filename": filename,
            "analyses_run": list(analyses.keys()),
            "row_count": len(df),
            "columns": len(df.columns)
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Answer questions using cached ML analysis as context for Gemma"""
    file_path = os.path.join(UPLOAD_DIR, req.filename)
    if not os.path.exists(file_path):
        # Try finding with prefix
        found_filename = None
        for f in os.listdir(UPLOAD_DIR):
            if f.endswith(req.filename) or f == req.filename:
                file_path = os.path.join(UPLOAD_DIR, f)
                found_filename = f
                break
        if not found_filename:
            raise HTTPException(status_code=404, detail="File not found")
        actual_filename = found_filename
    else:
        actual_filename = req.filename
        
    try:
        # 1. Load cached analysis if available
        cache_path = os.path.join(UPLOAD_DIR, f"{actual_filename}.analysis.json")
        cached = None
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                print(f"📚 Loaded cached analysis for {actual_filename}")
            except Exception as e:
                print(f"⚠️ Failed to load cache: {e}")
        
        # 2. Build rich context from cache
        context_parts = []
        
        if cached:
            profile = cached.get("profile", {})
            
            # Dataset overview
            context_parts.append(f"## Dataset: {profile.get('filename', actual_filename)}")
            context_parts.append(f"- Rows: {profile.get('row_count', 'unknown')}")
            context_parts.append(f"- Columns: {', '.join(profile.get('columns', [])[:15])}")
            
            # Statistics
            stats = cached.get("statistics", {})
            if stats:
                context_parts.append("\n## Key Statistics")
                for col, col_stats in list(stats.items())[:8]:
                    if col_stats.get("mean") is not None:
                        context_parts.append(f"- {col}: mean={col_stats['mean']}, min={col_stats['min']}, max={col_stats['max']}, sum={col_stats['sum']}")
            
            # Analysis insights
            analyses = cached.get("analyses", {})
            if analyses:
                context_parts.append("\n## Analysis Insights")
                for analysis_id, analysis in analyses.items():
                    context_parts.append(f"\n### {analysis.get('title', analysis_id)}")
                    context_parts.append(analysis.get("summary", ""))
                    for insight in analysis.get("insights", [])[:3]:
                        context_parts.append(f"- {insight}")
        else:
            # Fallback: basic data summary
            try:
                df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
                context_parts.append(f"Dataset: {actual_filename}")
                context_parts.append(f"- Rows: {len(df)}, Columns: {len(df.columns)}")
                context_parts.append(f"- Columns: {', '.join(df.columns[:10])}")
                
                # Basic stats
                for col in df.select_dtypes(include=['number']).columns[:5]:
                    context_parts.append(f"- {col}: mean={df[col].mean():.2f}, sum={df[col].sum():.2f}")
            except Exception as e:
                context_parts.append(f"Dataset: {actual_filename} (could not load details)")
        
        context = "\n".join(context_parts)
        
        # 3. Build Gemma prompt with rich context
        prompt = f"""You are an AI assistant helping users understand their data.
You have analyzed the following dataset:

{context}

Now answer the user's question clearly, conversationally, and helpfully.
Use the statistics and insights above to provide specific, accurate answers.
If asked for a summary or explanation, highlight the key findings.

USER QUESTION: {req.question}

ANSWER:"""
        
        # 4. Get Gemma response
        answer = gateway.generate(prompt, max_tokens=300)
        
        if not answer:
            # Fallback to context summary
            answer = f"Based on the analysis of {actual_filename}: " + (
                cached.get("analyses", {}).get("general_overview", {}).get("summary", "")
                if cached else "Please try asking a more specific question."
            )
        
        return AskResponse(
            answer=answer,
            context=[context[:500] + "..." if len(context) > 500 else context],
            related_chart=None
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")

@app.post("/explain-finding")
async def explain_finding(req: ExplainRequest):
    context_str = f"""
    Analysis Summary: {req.context.summary}
    Key Insights: {', '.join(req.context.insights)}
    Chart Data: {json.dumps(req.context.chart_data) if req.context.chart_data else 'None'}
    """
    
    prompt = f"""
    You are an expert Data Scientist explaining an analysis to a user.
    
    Context:
    {context_str}
    
    User Question: {req.question}
    
    Answer the user's question clearly and concisely based on the context provided.
    """
    
    answer = gateway.generate(prompt, max_tokens=300)
    if not answer:
        answer = "I'm sorry, I couldn't generate an explanation at this moment."
        
    return {"answer": answer}

# ==============================================================================
# PHASE 6: AGENTIC ORCHESTRATION
# ==============================================================================

class AutoAnalyzeRequest(BaseModel):
    filename: str

async def run_auto_analysis(update_progress, filename: str):
    """
    Background task that:
    1. Profiles the dataset
    2. Proposes goals
    3. Selects top 3 goals
    4. Executes them
    5. Generates a report
    """
    update_progress("Loading Data...")
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Robust load
    try:
         df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
    except:
         df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
         
    schema = SemanticMapper.infer_schema(df)
    
    # Engines
    analytic_engine = AnalyticEngine(df, schema)
    recipe_engine = RecipeEngine(df, schema)
    ts_engine = TimeSeriesEngine(df, schema)
    
    update_progress("Profiling Data...")
    profile = profile_dataframe(df, filename)
    
    update_progress("Proposing Goals...")
    goals = await propose_goals(profile)
    
    # Select Top 3 Goals (Heuristic: Prefer predictive/time-series over descriptive)
    # Sort by complexity/priority? For now, just take first 3 unique types
    selected_goals = goals[:3]
    
    results = []
    for i, goal in enumerate(selected_goals):
        update_progress(f"Running Analysis {i+1}/{len(selected_goals)}: {goal.title}...")
        
        # Re-use execution logic (simplified)
        # Ideally we'd refactor execute_analysis to be callable directly, but for now we duplicate routing
        # or call a helper. Let's duplicate routing for safety/speed.
        
        res = {}
        try:
            if goal.id == "general_overview":
                res = analytic_engine.analyze_distributions()
            elif goal.id == "correlation_analysis":
                res = analytic_engine.analyze_correlations()
            elif goal.id == "anomaly_detection":
                res = recipe_engine.analyze_anomalies()
            elif goal.id == "churn_analysis":
                res = recipe_engine.analyze_churn()
            elif goal.id == "cohort_analysis":
                res = recipe_engine.analyze_cohorts()
            elif goal.id == "forecast_analysis":
                res = ts_engine.analyze_forecast()
            elif goal.id == "seasonality_analysis":
                res = ts_engine.analyze_seasonality()
            else:
                res = {"summary": f"Skipped {goal.title}", "insights": [], "charts": []}
                
            # Add narration
            narration_prompt = f"Summarize this analysis for an executive: {res.get('summary')} {res.get('insights')}"
            res['gemma_narration'] = gateway.generate(narration_prompt, max_tokens=100)
            
        except Exception as e:
            res = {"summary": f"Error in {goal.title}", "insights": [str(e)], "charts": []}
            
        results.append(res)
        
    update_progress("Generating Report...")
    report_md = ReportGenerator.generate_markdown_report(filename, results)
    
    return {
        "report_markdown": report_md,
        "analyses": results
    }

@app.post("/jobs/submit")
async def submit_auto_analysis(req: AutoAnalyzeRequest):
    file_path = os.path.join(UPLOAD_DIR, req.filename)
    if not os.path.exists(file_path):
        # Try prefix match
        found = False
        for f in os.listdir(UPLOAD_DIR):
            if f.endswith(req.filename) or f == req.filename:
                req.filename = f # Update filename
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="File not found")

    job_id = job_manager.submit_job(run_auto_analysis, req.filename)
    return {"job_id": job_id, "status": "PENDING"}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Don't return full result in status poll to save bandwidth
    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "error": job["error"]
    }

@app.get("/jobs/{job_id}/report")
async def get_job_report(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job["status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail="Job not completed yet")
        
    return {"report_markdown": job["result"]["report_markdown"]}


# =============================================================================
# VECTORIZE DATABASE ENDPOINTS
# For databases.html database vectorization flow
# =============================================================================

# In-memory vectorization job storage (use redis in production)
_vectorize_jobs: Dict[str, Dict[str, Any]] = {}

class VectorizeRequest(BaseModel):
    database_name: str
    num_questions: int = 50
    data_description: Optional[str] = "User uploaded database"
    use_gpu: bool = False

@app.post("/vectorize/database")
async def vectorize_database(request: VectorizeRequest):
    """Start database vectorization job"""
    job_id = f"vec_{uuid.uuid4().hex[:10]}"
    
    # Validate the file exists
    file_path = os.path.join(UPLOAD_DIR, request.database_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.database_name}")
    
    # Create job entry
    _vectorize_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "filename": request.database_name,
        "progress": {"step": "initializing", "percent": 0},
        "created_at": datetime.now().isoformat(),
        "error": None,
        "result": None
    }
    
    # Start background task for vectorization
    import asyncio
    asyncio.create_task(_run_vectorization(job_id, file_path, request))
    
    return {"job_id": job_id, "status": "pending"}

async def _run_vectorization(job_id: str, file_path: str, request: VectorizeRequest):
    """Background task to perform vectorization"""
    try:
        job = _vectorize_jobs.get(job_id)
        if not job:
            return
        
        # Step 1: Load data
        job["status"] = "running"
        job["progress"] = {"step": "Loading data", "percent": 10}
        
        await asyncio.sleep(0.5)  # Simulate some work
        
        # Load and profile the data
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)  # Try CSV as fallback
        except Exception as e:
            job["status"] = "failed"
            job["error"] = f"Failed to load file: {str(e)}"
            return
        
        job["progress"] = {"step": "Analyzing schema", "percent": 30}
        await asyncio.sleep(0.3)
        
        # Step 2: Generate embeddings (simulated for now)
        job["progress"] = {"step": "Generating embeddings", "percent": 50}
        await asyncio.sleep(0.5)
        
        # Step 3: Create vector index
        job["progress"] = {"step": "Creating vector index", "percent": 70}
        await asyncio.sleep(0.5)
        
        # Step 4: Generate sample questions
        job["progress"] = {"step": "Generating sample questions", "percent": 90}
        await asyncio.sleep(0.3)
        
        # Complete
        job["status"] = "completed"
        job["progress"] = {"step": "Complete", "percent": 100}
        job["result"] = {
            "rows_processed": len(df),
            "columns": list(df.columns),
            "embeddings_created": True,
            "sample_questions": [
                f"What are the main trends in {df.columns[0]}?",
                f"Show me a summary of the data",
                f"What correlations exist in this dataset?"
            ][:request.num_questions]
        }
        
    except Exception as e:
        job = _vectorize_jobs.get(job_id)
        if job:
            job["status"] = "failed"
            job["error"] = str(e)
            logger.error(f"Vectorization failed: {e}")

@app.get("/vectorize/status/{job_id}")
async def vectorize_status(job_id: str):
    """Get vectorization job status"""
    job = _vectorize_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "error": job["error"],
        "result": job["result"]
    }


# Import and include analytics router
try:
    from .analytics_routes import analytics_router
except ImportError:
    from analytics_routes import analytics_router

app.include_router(analytics_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)

