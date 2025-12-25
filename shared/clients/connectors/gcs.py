"""
Google Cloud Storage Connector - GCS integration for RAG ingestion.

Provides secure object storage access with signed URLs and
streaming document retrieval for enterprise data pipelines.

Configuration:
- GCS_PROJECT_ID: Google Cloud project ID
- GCS_CREDENTIALS_FILE: Path to service account JSON (Docker secret)
- GCS_LOCATION: Default bucket location (default: us)

Usage:
    from shared.clients.connectors.gcs import GCSConnector, GCSConfig
    
    config = GCSConfig(bucket="my-bucket", prefix="documents/")
    connector = GCSConnector(config)
    
    async with connector:
        documents = await connector.fetch(limit=100)
        
        # Generate signed upload URL
        url = await connector.generate_signed_upload("report.pdf")
"""

import asyncio
import hashlib
import io
import logging
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO

try:
    from .base import BaseConnector, ConnectorConfig, ConnectorStatus, Document, DocumentType
except ImportError:
    from base import BaseConnector, ConnectorConfig, ConnectorStatus, Document, DocumentType

logger = logging.getLogger(__name__)


# Blocked extensions (same as S3 connector for consistency)
BLOCKED_EXTENSIONS = frozenset({
    ".exe", ".dll", ".bat", ".cmd", ".ps1", ".sh", ".vbs", ".js",
    ".msi", ".scr", ".com", ".pif", ".jar", ".app",
    ".php", ".asp", ".aspx", ".jsp", ".cgi",
})

# Max sizes per type
MAX_FILE_SIZES = {
    DocumentType.PDF: 100 * 1024 * 1024,
    DocumentType.IMAGE: 20 * 1024 * 1024,
    DocumentType.CSV: 500 * 1024 * 1024,
    DocumentType.JSON: 100 * 1024 * 1024,
    DocumentType.TEXT: 50 * 1024 * 1024,
    "default": 50 * 1024 * 1024,
}


def _detect_document_type(filename: str) -> DocumentType:
    """Detect document type from filename extension."""
    ext = Path(filename).suffix.lower()
    ext_map = {
        ".pdf": DocumentType.PDF,
        ".csv": DocumentType.CSV,
        ".json": DocumentType.JSON,
        ".html": DocumentType.HTML,
        ".htm": DocumentType.HTML,
        ".md": DocumentType.MARKDOWN,
        ".markdown": DocumentType.MARKDOWN,
        ".txt": DocumentType.TEXT,
        ".docx": DocumentType.DOCX,
        ".xlsx": DocumentType.XLSX,
        ".jpg": DocumentType.IMAGE,
        ".jpeg": DocumentType.IMAGE,
        ".png": DocumentType.IMAGE,
        ".gif": DocumentType.IMAGE,
    }
    return ext_map.get(ext, DocumentType.TEXT)


def _validate_extension(filename: str) -> tuple[bool, str | None]:
    """Validate file extension is not blocked."""
    ext = Path(filename).suffix.lower()
    if ext in BLOCKED_EXTENSIONS:
        return False, f"File type not allowed: {ext}"
    return True, None


def _compute_hash(content: bytes) -> str:
    """Compute SHA256 hash."""
    return hashlib.sha256(content).hexdigest()


@dataclass
class GCSConfig:
    """GCS connector configuration."""
    bucket: str
    prefix: str = ""
    project_id: str | None = None
    credentials_file: str | None = None
    location: str = "us"
    
    @classmethod
    def from_environment(cls, bucket: str, prefix: str = "") -> "GCSConfig":
        """Create config from environment variables."""
        return cls(
            bucket=bucket,
            prefix=prefix,
            project_id=os.getenv("GCS_PROJECT_ID"),
            credentials_file=os.getenv("GCS_CREDENTIALS_FILE"),
            location=os.getenv("GCS_LOCATION", "us"),
        )


class GCSConnector(BaseConnector):
    """
    Google Cloud Storage connector.
    
    Provides secure file storage with signed URLs and streaming retrieval.
    """
    
    def __init__(self, gcs_config: GCSConfig):
        """Initialize GCS connector."""
        connector_config = ConnectorConfig(
            name=f"gcs-{gcs_config.bucket}",
            connector_type="gcs",
            config={"bucket": gcs_config.bucket, "prefix": gcs_config.prefix},
        )
        super().__init__(connector_config)
        self.gcs_config = gcs_config
        self._client: Any = None
        self._bucket: Any = None
    
    async def connect(self) -> bool:
        """Establish connection to GCS."""
        try:
            from google.cloud import storage
            from google.oauth2 import service_account
            
            # Load credentials
            if self.gcs_config.credentials_file and os.path.exists(self.gcs_config.credentials_file):
                credentials = service_account.Credentials.from_service_account_file(
                    self.gcs_config.credentials_file
                )
                self._client = storage.Client(
                    project=self.gcs_config.project_id,
                    credentials=credentials,
                )
            else:
                # Use default credentials (ADC)
                self._client = storage.Client(project=self.gcs_config.project_id)
            
            # Get bucket reference
            self._bucket = self._client.bucket(self.gcs_config.bucket)
            
            # Verify bucket exists
            loop = asyncio.get_event_loop()
            exists = await loop.run_in_executor(None, self._bucket.exists)
            
            if not exists:
                raise ValueError(f"Bucket {self.gcs_config.bucket} does not exist")
            
            self.status = ConnectorStatus.CONNECTED
            self._connected = True
            logger.info(f"Connected to GCS bucket: {self.gcs_config.bucket}")
            return True
            
        except ImportError:
            self._error = "google-cloud-storage library not installed"
            self.status = ConnectorStatus.ERROR
            logger.error(self._error)
            return False
        except Exception as e:
            self._error = str(e)
            self.status = ConnectorStatus.ERROR
            logger.error(f"GCS connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close GCS connection."""
        self._client = None
        self._bucket = None
        self.status = ConnectorStatus.DISCONNECTED
        self._connected = False
        return True
    
    async def validate(self) -> dict[str, Any]:
        """Validate GCS configuration."""
        errors = []
        
        if not self.gcs_config.bucket:
            errors.append("Bucket name is required")
        
        if self._bucket:
            try:
                loop = asyncio.get_event_loop()
                exists = await loop.run_in_executor(None, self._bucket.exists)
                if not exists:
                    errors.append(f"Bucket {self.gcs_config.bucket} does not exist")
            except Exception as e:
                errors.append(f"Bucket validation failed: {e}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "bucket": self.gcs_config.bucket,
            "prefix": self.gcs_config.prefix,
        }
    
    async def fetch(
        self,
        query: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """Fetch documents from GCS bucket."""
        if not self._bucket:
            raise RuntimeError("Not connected to GCS")
        
        query = query or {}
        prefix = query.get("prefix", self.gcs_config.prefix)
        suffix = query.get("suffix")
        offset = query.get("offset", 0)
        
        loop = asyncio.get_event_loop()
        
        # List blobs
        blobs_iter = await loop.run_in_executor(
            None,
            lambda: list(self._bucket.list_blobs(prefix=prefix, max_results=offset + limit))
        )
        
        # Filter and apply offset
        blobs = []
        for blob in blobs_iter:
            # Skip directories
            if blob.name.endswith("/"):
                continue
            
            # Apply suffix filter
            if suffix and not blob.name.endswith(suffix):
                continue
            
            # Validate extension
            valid, _ = _validate_extension(blob.name)
            if not valid:
                logger.warning(f"Skipping blocked file: {blob.name}")
                continue
            
            blobs.append(blob)
        
        blobs = blobs[offset:offset + limit]
        
        # Fetch document contents
        documents = []
        for blob in blobs:
            try:
                doc = await self._fetch_blob(blob)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to fetch {blob.name}: {e}")
        
        return documents
    
    async def _fetch_blob(self, blob: Any) -> Document | None:
        """Fetch and parse a single GCS blob."""
        loop = asyncio.get_event_loop()
        
        content = await loop.run_in_executor(None, blob.download_as_bytes)
        
        doc_type = _detect_document_type(blob.name)
        max_size = MAX_FILE_SIZES.get(doc_type, MAX_FILE_SIZES["default"])
        
        if len(content) > max_size:
            logger.warning(f"Skipping oversized file {blob.name}")
            return None
        
        # Decode text content
        if doc_type in (DocumentType.TEXT, DocumentType.CSV, DocumentType.JSON,
                        DocumentType.HTML, DocumentType.MARKDOWN):
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                text_content = content.decode("latin-1", errors="replace")
        else:
            text_content = f"[Binary content: {len(content)} bytes]"
        
        from uuid import uuid4
        return Document(
            id=uuid4(),
            content=text_content,
            doc_type=doc_type,
            source_path=f"gs://{self.gcs_config.bucket}/{blob.name}",
            metadata={
                "bucket": self.gcs_config.bucket,
                "name": blob.name,
                "size": blob.size,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "md5_hash": blob.md5_hash,
                "content_hash": _compute_hash(content),
            },
            created_at=datetime.now(timezone.utc),
        )
    
    async def generate_signed_upload(
        self,
        filename: str,
        expires_in: int = 3600,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """Generate signed URL for upload."""
        if not self._bucket:
            raise RuntimeError("Not connected to GCS")
        
        valid, error = _validate_extension(filename)
        if not valid:
            raise ValueError(error)
        
        blob_name = f"{self.gcs_config.prefix}{filename}".lstrip("/")
        blob = self._bucket.blob(blob_name)
        
        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"
        
        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(
            None,
            lambda: blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expires_in),
                method="PUT",
                content_type=content_type,
            )
        )
        
        logger.info(f"Generated signed upload URL for: {blob_name}")
        
        return {
            "url": url,
            "blob_name": blob_name,
            "expires_in": expires_in,
            "content_type": content_type,
            "method": "PUT",
        }
    
    async def generate_signed_download(
        self,
        blob_name: str,
        expires_in: int = 3600,
    ) -> str:
        """Generate signed URL for download."""
        if not self._bucket:
            raise RuntimeError("Not connected to GCS")
        
        blob = self._bucket.blob(blob_name)
        
        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(
            None,
            lambda: blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expires_in),
                method="GET",
            )
        )
        
        return url
    
    async def upload(
        self,
        filename: str,
        content: bytes | BinaryIO,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """Upload file to GCS."""
        if not self._bucket:
            raise RuntimeError("Not connected to GCS")
        
        valid, error = _validate_extension(filename)
        if not valid:
            raise ValueError(error)
        
        if isinstance(content, bytes):
            content_bytes = content
        else:
            content.seek(0)
            content_bytes = content.read()
        
        doc_type = _detect_document_type(filename)
        max_size = MAX_FILE_SIZES.get(doc_type, MAX_FILE_SIZES["default"])
        if len(content_bytes) > max_size:
            raise ValueError(f"File exceeds maximum size of {max_size / (1024*1024):.1f}MB")
        
        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"
        
        blob_name = f"{self.gcs_config.prefix}{filename}".lstrip("/")
        blob = self._bucket.blob(blob_name)
        content_hash = _compute_hash(content_bytes)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: blob.upload_from_string(
                content_bytes,
                content_type=content_type,
            )
        )
        
        logger.info(f"Uploaded to GCS: {blob_name} ({len(content_bytes)} bytes)")
        
        return {
            "blob_name": blob_name,
            "bucket": self.gcs_config.bucket,
            "size": len(content_bytes),
            "content_hash": content_hash,
            "content_type": content_type,
        }
    
    async def delete(self, blob_name: str) -> bool:
        """Delete blob from GCS."""
        if not self._bucket:
            raise RuntimeError("Not connected to GCS")
        
        blob = self._bucket.blob(blob_name)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, blob.delete)
        
        logger.info(f"Deleted from GCS: {blob_name}")
        return True
