"""
S3-Compatible Connector - AWS S3 and MinIO storage integration.

Provides secure object storage access with presigned URLs and
streaming document retrieval for RAG ingestion.

Configuration:
- S3_ENDPOINT_URL: S3/MinIO endpoint (default: AWS S3)
- S3_ACCESS_KEY_FILE: Path to access key (Docker secret)
- S3_SECRET_KEY_FILE: Path to secret key (Docker secret)
- S3_REGION: AWS region (default: us-east-1)
- S3_USE_SSL: Whether to use HTTPS (default: True)

Usage:
    from shared.clients.connectors.s3 import S3Connector, S3Config
    
    config = S3Config(bucket="my-bucket", prefix="documents/")
    connector = S3Connector(config)
    
    async with connector:
        documents = await connector.fetch(limit=100)
        
        # Or stream for large datasets
        async for doc in connector.fetch_stream():
            process(doc)
            
        # Generate presigned upload URL
        url = await connector.generate_presigned_upload("report.pdf")
"""

import asyncio
import hashlib
import io
import logging
import mimetypes
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Optional
from uuid import UUID, uuid4

try:
    from .base import BaseConnector, ConnectorConfig, ConnectorStatus, Document, DocumentType
except ImportError:
    from base import BaseConnector, ConnectorConfig, ConnectorStatus, Document, DocumentType

logger = logging.getLogger(__name__)


# File type detection via magic bytes
MAGIC_BYTES = {
    b"%PDF": DocumentType.PDF,
    b"PK": DocumentType.DOCX,  # Also ZIP, XLSX, PPTX
    b"\xff\xd8\xff": DocumentType.IMAGE,  # JPEG
    b"\x89PNG": DocumentType.IMAGE,
    b"GIF8": DocumentType.IMAGE,
    b"<!DOCTYPE html": DocumentType.HTML,
    b"<html": DocumentType.HTML,
    b"# ": DocumentType.MARKDOWN,
    b"## ": DocumentType.MARKDOWN,
}

# Dangerous file extensions to block
BLOCKED_EXTENSIONS = {
    ".exe", ".dll", ".bat", ".cmd", ".ps1", ".sh", ".vbs", ".js",
    ".msi", ".scr", ".com", ".pif", ".jar", ".app",
}

# Maximum file sizes per type (bytes)
MAX_FILE_SIZES = {
    DocumentType.PDF: 100 * 1024 * 1024,  # 100MB
    DocumentType.IMAGE: 20 * 1024 * 1024,  # 20MB
    DocumentType.CSV: 500 * 1024 * 1024,  # 500MB
    DocumentType.JSON: 100 * 1024 * 1024,  # 100MB
    DocumentType.TEXT: 50 * 1024 * 1024,  # 50MB
    "default": 50 * 1024 * 1024,  # 50MB
}


def _load_secret(path: str | None, env_fallback: str | None = None) -> str | None:
    """Load secret from file or environment."""
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    if env_fallback:
        return os.getenv(env_fallback)
    return None


def _detect_document_type(filename: str, content: bytes | None = None) -> DocumentType:
    """Detect document type from filename and optional magic bytes."""
    # First check magic bytes if content provided
    if content:
        for magic, doc_type in MAGIC_BYTES.items():
            if content.startswith(magic):
                return doc_type
    
    # Fall back to extension
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


def _validate_file_extension(filename: str) -> tuple[bool, str | None]:
    """Validate file extension is not blocked."""
    ext = Path(filename).suffix.lower()
    if ext in BLOCKED_EXTENSIONS:
        return False, f"File type not allowed: {ext}"
    return True, None


def _validate_file_size(size: int, doc_type: DocumentType) -> tuple[bool, str | None]:
    """Validate file size against type-specific limits."""
    max_size = MAX_FILE_SIZES.get(doc_type, MAX_FILE_SIZES["default"])
    if size > max_size:
        max_mb = max_size / (1024 * 1024)
        return False, f"File exceeds maximum size of {max_mb:.1f}MB for {doc_type.value}"
    return True, None


def _compute_content_hash(content: bytes) -> str:
    """Compute SHA256 hash of content for integrity verification."""
    return hashlib.sha256(content).hexdigest()


@dataclass
class S3Config:
    """S3 connector configuration."""
    bucket: str
    prefix: str = ""
    region: str = "us-east-1"
    endpoint_url: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    use_ssl: bool = True
    signature_version: str = "s3v4"
    tenant_id: UUID | None = None
    
    @classmethod
    def from_environment(cls, bucket: str, prefix: str = "") -> "S3Config":
        """Create config from environment variables."""
        return cls(
            bucket=bucket,
            prefix=prefix,
            region=os.getenv("S3_REGION", "us-east-1"),
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            access_key=_load_secret(
                os.getenv("S3_ACCESS_KEY_FILE"),
                "S3_ACCESS_KEY"
            ),
            secret_key=_load_secret(
                os.getenv("S3_SECRET_KEY_FILE"),
                "S3_SECRET_KEY"
            ),
            use_ssl=os.getenv("S3_USE_SSL", "true").lower() == "true",
        )


class S3Connector(BaseConnector):
    """
    S3-compatible object storage connector.
    
    Supports AWS S3, MinIO, and other S3-compatible services.
    Provides secure file upload/download with validation.
    """
    
    def __init__(self, s3_config: S3Config):
        """
        Initialize S3 connector.
        
        Args:
            s3_config: S3-specific configuration
        """
        connector_config = ConnectorConfig(
            name=f"s3-{s3_config.bucket}",
            connector_type="s3",
            config={"bucket": s3_config.bucket, "prefix": s3_config.prefix},
            tenant_id=s3_config.tenant_id,
        )
        super().__init__(connector_config)
        self.s3_config = s3_config
        self._client: Any = None
    
    async def connect(self) -> bool:
        """Establish connection to S3."""
        try:
            # Import boto3 lazily to avoid hard dependency
            import boto3
            from botocore.config import Config as BotoConfig
            
            boto_config = BotoConfig(
                region_name=self.s3_config.region,
                signature_version=self.s3_config.signature_version,
                retries={"max_attempts": 3, "mode": "standard"},
            )
            
            self._client = boto3.client(
                "s3",
                endpoint_url=self.s3_config.endpoint_url,
                aws_access_key_id=self.s3_config.access_key,
                aws_secret_access_key=self.s3_config.secret_key,
                config=boto_config,
                use_ssl=self.s3_config.use_ssl,
            )
            
            # Verify bucket access
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._client.head_bucket(Bucket=self.s3_config.bucket)
            )
            
            self.status = ConnectorStatus.CONNECTED
            self._connected = True
            logger.info(f"Connected to S3 bucket: {self.s3_config.bucket}")
            return True
            
        except ImportError:
            self._error = "boto3 library not installed"
            self.status = ConnectorStatus.ERROR
            logger.error(self._error)
            return False
        except Exception as e:
            self._error = str(e)
            self.status = ConnectorStatus.ERROR
            logger.error(f"S3 connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close S3 connection."""
        self._client = None
        self.status = ConnectorStatus.DISCONNECTED
        self._connected = False
        return True
    
    async def validate(self) -> dict[str, Any]:
        """Validate S3 configuration and access."""
        errors = []
        
        if not self.s3_config.bucket:
            errors.append("Bucket name is required")
        if not self.s3_config.access_key:
            errors.append("Access key not configured")
        if not self.s3_config.secret_key:
            errors.append("Secret key not configured")
        
        if not errors and self._client:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self._client.head_bucket(Bucket=self.s3_config.bucket)
                )
            except Exception as e:
                errors.append(f"Bucket access failed: {e}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "bucket": self.s3_config.bucket,
            "prefix": self.s3_config.prefix,
        }
    
    async def fetch(
        self,
        query: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """
        Fetch documents from S3 bucket.
        
        Args:
            query: Optional query with keys: prefix, suffix, modified_after
            limit: Maximum documents to fetch
        """
        if not self._client:
            raise RuntimeError("Not connected to S3")
        
        query = query or {}
        prefix = query.get("prefix", self.s3_config.prefix)
        suffix = query.get("suffix")
        offset = query.get("offset", 0)
        
        loop = asyncio.get_event_loop()
        
        # List objects
        paginator = self._client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.s3_config.bucket,
            Prefix=prefix,
            PaginationConfig={"MaxItems": offset + limit},
        )
        
        objects = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                
                # Skip directories
                if key.endswith("/"):
                    continue
                
                # Apply suffix filter
                if suffix and not key.endswith(suffix):
                    continue
                
                # Validate extension
                valid, error = _validate_file_extension(key)
                if not valid:
                    logger.warning(f"Skipping blocked file: {key}")
                    continue
                
                objects.append(obj)
        
        # Apply offset
        objects = objects[offset:offset + limit]
        
        # Fetch document contents
        documents = []
        for obj in objects:
            try:
                doc = await self._fetch_object(obj["Key"], obj)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to fetch {obj['Key']}: {e}")
        
        return documents
    
    async def _fetch_object(self, key: str, metadata: dict) -> Document | None:
        """Fetch and parse a single S3 object."""
        loop = asyncio.get_event_loop()
        
        # Get object
        response = await loop.run_in_executor(
            None,
            lambda: self._client.get_object(
                Bucket=self.s3_config.bucket,
                Key=key,
            )
        )
        
        content = response["Body"].read()
        
        # Validate size
        doc_type = _detect_document_type(key, content[:16])
        valid, error = _validate_file_size(len(content), doc_type)
        if not valid:
            logger.warning(f"Skipping oversized file {key}: {error}")
            return None
        
        # Decode text content
        if doc_type in (DocumentType.TEXT, DocumentType.CSV, DocumentType.JSON,
                        DocumentType.HTML, DocumentType.MARKDOWN):
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                text_content = content.decode("latin-1", errors="replace")
        else:
            # For binary files, store base64 or reference
            text_content = f"[Binary content: {len(content)} bytes]"
        
        return Document(
            id=uuid4(),
            content=text_content,
            doc_type=doc_type,
            source_path=f"s3://{self.s3_config.bucket}/{key}",
            metadata={
                "bucket": self.s3_config.bucket,
                "key": key,
                "size": metadata.get("Size"),
                "last_modified": metadata.get("LastModified", "").isoformat() 
                    if hasattr(metadata.get("LastModified", ""), "isoformat") else str(metadata.get("LastModified")),
                "etag": metadata.get("ETag", "").strip('"'),
                "content_hash": _compute_content_hash(content),
            },
            created_at=datetime.now(timezone.utc),
        )
    
    async def generate_presigned_upload(
        self,
        filename: str,
        expires_in: int = 3600,
        max_size: int | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate presigned URL for secure upload.
        
        Args:
            filename: Target filename (will be prefixed)
            expires_in: URL expiration in seconds
            max_size: Maximum file size allowed
            content_type: Required content type
            
        Returns:
            Dict with url, fields, and upload instructions
        """
        if not self._client:
            raise RuntimeError("Not connected to S3")
        
        # Validate filename
        valid, error = _validate_file_extension(filename)
        if not valid:
            raise ValueError(error)
        
        # Build key with prefix
        key = f"{self.s3_config.prefix}{filename}".lstrip("/")
        
        # Determine content type
        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"
        
        # Build conditions
        conditions = [
            {"bucket": self.s3_config.bucket},
            {"key": key},
            {"Content-Type": content_type},
        ]
        
        if max_size:
            conditions.append(["content-length-range", 0, max_size])
        
        loop = asyncio.get_event_loop()
        presigned = await loop.run_in_executor(
            None,
            lambda: self._client.generate_presigned_post(
                Bucket=self.s3_config.bucket,
                Key=key,
                Conditions=conditions,
                ExpiresIn=expires_in,
            )
        )
        
        logger.info(f"Generated presigned upload URL for: {key}")
        
        return {
            "url": presigned["url"],
            "fields": presigned["fields"],
            "key": key,
            "expires_in": expires_in,
            "content_type": content_type,
            "max_size": max_size,
        }
    
    async def generate_presigned_download(
        self,
        key: str,
        expires_in: int = 3600,
    ) -> str:
        """
        Generate presigned URL for secure download.
        
        Args:
            key: Object key to download
            expires_in: URL expiration in seconds
            
        Returns:
            Presigned download URL
        """
        if not self._client:
            raise RuntimeError("Not connected to S3")
        
        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(
            None,
            lambda: self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.s3_config.bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        )
        
        return url
    
    async def upload(
        self,
        filename: str,
        content: bytes | BinaryIO,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Upload file directly to S3.
        
        Args:
            filename: Target filename
            content: File content as bytes or file-like object
            content_type: MIME type
            
        Returns:
            Upload result with key, etag, size
        """
        if not self._client:
            raise RuntimeError("Not connected to S3")
        
        # Validate filename
        valid, error = _validate_file_extension(filename)
        if not valid:
            raise ValueError(error)
        
        # Handle bytes vs file-like
        if isinstance(content, bytes):
            content_bytes = content
            body = io.BytesIO(content)
        else:
            content.seek(0)
            content_bytes = content.read()
            body = io.BytesIO(content_bytes)
        
        # Validate size
        doc_type = _detect_document_type(filename, content_bytes[:16])
        valid, error = _validate_file_size(len(content_bytes), doc_type)
        if not valid:
            raise ValueError(error)
        
        # Determine content type
        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"
        
        key = f"{self.s3_config.prefix}{filename}".lstrip("/")
        content_hash = _compute_content_hash(content_bytes)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.put_object(
                Bucket=self.s3_config.bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
                Metadata={"sha256": content_hash},
            )
        )
        
        logger.info(f"Uploaded to S3: {key} ({len(content_bytes)} bytes)")
        
        return {
            "key": key,
            "bucket": self.s3_config.bucket,
            "etag": response.get("ETag", "").strip('"'),
            "size": len(content_bytes),
            "content_hash": content_hash,
            "content_type": content_type,
        }
    
    async def delete(self, key: str) -> bool:
        """
        Delete object from S3.
        
        Args:
            key: Object key to delete
            
        Returns:
            True if deletion successful
        """
        if not self._client:
            raise RuntimeError("Not connected to S3")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._client.delete_object(
                Bucket=self.s3_config.bucket,
                Key=key,
            )
        )
        
        logger.info(f"Deleted from S3: {key}")
        return True
