"""
Azure Blob Storage Connector - Azure integration for RAG ingestion.

Provides secure object storage access with SAS tokens and
streaming document retrieval for enterprise data pipelines.

Configuration:
- AZURE_STORAGE_ACCOUNT: Storage account name
- AZURE_STORAGE_KEY_FILE: Path to access key (Docker secret)
- AZURE_STORAGE_CONNECTION_STRING_FILE: Alternative connection string

Usage:
    from shared.clients.connectors.azure_blob import AzureBlobConnector, AzureBlobConfig
    
    config = AzureBlobConfig(container="documents", prefix="uploads/")
    connector = AzureBlobConnector(config)
    
    async with connector:
        documents = await connector.fetch(limit=100)
        
        # Generate SAS URL for upload
        url = await connector.generate_sas_upload("report.pdf")
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


# Blocked extensions
BLOCKED_EXTENSIONS = frozenset({
    ".exe", ".dll", ".bat", ".cmd", ".ps1", ".sh", ".vbs", ".js",
    ".msi", ".scr", ".com", ".pif", ".jar", ".app",
    ".php", ".asp", ".aspx", ".jsp", ".cgi",
})

MAX_FILE_SIZES = {
    DocumentType.PDF: 100 * 1024 * 1024,
    DocumentType.IMAGE: 20 * 1024 * 1024,
    DocumentType.CSV: 500 * 1024 * 1024,
    DocumentType.JSON: 100 * 1024 * 1024,
    DocumentType.TEXT: 50 * 1024 * 1024,
    "default": 50 * 1024 * 1024,
}


def _load_secret(path: str | None) -> str | None:
    """Load secret from file."""
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return None


def _detect_document_type(filename: str) -> DocumentType:
    """Detect document type from filename."""
    ext = Path(filename).suffix.lower()
    ext_map = {
        ".pdf": DocumentType.PDF,
        ".csv": DocumentType.CSV,
        ".json": DocumentType.JSON,
        ".html": DocumentType.HTML,
        ".md": DocumentType.MARKDOWN,
        ".txt": DocumentType.TEXT,
        ".docx": DocumentType.DOCX,
        ".xlsx": DocumentType.XLSX,
        ".jpg": DocumentType.IMAGE,
        ".jpeg": DocumentType.IMAGE,
        ".png": DocumentType.IMAGE,
    }
    return ext_map.get(ext, DocumentType.TEXT)


def _validate_extension(filename: str) -> tuple[bool, str | None]:
    """Validate file extension."""
    ext = Path(filename).suffix.lower()
    if ext in BLOCKED_EXTENSIONS:
        return False, f"File type not allowed: {ext}"
    return True, None


def _compute_hash(content: bytes) -> str:
    """Compute SHA256 hash."""
    return hashlib.sha256(content).hexdigest()


@dataclass
class AzureBlobConfig:
    """Azure Blob Storage configuration."""
    container: str
    prefix: str = ""
    account_name: str | None = None
    account_key: str | None = None
    connection_string: str | None = None
    
    @classmethod
    def from_environment(cls, container: str, prefix: str = "") -> "AzureBlobConfig":
        """Create config from environment variables."""
        return cls(
            container=container,
            prefix=prefix,
            account_name=os.getenv("AZURE_STORAGE_ACCOUNT"),
            account_key=_load_secret(os.getenv("AZURE_STORAGE_KEY_FILE")),
            connection_string=_load_secret(os.getenv("AZURE_STORAGE_CONNECTION_STRING_FILE")),
        )


class AzureBlobConnector(BaseConnector):
    """
    Azure Blob Storage connector.
    
    Provides secure file storage with SAS tokens and streaming retrieval.
    """
    
    def __init__(self, config: AzureBlobConfig):
        """Initialize Azure Blob connector."""
        connector_config = ConnectorConfig(
            name=f"azure-{config.container}",
            connector_type="azure_blob",
            config={"container": config.container, "prefix": config.prefix},
        )
        super().__init__(connector_config)
        self.azure_config = config
        self._client: Any = None
        self._container_client: Any = None
    
    async def connect(self) -> bool:
        """Establish connection to Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
            
            # Create client from connection string or account key
            if self.azure_config.connection_string:
                self._client = BlobServiceClient.from_connection_string(
                    self.azure_config.connection_string
                )
            elif self.azure_config.account_name and self.azure_config.account_key:
                account_url = f"https://{self.azure_config.account_name}.blob.core.windows.net"
                self._client = BlobServiceClient(
                    account_url=account_url,
                    credential=self.azure_config.account_key,
                )
            else:
                # Use DefaultAzureCredential (managed identity, etc.)
                from azure.identity import DefaultAzureCredential
                account_url = f"https://{self.azure_config.account_name}.blob.core.windows.net"
                self._client = BlobServiceClient(
                    account_url=account_url,
                    credential=DefaultAzureCredential(),
                )
            
            # Get container client
            self._container_client = self._client.get_container_client(
                self.azure_config.container
            )
            
            # Verify container exists
            loop = asyncio.get_event_loop()
            exists = await loop.run_in_executor(
                None,
                self._container_client.exists
            )
            
            if not exists:
                raise ValueError(f"Container {self.azure_config.container} does not exist")
            
            self.status = ConnectorStatus.CONNECTED
            self._connected = True
            logger.info(f"Connected to Azure container: {self.azure_config.container}")
            return True
            
        except ImportError:
            self._error = "azure-storage-blob library not installed"
            self.status = ConnectorStatus.ERROR
            logger.error(self._error)
            return False
        except Exception as e:
            self._error = str(e)
            self.status = ConnectorStatus.ERROR
            logger.error(f"Azure connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close Azure connection."""
        self._client = None
        self._container_client = None
        self.status = ConnectorStatus.DISCONNECTED
        self._connected = False
        return True
    
    async def validate(self) -> dict[str, Any]:
        """Validate Azure configuration."""
        errors = []
        
        if not self.azure_config.container:
            errors.append("Container name is required")
        if not self.azure_config.account_name and not self.azure_config.connection_string:
            errors.append("Account name or connection string required")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "container": self.azure_config.container,
            "prefix": self.azure_config.prefix,
        }
    
    async def fetch(
        self,
        query: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """Fetch documents from Azure container."""
        if not self._container_client:
            raise RuntimeError("Not connected to Azure")
        
        query = query or {}
        prefix = query.get("prefix", self.azure_config.prefix)
        suffix = query.get("suffix")
        offset = query.get("offset", 0)
        
        loop = asyncio.get_event_loop()
        
        # List blobs
        blobs = await loop.run_in_executor(
            None,
            lambda: list(self._container_client.list_blobs(
                name_starts_with=prefix,
                results_per_page=offset + limit
            ))
        )
        
        # Filter
        filtered = []
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            if suffix and not blob.name.endswith(suffix):
                continue
            valid, _ = _validate_extension(blob.name)
            if not valid:
                continue
            filtered.append(blob)
        
        filtered = filtered[offset:offset + limit]
        
        # Fetch contents
        documents = []
        for blob in filtered:
            try:
                doc = await self._fetch_blob(blob)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to fetch {blob.name}: {e}")
        
        return documents
    
    async def _fetch_blob(self, blob: Any) -> Document | None:
        """Fetch and parse Azure blob."""
        loop = asyncio.get_event_loop()
        
        blob_client = self._container_client.get_blob_client(blob.name)
        download = await loop.run_in_executor(None, blob_client.download_blob)
        content = await loop.run_in_executor(None, download.readall)
        
        doc_type = _detect_document_type(blob.name)
        max_size = MAX_FILE_SIZES.get(doc_type, MAX_FILE_SIZES["default"])
        
        if len(content) > max_size:
            logger.warning(f"Skipping oversized file {blob.name}")
            return None
        
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
            source_path=f"azure://{self.azure_config.container}/{blob.name}",
            metadata={
                "container": self.azure_config.container,
                "name": blob.name,
                "size": blob.size,
                "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                "etag": blob.etag,
                "content_hash": _compute_hash(content),
            },
            created_at=datetime.now(timezone.utc),
        )
    
    async def generate_sas_upload(
        self,
        filename: str,
        expires_in: int = 3600,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """Generate SAS URL for upload."""
        if not self._container_client:
            raise RuntimeError("Not connected to Azure")
        
        valid, error = _validate_extension(filename)
        if not valid:
            raise ValueError(error)
        
        from azure.storage.blob import generate_blob_sas, BlobSasPermissions
        
        blob_name = f"{self.azure_config.prefix}{filename}".lstrip("/")
        expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        
        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"
        
        sas_token = generate_blob_sas(
            account_name=self.azure_config.account_name,
            container_name=self.azure_config.container,
            blob_name=blob_name,
            account_key=self.azure_config.account_key,
            permission=BlobSasPermissions(write=True, create=True),
            expiry=expiry,
            content_type=content_type,
        )
        
        url = f"https://{self.azure_config.account_name}.blob.core.windows.net/{self.azure_config.container}/{blob_name}?{sas_token}"
        
        logger.info(f"Generated SAS upload URL for: {blob_name}")
        
        return {
            "url": url,
            "blob_name": blob_name,
            "expires_in": expires_in,
            "content_type": content_type,
            "method": "PUT",
        }
    
    async def generate_sas_download(
        self,
        blob_name: str,
        expires_in: int = 3600,
    ) -> str:
        """Generate SAS URL for download."""
        if not self._container_client:
            raise RuntimeError("Not connected to Azure")
        
        from azure.storage.blob import generate_blob_sas, BlobSasPermissions
        
        expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        
        sas_token = generate_blob_sas(
            account_name=self.azure_config.account_name,
            container_name=self.azure_config.container,
            blob_name=blob_name,
            account_key=self.azure_config.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )
        
        return f"https://{self.azure_config.account_name}.blob.core.windows.net/{self.azure_config.container}/{blob_name}?{sas_token}"
    
    async def upload(
        self,
        filename: str,
        content: bytes | BinaryIO,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """Upload file to Azure Blob."""
        if not self._container_client:
            raise RuntimeError("Not connected to Azure")
        
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
            raise ValueError(f"File exceeds maximum size")
        
        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"
        
        blob_name = f"{self.azure_config.prefix}{filename}".lstrip("/")
        blob_client = self._container_client.get_blob_client(blob_name)
        content_hash = _compute_hash(content_bytes)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: blob_client.upload_blob(
                content_bytes,
                content_type=content_type,
                overwrite=True,
            )
        )
        
        logger.info(f"Uploaded to Azure: {blob_name} ({len(content_bytes)} bytes)")
        
        return {
            "blob_name": blob_name,
            "container": self.azure_config.container,
            "size": len(content_bytes),
            "content_hash": content_hash,
            "content_type": content_type,
        }
    
    async def delete(self, blob_name: str) -> bool:
        """Delete blob from Azure."""
        if not self._container_client:
            raise RuntimeError("Not connected to Azure")
        
        blob_client = self._container_client.get_blob_client(blob_name)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, blob_client.delete_blob)
        
        logger.info(f"Deleted from Azure: {blob_name}")
        return True
