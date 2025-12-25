"""
Base Connector Abstract Class.

All data source connectors inherit from this base class.
Provides a consistent interface for RAG ingestion pipelines.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from uuid import UUID, uuid4


logger = logging.getLogger(__name__)


class ConnectorStatus(str, Enum):
    """Connector connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    PENDING = "pending"


class DocumentType(str, Enum):
    """Supported document types."""

    TEXT = "text"
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    DOCX = "docx"
    XLSX = "xlsx"
    IMAGE = "image"


@dataclass
class Document:
    """
    A document extracted from a data source.

    Attributes:
        id: Unique document identifier
        content: Raw text content
        metadata: Document metadata (source, timestamps, etc.)
        doc_type: Type of document
        source_path: Original source path/URL
        embedding: Optional pre-computed embedding
    """

    id: UUID = field(default_factory=uuid4)
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_type: DocumentType = DocumentType.TEXT
    source_path: Optional[str] = None
    embedding: Optional[list[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def char_count(self) -> int:
        """Character count of content."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.content.split())

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "content": self.content,
            "metadata": self.metadata,
            "doc_type": self.doc_type.value,
            "source_path": self.source_path,
            "created_at": self.created_at.isoformat(),
            "char_count": self.char_count,
            "word_count": self.word_count,
        }


@dataclass
class ConnectorConfig:
    """Configuration for a connector."""

    name: str
    connector_type: str
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    tenant_id: Optional[UUID] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class BaseConnector(ABC):
    """
    Abstract base class for all data source connectors.

    Subclasses must implement:
        - connect(): Establish connection
        - disconnect(): Close connection
        - fetch(): Retrieve documents
        - validate(): Validate connection/config
    """

    def __init__(self, config: ConnectorConfig):
        """
        Initialize connector.

        Args:
            config: Connector configuration
        """
        self.config = config
        self.status = ConnectorStatus.DISCONNECTED
        self._connected = False
        self._error: Optional[str] = None

    @property
    def name(self) -> str:
        """Connector name."""
        return self.config.name

    @property
    def is_connected(self) -> bool:
        """Check if connector is connected."""
        return self._connected and self.status == ConnectorStatus.CONNECTED

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to data source.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Close connection to data source.

        Returns:
            True if disconnection successful
        """
        pass

    @abstractmethod
    async def fetch(
        self,
        query: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[Document]:
        """
        Fetch documents from data source.

        Args:
            query: Optional query parameters
            limit: Maximum documents to fetch

        Returns:
            List of Document objects
        """
        pass

    @abstractmethod
    async def validate(self) -> dict[str, Any]:
        """
        Validate connector configuration and connection.

        Returns:
            Validation result with status and errors
        """
        pass

    async def fetch_stream(
        self,
        query: Optional[dict[str, Any]] = None,
        batch_size: int = 10,
    ) -> AsyncIterator[Document]:
        """
        Stream documents one at a time.

        Args:
            query: Optional query parameters
            batch_size: Internal batch size for fetching

        Yields:
            Document objects
        """
        offset = 0
        while True:
            batch_query = {**(query or {}), "offset": offset, "limit": batch_size}
            documents = await self.fetch(batch_query, limit=batch_size)

            if not documents:
                break

            for doc in documents:
                yield doc

            offset += len(documents)
            if len(documents) < batch_size:
                break

    def get_status(self) -> dict[str, Any]:
        """Get connector status."""
        return {
            "name": self.name,
            "type": self.config.connector_type,
            "status": self.status.value,
            "connected": self.is_connected,
            "error": self._error,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class FileConnector(BaseConnector):
    """
    Base class for file-based connectors.

    Provides common file handling functionality.
    """

    def __init__(self, config: ConnectorConfig, base_path: Optional[Path] = None):
        """
        Initialize file connector.

        Args:
            config: Connector configuration
            base_path: Base directory for file operations
        """
        super().__init__(config)
        self.base_path = base_path or Path(".")

    def _resolve_path(self, file_path: str | Path) -> Path:
        """Resolve file path relative to base."""
        path = Path(file_path)
        if path.is_absolute():
            return path
        return self.base_path / path

    def _validate_file(self, file_path: Path) -> tuple[bool, Optional[str]]:
        """
        Validate file exists and is readable.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.exists():
            return False, f"File not found: {file_path}"
        if not file_path.is_file():
            return False, f"Not a file: {file_path}"
        if not file_path.stat().st_size:
            return False, f"File is empty: {file_path}"
        return True, None

    async def connect(self) -> bool:
        """File connectors are always 'connected'."""
        self.status = ConnectorStatus.CONNECTED
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        """No persistent connection to close."""
        self.status = ConnectorStatus.DISCONNECTED
        self._connected = False
        return True
