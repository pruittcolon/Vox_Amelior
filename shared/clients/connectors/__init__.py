"""
Connector Package for RAG Data Ingestion.

Provides pluggable connectors for various data sources:
- File-based: CSV, PDF
- Cloud storage: S3, GCS, Azure Blob
- Databases: PostgreSQL, Snowflake
"""

from shared.clients.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorStatus,
    Document,
    DocumentType,
    FileConnector,
)
from shared.clients.connectors.csv import CSVConnector
from shared.clients.connectors.pdf import PDFConnector

# Cloud storage connectors (lazy imports for optional deps)
try:
    from shared.clients.connectors.s3 import S3Connector, S3Config
except ImportError:
    S3Connector = None  # type: ignore
    S3Config = None  # type: ignore

try:
    from shared.clients.connectors.gcs import GCSConnector, GCSConfig
except ImportError:
    GCSConnector = None  # type: ignore
    GCSConfig = None  # type: ignore

try:
    from shared.clients.connectors.azure_blob import AzureBlobConnector, AzureBlobConfig
except ImportError:
    AzureBlobConnector = None  # type: ignore
    AzureBlobConfig = None  # type: ignore

# Database connectors (lazy imports for optional deps)
try:
    from shared.clients.connectors.postgres import PostgresConnector, PostgresConfig
except ImportError:
    PostgresConnector = None  # type: ignore
    PostgresConfig = None  # type: ignore

try:
    from shared.clients.connectors.snowflake import SnowflakeConnector, SnowflakeConfig
except ImportError:
    SnowflakeConnector = None  # type: ignore
    SnowflakeConfig = None  # type: ignore

__all__ = [
    # Base
    "BaseConnector",
    "ConnectorConfig",
    "ConnectorStatus",
    "Document",
    "DocumentType",
    "FileConnector",
    # File connectors
    "CSVConnector",
    "PDFConnector",
    # Cloud storage
    "S3Connector",
    "S3Config",
    "GCSConnector",
    "GCSConfig",
    "AzureBlobConnector",
    "AzureBlobConfig",
    # Databases
    "PostgresConnector",
    "PostgresConfig",
    "SnowflakeConnector",
    "SnowflakeConfig",
]
