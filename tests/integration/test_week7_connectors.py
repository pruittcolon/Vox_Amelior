"""
Integration tests for Week 7: Production Connectors.

Tests cover:
- GCS connector configuration
- Azure Blob connector configuration
- PostgreSQL connector validation
- Snowflake connector validation
- Blocked extension enforcement across connectors
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "clients" / "connectors"))


class TestGCSConnectorConfig:
    """Tests for GCS connector configuration."""
    
    def test_config_from_environment(self) -> None:
        """GCSConfig loads from environment variables."""
        from gcs import GCSConfig
        
        env = {
            "GCS_PROJECT_ID": "my-project",
            "GCS_CREDENTIALS_FILE": "/run/secrets/gcs_key.json",
            "GCS_LOCATION": "us-central1",
        }
        
        with patch.dict(os.environ, env):
            config = GCSConfig.from_environment("test-bucket", "uploads/")
        
        assert config.bucket == "test-bucket"
        assert config.prefix == "uploads/"
        assert config.project_id == "my-project"
        assert config.location == "us-central1"
    
    def test_config_defaults(self) -> None:
        """GCSConfig has sensible defaults."""
        from gcs import GCSConfig
        
        config = GCSConfig(bucket="my-bucket")
        
        assert config.prefix == ""
        assert config.location == "us"


class TestAzureBlobConnectorConfig:
    """Tests for Azure Blob connector configuration."""
    
    def test_config_from_environment(self) -> None:
        """AzureBlobConfig loads from environment variables."""
        from azure_blob import AzureBlobConfig
        
        env = {
            "AZURE_STORAGE_ACCOUNT": "mystorageaccount",
        }
        
        with patch.dict(os.environ, env):
            config = AzureBlobConfig.from_environment("documents", "uploads/")
        
        assert config.container == "documents"
        assert config.prefix == "uploads/"
        assert config.account_name == "mystorageaccount"
    
    def test_config_with_connection_string(self) -> None:
        """AzureBlobConfig supports connection string."""
        from azure_blob import AzureBlobConfig
        
        config = AzureBlobConfig(
            container="test",
            connection_string="DefaultEndpointsProtocol=https;AccountName=test;..."
        )
        
        assert config.connection_string is not None


class TestPostgresConnectorConfig:
    """Tests for PostgreSQL connector configuration."""
    
    def test_config_from_environment(self) -> None:
        """PostgresConfig loads from environment variables."""
        from postgres import PostgresConfig
        
        env = {
            "POSTGRES_HOST": "db.example.com",
            "POSTGRES_PORT": "5433",
            "POSTGRES_DB": "documents",
            "POSTGRES_USER": "reader",
            "POSTGRES_PASSWORD": "secret",
            "POSTGRES_SSL_MODE": "verify-full",
        }
        
        with patch.dict(os.environ, env):
            config = PostgresConfig.from_environment(
                table="articles",
                content_column="body"
            )
        
        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.database == "documents"
        assert config.table == "articles"
        assert config.content_column == "body"
        assert config.ssl_mode == "verify-full"
    
    def test_config_defaults(self) -> None:
        """PostgresConfig has sensible defaults."""
        from postgres import PostgresConfig
        
        config = PostgresConfig(
            host="localhost",
            database="test",
            table="docs",
            content_column="content"
        )
        
        assert config.port == 5432
        assert config.ssl_mode == "require"
        assert config.schema == "public"
        assert config.id_column == "id"


class TestSnowflakeConnectorConfig:
    """Tests for Snowflake connector configuration."""
    
    def test_config_from_environment(self) -> None:
        """SnowflakeConfig loads from environment variables."""
        from snowflake import SnowflakeConfig
        
        env = {
            "SNOWFLAKE_ACCOUNT": "xy12345.us-east-1",
            "SNOWFLAKE_WAREHOUSE": "COMPUTE_WH",
            "SNOWFLAKE_DATABASE": "ANALYTICS",
            "SNOWFLAKE_SCHEMA": "RAW",
            "SNOWFLAKE_USER": "etl_user",
            "SNOWFLAKE_PASSWORD": "secret",
            "SNOWFLAKE_ROLE": "ANALYST",
        }
        
        with patch.dict(os.environ, env):
            config = SnowflakeConfig.from_environment(
                table="DOCUMENTS",
                content_column="BODY"
            )
        
        assert config.account == "xy12345.us-east-1"
        assert config.warehouse == "COMPUTE_WH"
        assert config.database == "ANALYTICS"
        assert config.schema == "RAW"
        assert config.role == "ANALYST"
    
    def test_config_defaults(self) -> None:
        """SnowflakeConfig has sensible defaults."""
        from snowflake import SnowflakeConfig
        
        config = SnowflakeConfig(
            account="test",
            warehouse="WH",
            database="DB",
            table="TBL",
            content_column="COL"
        )
        
        assert config.schema == "PUBLIC"
        assert config.id_column == "ID"


class TestBlockedExtensions:
    """Tests for blocked extension enforcement across connectors."""
    
    def test_gcs_blocks_executables(self) -> None:
        """GCS connector blocks executable files."""
        from gcs import _validate_extension
        
        for ext in [".exe", ".bat", ".ps1", ".php", ".jsp"]:
            valid, error = _validate_extension(f"malware{ext}")
            assert valid is False, f"{ext} should be blocked"
    
    def test_azure_blocks_executables(self) -> None:
        """Azure connector blocks executable files."""
        from azure_blob import _validate_extension
        
        for ext in [".exe", ".bat", ".ps1", ".php", ".jsp"]:
            valid, error = _validate_extension(f"malware{ext}")
            assert valid is False, f"{ext} should be blocked"


class TestDocumentTypeDetection:
    """Tests for document type detection across connectors."""
    
    def test_gcs_detects_pdf(self) -> None:
        """GCS connector detects PDF files."""
        from gcs import _detect_document_type
        from base import DocumentType
        
        result = _detect_document_type("report.pdf")
        assert result == DocumentType.PDF
    
    def test_gcs_detects_csv(self) -> None:
        """GCS connector detects CSV files."""
        from gcs import _detect_document_type
        from base import DocumentType
        
        result = _detect_document_type("data.csv")
        assert result == DocumentType.CSV
    
    def test_azure_detects_images(self) -> None:
        """Azure connector detects image files."""
        from azure_blob import _detect_document_type
        from base import DocumentType
        
        for ext in [".jpg", ".jpeg", ".png"]:
            result = _detect_document_type(f"photo{ext}")
            assert result == DocumentType.IMAGE


class TestContentHashing:
    """Tests for content hash computation."""
    
    def test_gcs_hash_computation(self) -> None:
        """GCS connector computes correct SHA256 hash."""
        from gcs import _compute_hash
        import hashlib
        
        content = b"test content"
        computed = _compute_hash(content)
        expected = hashlib.sha256(content).hexdigest()
        
        assert computed == expected
    
    def test_azure_hash_computation(self) -> None:
        """Azure connector computes correct SHA256 hash."""
        from azure_blob import _compute_hash
        import hashlib
        
        content = b"test content"
        computed = _compute_hash(content)
        expected = hashlib.sha256(content).hexdigest()
        
        assert computed == expected
    
    def test_postgres_hash_computation(self) -> None:
        """PostgreSQL connector computes correct SHA256 hash."""
        from postgres import _compute_hash
        import hashlib
        
        content = "test content"
        computed = _compute_hash(content)
        expected = hashlib.sha256(content.encode()).hexdigest()
        
        assert computed == expected


class TestConnectorValidation:
    """Tests for connector validation logic."""
    
    @pytest.mark.asyncio
    async def test_gcs_validation_requires_bucket(self) -> None:
        """GCS connector requires bucket name."""
        from gcs import GCSConnector, GCSConfig
        
        config = GCSConfig(bucket="")
        connector = GCSConnector(config)
        
        result = await connector.validate()
        
        assert result["valid"] is False
        assert any("bucket" in e.lower() for e in result["errors"])
    
    @pytest.mark.asyncio
    async def test_azure_validation_requires_account(self) -> None:
        """Azure connector requires account name or connection string."""
        from azure_blob import AzureBlobConnector, AzureBlobConfig
        
        config = AzureBlobConfig(container="test")
        connector = AzureBlobConnector(config)
        
        result = await connector.validate()
        
        assert result["valid"] is False
    
    @pytest.mark.asyncio
    async def test_postgres_validation_requires_fields(self) -> None:
        """PostgreSQL connector requires host, database, table, content_column."""
        from postgres import PostgresConnector, PostgresConfig
        
        config = PostgresConfig(
            host="",
            database="",
            table="",
            content_column=""
        )
        connector = PostgresConnector(config)
        
        result = await connector.validate()
        
        assert result["valid"] is False
        assert len(result["errors"]) >= 4
    
    @pytest.mark.asyncio
    async def test_snowflake_validation_requires_fields(self) -> None:
        """Snowflake connector requires account, warehouse, database, table, user."""
        from snowflake import SnowflakeConnector, SnowflakeConfig
        
        config = SnowflakeConfig(
            account="",
            warehouse="",
            database="",
            table="",
            content_column=""
        )
        connector = SnowflakeConnector(config)
        
        result = await connector.validate()
        
        assert result["valid"] is False
        assert len(result["errors"]) >= 4


class TestConnectorStatus:
    """Tests for connector status tracking."""
    
    def test_initial_status_is_disconnected(self) -> None:
        """Connectors start in disconnected state."""
        from gcs import GCSConnector, GCSConfig
        from base import ConnectorStatus
        
        config = GCSConfig(bucket="test")
        connector = GCSConnector(config)
        
        assert connector.status == ConnectorStatus.DISCONNECTED
        assert connector.is_connected is False
    
    def test_connector_name_includes_resource(self) -> None:
        """Connector name includes the target resource."""
        from gcs import GCSConnector, GCSConfig
        from azure_blob import AzureBlobConnector, AzureBlobConfig
        from postgres import PostgresConnector, PostgresConfig
        
        gcs = GCSConnector(GCSConfig(bucket="my-bucket"))
        assert "my-bucket" in gcs.name
        
        azure = AzureBlobConnector(AzureBlobConfig(container="my-container"))
        assert "my-container" in azure.name
        
        pg = PostgresConnector(PostgresConfig(
            host="localhost",
            database="mydb",
            table="mytable",
            content_column="content"
        ))
        assert "mydb" in pg.name
        assert "mytable" in pg.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
