"""
Snowflake Connector - Data warehouse integration for RAG ingestion.

Provides secure connection to Snowflake for extracting documents
from tables and views for enterprise RAG pipelines.

Configuration:
- SNOWFLAKE_ACCOUNT: Snowflake account identifier
- SNOWFLAKE_USER: Username
- SNOWFLAKE_PASSWORD_FILE: Path to password (Docker secret)
- SNOWFLAKE_WAREHOUSE: Warehouse name
- SNOWFLAKE_DATABASE: Database name
- SNOWFLAKE_SCHEMA: Schema name (default: PUBLIC)
- SNOWFLAKE_ROLE: Role to use (optional)

Usage:
    from shared.clients.connectors.snowflake import SnowflakeConnector, SnowflakeConfig
    
    config = SnowflakeConfig(
        account="myaccount",
        warehouse="COMPUTE_WH",
        database="DOCUMENTS",
        table="ARTICLES",
        content_column="BODY",
    )
    connector = SnowflakeConnector(config)
    
    async with connector:
        documents = await connector.fetch(limit=100)
"""

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

try:
    from .base import BaseConnector, ConnectorConfig, ConnectorStatus, Document, DocumentType
except ImportError:
    from base import BaseConnector, ConnectorConfig, ConnectorStatus, Document, DocumentType

logger = logging.getLogger(__name__)


def _load_secret(path: str | None, env_fallback: str | None = None) -> str | None:
    """Load secret from file or environment."""
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    if env_fallback:
        return os.getenv(env_fallback)
    return None


def _compute_hash(content: str) -> str:
    """Compute SHA256 hash."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class SnowflakeConfig:
    """Snowflake connector configuration."""
    account: str
    warehouse: str
    database: str
    table: str
    content_column: str
    id_column: str = "ID"
    metadata_columns: list[str] = field(default_factory=list)
    schema: str = "PUBLIC"
    user: str | None = None
    password: str | None = None
    role: str | None = None
    private_key_file: str | None = None
    
    @classmethod
    def from_environment(
        cls,
        table: str,
        content_column: str,
        **kwargs: Any,
    ) -> "SnowflakeConfig":
        """Create config from environment variables."""
        return cls(
            account=os.getenv("SNOWFLAKE_ACCOUNT", ""),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", ""),
            database=os.getenv("SNOWFLAKE_DATABASE", ""),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
            user=os.getenv("SNOWFLAKE_USER"),
            password=_load_secret(
                os.getenv("SNOWFLAKE_PASSWORD_FILE"),
                "SNOWFLAKE_PASSWORD"
            ),
            role=os.getenv("SNOWFLAKE_ROLE"),
            private_key_file=os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE"),
            table=table,
            content_column=content_column,
            **kwargs,
        )


class SnowflakeConnector(BaseConnector):
    """
    Snowflake data warehouse connector.
    
    Extracts documents from Snowflake tables for RAG ingestion.
    Supports key-pair authentication and custom queries.
    """
    
    def __init__(self, sf_config: SnowflakeConfig):
        """Initialize Snowflake connector."""
        connector_config = ConnectorConfig(
            name=f"snowflake-{sf_config.database}-{sf_config.table}",
            connector_type="snowflake",
            config={
                "account": sf_config.account,
                "database": sf_config.database,
                "table": sf_config.table,
            },
        )
        super().__init__(connector_config)
        self.sf_config = sf_config
        self._conn: Any = None
    
    async def connect(self) -> bool:
        """Establish connection to Snowflake."""
        try:
            import snowflake.connector
            
            connect_params = {
                "account": self.sf_config.account,
                "user": self.sf_config.user,
                "warehouse": self.sf_config.warehouse,
                "database": self.sf_config.database,
                "schema": self.sf_config.schema,
            }
            
            if self.sf_config.role:
                connect_params["role"] = self.sf_config.role
            
            # Use key-pair auth if available
            if self.sf_config.private_key_file and os.path.exists(self.sf_config.private_key_file):
                from cryptography.hazmat.backends import default_backend
                from cryptography.hazmat.primitives import serialization
                
                with open(self.sf_config.private_key_file, "rb") as key_file:
                    private_key = serialization.load_pem_private_key(
                        key_file.read(),
                        password=None,
                        backend=default_backend(),
                    )
                
                connect_params["private_key"] = private_key
            else:
                connect_params["password"] = self.sf_config.password
            
            loop = asyncio.get_event_loop()
            self._conn = await loop.run_in_executor(
                None,
                lambda: snowflake.connector.connect(**connect_params)
            )
            
            # Verify connection
            cursor = self._conn.cursor()
            await loop.run_in_executor(None, lambda: cursor.execute("SELECT 1"))
            cursor.close()
            
            self.status = ConnectorStatus.CONNECTED
            self._connected = True
            logger.info(f"Connected to Snowflake: {self.sf_config.database}.{self.sf_config.table}")
            return True
            
        except ImportError:
            self._error = "snowflake-connector-python library not installed"
            self.status = ConnectorStatus.ERROR
            logger.error(self._error)
            return False
        except Exception as e:
            self._error = str(e)
            self.status = ConnectorStatus.ERROR
            logger.error(f"Snowflake connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close Snowflake connection."""
        if self._conn:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._conn.close)
        self._conn = None
        self.status = ConnectorStatus.DISCONNECTED
        self._connected = False
        return True
    
    async def validate(self) -> dict[str, Any]:
        """Validate Snowflake configuration."""
        errors = []
        
        if not self.sf_config.account:
            errors.append("Account is required")
        if not self.sf_config.warehouse:
            errors.append("Warehouse is required")
        if not self.sf_config.database:
            errors.append("Database is required")
        if not self.sf_config.table:
            errors.append("Table is required")
        if not self.sf_config.user:
            errors.append("User is required")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "account": self.sf_config.account,
            "database": self.sf_config.database,
            "table": self.sf_config.table,
        }
    
    async def fetch(
        self,
        query: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """
        Fetch documents from Snowflake table.
        
        Query options:
        - where: SQL WHERE clause
        - order_by: Column to order by
        - modified_after: Timestamp for incremental sync
        """
        if not self._conn:
            raise RuntimeError("Not connected to Snowflake")
        
        query = query or {}
        offset = query.get("offset", 0)
        where_clause = query.get("where", "")
        order_by = query.get("order_by", self.sf_config.id_column)
        
        # Build column list
        columns = [self.sf_config.id_column, self.sf_config.content_column]
        columns.extend(self.sf_config.metadata_columns)
        columns_sql = ", ".join(f'"{c}"' for c in columns)
        
        # Build query - Snowflake uses double quotes for identifiers
        sql = f"""
            SELECT {columns_sql}
            FROM "{self.sf_config.database}"."{self.sf_config.schema}"."{self.sf_config.table}"
        """
        
        if where_clause:
            sql += f" WHERE {where_clause}"
        
        sql += f' ORDER BY "{order_by}"'
        sql += f" LIMIT {limit} OFFSET {offset}"
        
        loop = asyncio.get_event_loop()
        cursor = self._conn.cursor()
        
        try:
            await loop.run_in_executor(None, lambda: cursor.execute(sql))
            rows = await loop.run_in_executor(None, cursor.fetchall)
            column_names = [desc[0] for desc in cursor.description]
        finally:
            cursor.close()
        
        # Convert to documents
        documents = []
        for row in rows:
            row_dict = dict(zip(column_names, row))
            content = row_dict.get(self.sf_config.content_column)
            if not content:
                continue
            
            metadata = {
                "table": self.sf_config.table,
                "schema": self.sf_config.schema,
                "database": self.sf_config.database,
                self.sf_config.id_column: str(row_dict[self.sf_config.id_column]),
            }
            for col in self.sf_config.metadata_columns:
                if col in row_dict:
                    value = row_dict[col]
                    if isinstance(value, datetime):
                        value = value.isoformat()
                    metadata[col] = value
            
            doc = Document(
                id=uuid4(),
                content=str(content),
                doc_type=DocumentType.TEXT,
                source_path=f"snowflake://{self.sf_config.account}/{self.sf_config.database}/{self.sf_config.table}/{row_dict[self.sf_config.id_column]}",
                metadata=metadata,
                created_at=datetime.now(timezone.utc),
            )
            doc.metadata["content_hash"] = _compute_hash(doc.content)
            documents.append(doc)
        
        logger.info(f"Fetched {len(documents)} documents from Snowflake")
        return documents
    
    async def execute_query(self, sql: str, params: list[Any] | None = None) -> list[dict]:
        """Execute custom SQL query and return results as dictionaries."""
        if not self._conn:
            raise RuntimeError("Not connected to Snowflake")
        
        loop = asyncio.get_event_loop()
        cursor = self._conn.cursor()
        
        try:
            if params:
                await loop.run_in_executor(None, lambda: cursor.execute(sql, params))
            else:
                await loop.run_in_executor(None, lambda: cursor.execute(sql))
            
            rows = await loop.run_in_executor(None, cursor.fetchall)
            column_names = [desc[0] for desc in cursor.description]
            
            return [dict(zip(column_names, row)) for row in rows]
        finally:
            cursor.close()
    
    async def count(self, where_clause: str | None = None) -> int:
        """Count documents matching criteria."""
        if not self._conn:
            raise RuntimeError("Not connected to Snowflake")
        
        sql = f"""
            SELECT COUNT(*)
            FROM "{self.sf_config.database}"."{self.sf_config.schema}"."{self.sf_config.table}"
        """
        if where_clause:
            sql += f" WHERE {where_clause}"
        
        loop = asyncio.get_event_loop()
        cursor = self._conn.cursor()
        
        try:
            await loop.run_in_executor(None, lambda: cursor.execute(sql))
            result = await loop.run_in_executor(None, cursor.fetchone)
            return result[0] if result else 0
        finally:
            cursor.close()
