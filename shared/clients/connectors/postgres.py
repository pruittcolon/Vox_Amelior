"""
PostgreSQL Connector - Database integration for RAG ingestion.

Provides secure database access for document extraction and
query-based data retrieval for enterprise data pipelines.

Configuration:
- POSTGRES_HOST: Database host
- POSTGRES_PORT: Database port (default: 5432)
- POSTGRES_DB: Database name
- POSTGRES_USER: Username
- POSTGRES_PASSWORD_FILE: Path to password (Docker secret)
- POSTGRES_SSL_MODE: SSL mode (default: require)

Usage:
    from shared.clients.connectors.postgres import PostgresConnector, PostgresConfig
    
    config = PostgresConfig(
        host="localhost",
        database="documents",
        table="articles",
        content_column="body",
    )
    connector = PostgresConnector(config)
    
    async with connector:
        documents = await connector.fetch(
            query={"where": "category = 'tech'"},
            limit=100
        )
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
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class PostgresConfig:
    """PostgreSQL connector configuration."""
    host: str
    database: str
    table: str
    content_column: str
    id_column: str = "id"
    metadata_columns: list[str] = field(default_factory=list)
    port: int = 5432
    user: str | None = None
    password: str | None = None
    ssl_mode: str = "require"
    schema: str = "public"
    
    @classmethod
    def from_environment(
        cls,
        table: str,
        content_column: str,
        **kwargs: Any,
    ) -> "PostgresConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", ""),
            user=os.getenv("POSTGRES_USER"),
            password=_load_secret(
                os.getenv("POSTGRES_PASSWORD_FILE"),
                "POSTGRES_PASSWORD"
            ),
            ssl_mode=os.getenv("POSTGRES_SSL_MODE", "require"),
            table=table,
            content_column=content_column,
            **kwargs,
        )


class PostgresConnector(BaseConnector):
    """
    PostgreSQL database connector.
    
    Extracts documents from database tables for RAG ingestion.
    Supports custom queries, metadata extraction, and incremental sync.
    """
    
    def __init__(self, pg_config: PostgresConfig):
        """Initialize PostgreSQL connector."""
        connector_config = ConnectorConfig(
            name=f"postgres-{pg_config.database}-{pg_config.table}",
            connector_type="postgres",
            config={
                "host": pg_config.host,
                "database": pg_config.database,
                "table": pg_config.table,
            },
        )
        super().__init__(connector_config)
        self.pg_config = pg_config
        self._pool: Any = None
    
    async def connect(self) -> bool:
        """Establish connection pool to PostgreSQL."""
        try:
            import asyncpg
            
            # Build connection string
            ssl_context = None
            if self.pg_config.ssl_mode != "disable":
                import ssl
                ssl_context = ssl.create_default_context()
                if self.pg_config.ssl_mode == "require":
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
            
            self._pool = await asyncpg.create_pool(
                host=self.pg_config.host,
                port=self.pg_config.port,
                database=self.pg_config.database,
                user=self.pg_config.user,
                password=self.pg_config.password,
                ssl=ssl_context,
                min_size=2,
                max_size=10,
                command_timeout=60,
            )
            
            # Verify connection
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            self.status = ConnectorStatus.CONNECTED
            self._connected = True
            logger.info(f"Connected to PostgreSQL: {self.pg_config.database}")
            return True
            
        except ImportError:
            self._error = "asyncpg library not installed"
            self.status = ConnectorStatus.ERROR
            logger.error(self._error)
            return False
        except Exception as e:
            self._error = str(e)
            self.status = ConnectorStatus.ERROR
            logger.error(f"PostgreSQL connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
        self._pool = None
        self.status = ConnectorStatus.DISCONNECTED
        self._connected = False
        return True
    
    async def validate(self) -> dict[str, Any]:
        """Validate PostgreSQL configuration."""
        errors = []
        
        if not self.pg_config.host:
            errors.append("Host is required")
        if not self.pg_config.database:
            errors.append("Database is required")
        if not self.pg_config.table:
            errors.append("Table is required")
        if not self.pg_config.content_column:
            errors.append("Content column is required")
        
        # Verify table exists
        if self._pool and not errors:
            try:
                async with self._pool.acquire() as conn:
                    exists = await conn.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = $1 AND table_name = $2
                        )
                        """,
                        self.pg_config.schema,
                        self.pg_config.table,
                    )
                    if not exists:
                        errors.append(f"Table {self.pg_config.table} does not exist")
            except Exception as e:
                errors.append(f"Validation query failed: {e}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "host": self.pg_config.host,
            "database": self.pg_config.database,
            "table": self.pg_config.table,
        }
    
    async def fetch(
        self,
        query: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """
        Fetch documents from database table.
        
        Query options:
        - where: SQL WHERE clause conditions
        - order_by: Column to order by (default: id_column)
        - modified_after: Timestamp for incremental sync
        """
        if not self._pool:
            raise RuntimeError("Not connected to PostgreSQL")
        
        query = query or {}
        offset = query.get("offset", 0)
        where_clause = query.get("where", "")
        order_by = query.get("order_by", self.pg_config.id_column)
        modified_after = query.get("modified_after")
        
        # Build column list
        columns = [self.pg_config.id_column, self.pg_config.content_column]
        columns.extend(self.pg_config.metadata_columns)
        columns_sql = ", ".join(f'"{c}"' for c in columns)
        
        # Build query
        sql = f"""
            SELECT {columns_sql}
            FROM "{self.pg_config.schema}"."{self.pg_config.table}"
        """
        
        conditions = []
        params = []
        param_idx = 1
        
        if where_clause:
            conditions.append(f"({where_clause})")
        
        if modified_after:
            conditions.append(f'"updated_at" > ${param_idx}')
            params.append(modified_after)
            param_idx += 1
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += f' ORDER BY "{order_by}"'
        sql += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([limit, offset])
        
        # Execute query
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        
        # Convert to documents
        documents = []
        for row in rows:
            content = row[self.pg_config.content_column]
            if not content:
                continue
            
            # Build metadata from additional columns
            metadata = {
                "table": self.pg_config.table,
                "schema": self.pg_config.schema,
                self.pg_config.id_column: str(row[self.pg_config.id_column]),
            }
            for col in self.pg_config.metadata_columns:
                if col in row:
                    value = row[col]
                    # Convert datetime to ISO format
                    if isinstance(value, datetime):
                        value = value.isoformat()
                    metadata[col] = value
            
            doc = Document(
                id=uuid4(),
                content=str(content),
                doc_type=DocumentType.TEXT,
                source_path=f"postgres://{self.pg_config.host}/{self.pg_config.database}/{self.pg_config.table}/{row[self.pg_config.id_column]}",
                metadata=metadata,
                created_at=datetime.now(timezone.utc),
            )
            doc.metadata["content_hash"] = _compute_hash(doc.content)
            documents.append(doc)
        
        logger.info(f"Fetched {len(documents)} documents from {self.pg_config.table}")
        return documents
    
    async def fetch_by_ids(self, ids: list[Any]) -> list[Document]:
        """Fetch specific documents by ID."""
        if not self._pool:
            raise RuntimeError("Not connected to PostgreSQL")
        
        if not ids:
            return []
        
        columns = [self.pg_config.id_column, self.pg_config.content_column]
        columns.extend(self.pg_config.metadata_columns)
        columns_sql = ", ".join(f'"{c}"' for c in columns)
        
        placeholders = ", ".join(f"${i+1}" for i in range(len(ids)))
        sql = f"""
            SELECT {columns_sql}
            FROM "{self.pg_config.schema}"."{self.pg_config.table}"
            WHERE "{self.pg_config.id_column}" IN ({placeholders})
        """
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *ids)
        
        documents = []
        for row in rows:
            content = row[self.pg_config.content_column]
            if not content:
                continue
            
            metadata = {
                "table": self.pg_config.table,
                self.pg_config.id_column: str(row[self.pg_config.id_column]),
            }
            
            doc = Document(
                id=uuid4(),
                content=str(content),
                doc_type=DocumentType.TEXT,
                source_path=f"postgres://{self.pg_config.host}/{self.pg_config.database}/{self.pg_config.table}/{row[self.pg_config.id_column]}",
                metadata=metadata,
                created_at=datetime.now(timezone.utc),
            )
            documents.append(doc)
        
        return documents
    
    async def count(self, where_clause: str | None = None) -> int:
        """Count documents matching criteria."""
        if not self._pool:
            raise RuntimeError("Not connected to PostgreSQL")
        
        sql = f'SELECT COUNT(*) FROM "{self.pg_config.schema}"."{self.pg_config.table}"'
        if where_clause:
            sql += f" WHERE {where_clause}"
        
        async with self._pool.acquire() as conn:
            return await conn.fetchval(sql)
    
    async def get_last_modified(self, timestamp_column: str = "updated_at") -> datetime | None:
        """Get most recent modification timestamp."""
        if not self._pool:
            raise RuntimeError("Not connected to PostgreSQL")
        
        sql = f"""
            SELECT MAX("{timestamp_column}")
            FROM "{self.pg_config.schema}"."{self.pg_config.table}"
        """
        
        async with self._pool.acquire() as conn:
            return await conn.fetchval(sql)
