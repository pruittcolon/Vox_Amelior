"""
Database Adapter - Unified interface for SQLite and PostgreSQL

Allows services to switch between database backends via environment variable.
Supports both SQLite (for local development) and PostgreSQL (for distributed clusters).

Usage:
    from shared.db_adapter import get_connection, DatabaseBackend

    # Environment: DB_BACKEND=postgres DATABASE_URL=postgresql://user:pass@host:5432/db
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
"""

import os
import sqlite3
from contextlib import contextmanager
from enum import Enum
from typing import Any

# Optional PostgreSQL support
try:
    import psycopg2
    import psycopg2.extras

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import structlog

    logger = structlog.get_logger("db_adapter")
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("db_adapter")


class DatabaseBackend(str, Enum):
    """Supported database backends."""

    SQLITE = "sqlite"
    POSTGRES = "postgres"


def get_backend() -> DatabaseBackend:
    """Get configured database backend from environment."""
    backend = os.getenv("DB_BACKEND", "sqlite").lower()
    if backend in ("postgres", "postgresql"):
        return DatabaseBackend.POSTGRES
    return DatabaseBackend.SQLITE


def get_database_url() -> str | None:
    """Get PostgreSQL connection URL from environment."""
    return os.getenv("DATABASE_URL")


class PostgresRowWrapper:
    """Wrapper to make psycopg2 rows behave like sqlite3.Row."""

    def __init__(self, row: tuple, description: list):
        self._data = dict(zip([col[0] for col in description], row))

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return list(self._data.values())[key]
        return self._data[key]

    def keys(self) -> list:
        return list(self._data.keys())

    def values(self) -> list:
        return list(self._data.values())

    def items(self) -> list:
        return list(self._data.items())


class PostgresCursorWrapper:
    """Wrapper to make psycopg2 cursor behave more like sqlite3 cursor."""

    def __init__(self, cursor):
        self._cursor = cursor
        self._description = None

    def execute(self, sql: str, params: tuple = None) -> "PostgresCursorWrapper":
        # Convert SQLite-style ? placeholders to PostgreSQL %s
        converted_sql = sql.replace("?", "%s")
        # Convert COLLATE NOCASE to ILIKE pattern (for username lookups)
        converted_sql = converted_sql.replace("COLLATE NOCASE", "")
        self._cursor.execute(converted_sql, params)
        self._description = self._cursor.description
        return self

    def executemany(self, sql: str, params_list: list) -> "PostgresCursorWrapper":
        converted_sql = sql.replace("?", "%s")
        for params in params_list:
            self._cursor.execute(converted_sql, params)
        return self

    def fetchone(self) -> PostgresRowWrapper | None:
        row = self._cursor.fetchone()
        if row is None:
            return None
        return PostgresRowWrapper(row, self._description)

    def fetchall(self) -> list:
        rows = self._cursor.fetchall()
        return [PostgresRowWrapper(row, self._description) for row in rows]

    def fetchmany(self, size: int = None) -> list:
        rows = self._cursor.fetchmany(size)
        return [PostgresRowWrapper(row, self._description) for row in rows]

    @property
    def lastrowid(self) -> int | None:
        # PostgreSQL doesn't have lastrowid, use RETURNING in INSERT
        return None

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def close(self) -> None:
        self._cursor.close()


class PostgresConnectionWrapper:
    """Wrapper to make psycopg2 connection behave like sqlite3 connection."""

    def __init__(self, conn):
        self._conn = conn
        self.row_factory = None  # Compatibility attribute

    def cursor(self) -> PostgresCursorWrapper:
        return PostgresCursorWrapper(self._conn.cursor())

    def execute(self, sql: str, params: tuple = None) -> PostgresCursorWrapper:
        cursor = self.cursor()
        cursor.execute(sql, params)
        return cursor

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.close()


class DatabaseAdapter:
    """
    Unified database adapter supporting SQLite and PostgreSQL.

    Automatically selects backend based on environment variables:
    - DB_BACKEND: "sqlite" or "postgres"
    - DATABASE_URL: PostgreSQL connection string (for postgres backend)
    - DB_PATH: SQLite file path (for sqlite backend)
    """

    def __init__(
        self, backend: DatabaseBackend | None = None, db_path: str | None = None, database_url: str | None = None
    ):
        self.backend = backend or get_backend()
        self.db_path = db_path or os.getenv("DB_PATH", "/app/instance/data.db")
        self.database_url = database_url or get_database_url()

        if self.backend == DatabaseBackend.POSTGRES:
            if not POSTGRES_AVAILABLE:
                raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
            if not self.database_url:
                # Build from individual env vars
                host = os.getenv("POSTGRES_HOST", "100.68.213.84")
                port = os.getenv("POSTGRES_PORT", "5432")
                db = os.getenv("POSTGRES_DB", "nemo_queue")
                user = os.getenv("POSTGRES_USER", "nemo")
                password = os.getenv("POSTGRES_PASSWORD", "")
                self.database_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"

        logger.info(
            "db_adapter_initialized",
            backend=self.backend.value,
            path=self.db_path if self.backend == DatabaseBackend.SQLITE else "[postgres]",
        )

    def connect(self) -> sqlite3.Connection | PostgresConnectionWrapper:
        """Get a database connection."""
        if self.backend == DatabaseBackend.POSTGRES:
            conn = psycopg2.connect(self.database_url)
            return PostgresConnectionWrapper(conn)
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn

    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, sql: str, params: tuple = None) -> Any:
        """Execute a single SQL statement."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params or ())
            return cursor

    def query(self, sql: str, params: tuple = None) -> list:
        """Execute a query and return all results."""
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params or ())
            return cursor.fetchall()
        finally:
            conn.close()

    def query_one(self, sql: str, params: tuple = None) -> Any | None:
        """Execute a query and return first result."""
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params or ())
            return cursor.fetchone()
        finally:
            conn.close()


# Global adapter instance
_adapter: DatabaseAdapter | None = None


def get_adapter() -> DatabaseAdapter:
    """Get or create the global database adapter."""
    global _adapter
    if _adapter is None:
        _adapter = DatabaseAdapter()
    return _adapter


def get_connection() -> sqlite3.Connection | PostgresConnectionWrapper:
    """Convenience function to get a database connection."""
    return get_adapter().connect()


# Compatibility aliases
def init_adapter(**kwargs) -> DatabaseAdapter:
    """Initialize the global adapter with custom settings."""
    global _adapter
    _adapter = DatabaseAdapter(**kwargs)
    return _adapter
