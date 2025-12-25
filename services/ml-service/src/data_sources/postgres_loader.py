from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text


class PostgresLoader:
    """
    Universal PostgreSQL Loader using SQLAlchemy.
    Supports schema introspection, chunked loading, and safe connection handling.
    """

    def __init__(self, connection_string: str):
        """
        Initialize with a connection string.
        Format: postgresql+psycopg2://user:pass@host:port/dbname
        """
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)

    def test_connection(self) -> bool:
        """Verify connection is alive."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def list_schemas(self) -> list[str]:
        """List all schemas in the database."""
        inspector = inspect(self.engine)
        return inspector.get_schema_names()

    def list_tables(self, schema: str = "public") -> list[str]:
        """List all tables in a specific schema."""
        inspector = inspect(self.engine)
        return inspector.get_table_names(schema=schema)

    def get_schema(self, table_name: str, schema: str = "public") -> dict[str, Any]:
        """Get column metadata for a table."""
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name, schema=schema)
        # Convert SQLAlchemy types to string representation for JSON serialization
        return {col["name"]: {"type": str(col["type"]), "nullable": col["nullable"]} for col in columns}

    def load_query(self, query: str, params: dict | None = None) -> pd.DataFrame:
        """Load data from a raw SQL query."""
        return pd.read_sql_query(query, self.engine, params=params)

    def load_table(self, table_name: str, schema: str = "public", limit: int | None = None) -> pd.DataFrame:
        """
        Load data from a table.

        Args:
            table_name: Name of the table
            schema: Schema name (default 'public')
            limit: Optional row limit for sampling
        """
        query = f'SELECT * FROM "{schema}"."{table_name}"'
        if limit:
            query += f" LIMIT {limit}"

        return pd.read_sql_query(query, self.engine)

    def estimate_count(self, table_name: str, schema: str = "public") -> int:
        """Fast estimate of row count using system catalogs."""
        query = text("""
            SELECT reltuples::bigint AS estimate
            FROM pg_class
            JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
            WHERE relname = :table AND nspname = :schema
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"table": table_name, "schema": schema}).scalar()
            if result is None or result == -1:
                # Fallback to exact count if stats missing (slower but accurate)
                return self.load_query(f'SELECT COUNT(*) FROM "{schema}"."{table_name}"').iloc[0, 0]
            return int(result)
