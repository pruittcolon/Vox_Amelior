from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text


class MySQLLoader:
    """
    Universal MySQL/MariaDB Loader using SQLAlchemy + PyMySQL.
    """

    def __init__(self, connection_string: str):
        """
        Initialize with a connection string.
        Format: mysql+pymysql://user:pass@host:port/dbname
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

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_schema(self, table_name: str) -> dict[str, Any]:
        """Get column metadata for a table."""
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        return {col["name"]: {"type": str(col["type"]), "nullable": col["nullable"]} for col in columns}

    def load_query(self, query: str, params: dict | None = None) -> pd.DataFrame:
        """Load data from a raw SQL query."""
        return pd.read_sql_query(query, self.engine, params=params)

    def load_table(self, table_name: str, limit: int | None = None) -> pd.DataFrame:
        """
        Load data from a table.
        """
        query = f"SELECT * FROM `{table_name}`"
        if limit:
            query += f" LIMIT {limit}"

        return pd.read_sql_query(query, self.engine)

    def estimate_count(self, table_name: str) -> int:
        """Fast estimate of row count using information_schema."""
        # MySQL/MariaDB stores approximate counts in information_schema.tables
        # Note: This is an estimate for InnoDB, exact for MyISAM (rarely used now)
        db_name = self.engine.url.database
        query = text("""
            SELECT table_rows 
            FROM information_schema.tables 
            WHERE table_schema = :db AND table_name = :table
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"db": db_name, "table": table_name}).scalar()
            if result is None:
                # Fallback
                return self.load_query(f"SELECT COUNT(*) FROM `{table_name}`").iloc[0, 0]
            return int(result)
