import pandas as pd
from sqlalchemy import create_engine, inspect, text
from typing import List, Dict, Optional, Any
import os

class SQLiteLoader:
    """
    Universal SQLite Loader using SQLAlchemy.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize with a file path.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite DB not found: {db_path}")
            
        # SQLite connection string
        self.connection_string = f"sqlite:///{db_path}"
        self.engine = create_engine(self.connection_string)

    def test_connection(self) -> bool:
        """Verify connection is alive."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_schema(self, table_name: str) -> Dict[str, Any]:
        """Get column metadata for a table."""
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        return {
            col['name']: {
                'type': str(col['type']),
                'nullable': col['nullable']
            }
            for col in columns
        }

    def load_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Load data from a raw SQL query."""
        return pd.read_sql_query(query, self.engine, params=params)

    def load_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from a table.
        """
        query = f'SELECT * FROM "{table_name}"'
        if limit:
            query += f" LIMIT {limit}"
        
        return pd.read_sql_query(query, self.engine)

    def estimate_count(self, table_name: str) -> int:
        """Fast estimate of row count."""
        # SQLite doesn't have system catalogs for row counts like Postgres/MySQL
        # But COUNT(*) is relatively fast on SQLite
        return self.load_query(f'SELECT COUNT(*) FROM "{table_name}"').iloc[0, 0]
