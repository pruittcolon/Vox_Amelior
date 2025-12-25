import pandas as pd
import snowflake.connector


class SnowflakeLoader:
    """
    Universal Snowflake Loader.
    """

    def __init__(
        self,
        user: str,
        password: str,
        account: str,
        warehouse: str | None = None,
        database: str | None = None,
        schema: str | None = None,
    ):
        """
        Initialize Snowflake connection.
        """
        self.conn = snowflake.connector.connect(
            user=user, password=password, account=account, warehouse=warehouse, database=database, schema=schema
        )

    def test_connection(self) -> bool:
        """Verify connection is alive."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def list_databases(self) -> list[str]:
        """List all databases."""
        cursor = self.conn.cursor()
        cursor.execute("SHOW DATABASES")
        return [row[1] for row in cursor.fetchall()]  # Name is usually 2nd col

    def _validate_identifier(self, identifier: str):
        """Ensure identifier contains only alphanumeric chars and underscores."""
        if not identifier.replace("_", "").isalnum():
            raise ValueError(f"Invalid identifier: {identifier}")

    def list_schemas(self, database: str) -> list[str]:
        """List schemas in a database."""
        self._validate_identifier(database)
        cursor = self.conn.cursor()
        cursor.execute(f"SHOW SCHEMAS IN DATABASE {database}")
        return [row[1] for row in cursor.fetchall()]

    def list_tables(self, database: str, schema: str) -> list[str]:
        """List tables in a schema."""
        self._validate_identifier(database)
        self._validate_identifier(schema)
        cursor = self.conn.cursor()
        cursor.execute(f"SHOW TABLES IN SCHEMA {database}.{schema}")
        return [row[1] for row in cursor.fetchall()]

    def get_schema(self, database: str, schema: str, table: str) -> list[dict[str, str]]:
        """Get column metadata."""
        self._validate_identifier(database)
        self._validate_identifier(schema)
        self._validate_identifier(table)
        cursor = self.conn.cursor()
        cursor.execute(f"DESCRIBE TABLE {database}.{schema}.{table}")
        # Result cols: name, type, kind, null?, default, primary key, unique key, check, expression, comment, policy name
        return [{"name": row[0], "type": row[1], "nullable": row[3] == "Y"} for row in cursor.fetchall()]

    def load_query(self, query: str) -> pd.DataFrame:
        """
        Load data from a SQL query.
        """
        cursor = self.conn.cursor()
        cursor.execute(query)
        # Fetch as pandas DataFrame
        return cursor.fetch_pandas_all()

    def load_table(self, database: str, schema: str, table: str, limit: int | None = None) -> pd.DataFrame:
        """
        Load data from a specific table.
        """
        self._validate_identifier(database)
        self._validate_identifier(schema)
        self._validate_identifier(table)
        query = f"SELECT * FROM {database}.{schema}.{table}"
        if limit:
            query += f" LIMIT {limit}"

        return self.load_query(query)

    def estimate_count(self, database: str, schema: str, table: str) -> int:
        """Fast estimate of row count."""
        self._validate_identifier(database)
        self._validate_identifier(schema)
        self._validate_identifier(table)
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {database}.{schema}.{table}")
        return cursor.fetchone()[0]
