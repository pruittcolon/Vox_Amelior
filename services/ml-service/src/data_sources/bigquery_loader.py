import os

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account


class BigQueryLoader:
    """
    Universal BigQuery Loader.
    """

    def __init__(self, credentials_path: str | None = None, project_id: str | None = None):
        """
        Initialize BigQuery client.
        If credentials_path is provided, uses Service Account.
        Otherwise, falls back to default environment credentials (GOOGLE_APPLICATION_CREDENTIALS).
        """
        self.project_id = project_id

        if credentials_path and os.path.exists(credentials_path):
            self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = bigquery.Client(credentials=self.credentials, project=project_id)
        else:
            # Fallback to default auth
            self.client = bigquery.Client(project=project_id)

    def test_connection(self) -> bool:
        """Verify connection is alive by running a lightweight query."""
        try:
            query = "SELECT 1"
            query_job = self.client.query(query)
            query_job.result()
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def list_datasets(self) -> list[str]:
        """List all datasets in the project."""
        return [dataset.dataset_id for dataset in self.client.list_datasets()]

    def list_tables(self, dataset_id: str) -> list[str]:
        """List all tables in a dataset."""
        dataset_ref = self.client.dataset(dataset_id)
        return [table.table_id for table in self.client.list_tables(dataset_ref)]

    def get_schema(self, dataset_id: str, table_id: str) -> list[dict[str, str]]:
        """Get schema of a table."""
        table_ref = self.client.dataset(dataset_id).table(table_id)
        table = self.client.get_table(table_ref)
        return [{"name": field.name, "type": field.field_type, "mode": field.mode} for field in table.schema]

    def load_query(self, query: str) -> pd.DataFrame:
        """
        Load data from a SQL query.
        """
        query_job = self.client.query(query)
        return query_job.to_dataframe()

    def load_table(self, dataset_id: str, table_id: str, limit: int | None = None) -> pd.DataFrame:
        """
        Load data from a specific table.
        """
        table_ref = f"{self.client.project}.{dataset_id}.{table_id}"
        query = f"SELECT * FROM `{table_ref}`"
        if limit:
            query += f" LIMIT {limit}"

        return self.load_query(query)

    def estimate_count(self, dataset_id: str, table_id: str) -> int:
        """Fast estimate of row count using metadata."""
        table_ref = self.client.dataset(dataset_id).table(table_id)
        table = self.client.get_table(table_ref)
        return table.num_rows
