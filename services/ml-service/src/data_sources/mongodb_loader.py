from typing import Any

import pandas as pd
from pymongo import MongoClient


class MongoDBLoader:
    """
    Universal MongoDB Loader using PyMongo.
    """

    def __init__(self, connection_string: str, database_name: str | None = None):
        """
        Initialize with a connection string.
        Format: mongodb://user:pass@host:port/dbname
        """
        self.connection_string = connection_string
        self.client = MongoClient(connection_string)

        # If database is not specified in init, try to get from connection string or default
        if database_name:
            self.db = self.client[database_name]
        else:
            try:
                self.db = self.client.get_database()
            except:
                # Fallback if no default DB in URI
                self.db = None

    def test_connection(self) -> bool:
        """Verify connection is alive."""
        try:
            # The ismaster command is cheap and does not require auth.
            self.client.admin.command("ismaster")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def list_collections(self) -> list[str]:
        """List all collections in the database."""
        if self.db is None:
            raise ValueError("No database selected.")
        return self.db.list_collection_names()

    def get_schema(self, collection_name: str, sample_size: int = 100) -> dict[str, Any]:
        """
        Infer schema by sampling documents.
        MongoDB is schemaless, so we scan a sample to find fields and types.
        """
        if self.db is None:
            raise ValueError("No database selected.")

        collection = self.db[collection_name]
        cursor = collection.find().limit(sample_size)

        schema = {}
        for doc in cursor:
            for key, value in doc.items():
                if key not in schema:
                    schema[key] = {"type": type(value).__name__, "nullable": False}
                else:
                    # If we see different types, mark as Mixed? For now, keep first seen.
                    pass

        return schema

    def load_collection(self, collection_name: str, query: dict = {}, limit: int | None = None) -> pd.DataFrame:
        """
        Load data from a collection with optional query filter.
        """
        if self.db is None:
            raise ValueError("No database selected.")

        collection = self.db[collection_name]

        cursor = collection.find(query)
        if limit:
            cursor = cursor.limit(limit)

        # Convert to list then DataFrame
        data = list(cursor)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Convert _id to string usually
        if "_id" in df.columns:
            df["_id"] = df["_id"].astype(str)

        return df

    def estimate_count(self, collection_name: str) -> int:
        """Fast estimate of document count."""
        if self.db is None:
            raise ValueError("No database selected.")
        return self.db[collection_name].estimated_document_count()
