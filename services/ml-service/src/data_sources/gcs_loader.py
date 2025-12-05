import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
from typing import List, Dict, Optional, Any, Union
import os
import io

class GCSLoader:
    """
    Universal Google Cloud Storage Loader.
    Supports listing buckets/blobs and reading CSV, Parquet, JSON, Excel directly into Pandas.
    """
    
    def __init__(self, 
                 project_id: Optional[str] = None,
                 credentials_path: Optional[str] = None,
                 endpoint_url: Optional[str] = None): # endpoint_url for fake-gcs-server
        """
        Initialize GCS client.
        """
        self.project_id = project_id
        
        # Setup Client
        client_kwargs = {"project": project_id}
        
        if credentials_path and os.path.exists(credentials_path):
            self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
            client_kwargs["credentials"] = self.credentials
            
        if endpoint_url:
            # For testing with fake-gcs-server
            from google.api_core.client_options import ClientOptions
            client_kwargs["client_options"] = ClientOptions(api_endpoint=endpoint_url)
            
            # Also set for gcsfs
            self.storage_options = {"endpoint_url": endpoint_url}
            
            # If the env var is set, it might interfere if it lacks scheme, 
            # but we are using client_options which should take precedence.
        else:
            self.storage_options = {}
            if credentials_path:
                 self.storage_options["token"] = credentials_path

        self.client = storage.Client(**client_kwargs)

    def test_connection(self) -> bool:
        """Verify connection is alive."""
        try:
            self.client.list_buckets()
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def list_buckets(self) -> List[str]:
        """List all buckets."""
        return [bucket.name for bucket in self.client.list_buckets()]

    def list_blobs(self, bucket_name: str, prefix: str = '', max_results: int = 100) -> List[Dict[str, Any]]:
        """List blobs in a bucket with metadata."""
        bucket = self.client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)
        
        results = []
        for blob in blobs:
            results.append({
                "name": blob.name,
                "size": blob.size,
                "updated": blob.updated
            })
        return results

    def read_file(self, bucket_name: str, blob_name: str, file_type: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Read a file from GCS into a DataFrame.
        """
        if not file_type:
            ext = blob_name.split('.')[-1].lower()
            if ext in ['csv', 'txt']:
                file_type = 'csv'
            elif ext == 'parquet':
                file_type = 'parquet'
            elif ext in ['json', 'jsonl', 'ndjson']:
                file_type = 'json'
            elif ext in ['xlsx', 'xls']:
                file_type = 'excel'
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        path = f"gs://{bucket_name}/{blob_name}"
        
        # Merge storage_options
        storage_opts = self.storage_options.copy()
        
        if file_type == 'csv':
            return pd.read_csv(path, storage_options=storage_opts, **kwargs)
        elif file_type == 'parquet':
            return pd.read_parquet(path, storage_options=storage_opts, **kwargs)
        elif file_type == 'json':
            return pd.read_json(path, storage_options=storage_opts, **kwargs)
        elif file_type == 'excel':
            return pd.read_excel(path, storage_options=storage_opts, **kwargs)
        else:
            raise ValueError(f"Unknown file type: {file_type}")

    def get_blob_metadata(self, bucket_name: str, blob_name: str) -> Dict[str, Any]:
        """Get metadata for a specific blob."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.get_blob(blob_name)
        if not blob:
            return {}
            
        return {
            "content_type": blob.content_type,
            "size": blob.size,
            "updated": blob.updated,
            "metadata": blob.metadata
        }
