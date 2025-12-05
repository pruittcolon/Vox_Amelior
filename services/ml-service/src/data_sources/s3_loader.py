import pandas as pd
import boto3
import s3fs
from typing import List, Dict, Optional, Any, Union
import os
import io

class S3Loader:
    """
    Universal AWS S3 Loader.
    Supports listing buckets/objects and reading CSV, Parquet, JSON, Excel directly into Pandas.
    """
    
    def __init__(self, 
                 aws_access_key_id: Optional[str] = None, 
                 aws_secret_access_key: Optional[str] = None, 
                 region_name: Optional[str] = None,
                 endpoint_url: Optional[str] = None): # endpoint_url for MinIO/LocalStack support
        """
        Initialize S3 client.
        """
        self.session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.client = self.session.client('s3', endpoint_url=endpoint_url)
        self.resource = self.session.resource('s3', endpoint_url=endpoint_url)
        
        # For pandas s3fs integration
        self.storage_options = {}
        if aws_access_key_id and aws_secret_access_key:
            self.storage_options = {
                "key": aws_access_key_id,
                "secret": aws_secret_access_key,
                "client_kwargs": {}
            }
            if endpoint_url:
                self.storage_options["client_kwargs"]["endpoint_url"] = endpoint_url
            if region_name:
                self.storage_options["client_kwargs"]["region_name"] = region_name

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
        response = self.client.list_buckets()
        return [bucket['Name'] for bucket in response.get('Buckets', [])]

    def list_objects(self, bucket: str, prefix: str = '', max_keys: int = 100) -> List[Dict[str, Any]]:
        """List objects in a bucket with metadata."""
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
        objects = []
        for obj in response.get('Contents', []):
            objects.append({
                "key": obj['Key'],
                "size": obj['Size'],
                "last_modified": obj['LastModified']
            })
        return objects

    def read_file(self, bucket: str, key: str, file_type: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Read a file from S3 into a DataFrame.
        Auto-detects type from extension if file_type is not provided.
        """
        if not file_type:
            ext = key.split('.')[-1].lower()
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

        path = f"s3://{bucket}/{key}"
        
        if file_type == 'csv':
            return pd.read_csv(path, storage_options=self.storage_options, **kwargs)
        elif file_type == 'parquet':
            return pd.read_parquet(path, storage_options=self.storage_options, **kwargs)
        elif file_type == 'json':
            return pd.read_json(path, storage_options=self.storage_options, **kwargs)
        elif file_type == 'excel':
            return pd.read_excel(path, storage_options=self.storage_options, **kwargs)
        else:
            raise ValueError(f"Unknown file type: {file_type}")

    def get_object_metadata(self, bucket: str, key: str) -> Dict[str, Any]:
        """Get metadata for a specific object."""
        response = self.client.head_object(Bucket=bucket, Key=key)
        return {
            "content_type": response.get('ContentType'),
            "content_length": response.get('ContentLength'),
            "last_modified": response.get('LastModified'),
            "metadata": response.get('Metadata', {})
        }
