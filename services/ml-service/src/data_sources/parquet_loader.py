import os

import pandas as pd


class ParquetLoader:
    """
    Universal Parquet Loader supporting single files and partitioned datasets.
    """

    def __init__(self):
        pass

    def load(self, file_path: str, columns: list[str] | None = None, **kwargs) -> pd.DataFrame:
        """
        Load data from Parquet file or directory (partitioned).

        Args:
            file_path: Path to Parquet file or directory.
            columns: List of columns to read (optimization).
            **kwargs: Additional arguments passed to pd.read_parquet.

        Returns:
            DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Path not found: {file_path}")

        try:
            # pandas read_parquet handles both single files and directories (partitioned)
            # It uses pyarrow or fastparquet engine
            return pd.read_parquet(file_path, columns=columns, engine="pyarrow", **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load Parquet: {str(e)}")

    def get_schema(self, file_path: str) -> dict[str, str]:
        """Get schema without loading full data."""
        try:
            import pyarrow.parquet as pq

            schema = pq.read_schema(file_path)
            return {name: str(field.type) for name, field in zip(schema.names, schema)}
        except Exception as e:
            # Fallback: load empty
            try:
                df = pd.read_parquet(file_path, engine="pyarrow")
                return {col: str(dtype) for col, dtype in df.dtypes.items()}
            except Exception:
                raise RuntimeError(f"Failed to read schema: {str(e)}")

    def load_sample(self, file_path: str, rows: int = 1000) -> pd.DataFrame:
        """Load a sample of rows."""
        # Parquet doesn't support 'nrows' in read_parquet directly efficiently without reading row groups
        # But we can try to read just the first row group if using pyarrow directly
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(file_path)
            # Read first row group
            first_batch = next(parquet_file.iter_batches(batch_size=rows))
            return first_batch.to_pandas()
        except (StopIteration, Exception):
            # Fallback
            df = self.load(file_path)
            return df.head(rows)
