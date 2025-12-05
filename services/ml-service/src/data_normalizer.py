"""
Data Normalizer - Universal Dataset Preprocessing

Standardizes any incoming dataset to work with all ML engines.
Uses temp_test_data/ as a staging area for normalized datasets.

Features:
- Multi-format support (CSV, Excel, JSON, Parquet, SQLite)
- Column name standardization
- Type inference and conversion
- Missing value handling
- Outlier detection and flagging
- Schema validation

Usage:
    from data_normalizer import DataNormalizer, normalize_dataset
    
    # Quick normalize
    df = normalize_dataset("path/to/data.csv")
    
    # Full pipeline with caching
    normalizer = DataNormalizer(temp_dir="/path/to/temp_test_data")
    df = normalizer.process("path/to/data.csv")
    
Author: Nemo Analytics Team
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NormalizationConfig:
    """Configuration for normalization process."""
    
    # Column name cleaning
    lowercase_columns: bool = True
    replace_spaces: str = "_"
    remove_special_chars: bool = True
    max_column_length: int = 64
    
    # Type handling
    infer_numeric: bool = True
    infer_datetime: bool = True
    infer_categorical: bool = True
    categorical_threshold: int = 20  # Max unique values to consider categorical
    
    # Missing values
    missing_representations: List[str] = field(default_factory=lambda: [
        'NA', 'N/A', 'null', 'NULL', 'None', 'none', 
        '', ' ', '-', '--', '?', 'n/a', 'NaN', 'nan'
    ])
    drop_all_null_columns: bool = True
    drop_all_null_rows: bool = True
    fill_numeric_na: Optional[float] = None  # None = don't fill
    fill_categorical_na: Optional[str] = None  # None = don't fill
    
    # Size limits
    max_rows: Optional[int] = None  # None = no limit
    max_columns: int = 500
    sample_large_datasets: bool = True
    sample_size: int = 100000
    
    # Output
    preserve_original_index: bool = False
    add_metadata_columns: bool = False


# =============================================================================
# DATA NORMALIZER CLASS
# =============================================================================

class DataNormalizer:
    """
    Universal dataset normalizer with caching support.
    
    Processes datasets through a temp directory for consistent results.
    """
    
    def __init__(
        self,
        temp_dir: Optional[Union[str, Path]] = None,
        config: Optional[NormalizationConfig] = None,
        cache_enabled: bool = True
    ):
        self.temp_dir = Path(temp_dir) if temp_dir else Path("data/temp_test_data")
        self.config = config or NormalizationConfig()
        self.cache_enabled = cache_enabled
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.temp_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def process(
        self,
        source: Union[str, Path, pd.DataFrame],
        output_name: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Process a dataset through the normalization pipeline.
        
        Args:
            source: File path or DataFrame to normalize
            output_name: Name for cached output (auto-generated if not provided)
            config_override: Override specific config options
            
        Returns:
            Normalized DataFrame
        """
        # Apply config overrides
        config = self.config
        if config_override:
            config = NormalizationConfig(**{
                **vars(self.config),
                **config_override
            })
        
        # Load data
        if isinstance(source, pd.DataFrame):
            df = source.copy()
            source_id = f"dataframe_{id(source)}"
        else:
            source_path = Path(source)
            source_id = self._get_source_hash(source_path)
            
            # Check cache
            if self.cache_enabled:
                cached = self._load_from_cache(source_id)
                if cached is not None:
                    logger.info(f"Loaded from cache: {source_path.name}")
                    return cached
            
            df = self._load_file(source_path)
        
        # Run normalization pipeline
        df = self._normalize(df, config)
        
        # Save to cache
        if self.cache_enabled and not isinstance(source, pd.DataFrame):
            self._save_to_cache(df, source_id, source_path.name)
        
        return df
    
    def _load_file(self, path: Path) -> pd.DataFrame:
        """Load data from various file formats."""
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            return pd.read_csv(path, low_memory=False)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        elif suffix == '.json':
            return pd.read_json(path)
        elif suffix == '.parquet':
            return pd.read_parquet(path)
        elif suffix in ['.sqlite', '.db']:
            return self._load_sqlite(path)
        elif suffix == '.jsonl':
            return pd.read_json(path, lines=True)
        elif suffix == '.tsv':
            return pd.read_csv(path, sep='\t', low_memory=False)
        else:
            # Try CSV as default
            return pd.read_csv(path, low_memory=False)
    
    def _load_sqlite(self, path: Path) -> pd.DataFrame:
        """Load first table from SQLite database."""
        import sqlite3
        conn = sqlite3.connect(str(path))
        
        # Get first table
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )['name'].tolist()
        
        if not tables:
            raise ValueError(f"No tables found in {path}")
        
        df = pd.read_sql_query(f"SELECT * FROM {tables[0]}", conn)
        conn.close()
        return df
    
    def _normalize(self, df: pd.DataFrame, config: NormalizationConfig) -> pd.DataFrame:
        """Run the full normalization pipeline."""
        
        # 1. Sample large datasets if needed
        if config.sample_large_datasets and config.max_rows:
            if len(df) > config.max_rows:
                logger.info(f"Sampling {config.sample_size} rows from {len(df)}")
                df = df.sample(n=config.sample_size, random_state=42)
        
        # 2. Clean column names
        df = self._clean_column_names(df, config)
        
        # 3. Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 4. Limit columns
        if len(df.columns) > config.max_columns:
            logger.warning(f"Truncating columns from {len(df.columns)} to {config.max_columns}")
            df = df.iloc[:, :config.max_columns]
        
        # 5. Handle missing value representations
        df = self._standardize_missing(df, config)
        
        # 6. Drop all-null columns and rows
        if config.drop_all_null_columns:
            null_cols = df.columns[df.isnull().all()].tolist()
            if null_cols:
                logger.info(f"Dropping {len(null_cols)} all-null columns")
                df = df.drop(columns=null_cols)
        
        if config.drop_all_null_rows:
            null_rows = df.isnull().all(axis=1).sum()
            if null_rows > 0:
                logger.info(f"Dropping {null_rows} all-null rows")
                df = df.dropna(how='all')
        
        # 7. Infer and convert types
        if config.infer_numeric:
            df = self._infer_numeric(df)
        
        if config.infer_datetime:
            df = self._infer_datetime(df)
        
        if config.infer_categorical:
            df = self._infer_categorical(df, config.categorical_threshold)
        
        # 8. Fill missing values if configured
        if config.fill_numeric_na is not None:
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(config.fill_numeric_na)
        
        if config.fill_categorical_na is not None:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            df[cat_cols] = df[cat_cols].fillna(config.fill_categorical_na)
        
        # 9. Reset index
        if not config.preserve_original_index:
            df = df.reset_index(drop=True)
        
        return df
    
    def _clean_column_names(self, df: pd.DataFrame, config: NormalizationConfig) -> pd.DataFrame:
        """Clean and standardize column names."""
        new_columns = []
        
        for col in df.columns:
            new_col = str(col)
            
            if config.lowercase_columns:
                new_col = new_col.lower()
            
            if config.replace_spaces:
                new_col = new_col.replace(" ", config.replace_spaces)
                new_col = new_col.replace("\t", config.replace_spaces)
            
            if config.remove_special_chars:
                new_col = re.sub(r'[^\w\s]', '', new_col)
                new_col = re.sub(r'_+', '_', new_col)  # Collapse multiple underscores
                new_col = new_col.strip('_')
            
            if config.max_column_length:
                new_col = new_col[:config.max_column_length]
            
            # Ensure unique names
            base_col = new_col
            counter = 1
            while new_col in new_columns:
                new_col = f"{base_col}_{counter}"
                counter += 1
            
            new_columns.append(new_col)
        
        df.columns = new_columns
        return df
    
    def _standardize_missing(self, df: pd.DataFrame, config: NormalizationConfig) -> pd.DataFrame:
        """Convert all missing value representations to pd.NA."""
        for rep in config.missing_representations:
            df = df.replace(rep, pd.NA)
        return df
    
    def _infer_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Try to convert object columns to numeric."""
        for col in df.select_dtypes(include=['object']).columns:
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                # Only convert if most values are valid numbers
                valid_ratio = converted.notna().sum() / len(df)
                if valid_ratio > 0.8:  # 80% threshold
                    df[col] = converted
            except:
                pass
        return df
    
    def _infer_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Try to convert columns to datetime."""
        date_patterns = ['date', 'time', 'created', 'updated', 'timestamp', 'dt', 'dob']
        
        for col in df.select_dtypes(include=['object']).columns:
            # Check column name hints
            if any(pattern in col.lower() for pattern in date_patterns):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        return df
    
    def _infer_categorical(self, df: pd.DataFrame, threshold: int) -> pd.DataFrame:
        """Convert low-cardinality object columns to category."""
        for col in df.select_dtypes(include=['object']).columns:
            unique_count = df[col].nunique()
            if unique_count <= threshold:
                df[col] = df[col].astype('category')
        return df
    
    def _get_source_hash(self, path: Path) -> str:
        """Generate hash for a source file."""
        stat = path.stat()
        content = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _load_from_cache(self, source_id: str) -> Optional[pd.DataFrame]:
        """Load normalized data from cache."""
        cache_path = self.cache_dir / f"{source_id}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, source_id: str, original_name: str):
        """Save normalized data to cache."""
        cache_path = self.cache_dir / f"{source_id}.parquet"
        df.to_parquet(cache_path, index=False)
        
        # Save metadata
        meta_path = self.cache_dir / f"{source_id}.meta.json"
        with open(meta_path, 'w') as f:
            json.dump({
                "original_name": original_name,
                "normalized_at": datetime.now().isoformat(),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist()
            }, f, indent=2)
    
    def clear_cache(self):
        """Clear all cached data."""
        for f in self.cache_dir.glob("*"):
            f.unlink()
        logger.info("Cache cleared")
    
    def get_cache_info(self) -> List[Dict[str, Any]]:
        """Get information about cached datasets."""
        info = []
        for meta_path in self.cache_dir.glob("*.meta.json"):
            with open(meta_path) as f:
                info.append(json.load(f))
        return info


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def normalize_dataset(
    source: Union[str, Path, pd.DataFrame],
    **config_kwargs
) -> pd.DataFrame:
    """
    Quick function to normalize a dataset.
    
    Args:
        source: File path or DataFrame
        **config_kwargs: Override default config options
        
    Returns:
        Normalized DataFrame
    """
    config = NormalizationConfig(**config_kwargs) if config_kwargs else None
    normalizer = DataNormalizer(config=config, cache_enabled=False)
    return normalizer.process(source)


def get_dataset_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a quick profile of a dataset.
    
    Returns:
        Dict with dataset statistics and column info
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "missing_values": int(df.isnull().sum().sum()),
        "missing_percent": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "datetime_columns": len(datetime_cols),
        "column_types": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols
        },
        "suggested_targets": _suggest_targets(df, numeric_cols, categorical_cols)
    }


def _suggest_targets(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> List[str]:
    """Suggest potential target columns for ML."""
    suggestions = []
    
    # Known target column names
    target_patterns = ['target', 'label', 'y', 'class', 'outcome', 'result', 'price', 'value', 'score']
    
    for col in df.columns:
        col_lower = col.lower()
        for pattern in target_patterns:
            if pattern in col_lower:
                suggestions.append(col)
                break
    
    # If no named targets, suggest last numeric or categorical column
    if not suggestions:
        if numeric_cols:
            # Prefer columns without 'id' in name
            non_id = [c for c in numeric_cols if 'id' not in c.lower()]
            if non_id:
                suggestions.append(non_id[-1])
        elif categorical_cols:
            suggestions.append(categorical_cols[-1])
    
    return suggestions[:3]  # Return top 3


def validate_for_engine(
    df: pd.DataFrame,
    engine_name: str,
    target_column: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate that a dataset is suitable for a specific engine.
    
    Args:
        df: DataFrame to validate
        engine_name: Name of engine to validate for
        target_column: Target column if required
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Basic checks
    if len(df) < 10:
        issues.append(f"Too few rows ({len(df)}), minimum is 10")
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 1:
        issues.append("No numeric columns found")
    
    # Engine-specific checks
    if engine_name == "titan":
        if not target_column and not _suggest_targets(df, list(numeric_cols), []):
            issues.append("Titan requires a target column but none detected")
        if len(df) < 50:
            issues.append("Titan works best with 50+ rows")
    
    elif engine_name == "predictive":
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) == 0:
            # Check for potential date columns
            potential = [c for c in df.columns if any(p in c.lower() for p in ['date', 'time', 'year', 'month'])]
            if not potential:
                issues.append("No datetime column detected for time series")
    
    elif engine_name == "clustering":
        if len(numeric_cols) < 2:
            issues.append("Clustering needs at least 2 numeric columns")
    
    return len(issues) == 0, issues
