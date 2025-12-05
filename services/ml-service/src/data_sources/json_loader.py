import pandas as pd
import json
import os
from typing import Union, Dict, List, Optional

class JSONLoader:
    """
    Universal JSON Loader supporting standard JSON, JSON Lines, and nested structures.
    """
    
    def __init__(self):
        pass

    def detect_format(self, file_path: str) -> str:
        """
        Detect if file is standard JSON or JSON Lines.
        Returns: 'json' or 'jsonl'
        """
        try:
            with open(file_path, 'r') as f:
                first_char = f.read(1).strip()
                if first_char == '[':
                    return 'json' # Array of objects
                elif first_char == '{':
                    # Could be single object or JSON lines
                    # Read first line and try to parse
                    f.seek(0)
                    line = f.readline()
                    try:
                        json.loads(line)
                        # Check second line
                        line2 = f.readline()
                        if line2:
                            try:
                                json.loads(line2)
                                return 'jsonl'
                            except:
                                return 'json' # Single object?
                        return 'json' # Single line JSON
                    except:
                        return 'json' # Fallback
        except:
            pass
        return 'json'

    def load(self, file_path: str, orient: str = 'records', lines: bool = False, flatten: bool = True) -> pd.DataFrame:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file.
            orient: Expected JSON orientation (default 'records').
            lines: If True, read as JSON Lines.
            flatten: If True, use json_normalize to flatten nested structures.
            
        Returns:
            DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect lines if not specified
        if not lines and file_path.endswith('.jsonl'):
            lines = True
        elif not lines:
            fmt = self.detect_format(file_path)
            if fmt == 'jsonl':
                lines = True

        try:
            if flatten:
                # For flattening, we often need to load raw data first
                if lines:
                    data = []
                    with open(file_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                data.append(json.loads(line))
                else:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                
                # Normalize
                df = pd.json_normalize(data)
                return df
            else:
                # Standard pandas load
                return pd.read_json(file_path, orient=orient, lines=lines)
                
        except ValueError as e:
            raise ValueError(f"Failed to parse JSON: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error loading JSON file: {str(e)}")

    def load_sample(self, file_path: str, rows: int = 1000) -> pd.DataFrame:
        """Load a sample of rows."""
        # For JSONL, we can stream. For standard JSON, we might need to load all (or use ijson for streaming).
        # For now, simple implementation.
        
        fmt = self.detect_format(file_path)
        
        if fmt == 'jsonl' or file_path.endswith('.jsonl'):
            data = []
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= rows:
                        break
                    if line.strip():
                        data.append(json.loads(line))
            return pd.json_normalize(data)
        else:
            # Standard JSON - load all and head (optimization needed for massive files later)
            df = self.load(file_path)
            return df.head(rows)
