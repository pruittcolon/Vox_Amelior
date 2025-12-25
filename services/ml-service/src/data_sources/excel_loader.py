import os

import pandas as pd


class ExcelLoader:
    """
    Universal Excel Loader supporting .xlsx, .xls, .xlsm, .xlsb
    """

    ENGINE_MAP = {"xlsx": "openpyxl", "xlsm": "openpyxl", "xls": "xlrd", "xlsb": "pyxlsb"}

    def __init__(self):
        pass

    def _get_engine(self, file_path: str) -> str:
        ext = file_path.split(".")[-1].lower()
        return self.ENGINE_MAP.get(ext, "openpyxl")

    def list_sheets(self, file_path: str) -> list[str]:
        """List all sheet names in the Excel file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        engine = self._get_engine(file_path)
        excel_file = pd.ExcelFile(file_path, engine=engine)
        return excel_file.sheet_names

    def load(
        self, file_path: str, sheet_name: str | int | None | list[str | int] = 0, **kwargs
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Load data from Excel file.

        Args:
            file_path: Path to the Excel file.
            sheet_name: Sheet to load.
                        - 0 (default): First sheet
                        - 'Sheet1': Specific sheet
                        - None: All sheets (returns dict)
                        - [0, 'Sheet2']: Multiple sheets
            **kwargs: Additional arguments passed to pd.read_excel (e.g., skiprows, header)

        Returns:
            DataFrame or Dict of DataFrames
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        engine = self._get_engine(file_path)

        try:
            # If sheet_name is None (load all), we might want to inspect first to avoid OOM on massive files
            # But for now, we rely on pandas implementation
            data = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine, **kwargs)
            return data
        except ValueError as e:
            if "Worksheet" in str(e) and "does not exist" in str(e):
                raise ValueError(f"Sheet '{sheet_name}' not found in {file_path}")
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load Excel file: {str(e)}")

    def load_sample(self, file_path: str, sheet_name: str | int = 0, rows: int = 1000) -> pd.DataFrame:
        """Load a sample of rows from a specific sheet."""
        return self.load(file_path, sheet_name=sheet_name, nrows=rows)
