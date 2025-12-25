"""
CSV File Connector for RAG Ingestion.

Parses CSV files and converts rows to documents for vectorization.
"""

import csv
import logging
from io import StringIO
from pathlib import Path
from typing import Any, Optional

from shared.clients.connectors.base import (
    ConnectorConfig,
    Document,
    DocumentType,
    FileConnector,
)


logger = logging.getLogger(__name__)


class CSVConnector(FileConnector):
    """
    Connector for CSV file ingestion.

    Supports:
        - Single file or directory of CSVs
        - Row-level or grouped documents
        - Custom column selection
        - Header/no-header files
    """

    def __init__(
        self,
        config: ConnectorConfig,
        base_path: Optional[Path] = None,
        text_columns: Optional[list[str]] = None,
        id_column: Optional[str] = None,
        combine_rows: bool = False,
        rows_per_doc: int = 1,
    ):
        """
        Initialize CSV connector.

        Args:
            config: Connector configuration
            base_path: Base directory for files
            text_columns: Columns to use for content (None = all)
            id_column: Column to use for document ID
            combine_rows: Combine multiple rows into one document
            rows_per_doc: Rows per document if combining
        """
        super().__init__(config, base_path)
        self.text_columns = text_columns
        self.id_column = id_column
        self.combine_rows = combine_rows
        self.rows_per_doc = rows_per_doc

    async def fetch(
        self,
        query: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[Document]:
        """
        Fetch documents from CSV file(s).

        Args:
            query: Query with 'file_path' or 'content' key
            limit: Maximum documents to return

        Returns:
            List of Document objects
        """
        query = query or {}
        file_path = query.get("file_path")
        content = query.get("content")
        offset = query.get("offset", 0)

        if content:
            # Parse from string content
            return await self._parse_csv_content(content, limit, offset)
        elif file_path:
            # Parse from file
            resolved = self._resolve_path(file_path)
            valid, error = self._validate_file(resolved)
            if not valid:
                logger.error(error)
                return []
            return await self._parse_csv_file(resolved, limit, offset)
        else:
            logger.warning("CSVConnector.fetch requires 'file_path' or 'content' in query")
            return []

    async def _parse_csv_content(
        self,
        content: str,
        limit: int,
        offset: int = 0,
    ) -> list[Document]:
        """Parse CSV from string content."""
        reader = csv.DictReader(StringIO(content))
        return self._rows_to_documents(reader, limit, offset)

    async def _parse_csv_file(
        self,
        file_path: Path,
        limit: int,
        offset: int = 0,
    ) -> list[Document]:
        """Parse CSV from file path."""
        documents = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                documents = self._rows_to_documents(
                    reader, limit, offset, source_path=str(file_path)
                )
        except Exception as e:
            logger.error(f"Error parsing CSV {file_path}: {e}")
            self._error = str(e)

        return documents

    def _rows_to_documents(
        self,
        reader: csv.DictReader,
        limit: int,
        offset: int = 0,
        source_path: Optional[str] = None,
    ) -> list[Document]:
        """Convert CSV rows to documents."""
        documents = []
        rows = list(reader)

        if self.combine_rows:
            # Group rows into documents
            for i in range(offset, len(rows), self.rows_per_doc):
                if len(documents) >= limit:
                    break

                batch = rows[i : i + self.rows_per_doc]
                content = self._rows_to_content(batch)

                documents.append(
                    Document(
                        content=content,
                        doc_type=DocumentType.CSV,
                        source_path=source_path,
                        metadata={
                            "row_start": i,
                            "row_end": i + len(batch),
                            "row_count": len(batch),
                        },
                    )
                )
        else:
            # One document per row
            for i, row in enumerate(rows[offset:], start=offset):
                if len(documents) >= limit:
                    break

                content = self._row_to_content(row)
                doc_id = row.get(self.id_column) if self.id_column else None

                documents.append(
                    Document(
                        content=content,
                        doc_type=DocumentType.CSV,
                        source_path=source_path,
                        metadata={
                            "row_index": i,
                            "columns": list(row.keys()),
                            "source_id": doc_id,
                        },
                    )
                )

        return documents

    def _row_to_content(self, row: dict) -> str:
        """Convert single row to text content."""
        if self.text_columns:
            # Use only specified columns
            parts = [
                f"{col}: {row.get(col, '')}"
                for col in self.text_columns
                if col in row
            ]
        else:
            # Use all columns
            parts = [f"{k}: {v}" for k, v in row.items()]

        return "\n".join(parts)

    def _rows_to_content(self, rows: list[dict]) -> str:
        """Convert multiple rows to text content."""
        return "\n---\n".join(self._row_to_content(row) for row in rows)

    async def validate(self) -> dict[str, Any]:
        """Validate CSV connector configuration."""
        errors = []

        # Check base path
        if self.base_path and not self.base_path.exists():
            errors.append(f"Base path does not exist: {self.base_path}")

        # Check text columns on sample file
        sample_file = self.config.config.get("sample_file")
        if sample_file:
            resolved = self._resolve_path(sample_file)
            valid, error = self._validate_file(resolved)
            if not valid:
                errors.append(error)
            else:
                try:
                    with open(resolved, "r") as f:
                        reader = csv.DictReader(f)
                        headers = reader.fieldnames or []

                        if self.text_columns:
                            missing = [c for c in self.text_columns if c not in headers]
                            if missing:
                                errors.append(f"Missing columns: {missing}")

                        if self.id_column and self.id_column not in headers:
                            errors.append(f"ID column not found: {self.id_column}")

                except Exception as e:
                    errors.append(f"Validation error: {e}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "connector_type": "csv",
        }
