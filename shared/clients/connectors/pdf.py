"""
PDF File Connector for RAG Ingestion.

Extracts text from PDF files for vectorization.
Supports multiple extraction backends (pdfplumber, PyPDF2).
"""

import logging
from pathlib import Path
from typing import Any, Optional

from shared.clients.connectors.base import (
    ConnectorConfig,
    Document,
    DocumentType,
    FileConnector,
)


logger = logging.getLogger(__name__)


# Try to import PDF libraries
_PDFPLUMBER_AVAILABLE = False
_PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    pass

try:
    from PyPDF2 import PdfReader
    _PYPDF2_AVAILABLE = True
except ImportError:
    pass


class PDFConnector(FileConnector):
    """
    Connector for PDF file ingestion.

    Supports:
        - Page-level or full-document extraction
        - Multiple PDF backends (pdfplumber, PyPDF2)
        - Table extraction (with pdfplumber)
        - OCR integration ready
    """

    def __init__(
        self,
        config: ConnectorConfig,
        base_path: Optional[Path] = None,
        pages_per_doc: int = 1,
        extract_tables: bool = False,
        preferred_backend: str = "auto",
    ):
        """
        Initialize PDF connector.

        Args:
            config: Connector configuration
            base_path: Base directory for files
            pages_per_doc: Pages per document (0 = full doc)
            extract_tables: Extract tables as structured data
            preferred_backend: 'pdfplumber', 'pypdf2', or 'auto'
        """
        super().__init__(config, base_path)
        self.pages_per_doc = pages_per_doc
        self.extract_tables = extract_tables
        self.preferred_backend = preferred_backend

        # Select backend
        self._backend = self._select_backend()

    def _select_backend(self) -> str:
        """Select available PDF extraction backend."""
        if self.preferred_backend == "pdfplumber" and _PDFPLUMBER_AVAILABLE:
            return "pdfplumber"
        if self.preferred_backend == "pypdf2" and _PYPDF2_AVAILABLE:
            return "pypdf2"
        if self.preferred_backend == "auto":
            if _PDFPLUMBER_AVAILABLE:
                return "pdfplumber"
            if _PYPDF2_AVAILABLE:
                return "pypdf2"
        return "none"

    async def fetch(
        self,
        query: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[Document]:
        """
        Fetch documents from PDF file.

        Args:
            query: Query with 'file_path' or 'content' (bytes)
            limit: Maximum documents to return

        Returns:
            List of Document objects
        """
        query = query or {}
        file_path = query.get("file_path")
        content = query.get("content")  # bytes for in-memory PDF
        offset = query.get("offset", 0)

        if self._backend == "none":
            logger.error("No PDF backend available. Install pdfplumber or PyPDF2.")
            return []

        if file_path:
            resolved = self._resolve_path(file_path)
            valid, error = self._validate_file(resolved)
            if not valid:
                logger.error(error)
                return []
            return await self._extract_pdf(resolved, limit, offset)
        elif content:
            return await self._extract_pdf_bytes(content, limit, offset)
        else:
            logger.warning("PDFConnector.fetch requires 'file_path' or 'content'")
            return []

    async def _extract_pdf(
        self,
        file_path: Path,
        limit: int,
        offset: int = 0,
    ) -> list[Document]:
        """Extract text from PDF file."""
        if self._backend == "pdfplumber":
            return await self._extract_with_pdfplumber(file_path, limit, offset)
        elif self._backend == "pypdf2":
            return await self._extract_with_pypdf2(file_path, limit, offset)
        return []

    async def _extract_pdf_bytes(
        self,
        content: bytes,
        limit: int,
        offset: int = 0,
    ) -> list[Document]:
        """Extract text from PDF bytes."""
        # Write to temp file for processing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            return await self._extract_pdf(temp_path, limit, offset)
        finally:
            temp_path.unlink(missing_ok=True)

    async def _extract_with_pdfplumber(
        self,
        file_path: Path,
        limit: int,
        offset: int = 0,
    ) -> list[Document]:
        """Extract using pdfplumber (preferred for accuracy)."""
        documents = []

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)

                if self.pages_per_doc == 0:
                    # Full document as one Document
                    text_parts = []
                    tables = []

                    for page in pdf.pages:
                        text_parts.append(page.extract_text() or "")
                        if self.extract_tables:
                            tables.extend(page.extract_tables() or [])

                    content = "\n\n".join(text_parts)

                    documents.append(
                        Document(
                            content=content,
                            doc_type=DocumentType.PDF,
                            source_path=str(file_path),
                            metadata={
                                "total_pages": total_pages,
                                "table_count": len(tables),
                                "tables": tables if self.extract_tables else None,
                            },
                        )
                    )
                else:
                    # Page-level documents
                    for i in range(offset, total_pages, self.pages_per_doc):
                        if len(documents) >= limit:
                            break

                        pages_slice = pdf.pages[i : i + self.pages_per_doc]
                        text_parts = [p.extract_text() or "" for p in pages_slice]
                        content = "\n\n".join(text_parts)

                        documents.append(
                            Document(
                                content=content,
                                doc_type=DocumentType.PDF,
                                source_path=str(file_path),
                                metadata={
                                    "page_start": i + 1,
                                    "page_end": i + len(pages_slice),
                                    "total_pages": total_pages,
                                },
                            )
                        )

        except Exception as e:
            logger.error(f"pdfplumber extraction error for {file_path}: {e}")
            self._error = str(e)

        return documents

    async def _extract_with_pypdf2(
        self,
        file_path: Path,
        limit: int,
        offset: int = 0,
    ) -> list[Document]:
        """Extract using PyPDF2 (fallback)."""
        documents = []

        try:
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)

            if self.pages_per_doc == 0:
                # Full document
                text_parts = [page.extract_text() or "" for page in reader.pages]
                content = "\n\n".join(text_parts)

                documents.append(
                    Document(
                        content=content,
                        doc_type=DocumentType.PDF,
                        source_path=str(file_path),
                        metadata={"total_pages": total_pages},
                    )
                )
            else:
                # Page-level
                for i in range(offset, total_pages, self.pages_per_doc):
                    if len(documents) >= limit:
                        break

                    pages_slice = reader.pages[i : i + self.pages_per_doc]
                    text_parts = [p.extract_text() or "" for p in pages_slice]
                    content = "\n\n".join(text_parts)

                    documents.append(
                        Document(
                            content=content,
                            doc_type=DocumentType.PDF,
                            source_path=str(file_path),
                            metadata={
                                "page_start": i + 1,
                                "page_end": i + len(pages_slice),
                                "total_pages": total_pages,
                            },
                        )
                    )

        except Exception as e:
            logger.error(f"PyPDF2 extraction error for {file_path}: {e}")
            self._error = str(e)

        return documents

    async def validate(self) -> dict[str, Any]:
        """Validate PDF connector configuration."""
        errors = []
        warnings = []

        # Check backend availability
        if self._backend == "none":
            errors.append("No PDF backend available. Install pdfplumber or PyPDF2.")
        elif not _PDFPLUMBER_AVAILABLE and self.extract_tables:
            warnings.append("Table extraction requires pdfplumber. Using PyPDF2 fallback.")

        # Check base path
        if self.base_path and not self.base_path.exists():
            errors.append(f"Base path does not exist: {self.base_path}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "backend": self._backend,
            "connector_type": "pdf",
        }
