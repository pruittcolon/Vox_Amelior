"""
RAG Ingestion Pipeline.

Provides async document ingestion workflow:
1. Accept file upload or content
2. Extract text (via connectors)
3. Detect and redact PII
4. Chunk for vectorization
5. Generate embeddings
6. Store in vector DB

Follows 2024 RAG best practices for secure, scalable ingestion.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4


logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Ingestion job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class IngestionJob:
    """Ingestion job tracking."""

    id: UUID = field(default_factory=uuid4)
    tenant_id: Optional[UUID] = None
    status: JobStatus = JobStatus.PENDING
    source_type: str = "file"
    source_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    progress: float = 0.0
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id) if self.tenant_id else None,
            "status": self.status.value,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "progress": self.progress,
            "stats": self.stats,
        }


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline."""

    chunk_size: int = 500
    chunk_overlap: int = 50
    chunk_strategy: str = "fixed"  # fixed, sentence, paragraph, recursive
    detect_pii: bool = True
    redact_pii: bool = True
    generate_embeddings: bool = True
    tenant_id: Optional[UUID] = None
    metadata: dict = field(default_factory=dict)


class IngestionPipeline:
    """
    Async document ingestion pipeline for RAG.

    Usage:
        pipeline = IngestionPipeline()
        job = await pipeline.ingest_file(Path("document.pdf"))
        status = await pipeline.get_job_status(job.id)
    """

    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        embedding_service_url: Optional[str] = None,
        vector_store_url: Optional[str] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Ingestion configuration
            embedding_service_url: URL for embedding generation
            vector_store_url: URL for vector database
        """
        self.config = config or IngestionConfig()
        self.embedding_url = embedding_service_url
        self.vector_store_url = vector_store_url

        # Job tracking
        self._jobs: dict[UUID, IngestionJob] = {}

        # Import utilities (lazy to avoid circular imports)
        self._chunker = None
        self._pii_detector = None

    def _get_chunker(self):
        """Lazy load chunking module."""
        if self._chunker is None:
            from shared.utils import chunking
            self._chunker = chunking
        return self._chunker

    def _get_pii_detector(self):
        """Lazy load PII detector."""
        if self._pii_detector is None:
            from shared.security.pii_detector import PIIDetector
            self._pii_detector = PIIDetector()
        return self._pii_detector

    async def ingest_file(
        self,
        file_path: Path,
        config: Optional[IngestionConfig] = None,
    ) -> IngestionJob:
        """
        Ingest a file into the knowledge base.

        Args:
            file_path: Path to file to ingest
            config: Override default config

        Returns:
            IngestionJob for tracking
        """
        cfg = config or self.config

        # Create job
        job = IngestionJob(
            tenant_id=cfg.tenant_id,
            source_type="file",
            source_path=str(file_path),
        )
        self._jobs[job.id] = job

        # Start async processing
        asyncio.create_task(self._process_file(job, file_path, cfg))

        return job

    async def ingest_content(
        self,
        content: str,
        source_name: str = "inline",
        config: Optional[IngestionConfig] = None,
    ) -> IngestionJob:
        """
        Ingest raw text content.

        Args:
            content: Text content to ingest
            source_name: Name for the content source
            config: Override default config

        Returns:
            IngestionJob for tracking
        """
        cfg = config or self.config

        job = IngestionJob(
            tenant_id=cfg.tenant_id,
            source_type="content",
            source_path=source_name,
        )
        self._jobs[job.id] = job

        asyncio.create_task(self._process_content(job, content, cfg))

        return job

    async def _process_file(
        self,
        job: IngestionJob,
        file_path: Path,
        config: IngestionConfig,
    ) -> None:
        """Process a file through the pipeline."""
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()

            # 1. Determine file type and load connector
            suffix = file_path.suffix.lower()
            content = ""

            if suffix == ".csv":
                from shared.clients.connectors.csv import CSVConnector, ConnectorConfig
                connector_cfg = ConnectorConfig(name="csv", connector_type="csv")
                connector = CSVConnector(connector_cfg)
                docs = await connector.fetch({"file_path": str(file_path)})
                content = "\n\n".join(d.content for d in docs)

            elif suffix == ".pdf":
                from shared.clients.connectors.pdf import PDFConnector, ConnectorConfig
                connector_cfg = ConnectorConfig(name="pdf", connector_type="pdf")
                connector = PDFConnector(connector_cfg, pages_per_doc=0)
                docs = await connector.fetch({"file_path": str(file_path)})
                content = "\n\n".join(d.content for d in docs)

            else:
                # Plain text
                content = file_path.read_text(encoding="utf-8")

            job.progress = 0.2
            job.stats["raw_chars"] = len(content)

            # Process content
            await self._process_content_internal(job, content, config)

        except Exception as e:
            logger.error(f"Ingestion failed for {file_path}: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.utcnow()

    async def _process_content(
        self,
        job: IngestionJob,
        content: str,
        config: IngestionConfig,
    ) -> None:
        """Process content through the pipeline."""
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            job.stats["raw_chars"] = len(content)

            await self._process_content_internal(job, content, config)

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.utcnow()

    async def _process_content_internal(
        self,
        job: IngestionJob,
        content: str,
        config: IngestionConfig,
    ) -> None:
        """Internal content processing."""
        try:
            # 2. PII Detection and Redaction
            if config.detect_pii:
                detector = self._get_pii_detector()
                report = detector.scan_and_report(content)
                job.stats["pii_found"] = report["total_pii_found"]
                job.stats["pii_types"] = report["types_found"]
                job.stats["risk_level"] = report["risk_level"]

                if config.redact_pii and report["total_pii_found"] > 0:
                    content = detector.redact(content)
                    job.stats["pii_redacted"] = True

            job.progress = 0.4

            # 3. Chunking
            chunker = self._get_chunker()

            if config.chunk_strategy == "sentence":
                chunks = chunker.sentence_chunk(content)
            elif config.chunk_strategy == "paragraph":
                chunks = chunker.paragraph_chunk(content)
            elif config.chunk_strategy == "recursive":
                chunks = chunker.recursive_chunk(content)
            else:
                chunks = chunker.fixed_size_chunk(
                    content, config.chunk_size, config.chunk_overlap
                )

            job.stats["chunk_count"] = len(chunks)
            job.progress = 0.6

            # 4. Generate embeddings (if service available)
            embeddings = []
            if config.generate_embeddings and self.embedding_url:
                embeddings = await self._generate_embeddings(
                    [c.content for c in chunks]
                )
                job.stats["embeddings_generated"] = len(embeddings)

            job.progress = 0.8

            # 5. Store in vector DB (if available)
            if self.vector_store_url and embeddings:
                await self._store_vectors(chunks, embeddings, config)
                job.stats["vectors_stored"] = len(embeddings)

            job.progress = 1.0
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()

            logger.info(f"Ingestion completed: {job.id}, {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.utcnow()

    async def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for text chunks."""
        if not self.embedding_url:
            return []

        try:
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.embedding_url}/embed",
                    json={"texts": texts},
                )
                response.raise_for_status()
                return response.json().get("embeddings", [])
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

    async def _store_vectors(
        self,
        chunks,
        embeddings: list[list[float]],
        config: IngestionConfig,
    ) -> None:
        """Store vectors in vector database."""
        if not self.vector_store_url:
            return

        try:
            import httpx
            documents = [
                {
                    "id": str(uuid4()),
                    "content": chunk.content,
                    "embedding": emb,
                    "metadata": {
                        **chunk.metadata,
                        **config.metadata,
                        "tenant_id": str(config.tenant_id) if config.tenant_id else None,
                    },
                }
                for chunk, emb in zip(chunks, embeddings)
            ]

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.vector_store_url}/upsert",
                    json={"documents": documents},
                )
                response.raise_for_status()

        except Exception as e:
            logger.error(f"Vector storage failed: {e}")

    async def get_job_status(self, job_id: UUID) -> Optional[IngestionJob]:
        """Get job status by ID."""
        return self._jobs.get(job_id)

    async def cancel_job(self, job_id: UUID) -> bool:
        """Cancel a pending or running job."""
        job = self._jobs.get(job_id)
        if job and job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            return True
        return False

    def list_jobs(
        self,
        tenant_id: Optional[UUID] = None,
        status: Optional[JobStatus] = None,
    ) -> list[IngestionJob]:
        """List jobs with optional filtering."""
        jobs = list(self._jobs.values())

        if tenant_id:
            jobs = [j for j in jobs if j.tenant_id == tenant_id]
        if status:
            jobs = [j for j in jobs if j.status == status]

        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
