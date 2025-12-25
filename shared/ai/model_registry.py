"""
AI Model Registry - Centralized model management for enterprise AI governance.

Provides model versioning, deployment tracking, and configuration management
for all AI/ML models in the platform.

Configuration:
- MODEL_REGISTRY_DB: Path to SQLite database (default: /data/model_registry.db)
- MODEL_CACHE_DIR: Directory for model artifacts

Usage:
    from shared.ai.model_registry import ModelRegistry, get_registry
    
    registry = get_registry()
    
    # Register a model
    model = registry.register_model(
        name="gemma-2b",
        version="1.0.0",
        model_type="llm",
        framework="llama.cpp",
    )
    
    # Get active model
    active = registry.get_active_model("gemma-2b")
"""

import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model deployment status."""
    REGISTERED = "registered"
    VALIDATING = "validating"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class ModelType(str, Enum):
    """Types of AI models."""
    LLM = "llm"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    NER = "ner"
    SPEAKER_ID = "speaker_id"
    TRANSCRIPTION = "transcription"
    CUSTOM = "custom"


@dataclass
class ModelMetadata:
    """Model metadata for governance and tracking."""
    id: str
    name: str
    version: str
    model_type: ModelType
    framework: str
    status: ModelStatus
    
    # Artifact tracking
    artifact_path: str | None = None
    artifact_hash: str | None = None
    artifact_size: int = 0
    
    # Governance
    owner: str = "system"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    
    # Performance
    avg_latency_ms: float | None = None
    throughput_rps: float | None = None
    
    # Resource requirements
    min_memory_mb: int = 0
    min_vram_mb: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    activated_at: datetime | None = None
    
    # Configuration
    config: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type.value,
            "framework": self.framework,
            "status": self.status.value,
            "artifact_path": self.artifact_path,
            "artifact_hash": self.artifact_hash,
            "artifact_size": self.artifact_size,
            "owner": self.owner,
            "description": self.description,
            "tags": self.tags,
            "avg_latency_ms": self.avg_latency_ms,
            "throughput_rps": self.throughput_rps,
            "min_memory_mb": self.min_memory_mb,
            "min_vram_mb": self.min_vram_mb,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "config": self.config,
        }


class ModelRegistry:
    """
    Centralized model registry for AI governance.
    
    Tracks model versions, deployments, and configurations.
    Supports model promotion workflows and rollback.
    """
    
    def __init__(self, db_path: str | None = None):
        """Initialize model registry."""
        self.db_path = db_path or os.getenv(
            "MODEL_REGISTRY_DB", "/data/model_registry.db"
        )
        self._ensure_db()
        logger.info(f"Model registry initialized: {self.db_path}")
    
    def _ensure_db(self) -> None:
        """Ensure database and tables exist."""
        db_dir = Path(self.db_path).parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'registered',
                    artifact_path TEXT,
                    artifact_hash TEXT,
                    artifact_size INTEGER DEFAULT 0,
                    owner TEXT DEFAULT 'system',
                    description TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    avg_latency_ms REAL,
                    throughput_rps REAL,
                    min_memory_mb INTEGER DEFAULT 0,
                    min_vram_mb INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    activated_at TEXT,
                    config TEXT DEFAULT '{}',
                    UNIQUE(name, version)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_events (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES models(id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_name ON models(name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_status ON models(status)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def register_model(
        self,
        name: str,
        version: str,
        model_type: ModelType | str,
        framework: str,
        artifact_path: str | None = None,
        owner: str = "system",
        description: str = "",
        tags: list[str] | None = None,
        min_memory_mb: int = 0,
        min_vram_mb: int = 0,
        config: dict | None = None,
    ) -> ModelMetadata:
        """
        Register a new model version.
        
        Args:
            name: Model name (e.g., "gemma-2b")
            version: Semantic version (e.g., "1.0.0")
            model_type: Type of model
            framework: ML framework (e.g., "llama.cpp", "transformers")
            artifact_path: Path to model files
            owner: Model owner/team
            description: Model description
            tags: Tags for categorization
            min_memory_mb: Minimum memory requirement
            min_vram_mb: Minimum VRAM requirement
            config: Model-specific configuration
            
        Returns:
            ModelMetadata for registered model
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        model_id = str(uuid4())
        now = datetime.now(timezone.utc)
        
        # Calculate artifact hash if path provided
        artifact_hash = None
        artifact_size = 0
        if artifact_path and os.path.exists(artifact_path):
            artifact_size = os.path.getsize(artifact_path)
            # Hash first 1MB for large files
            with open(artifact_path, "rb") as f:
                artifact_hash = hashlib.sha256(f.read(1024 * 1024)).hexdigest()
        
        model = ModelMetadata(
            id=model_id,
            name=name,
            version=version,
            model_type=model_type,
            framework=framework,
            status=ModelStatus.REGISTERED,
            artifact_path=artifact_path,
            artifact_hash=artifact_hash,
            artifact_size=artifact_size,
            owner=owner,
            description=description,
            tags=tags or [],
            min_memory_mb=min_memory_mb,
            min_vram_mb=min_vram_mb,
            created_at=now,
            updated_at=now,
            config=config or {},
        )
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO models (
                    id, name, version, model_type, framework, status,
                    artifact_path, artifact_hash, artifact_size,
                    owner, description, tags,
                    min_memory_mb, min_vram_mb,
                    created_at, updated_at, config
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model.id, model.name, model.version,
                model.model_type.value, model.framework, model.status.value,
                model.artifact_path, model.artifact_hash, model.artifact_size,
                model.owner, model.description, json.dumps(model.tags),
                model.min_memory_mb, model.min_vram_mb,
                model.created_at.isoformat(), model.updated_at.isoformat(),
                json.dumps(model.config),
            ))
            
            # Log event
            self._log_event(conn, model.id, "registered", f"Model {name}:{version} registered")
            
            conn.commit()
            logger.info(f"Registered model: {name}:{version}")
        finally:
            conn.close()
        
        return model
    
    def get_model(self, name: str, version: str) -> ModelMetadata | None:
        """Get specific model version."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT * FROM models WHERE name = ? AND version = ?
            """, (name, version))
            row = cursor.fetchone()
            if row:
                return self._row_to_model(cursor.description, row)
        finally:
            conn.close()
        return None
    
    def get_active_model(self, name: str) -> ModelMetadata | None:
        """Get currently active version of a model."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT * FROM models 
                WHERE name = ? AND status = 'active'
                ORDER BY activated_at DESC
                LIMIT 1
            """, (name,))
            row = cursor.fetchone()
            if row:
                return self._row_to_model(cursor.description, row)
        finally:
            conn.close()
        return None
    
    def list_models(
        self,
        name: str | None = None,
        model_type: ModelType | None = None,
        status: ModelStatus | None = None,
    ) -> list[ModelMetadata]:
        """List models with optional filters."""
        query = "SELECT * FROM models WHERE 1=1"
        params = []
        
        if name:
            query += " AND name = ?"
            params.append(name)
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type.value)
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY name, version DESC"
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(query, params)
            return [self._row_to_model(cursor.description, row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def activate_model(self, name: str, version: str) -> bool:
        """
        Activate a model version (promotes to production).
        
        Deactivates any currently active version.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            # Deactivate current active version
            conn.execute("""
                UPDATE models SET status = 'deprecated', updated_at = ?
                WHERE name = ? AND status = 'active'
            """, (now, name))
            
            # Activate new version
            cursor = conn.execute("""
                UPDATE models 
                SET status = 'active', activated_at = ?, updated_at = ?
                WHERE name = ? AND version = ?
            """, (now, now, name, version))
            
            if cursor.rowcount > 0:
                # Log events
                cursor = conn.execute(
                    "SELECT id FROM models WHERE name = ? AND version = ?",
                    (name, version)
                )
                row = cursor.fetchone()
                if row:
                    self._log_event(conn, row[0], "activated", f"Model activated")
                
                conn.commit()
                logger.info(f"Activated model: {name}:{version}")
                return True
        finally:
            conn.close()
        return False
    
    def deprecate_model(self, name: str, version: str) -> bool:
        """Mark a model version as deprecated."""
        return self._update_status(name, version, ModelStatus.DEPRECATED)
    
    def retire_model(self, name: str, version: str) -> bool:
        """Mark a model version as retired (no longer usable)."""
        return self._update_status(name, version, ModelStatus.RETIRED)
    
    def _update_status(self, name: str, version: str, status: ModelStatus) -> bool:
        """Update model status."""
        conn = sqlite3.connect(self.db_path)
        try:
            now = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute("""
                UPDATE models SET status = ?, updated_at = ?
                WHERE name = ? AND version = ?
            """, (status.value, now, name, version))
            
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"Model {name}:{version} status -> {status.value}")
                return True
        finally:
            conn.close()
        return False
    
    def update_metrics(
        self,
        name: str,
        version: str,
        avg_latency_ms: float | None = None,
        throughput_rps: float | None = None,
    ) -> bool:
        """Update model performance metrics."""
        conn = sqlite3.connect(self.db_path)
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            updates = ["updated_at = ?"]
            params = [now]
            
            if avg_latency_ms is not None:
                updates.append("avg_latency_ms = ?")
                params.append(avg_latency_ms)
            if throughput_rps is not None:
                updates.append("throughput_rps = ?")
                params.append(throughput_rps)
            
            params.extend([name, version])
            
            cursor = conn.execute(f"""
                UPDATE models SET {', '.join(updates)}
                WHERE name = ? AND version = ?
            """, params)
            
            if cursor.rowcount > 0:
                conn.commit()
                return True
        finally:
            conn.close()
        return False
    
    def _log_event(self, conn: sqlite3.Connection, model_id: str, event_type: str, details: str) -> None:
        """Log a model event."""
        conn.execute("""
            INSERT INTO model_events (id, model_id, event_type, details, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (str(uuid4()), model_id, event_type, details, datetime.now(timezone.utc).isoformat()))
    
    def get_model_events(self, model_id: str, limit: int = 50) -> list[dict]:
        """Get events for a model."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT id, model_id, event_type, details, created_at
                FROM model_events
                WHERE model_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (model_id, limit))
            
            return [
                {"id": r[0], "model_id": r[1], "event_type": r[2], "details": r[3], "created_at": r[4]}
                for r in cursor.fetchall()
            ]
        finally:
            conn.close()
    
    def _row_to_model(self, description: Any, row: tuple) -> ModelMetadata:
        """Convert database row to ModelMetadata."""
        columns = [d[0] for d in description]
        data = dict(zip(columns, row))
        
        return ModelMetadata(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            model_type=ModelType(data["model_type"]),
            framework=data["framework"],
            status=ModelStatus(data["status"]),
            artifact_path=data.get("artifact_path"),
            artifact_hash=data.get("artifact_hash"),
            artifact_size=data.get("artifact_size", 0),
            owner=data.get("owner", "system"),
            description=data.get("description", ""),
            tags=json.loads(data.get("tags", "[]")),
            avg_latency_ms=data.get("avg_latency_ms"),
            throughput_rps=data.get("throughput_rps"),
            min_memory_mb=data.get("min_memory_mb", 0),
            min_vram_mb=data.get("min_vram_mb", 0),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            activated_at=datetime.fromisoformat(data["activated_at"]) if data.get("activated_at") else None,
            config=json.loads(data.get("config", "{}")),
        )


# Singleton instance
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get or create model registry singleton."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
