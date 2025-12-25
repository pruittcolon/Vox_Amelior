"""
Prompt Version Management - Versioned prompt templates for AI governance.

Provides version-controlled prompt templates with A/B testing support,
rollback capabilities, and audit logging.

Usage:
    from shared.ai.prompts import PromptManager, get_prompt_manager
    
    manager = get_prompt_manager()
    
    # Create a prompt template
    prompt = manager.create_prompt(
        name="qa_assistant",
        template="You are a helpful assistant. Context: {context}\n\nQuestion: {question}",
        variables=["context", "question"],
    )
    
    # Render prompt with variables
    rendered = manager.render("qa_assistant", context="...", question="...")
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class PromptStatus(str, Enum):
    """Prompt template status."""
    DRAFT = "draft"
    ACTIVE = "active"
    TESTING = "testing"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class PromptTemplate:
    """A versioned prompt template."""
    id: str
    name: str
    version: int
    template: str
    variables: list[str]
    status: PromptStatus
    
    # Metadata
    description: str = ""
    owner: str = "system"
    tags: list[str] = field(default_factory=list)
    
    # A/B testing
    weight: float = 1.0  # For A/B testing distribution
    
    # Configuration
    max_tokens: int | None = None
    temperature: float | None = None
    model_hint: str | None = None
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Hash for integrity
    content_hash: str = ""
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.template.encode()).hexdigest()[:16]
    
    def render(self, **kwargs: Any) -> str:
        """Render template with variables."""
        result = self.template
        for var in self.variables:
            placeholder = f"{{{var}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(kwargs.get(var, "")))
        return result
    
    def validate_variables(self, **kwargs: Any) -> list[str]:
        """Check for missing required variables."""
        missing = []
        for var in self.variables:
            if var not in kwargs:
                missing.append(var)
        return missing
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "template": self.template,
            "variables": self.variables,
            "status": self.status.value,
            "description": self.description,
            "owner": self.owner,
            "tags": self.tags,
            "weight": self.weight,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "model_hint": self.model_hint,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "content_hash": self.content_hash,
        }


class PromptManager:
    """
    Versioned prompt template manager.
    
    Supports prompt versioning, A/B testing, and audit logging.
    """
    
    def __init__(self, db_path: str | None = None):
        """Initialize prompt manager."""
        self.db_path = db_path or os.getenv(
            "PROMPT_DB", "/data/prompts.db"
        )
        self._ensure_db()
        self._cache: dict[str, PromptTemplate] = {}  # name -> active prompt
        logger.info(f"Prompt manager initialized: {self.db_path}")
    
    def _ensure_db(self) -> None:
        """Ensure database and tables exist."""
        from pathlib import Path
        db_dir = Path(self.db_path).parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    template TEXT NOT NULL,
                    variables TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'draft',
                    description TEXT DEFAULT '',
                    owner TEXT DEFAULT 'system',
                    tags TEXT DEFAULT '[]',
                    weight REAL DEFAULT 1.0,
                    max_tokens INTEGER,
                    temperature REAL,
                    model_hint TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    UNIQUE(name, version)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_usage (
                    id TEXT PRIMARY KEY,
                    prompt_id TEXT NOT NULL,
                    rendered_hash TEXT,
                    client_id TEXT,
                    success INTEGER,
                    latency_ms REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (prompt_id) REFERENCES prompts(id)
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_name ON prompts(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_status ON prompts(status)")
            
            conn.commit()
        finally:
            conn.close()
    
    def create_prompt(
        self,
        name: str,
        template: str,
        variables: list[str] | None = None,
        description: str = "",
        owner: str = "system",
        tags: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model_hint: str | None = None,
    ) -> PromptTemplate:
        """
        Create a new prompt template.
        
        Automatically increments version if prompt with name exists.
        """
        # Auto-detect variables from template
        if variables is None:
            variables = self._extract_variables(template)
        
        # Get next version
        next_version = self._get_next_version(name)
        
        prompt = PromptTemplate(
            id=str(uuid4()),
            name=name,
            version=next_version,
            template=template,
            variables=variables,
            status=PromptStatus.DRAFT,
            description=description,
            owner=owner,
            tags=tags or [],
            max_tokens=max_tokens,
            temperature=temperature,
            model_hint=model_hint,
        )
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO prompts (
                    id, name, version, template, variables, status,
                    description, owner, tags, weight,
                    max_tokens, temperature, model_hint,
                    created_at, updated_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prompt.id, prompt.name, prompt.version, prompt.template,
                json.dumps(prompt.variables), prompt.status.value,
                prompt.description, prompt.owner, json.dumps(prompt.tags),
                prompt.weight, prompt.max_tokens, prompt.temperature,
                prompt.model_hint, prompt.created_at.isoformat(),
                prompt.updated_at.isoformat(), prompt.content_hash,
            ))
            conn.commit()
            logger.info(f"Created prompt: {name} v{next_version}")
        finally:
            conn.close()
        
        return prompt
    
    def get_prompt(self, name: str, version: int | None = None) -> PromptTemplate | None:
        """Get a prompt by name and optional version."""
        conn = sqlite3.connect(self.db_path)
        try:
            if version:
                cursor = conn.execute(
                    "SELECT * FROM prompts WHERE name = ? AND version = ?",
                    (name, version)
                )
            else:
                # Get active version
                cursor = conn.execute(
                    "SELECT * FROM prompts WHERE name = ? AND status = 'active' ORDER BY version DESC LIMIT 1",
                    (name,)
                )
            
            row = cursor.fetchone()
            if row:
                return self._row_to_prompt(cursor.description, row)
        finally:
            conn.close()
        return None
    
    def get_active_prompt(self, name: str) -> PromptTemplate | None:
        """Get the active version of a prompt."""
        # Check cache first
        if name in self._cache:
            return self._cache[name]
        
        prompt = self.get_prompt(name)
        if prompt:
            self._cache[name] = prompt
        return prompt
    
    def activate_prompt(self, name: str, version: int) -> bool:
        """Activate a prompt version (deprecates current active)."""
        conn = sqlite3.connect(self.db_path)
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            # Deprecate current active
            conn.execute("""
                UPDATE prompts SET status = 'deprecated', updated_at = ?
                WHERE name = ? AND status = 'active'
            """, (now, name))
            
            # Activate new version
            cursor = conn.execute("""
                UPDATE prompts SET status = 'active', updated_at = ?
                WHERE name = ? AND version = ?
            """, (now, name, version))
            
            if cursor.rowcount > 0:
                conn.commit()
                # Clear cache
                self._cache.pop(name, None)
                logger.info(f"Activated prompt: {name} v{version}")
                return True
        finally:
            conn.close()
        return False
    
    def render(self, name: str, version: int | None = None, **kwargs: Any) -> str:
        """Render a prompt template with variables."""
        prompt = self.get_active_prompt(name) if version is None else self.get_prompt(name, version)
        
        if not prompt:
            raise ValueError(f"Prompt not found: {name}")
        
        missing = prompt.validate_variables(**kwargs)
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        
        return prompt.render(**kwargs)
    
    def list_prompts(
        self,
        status: PromptStatus | None = None,
        tag: str | None = None,
    ) -> list[PromptTemplate]:
        """List prompts with optional filters."""
        query = "SELECT * FROM prompts WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        if tag:
            query += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')
        
        query += " ORDER BY name, version DESC"
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(query, params)
            return [self._row_to_prompt(cursor.description, row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def log_usage(
        self,
        prompt_id: str,
        client_id: str | None = None,
        success: bool = True,
        latency_ms: float | None = None,
        rendered_hash: str | None = None,
    ) -> None:
        """Log prompt usage for analytics."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO prompt_usage (id, prompt_id, rendered_hash, client_id, success, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid4()), prompt_id, rendered_hash, client_id,
                1 if success else 0, latency_ms,
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()
        finally:
            conn.close()
    
    def get_usage_stats(self, prompt_name: str) -> dict:
        """Get usage statistics for a prompt."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_uses,
                    SUM(success) as successes,
                    AVG(latency_ms) as avg_latency
                FROM prompt_usage pu
                JOIN prompts p ON pu.prompt_id = p.id
                WHERE p.name = ?
            """, (prompt_name,))
            row = cursor.fetchone()
            
            return {
                "prompt_name": prompt_name,
                "total_uses": row[0] or 0,
                "successes": row[1] or 0,
                "success_rate": (row[1] / row[0] * 100) if row[0] else 0,
                "avg_latency_ms": round(row[2], 2) if row[2] else None,
            }
        finally:
            conn.close()
    
    def _get_next_version(self, name: str) -> int:
        """Get next version number for a prompt name."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT MAX(version) FROM prompts WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()
            return (row[0] or 0) + 1
        finally:
            conn.close()
    
    def _extract_variables(self, template: str) -> list[str]:
        """Extract variable names from template."""
        pattern = re.compile(r'\{(\w+)\}')
        return list(set(pattern.findall(template)))
    
    def _row_to_prompt(self, description: Any, row: tuple) -> PromptTemplate:
        """Convert database row to PromptTemplate."""
        columns = [d[0] for d in description]
        data = dict(zip(columns, row))
        
        return PromptTemplate(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            template=data["template"],
            variables=json.loads(data["variables"]),
            status=PromptStatus(data["status"]),
            description=data.get("description", ""),
            owner=data.get("owner", "system"),
            tags=json.loads(data.get("tags", "[]")),
            weight=data.get("weight", 1.0),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            model_hint=data.get("model_hint"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            content_hash=data["content_hash"],
        )


# Singleton instance
_manager: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    """Get or create prompt manager singleton."""
    global _manager
    if _manager is None:
        _manager = PromptManager()
    return _manager
