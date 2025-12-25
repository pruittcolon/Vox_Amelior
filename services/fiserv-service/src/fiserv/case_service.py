"""
Case Management Service

Real database-backed case management for fraud, disputes, complaints, and compliance.
NO DEMO DATA - All cases stored in PostgreSQL.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from uuid import uuid4

import asyncpg

logger = logging.getLogger(__name__)


def _read_secret(secret_name: str, default: str = "") -> str:
    """Read a Docker secret from /run/secrets/ or environment variable."""
    secret_path = Path(f"/run/secrets/{secret_name}")
    if secret_path.exists():
        return secret_path.read_text().strip()
    return os.getenv(secret_name.upper(), default)


# Build DATABASE_URL from Docker secrets
_db_user = _read_secret("postgres_user", "nemo_user")
_db_password = quote_plus(_read_secret("postgres_password", "password"))  # URL-encode special chars
_db_host = os.getenv("POSTGRES_HOST", "postgres")
_db_name = os.getenv("POSTGRES_DB", "nemo_queue")

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"postgresql://{_db_user}:{_db_password}@{_db_host}:5432/{_db_name}"
)


class CaseService:
    """
    Database-backed case management service.
    
    Handles CRUD operations for cases with full audit trail.
    """
    
    _pool: asyncpg.Pool | None = None
    
    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        """Get or create database connection pool."""
        if cls._pool is None:
            cls._pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
            await cls._ensure_tables_exist()
        return cls._pool
    
    @classmethod
    async def _ensure_tables_exist(cls) -> None:
        """Create tables if they don't exist."""
        pool = cls._pool
        if not pool:
            return
            
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cases (
                    id VARCHAR(50) PRIMARY KEY,
                    type VARCHAR(50) NOT NULL,
                    subject VARCHAR(500) NOT NULL,
                    description TEXT,
                    member_id VARCHAR(50),
                    account_id VARCHAR(50),
                    priority VARCHAR(20) NOT NULL DEFAULT 'medium',
                    status VARCHAR(30) NOT NULL DEFAULT 'open',
                    assignee_id VARCHAR(50),
                    assignee_name VARCHAR(200),
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    due_date TIMESTAMP,
                    resolved_at TIMESTAMP,
                    resolution_summary TEXT
                );
                
                CREATE TABLE IF NOT EXISTS case_timeline (
                    id SERIAL PRIMARY KEY,
                    case_id VARCHAR(50) REFERENCES cases(id) ON DELETE CASCADE,
                    event_type VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    user_name VARCHAR(200) NOT NULL,
                    note TEXT,
                    metadata JSONB
                );
                
                CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
                CREATE INDEX IF NOT EXISTS idx_cases_priority ON cases(priority);
                CREATE INDEX IF NOT EXISTS idx_cases_assignee ON cases(assignee_id);
                CREATE INDEX IF NOT EXISTS idx_timeline_case ON case_timeline(case_id);
            """)
            logger.info("[CaseService] Database tables verified")
    
    async def create_case(
        self,
        case_type: str,
        subject: str,
        description: str | None = None,
        member_id: str | None = None,
        account_id: str | None = None,
        priority: str = "medium",
        assignee_id: str | None = None,
        assignee_name: str | None = None,
        due_date: datetime | None = None,
        created_by: str = "System"
    ) -> dict[str, Any]:
        """Create a new case."""
        pool = await self.get_pool()
        
        case_id = f"CASE-{datetime.now().year}-{uuid4().hex[:6].upper()}"
        now = datetime.now()
        
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO cases (id, type, subject, description, member_id, account_id, 
                                   priority, status, assignee_id, assignee_name, created_at, updated_at, due_date)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $11, $12)
            """, case_id, case_type, subject, description, member_id, account_id,
                priority, "open", assignee_id, assignee_name, now, due_date)
            
            # Add creation timeline entry
            await conn.execute("""
                INSERT INTO case_timeline (case_id, event_type, timestamp, user_name, note)
                VALUES ($1, 'created', $2, $3, $4)
            """, case_id, now, created_by, f"Case created: {subject}")
            
            if assignee_name:
                await conn.execute("""
                    INSERT INTO case_timeline (case_id, event_type, timestamp, user_name, note)
                    VALUES ($1, 'assigned', $2, $3, $4)
                """, case_id, now, created_by, f"Assigned to {assignee_name}")
        
        logger.info(f"[CaseService] Created case {case_id}")
        return await self.get_case(case_id)
    
    async def get_case(self, case_id: str) -> dict[str, Any] | None:
        """Get a single case by ID with timeline."""
        pool = await self.get_pool()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM cases WHERE id = $1", case_id)
            if not row:
                return None
            
            timeline_rows = await conn.fetch(
                "SELECT * FROM case_timeline WHERE case_id = $1 ORDER BY timestamp ASC",
                case_id
            )
            
            return {
                **dict(row),
                "timeline": [dict(t) for t in timeline_rows]
            }
    
    async def list_cases(
        self,
        status: str | None = None,
        priority: str | None = None,
        case_type: str | None = None,
        assignee_id: str | None = None,
        limit: int = 50
    ) -> list[dict[str, Any]]:
        """List cases with optional filters."""
        pool = await self.get_pool()
        
        conditions = []
        params = []
        param_idx = 1
        
        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1
        if priority:
            conditions.append(f"priority = ${param_idx}")
            params.append(priority)
            param_idx += 1
        if case_type:
            conditions.append(f"type = ${param_idx}")
            params.append(case_type)
            param_idx += 1
        if assignee_id:
            conditions.append(f"assignee_id = ${param_idx}")
            params.append(assignee_id)
            param_idx += 1
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT * FROM cases {where_clause}
                ORDER BY 
                    CASE priority 
                        WHEN 'critical' THEN 1 
                        WHEN 'high' THEN 2 
                        WHEN 'medium' THEN 3 
                        ELSE 4 
                    END,
                    created_at DESC
                LIMIT ${param_idx}
            """, *params)
            
            return [dict(r) for r in rows]
    
    async def update_case(
        self,
        case_id: str,
        updated_by: str,
        status: str | None = None,
        priority: str | None = None,
        assignee_id: str | None = None,
        assignee_name: str | None = None,
        resolution_summary: str | None = None
    ) -> dict[str, Any] | None:
        """Update a case."""
        pool = await self.get_pool()
        
        updates = ["updated_at = NOW()"]
        params = []
        param_idx = 1
        timeline_notes = []
        
        if status:
            updates.append(f"status = ${param_idx}")
            params.append(status)
            timeline_notes.append(f"Status changed to {status}")
            if status == "closed":
                updates.append("resolved_at = NOW()")
            param_idx += 1
        
        if priority:
            updates.append(f"priority = ${param_idx}")
            params.append(priority)
            timeline_notes.append(f"Priority changed to {priority}")
            param_idx += 1
        
        if assignee_id is not None:
            updates.append(f"assignee_id = ${param_idx}")
            params.append(assignee_id)
            param_idx += 1
            updates.append(f"assignee_name = ${param_idx}")
            params.append(assignee_name)
            timeline_notes.append(f"Assigned to {assignee_name or 'Unassigned'}")
            param_idx += 1
        
        if resolution_summary:
            updates.append(f"resolution_summary = ${param_idx}")
            params.append(resolution_summary)
            timeline_notes.append(f"Resolution: {resolution_summary}")
            param_idx += 1
        
        params.append(case_id)
        
        async with pool.acquire() as conn:
            await conn.execute(f"""
                UPDATE cases SET {', '.join(updates)} WHERE id = ${param_idx}
            """, *params)
            
            # Add timeline entry for each change
            for note in timeline_notes:
                event_type = "note"
                if "Status" in note:
                    event_type = "status_change"
                elif "Assigned" in note:
                    event_type = "assigned"
                elif "Resolution" in note:
                    event_type = "resolved"
                
                await conn.execute("""
                    INSERT INTO case_timeline (case_id, event_type, timestamp, user_name, note)
                    VALUES ($1, $2, NOW(), $3, $4)
                """, case_id, event_type, updated_by, note)
        
        logger.info(f"[CaseService] Updated case {case_id}")
        return await self.get_case(case_id)
    
    async def add_note(self, case_id: str, user_name: str, note: str) -> dict[str, Any]:
        """Add a note to a case timeline."""
        pool = await self.get_pool()
        
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO case_timeline (case_id, event_type, timestamp, user_name, note)
                VALUES ($1, 'note', NOW(), $2, $3)
            """, case_id, user_name, note)
            
            await conn.execute("UPDATE cases SET updated_at = NOW() WHERE id = $1", case_id)
        
        logger.info(f"[CaseService] Added note to case {case_id}")
        return await self.get_case(case_id)
    
    async def get_stats(self) -> dict[str, Any]:
        """Get case statistics."""
        pool = await self.get_pool()
        
        async with pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) FILTER (WHERE priority = 'critical' AND status != 'closed') as critical_count,
                    COUNT(*) FILTER (WHERE priority = 'high' AND status != 'closed') as high_count,
                    COUNT(*) FILTER (WHERE status IN ('open', 'in_progress', 'escalated')) as open_count,
                    COUNT(*) FILTER (WHERE status = 'closed') as resolved_count,
                    AVG(EXTRACT(EPOCH FROM (resolved_at - created_at)) / 3600) 
                        FILTER (WHERE resolved_at IS NOT NULL) as avg_resolution_hours
                FROM cases
            """)
            
            return {
                "critical": stats["critical_count"] or 0,
                "high": stats["high_count"] or 0,
                "open": stats["open_count"] or 0,
                "resolved": stats["resolved_count"] or 0,
                "avg_resolution_hours": round(stats["avg_resolution_hours"] or 0, 1)
            }


# Singleton instance
_case_service: CaseService | None = None


def get_case_service() -> CaseService:
    """Get case service singleton."""
    global _case_service
    if _case_service is None:
        _case_service = CaseService()
    return _case_service
