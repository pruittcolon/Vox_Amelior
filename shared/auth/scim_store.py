"""
SCIM Database Store - Persistent storage for SCIM provisioning.

Provides database-backed storage for SCIM users and groups, replacing
the in-memory store for production use.

Uses SQLite with optional SQLCipher encryption (consistent with auth_manager.py).

Tables:
- scim_users: User records from IdP provisioning
- scim_groups: Group memberships
- scim_tokens: API tokens for IdP authentication

Configuration:
- SCIM_DB_PATH: Path to SQLite database (default: instance/scim.db)
- SCIM_DB_KEY_FILE: Path to encryption key (Docker secret, optional)
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator
from uuid import uuid4

logger = logging.getLogger(__name__)


def _load_secret(path: str | None) -> str | None:
    """Load secret from file."""
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return None


class SCIMStore:
    """
    Database-backed SCIM storage.
    
    Provides persistent storage for SCIM users and groups with tenant isolation.
    """
    
    def __init__(self, db_path: str, encryption_key: str | None = None):
        """Initialize SCIM store with database path."""
        self.db_path = db_path
        self.encryption_key = encryption_key
        self._ensure_schema()
        logger.info(f"SCIM store initialized: {db_path}")
    
    @classmethod
    def from_environment(cls) -> "SCIMStore":
        """Create SCIM store from environment variables."""
        db_path = os.getenv("SCIM_DB_PATH", "instance/scim.db")
        key = _load_secret(os.getenv("SCIM_DB_KEY_FILE"))
        return cls(db_path, key)
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with proper cleanup."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Enable encryption if key provided and sqlcipher available
        if self.encryption_key:
            try:
                conn.execute(f"PRAGMA key='{self.encryption_key}'")
            except sqlite3.OperationalError:
                logger.warning("SQLCipher not available, using unencrypted database")
        
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _ensure_schema(self) -> None:
        """Create database schema if not exists."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scim_users (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_name TEXT NOT NULL,
                    external_id TEXT,
                    given_name TEXT,
                    family_name TEXT,
                    display_name TEXT,
                    email TEXT,
                    active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(tenant_id, user_name)
                );
                
                CREATE INDEX IF NOT EXISTS idx_scim_users_tenant 
                    ON scim_users(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_scim_users_email 
                    ON scim_users(email);
                CREATE INDEX IF NOT EXISTS idx_scim_users_external 
                    ON scim_users(external_id);
                
                CREATE TABLE IF NOT EXISTS scim_groups (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    external_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(tenant_id, display_name)
                );
                
                CREATE TABLE IF NOT EXISTS scim_group_members (
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    PRIMARY KEY (group_id, user_id),
                    FOREIGN KEY (group_id) REFERENCES scim_groups(id),
                    FOREIGN KEY (user_id) REFERENCES scim_users(id)
                );
                
                CREATE TABLE IF NOT EXISTS scim_tokens (
                    token_hash TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    last_used_at TEXT
                );
            """)
    
    # =========================================================================
    # User Operations
    # =========================================================================
    
    def create_user(
        self,
        tenant_id: str,
        user_name: str,
        *,
        external_id: str | None = None,
        given_name: str | None = None,
        family_name: str | None = None,
        display_name: str | None = None,
        email: str | None = None,
        active: bool = True,
    ) -> dict[str, Any]:
        """Create a new SCIM user."""
        user_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO scim_users 
                (id, tenant_id, user_name, external_id, given_name, family_name, 
                 display_name, email, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, tenant_id, user_name, external_id, given_name,
                 family_name, display_name, email, int(active), now, now),
            )
        
        logger.info(f"SCIM created user {user_id} ({user_name}) for tenant {tenant_id}")
        return self.get_user(tenant_id, user_id)  # type: ignore
    
    def get_user(self, tenant_id: str, user_id: str) -> dict[str, Any] | None:
        """Get user by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM scim_users WHERE id = ? AND tenant_id = ?",
                (user_id, tenant_id),
            ).fetchone()
            
            if row:
                return self._row_to_user(row)
        return None
    
    def get_user_by_username(self, tenant_id: str, user_name: str) -> dict[str, Any] | None:
        """Get user by username."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM scim_users WHERE user_name = ? AND tenant_id = ?",
                (user_name, tenant_id),
            ).fetchone()
            
            if row:
                return self._row_to_user(row)
        return None
    
    def list_users(
        self,
        tenant_id: str,
        start_index: int = 1,
        count: int = 100,
        filter_field: str | None = None,
        filter_value: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """List users with pagination and optional filter."""
        with self._get_connection() as conn:
            # Build query
            base_query = "FROM scim_users WHERE tenant_id = ?"
            params: list[Any] = [tenant_id]
            
            if filter_field and filter_value:
                # Map SCIM field names to DB columns
                field_map = {
                    "userName": "user_name",
                    "email": "email",
                    "externalId": "external_id",
                    "active": "active",
                }
                db_field = field_map.get(filter_field, filter_field)
                if db_field in ("user_name", "email", "external_id"):
                    base_query += f" AND {db_field} = ?"
                    params.append(filter_value)
                elif db_field == "active":
                    base_query += " AND active = ?"
                    params.append(1 if filter_value.lower() == "true" else 0)
            
            # Get total count
            total = conn.execute(f"SELECT COUNT(*) {base_query}", params).fetchone()[0]
            
            # Get page
            offset = start_index - 1
            rows = conn.execute(
                f"SELECT * {base_query} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [count, offset],
            ).fetchall()
            
            users = [self._row_to_user(row) for row in rows]
            return users, total
    
    def update_user(
        self,
        tenant_id: str,
        user_id: str,
        **updates: Any,
    ) -> dict[str, Any] | None:
        """Update user fields."""
        # Map SCIM fields to DB columns
        field_map = {
            "userName": "user_name",
            "externalId": "external_id",
            "givenName": "given_name",
            "familyName": "family_name",
            "displayName": "display_name",
            "email": "email",
            "active": "active",
        }
        
        db_updates = {}
        for key, value in updates.items():
            db_key = field_map.get(key, key)
            if db_key in ("user_name", "external_id", "given_name", "family_name",
                          "display_name", "email"):
                db_updates[db_key] = value
            elif db_key == "active":
                db_updates[db_key] = int(value)
        
        if not db_updates:
            return self.get_user(tenant_id, user_id)
        
        db_updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        set_clause = ", ".join(f"{k} = ?" for k in db_updates.keys())
        values = list(db_updates.values()) + [user_id, tenant_id]
        
        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE scim_users SET {set_clause} WHERE id = ? AND tenant_id = ?",
                values,
            )
        
        logger.info(f"SCIM updated user {user_id}")
        return self.get_user(tenant_id, user_id)
    
    def delete_user(self, tenant_id: str, user_id: str, soft: bool = True) -> bool:
        """Delete or deactivate user."""
        with self._get_connection() as conn:
            if soft:
                # Soft delete - mark as inactive
                result = conn.execute(
                    """
                    UPDATE scim_users 
                    SET active = 0, updated_at = ? 
                    WHERE id = ? AND tenant_id = ?
                    """,
                    (datetime.now(timezone.utc).isoformat(), user_id, tenant_id),
                )
            else:
                # Hard delete
                conn.execute(
                    "DELETE FROM scim_group_members WHERE user_id = ?",
                    (user_id,),
                )
                result = conn.execute(
                    "DELETE FROM scim_users WHERE id = ? AND tenant_id = ?",
                    (user_id, tenant_id),
                )
            
            deleted = result.rowcount > 0
        
        if deleted:
            logger.info(f"SCIM {'deactivated' if soft else 'deleted'} user {user_id}")
        return deleted
    
    def _row_to_user(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert database row to SCIM user dict."""
        return {
            "id": row["id"],
            "userName": row["user_name"],
            "externalId": row["external_id"],
            "name": {
                "givenName": row["given_name"],
                "familyName": row["family_name"],
                "formatted": f"{row['given_name'] or ''} {row['family_name'] or ''}".strip() or None,
            } if row["given_name"] or row["family_name"] else None,
            "displayName": row["display_name"],
            "emails": [{"value": row["email"], "type": "work", "primary": True}] if row["email"] else [],
            "active": bool(row["active"]),
            "meta": {
                "resourceType": "User",
                "created": row["created_at"],
                "lastModified": row["updated_at"],
            },
        }
    
    # =========================================================================
    # Group Operations
    # =========================================================================
    
    def create_group(
        self,
        tenant_id: str,
        display_name: str,
        external_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new SCIM group."""
        group_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO scim_groups 
                (id, tenant_id, display_name, external_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (group_id, tenant_id, display_name, external_id, now, now),
            )
        
        logger.info(f"SCIM created group {group_id} ({display_name})")
        return self.get_group(tenant_id, group_id)  # type: ignore
    
    def get_group(self, tenant_id: str, group_id: str) -> dict[str, Any] | None:
        """Get group by ID with members."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM scim_groups WHERE id = ? AND tenant_id = ?",
                (group_id, tenant_id),
            ).fetchone()
            
            if not row:
                return None
            
            members = conn.execute(
                """
                SELECT u.id, u.user_name 
                FROM scim_group_members gm
                JOIN scim_users u ON gm.user_id = u.id
                WHERE gm.group_id = ?
                """,
                (group_id,),
            ).fetchall()
            
            return {
                "id": row["id"],
                "displayName": row["display_name"],
                "externalId": row["external_id"],
                "members": [{"value": m["id"], "display": m["user_name"]} for m in members],
                "meta": {
                    "resourceType": "Group",
                    "created": row["created_at"],
                    "lastModified": row["updated_at"],
                },
            }
    
    def add_user_to_group(self, group_id: str, user_id: str) -> bool:
        """Add user to group."""
        with self._get_connection() as conn:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO scim_group_members (group_id, user_id) VALUES (?, ?)",
                    (group_id, user_id),
                )
                return True
            except sqlite3.IntegrityError:
                return False
    
    def remove_user_from_group(self, group_id: str, user_id: str) -> bool:
        """Remove user from group."""
        with self._get_connection() as conn:
            result = conn.execute(
                "DELETE FROM scim_group_members WHERE group_id = ? AND user_id = ?",
                (group_id, user_id),
            )
            return result.rowcount > 0


# Singleton instance
_scim_store: SCIMStore | None = None


def get_scim_store() -> SCIMStore:
    """Get or create global SCIM store instance."""
    global _scim_store
    if _scim_store is None:
        _scim_store = SCIMStore.from_environment()
    return _scim_store
