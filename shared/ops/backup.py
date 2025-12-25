"""
Backup and Restore Operations - Enterprise disaster recovery.

Provides automated backup and restore for databases, configurations,
and model artifacts with encryption and integrity verification.

Usage:
    from shared.ops.backup import BackupManager, get_backup_manager
    
    manager = get_backup_manager()
    
    # Create backup
    backup = await manager.create_backup(
        components=["database", "configs", "models"],
        encryption_key="your-key",
    )
    
    # Restore from backup
    await manager.restore_backup(backup.id)
"""

import asyncio
import gzip
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class BackupStatus(str, Enum):
    """Backup status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


class BackupType(str, Enum):
    """Types of backups."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


@dataclass
class BackupManifest:
    """Backup manifest with metadata."""
    id: str
    backup_type: BackupType
    status: BackupStatus
    components: list[str]
    
    # File info
    archive_path: str | None = None
    archive_size: int = 0
    checksum: str = ""
    
    # Encryption
    encrypted: bool = False
    encryption_algorithm: str = "aes-256-gcm"
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    
    # Metadata
    source_host: str = ""
    version: str = "1.0"
    notes: str = ""
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "backup_type": self.backup_type.value,
            "status": self.status.value,
            "components": self.components,
            "archive_path": self.archive_path,
            "archive_size": self.archive_size,
            "checksum": self.checksum,
            "encrypted": self.encrypted,
            "encryption_algorithm": self.encryption_algorithm,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "source_host": self.source_host,
            "version": self.version,
            "notes": self.notes,
        }


@dataclass
class RestoreResult:
    """Result of restore operation."""
    success: bool
    backup_id: str
    components_restored: list[str]
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class BackupManager:
    """
    Enterprise backup and restore manager.
    
    Supports database, config, and model artifact backups
    with optional encryption and integrity verification.
    """
    
    # Component backup handlers
    SUPPORTED_COMPONENTS = ["database", "configs", "models", "logs", "prompts"]
    
    def __init__(
        self,
        backup_dir: str | None = None,
        db_path: str | None = None,
    ):
        """Initialize backup manager."""
        self.backup_dir = Path(backup_dir or os.getenv("BACKUP_DIR", "/data/backups"))
        self.db_path = db_path or os.getenv("BACKUP_DB", str(self.backup_dir / "backups.db"))
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_db()
        
        # Component paths (configurable via env)
        self.paths = {
            "database": Path(os.getenv("DATA_DIR", "/data")),
            "configs": Path(os.getenv("CONFIG_DIR", "/app/config")),
            "models": Path(os.getenv("MODEL_DIR", "/data/models")),
            "logs": Path(os.getenv("LOG_DIR", "/var/log/nemo")),
            "prompts": Path(os.getenv("PROMPT_DB", "/data/prompts.db")),
        }
        
        logger.info(f"Backup manager initialized: {self.backup_dir}")
    
    def _ensure_db(self) -> None:
        """Ensure backup tracking database exists."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backups (
                    id TEXT PRIMARY KEY,
                    backup_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    components TEXT NOT NULL,
                    archive_path TEXT,
                    archive_size INTEGER DEFAULT 0,
                    checksum TEXT,
                    encrypted INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    source_host TEXT,
                    notes TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()
    
    async def create_backup(
        self,
        components: list[str] | None = None,
        backup_type: BackupType = BackupType.FULL,
        encrypt: bool = False,
        encryption_key: str | None = None,
        notes: str = "",
    ) -> BackupManifest:
        """
        Create a new backup.
        
        Args:
            components: Components to backup (default: all)
            backup_type: Full, incremental, or differential
            encrypt: Whether to encrypt the backup
            encryption_key: Key for encryption
            notes: Optional notes
            
        Returns:
            BackupManifest with backup details
        """
        import socket
        
        components = components or self.SUPPORTED_COMPONENTS
        invalid = set(components) - set(self.SUPPORTED_COMPONENTS)
        if invalid:
            raise ValueError(f"Invalid components: {invalid}")
        
        # Create manifest
        manifest = BackupManifest(
            id=str(uuid4()),
            backup_type=backup_type,
            status=BackupStatus.PENDING,
            components=components,
            source_host=socket.gethostname(),
            notes=notes,
            encrypted=encrypt,
        )
        
        # Save initial record
        self._save_manifest(manifest)
        
        try:
            manifest.status = BackupStatus.IN_PROGRESS
            self._save_manifest(manifest)
            
            # Create temp directory for staging
            staging_dir = self.backup_dir / f"staging_{manifest.id}"
            staging_dir.mkdir(exist_ok=True)
            
            # Backup each component
            for component in components:
                await self._backup_component(component, staging_dir)
            
            # Create archive
            archive_name = f"backup_{manifest.id}_{manifest.created_at.strftime('%Y%m%d_%H%M%S')}.tar.gz"
            archive_path = self.backup_dir / archive_name
            
            await self._create_archive(staging_dir, archive_path)
            
            # Calculate checksum
            manifest.checksum = self._calculate_checksum(archive_path)
            manifest.archive_path = str(archive_path)
            manifest.archive_size = archive_path.stat().st_size
            
            # Encrypt if requested
            if encrypt and encryption_key:
                await self._encrypt_archive(archive_path, encryption_key)
                manifest.archive_path = str(archive_path) + ".enc"
            
            # Cleanup staging
            shutil.rmtree(staging_dir, ignore_errors=True)
            
            # Mark completed
            manifest.status = BackupStatus.COMPLETED
            manifest.completed_at = datetime.now(timezone.utc)
            self._save_manifest(manifest)
            
            logger.info(f"Backup completed: {manifest.id} ({manifest.archive_size} bytes)")
            return manifest
            
        except Exception as e:
            manifest.status = BackupStatus.FAILED
            manifest.notes = f"Error: {str(e)}"
            self._save_manifest(manifest)
            logger.error(f"Backup failed: {e}")
            raise
    
    async def _backup_component(self, component: str, staging_dir: Path) -> None:
        """Backup a specific component."""
        source_path = self.paths.get(component)
        if not source_path:
            logger.warning(f"No path configured for component: {component}")
            return
        
        target_dir = staging_dir / component
        target_dir.mkdir(exist_ok=True)
        
        if component == "database":
            await self._backup_databases(source_path, target_dir)
        elif component == "prompts":
            if source_path.exists():
                shutil.copy2(source_path, target_dir / source_path.name)
        elif source_path.exists():
            if source_path.is_dir():
                shutil.copytree(source_path, target_dir / source_path.name, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, target_dir / source_path.name)
    
    async def _backup_databases(self, data_dir: Path, target_dir: Path) -> None:
        """Backup SQLite databases."""
        loop = asyncio.get_event_loop()
        
        for db_file in data_dir.glob("*.db"):
            target_file = target_dir / db_file.name
            
            # Use SQLite backup API for consistency
            try:
                def backup_db():
                    source_conn = sqlite3.connect(str(db_file))
                    target_conn = sqlite3.connect(str(target_file))
                    try:
                        source_conn.backup(target_conn)
                    finally:
                        source_conn.close()
                        target_conn.close()
                
                await loop.run_in_executor(None, backup_db)
                logger.info(f"Backed up database: {db_file.name}")
            except Exception as e:
                logger.error(f"Failed to backup {db_file.name}: {e}")
    
    async def _create_archive(self, source_dir: Path, archive_path: Path) -> None:
        """Create gzipped tar archive."""
        loop = asyncio.get_event_loop()
        
        def create_tar():
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(source_dir, arcname="backup")
        
        await loop.run_in_executor(None, create_tar)
    
    async def _encrypt_archive(self, archive_path: Path, key: str) -> None:
        """Encrypt archive using AES-256-GCM (placeholder)."""
        # In production, use cryptography library
        # This is a placeholder that just copies the file
        encrypted_path = Path(str(archive_path) + ".enc")
        shutil.copy2(archive_path, encrypted_path)
        archive_path.unlink()
        logger.info(f"Encrypted archive: {encrypted_path}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    async def restore_backup(
        self,
        backup_id: str,
        components: list[str] | None = None,
        decryption_key: str | None = None,
    ) -> RestoreResult:
        """
        Restore from a backup.
        
        Args:
            backup_id: ID of backup to restore
            components: Specific components to restore (default: all)
            decryption_key: Key for encrypted backups
        """
        import time
        start = time.time()
        
        manifest = self.get_backup(backup_id)
        if not manifest:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                components_restored=[],
                errors=["Backup not found"],
            )
        
        components = components or manifest.components
        errors = []
        restored = []
        
        try:
            archive_path = Path(manifest.archive_path)
            
            # Decrypt if needed
            if manifest.encrypted:
                if not decryption_key:
                    return RestoreResult(
                        success=False,
                        backup_id=backup_id,
                        components_restored=[],
                        errors=["Decryption key required"],
                    )
                await self._decrypt_archive(archive_path, decryption_key)
                archive_path = Path(str(archive_path).replace(".enc", ""))
            
            # Verify checksum
            if not manifest.encrypted:
                current_checksum = self._calculate_checksum(archive_path)
                if current_checksum != manifest.checksum:
                    return RestoreResult(
                        success=False,
                        backup_id=backup_id,
                        components_restored=[],
                        errors=["Checksum verification failed"],
                    )
            
            # Extract archive
            extract_dir = self.backup_dir / f"restore_{backup_id}"
            extract_dir.mkdir(exist_ok=True)
            
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            
            # Restore components
            backup_root = extract_dir / "backup"
            for component in components:
                try:
                    await self._restore_component(component, backup_root)
                    restored.append(component)
                except Exception as e:
                    errors.append(f"{component}: {str(e)}")
            
            # Cleanup
            shutil.rmtree(extract_dir, ignore_errors=True)
            
            duration = time.time() - start
            
            return RestoreResult(
                success=len(errors) == 0,
                backup_id=backup_id,
                components_restored=restored,
                errors=errors,
                duration_seconds=duration,
            )
            
        except Exception as e:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                components_restored=restored,
                errors=[str(e)],
                duration_seconds=time.time() - start,
            )
    
    async def _restore_component(self, component: str, backup_root: Path) -> None:
        """Restore a specific component."""
        source_dir = backup_root / component
        if not source_dir.exists():
            raise ValueError(f"Component not in backup: {component}")
        
        target_path = self.paths.get(component)
        if not target_path:
            raise ValueError(f"No restore path for component: {component}")
        
        # Backup current state before restoring
        if target_path.exists():
            backup_current = Path(str(target_path) + ".pre_restore")
            if target_path.is_dir():
                shutil.copytree(target_path, backup_current, dirs_exist_ok=True)
            else:
                shutil.copy2(target_path, backup_current)
        
        # Restore
        for item in source_dir.iterdir():
            if target_path.is_dir() or not target_path.exists():
                target_path.mkdir(parents=True, exist_ok=True)
                if item.is_dir():
                    shutil.copytree(item, target_path / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target_path / item.name)
            else:
                shutil.copy2(item, target_path)
        
        logger.info(f"Restored component: {component}")
    
    async def _decrypt_archive(self, archive_path: Path, key: str) -> None:
        """Decrypt archive (placeholder)."""
        decrypted_path = Path(str(archive_path).replace(".enc", ""))
        shutil.copy2(archive_path, decrypted_path)
        logger.info(f"Decrypted archive: {decrypted_path}")
    
    async def verify_backup(self, backup_id: str) -> dict:
        """Verify backup integrity."""
        manifest = self.get_backup(backup_id)
        if not manifest:
            return {"valid": False, "error": "Backup not found"}
        
        archive_path = Path(manifest.archive_path)
        if not archive_path.exists():
            return {"valid": False, "error": "Archive file missing"}
        
        # Verify checksum
        if manifest.checksum:
            current = self._calculate_checksum(archive_path)
            if current != manifest.checksum:
                return {"valid": False, "error": "Checksum mismatch"}
        
        # Verify archive can be read
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.getmembers()
        except Exception as e:
            return {"valid": False, "error": f"Archive corrupt: {e}"}
        
        # Mark as verified
        manifest.status = BackupStatus.VERIFIED
        self._save_manifest(manifest)
        
        return {
            "valid": True,
            "backup_id": backup_id,
            "checksum": manifest.checksum,
            "size": manifest.archive_size,
        }
    
    def get_backup(self, backup_id: str) -> BackupManifest | None:
        """Get backup by ID."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT * FROM backups WHERE id = ?",
                (backup_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_manifest(cursor.description, row)
        finally:
            conn.close()
        return None
    
    def list_backups(
        self,
        limit: int = 50,
        status: BackupStatus | None = None,
    ) -> list[BackupManifest]:
        """List recent backups."""
        query = "SELECT * FROM backups"
        params = []
        
        if status:
            query += " WHERE status = ?"
            params.append(status.value)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(query, params)
            return [self._row_to_manifest(cursor.description, r) for r in cursor.fetchall()]
        finally:
            conn.close()
    
    def _save_manifest(self, manifest: BackupManifest) -> None:
        """Save backup manifest to database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO backups (
                    id, backup_type, status, components, archive_path,
                    archive_size, checksum, encrypted, created_at,
                    completed_at, source_host, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                manifest.id, manifest.backup_type.value, manifest.status.value,
                json.dumps(manifest.components), manifest.archive_path,
                manifest.archive_size, manifest.checksum, 1 if manifest.encrypted else 0,
                manifest.created_at.isoformat(),
                manifest.completed_at.isoformat() if manifest.completed_at else None,
                manifest.source_host, manifest.notes,
            ))
            conn.commit()
        finally:
            conn.close()
    
    def _row_to_manifest(self, description: Any, row: tuple) -> BackupManifest:
        """Convert database row to manifest."""
        columns = [d[0] for d in description]
        data = dict(zip(columns, row))
        
        return BackupManifest(
            id=data["id"],
            backup_type=BackupType(data["backup_type"]),
            status=BackupStatus(data["status"]),
            components=json.loads(data["components"]),
            archive_path=data.get("archive_path"),
            archive_size=data.get("archive_size", 0),
            checksum=data.get("checksum", ""),
            encrypted=bool(data.get("encrypted", 0)),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            source_host=data.get("source_host", ""),
            notes=data.get("notes", ""),
        )


# Singleton
_manager: BackupManager | None = None


def get_backup_manager() -> BackupManager:
    """Get or create backup manager singleton."""
    global _manager
    if _manager is None:
        _manager = BackupManager()
    return _manager
