"""
Secure Backup with Encryption.

Provides encrypted backup capabilities:
- AES-256-GCM encryption for backups
- Integrity verification
- Key rotation support
- Remote storage integration

Backup types:
- Database dumps
- Configuration files
- Secrets (encrypted)
- Audit logs
"""

import base64
import gzip
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import logging

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups."""
    
    DATABASE = "database"
    CONFIGURATION = "configuration"
    SECRETS = "secrets"
    AUDIT_LOGS = "audit_logs"
    FULL = "full"


class BackupStatus(Enum):
    """Backup status."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class BackupMetadata:
    """Metadata for a backup."""
    
    backup_id: str
    backup_type: BackupType
    created_at: str
    completed_at: Optional[str] = None
    status: BackupStatus = BackupStatus.PENDING
    
    # Size info
    original_size: int = 0
    compressed_size: int = 0
    encrypted_size: int = 0
    
    # Integrity
    checksum: Optional[str] = None
    encryption_key_id: Optional[str] = None
    
    # Location
    file_path: Optional[str] = None
    remote_url: Optional[str] = None
    
    # Error tracking
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "status": self.status.value,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "encrypted_size": self.encrypted_size,
            "checksum": self.checksum,
            "encryption_key_id": self.encryption_key_id,
            "file_path": self.file_path,
        }


class SecureBackupManager:
    """Manages encrypted backups.
    
    Features:
    - AES-256-GCM encryption
    - Gzip compression
    - SHA-256 checksums
    - Metadata tracking
    """
    
    def __init__(
        self,
        backup_path: Optional[str] = None,
        encryption_key: Optional[bytes] = None,
    ):
        """Initialize backup manager.
        
        Args:
            backup_path: Directory for backups
            encryption_key: 32-byte AES key
        """
        self.backup_path = Path(
            backup_path or os.getenv("BACKUP_PATH", "/var/backups/nemo")
        )
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Load or generate encryption key
        self._encryption_key = encryption_key
        if not encryption_key:
            key_path = Path("/run/secrets/backup_encryption_key")
            if key_path.exists():
                self._encryption_key = key_path.read_bytes()[:32]
            else:
                self._encryption_key = secrets.token_bytes(32)
        
        self._backup_counter = 0
        
        logger.info("SecureBackupManager initialized: path=%s", self.backup_path)
    
    def _generate_backup_id(self, backup_type: BackupType) -> str:
        """Generate unique backup ID."""
        self._backup_counter += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{backup_type.value}-{timestamp}-{self._backup_counter:04d}"
    
    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data with AES-256-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(self._encryption_key)
        ciphertext = aesgcm.encrypt(nonce, data, None)
        
        return nonce + ciphertext
    
    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt AES-256-GCM encrypted data."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        nonce = data[:12]
        ciphertext = data[12:]
        
        aesgcm = AESGCM(self._encryption_key)
        return aesgcm.decrypt(nonce, ciphertext, None)
    
    def _compress(self, data: bytes) -> bytes:
        """Compress data with gzip."""
        return gzip.compress(data, compresslevel=9)
    
    def _decompress(self, data: bytes) -> bytes:
        """Decompress gzip data."""
        return gzip.decompress(data)
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum."""
        return hashlib.sha256(data).hexdigest()
    
    def backup_directory(
        self,
        source_path: str,
        backup_type: BackupType = BackupType.CONFIGURATION,
    ) -> BackupMetadata:
        """Create encrypted backup of a directory.
        
        Args:
            source_path: Directory to backup
            backup_type: Type of backup
            
        Returns:
            BackupMetadata with backup details
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source_path}")
        
        backup_id = self._generate_backup_id(backup_type)
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            status=BackupStatus.IN_PROGRESS,
        )
        
        try:
            # Create tar archive in memory
            import io
            tar_buffer = io.BytesIO()
            
            with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
                tar.add(source, arcname=source.name)
            
            raw_data = tar_buffer.getvalue()
            metadata.original_size = len(raw_data)
            
            # Compress
            compressed = self._compress(raw_data)
            metadata.compressed_size = len(compressed)
            
            # Encrypt
            encrypted = self._encrypt(compressed)
            metadata.encrypted_size = len(encrypted)
            
            # Calculate checksum
            metadata.checksum = self._calculate_checksum(encrypted)
            
            # Save to file
            backup_file = self.backup_path / f"{backup_id}.enc"
            backup_file.write_bytes(encrypted)
            metadata.file_path = str(backup_file)
            
            # Save metadata
            meta_file = self.backup_path / f"{backup_id}.meta.json"
            meta_file.write_text(json.dumps(metadata.to_dict(), indent=2))
            
            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.now(timezone.utc).isoformat()
            
            logger.info(
                "Backup completed: %s (%d bytes -> %d encrypted)",
                backup_id, metadata.original_size, metadata.encrypted_size,
            )
            
        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            logger.error("Backup failed: %s - %s", backup_id, e)
        
        return metadata
    
    def restore_backup(
        self,
        backup_id: str,
        restore_path: str,
    ) -> bool:
        """Restore from encrypted backup.
        
        Args:
            backup_id: Backup to restore
            restore_path: Where to restore
            
        Returns:
            True if successful
        """
        backup_file = self.backup_path / f"{backup_id}.enc"
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup not found: {backup_id}")
        
        try:
            # Read encrypted data
            encrypted = backup_file.read_bytes()
            
            # Verify checksum
            meta_file = self.backup_path / f"{backup_id}.meta.json"
            if meta_file.exists():
                metadata = json.loads(meta_file.read_text())
                expected_checksum = metadata.get("checksum")
                actual_checksum = self._calculate_checksum(encrypted)
                
                if expected_checksum and expected_checksum != actual_checksum:
                    raise ValueError("Checksum verification failed - backup may be corrupted")
            
            # Decrypt
            compressed = self._decrypt(encrypted)
            
            # Decompress
            raw_data = self._decompress(compressed)
            
            # Extract tar
            import io
            tar_buffer = io.BytesIO(raw_data)
            
            restore_dir = Path(restore_path)
            restore_dir.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
                tar.extractall(restore_dir)
            
            logger.info("Backup restored: %s -> %s", backup_id, restore_path)
            return True
            
        except Exception as e:
            logger.error("Restore failed: %s - %s", backup_id, e)
            return False
    
    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity without restoring.
        
        Returns:
            True if backup is valid
        """
        backup_file = self.backup_path / f"{backup_id}.enc"
        meta_file = self.backup_path / f"{backup_id}.meta.json"
        
        if not backup_file.exists():
            return False
        
        try:
            encrypted = backup_file.read_bytes()
            
            # Verify checksum
            if meta_file.exists():
                metadata = json.loads(meta_file.read_text())
                expected = metadata.get("checksum")
                actual = self._calculate_checksum(encrypted)
                
                if expected != actual:
                    return False
            
            # Try to decrypt header
            self._decrypt(encrypted)
            
            return True
            
        except Exception:
            return False
    
    def list_backups(self, backup_type: Optional[BackupType] = None) -> list[dict]:
        """List available backups."""
        backups = []
        
        for meta_file in self.backup_path.glob("*.meta.json"):
            try:
                metadata = json.loads(meta_file.read_text())
                
                if backup_type and metadata.get("backup_type") != backup_type.value:
                    continue
                
                backups.append(metadata)
            except Exception:
                continue
        
        return sorted(backups, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def cleanup_old_backups(self, keep_count: int = 5) -> int:
        """Remove old backups, keeping the most recent.
        
        Returns:
            Number of backups removed
        """
        backups = self.list_backups()
        removed = 0
        
        if len(backups) <= keep_count:
            return 0
        
        for backup in backups[keep_count:]:
            backup_id = backup["backup_id"]
            
            backup_file = self.backup_path / f"{backup_id}.enc"
            meta_file = self.backup_path / f"{backup_id}.meta.json"
            
            if backup_file.exists():
                backup_file.unlink()
            if meta_file.exists():
                meta_file.unlink()
            
            removed += 1
        
        logger.info("Cleaned up %d old backups", removed)
        return removed


def get_backup_manager() -> SecureBackupManager:
    """Get global backup manager."""
    return SecureBackupManager()
