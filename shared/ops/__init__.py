"""
Operations Package for Enterprise Reliability.

Provides:
- Backup/Restore: Automated disaster recovery
- Health checks and monitoring utilities
"""

from shared.ops.backup import (
    BackupManager,
    BackupManifest,
    BackupStatus,
    BackupType,
    RestoreResult,
    get_backup_manager,
)

__all__ = [
    "BackupManager",
    "BackupManifest",
    "BackupStatus",
    "BackupType",
    "RestoreResult",
    "get_backup_manager",
]
