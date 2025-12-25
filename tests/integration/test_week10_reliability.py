"""
Integration tests for Week 10: Reliability & DR.

Tests cover:
- Backup creation and verification
- Restore operations
- Checksum validation
- Backup listing and status tracking
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "ops"))


class TestBackupManager:
    """Tests for backup manager."""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with temp directories."""
        from backup import BackupManager
        
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        
        # Create mock data directories
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create a test database
        import sqlite3
        db_path = data_dir / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO items (name) VALUES ('test1'), ('test2')")
        conn.commit()
        conn.close()
        
        # Create manager with test paths
        manager = BackupManager(
            backup_dir=str(backup_dir),
            db_path=str(backup_dir / "backups.db"),
        )
        manager.paths["database"] = data_dir
        
        return manager
    
    @pytest.mark.asyncio
    async def test_create_backup(self, manager) -> None:
        """Backups can be created."""
        from backup import BackupStatus
        
        backup = await manager.create_backup(
            components=["database"],
            notes="Test backup",
        )
        
        assert backup.id is not None
        assert backup.status == BackupStatus.COMPLETED
        assert "database" in backup.components
        assert backup.archive_size > 0
    
    @pytest.mark.asyncio
    async def test_backup_checksum(self, manager) -> None:
        """Backups have valid checksums."""
        backup = await manager.create_backup(components=["database"])
        
        assert backup.checksum is not None
        assert len(backup.checksum) == 64  # SHA-256 hex
        
        # Verify checksum
        from backup import Path
        current = manager._calculate_checksum(Path(backup.archive_path))
        assert current == backup.checksum
    
    @pytest.mark.asyncio
    async def test_verify_backup(self, manager) -> None:
        """Backups can be verified."""
        backup = await manager.create_backup(components=["database"])
        
        result = await manager.verify_backup(backup.id)
        
        assert result["valid"] is True
        assert result["checksum"] == backup.checksum
    
    @pytest.mark.asyncio
    async def test_list_backups(self, manager) -> None:
        """Backups are listed correctly."""
        await manager.create_backup(components=["database"], notes="First")
        await manager.create_backup(components=["database"], notes="Second")
        
        backups = manager.list_backups()
        
        assert len(backups) == 2
        # Most recent first
        assert backups[0].notes == "Second"
    
    @pytest.mark.asyncio
    async def test_get_backup_by_id(self, manager) -> None:
        """Backups can be retrieved by ID."""
        original = await manager.create_backup(components=["database"])
        
        retrieved = manager.get_backup(original.id)
        
        assert retrieved is not None
        assert retrieved.id == original.id
        assert retrieved.checksum == original.checksum
    
    @pytest.mark.asyncio
    async def test_restore_backup(self, manager, tmp_path) -> None:
        """Backups can be restored."""
        from backup import BackupType
        import sqlite3
        
        # Create backup
        backup = await manager.create_backup(components=["database"])
        
        # Corrupt the database
        db_path = manager.paths["database"] / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM items")
        conn.commit()
        conn.close()
        
        # Verify corruption
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        conn.close()
        assert count == 0
        
        # Restore
        result = await manager.restore_backup(backup.id, components=["database"])
        
        assert result.success is True
        assert "database" in result.components_restored
    
    def test_backup_not_found(self, manager) -> None:
        """Returns None for non-existent backup."""
        result = manager.get_backup("non-existent-id")
        
        assert result is None


class TestBackupStatus:
    """Tests for backup status tracking."""
    
    def test_status_values(self) -> None:
        """All backup statuses are defined."""
        from backup import BackupStatus
        
        assert BackupStatus.PENDING.value == "pending"
        assert BackupStatus.IN_PROGRESS.value == "in_progress"
        assert BackupStatus.COMPLETED.value == "completed"
        assert BackupStatus.FAILED.value == "failed"
        assert BackupStatus.VERIFIED.value == "verified"
    
    def test_backup_types(self) -> None:
        """All backup types are defined."""
        from backup import BackupType
        
        assert BackupType.FULL.value == "full"
        assert BackupType.INCREMENTAL.value == "incremental"
        assert BackupType.DIFFERENTIAL.value == "differential"


class TestBackupManifest:
    """Tests for backup manifest."""
    
    def test_manifest_to_dict(self) -> None:
        """Manifest can be serialized."""
        from backup import BackupManifest, BackupType, BackupStatus
        
        manifest = BackupManifest(
            id="test-id",
            backup_type=BackupType.FULL,
            status=BackupStatus.COMPLETED,
            components=["database", "configs"],
        )
        
        data = manifest.to_dict()
        
        assert data["id"] == "test-id"
        assert data["backup_type"] == "full"
        assert data["status"] == "completed"
        assert data["components"] == ["database", "configs"]


class TestRestoreResult:
    """Tests for restore result."""
    
    def test_restore_success_result(self) -> None:
        """Successful restore result."""
        from backup import RestoreResult
        
        result = RestoreResult(
            success=True,
            backup_id="test-id",
            components_restored=["database", "configs"],
            duration_seconds=5.5,
        )
        
        assert result.success is True
        assert len(result.components_restored) == 2
        assert len(result.errors) == 0
    
    def test_restore_failure_result(self) -> None:
        """Failed restore result with errors."""
        from backup import RestoreResult
        
        result = RestoreResult(
            success=False,
            backup_id="test-id",
            components_restored=[],
            errors=["Checksum mismatch", "File not found"],
        )
        
        assert result.success is False
        assert len(result.errors) == 2


class TestChecksumCalculation:
    """Tests for checksum functionality."""
    
    def test_checksum_calculation(self, tmp_path) -> None:
        """Checksums are calculated correctly."""
        from backup import BackupManager
        import hashlib
        
        # Create a test file
        test_file = tmp_path / "test.txt"
        content = b"Hello, World!"
        test_file.write_bytes(content)
        
        manager = BackupManager(
            backup_dir=str(tmp_path),
            db_path=str(tmp_path / "backups.db"),
        )
        
        checksum = manager._calculate_checksum(test_file)
        expected = hashlib.sha256(content).hexdigest()
        
        assert checksum == expected
    
    def test_checksum_detects_changes(self, tmp_path) -> None:
        """Checksum changes when file changes."""
        from backup import BackupManager
        
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Original content")
        
        manager = BackupManager(
            backup_dir=str(tmp_path),
            db_path=str(tmp_path / "backups.db"),
        )
        
        checksum1 = manager._calculate_checksum(test_file)
        
        # Modify file
        test_file.write_bytes(b"Modified content")
        
        checksum2 = manager._calculate_checksum(test_file)
        
        assert checksum1 != checksum2


class TestSupportedComponents:
    """Tests for supported backup components."""
    
    def test_supported_components_list(self) -> None:
        """All expected components are supported."""
        from backup import BackupManager
        
        expected = ["database", "configs", "models", "logs", "prompts"]
        
        for component in expected:
            assert component in BackupManager.SUPPORTED_COMPONENTS
    
    @pytest.mark.asyncio
    async def test_invalid_component_raises_error(self, tmp_path) -> None:
        """Invalid components raise error."""
        from backup import BackupManager
        
        manager = BackupManager(
            backup_dir=str(tmp_path),
            db_path=str(tmp_path / "backups.db"),
        )
        
        with pytest.raises(ValueError) as exc:
            await manager.create_backup(components=["invalid_component"])
        
        assert "invalid_component" in str(exc.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
