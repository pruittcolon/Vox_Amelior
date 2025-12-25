"""
Integration tests for Week 6: Secure Ingestion Pipeline.

Tests cover:
- S3 connector configuration and validation
- File type detection via magic bytes
- Blocked extension enforcement
- File size limits per type
- Filename sanitization
- Upload validation workflow
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "clients" / "connectors"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "security"))


class TestS3ConnectorConfig:
    """Tests for S3 connector configuration."""
    
    def test_config_from_environment(self) -> None:
        """S3Config loads from environment variables."""
        from s3 import S3Config
        
        env = {
            "S3_REGION": "us-west-2",
            "S3_ENDPOINT_URL": "http://minio:9000",
            "S3_ACCESS_KEY": "test-key",
            "S3_SECRET_KEY": "test-secret",
            "S3_USE_SSL": "false",
        }
        
        with patch.dict(os.environ, env):
            config = S3Config.from_environment("test-bucket", "uploads/")
        
        assert config.bucket == "test-bucket"
        assert config.prefix == "uploads/"
        assert config.region == "us-west-2"
        assert config.endpoint_url == "http://minio:9000"
        assert config.use_ssl is False
    
    def test_config_defaults(self) -> None:
        """S3Config has sensible defaults."""
        from s3 import S3Config
        
        config = S3Config(bucket="my-bucket")
        
        assert config.region == "us-east-1"
        assert config.prefix == ""
        assert config.use_ssl is True
        assert config.signature_version == "s3v4"


class TestFileTypeDetection:
    """Tests for file type detection via magic bytes."""
    
    def test_pdf_detection(self) -> None:
        """PDF files detected by magic bytes."""
        from s3 import _detect_document_type
        from base import DocumentType
        
        pdf_content = b"%PDF-1.4 test content"
        result = _detect_document_type("document.pdf", pdf_content)
        
        assert result == DocumentType.PDF
    
    def test_jpeg_detection(self) -> None:
        """JPEG files detected by magic bytes."""
        from s3 import _detect_document_type
        from base import DocumentType
        
        jpeg_magic = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        result = _detect_document_type("image.jpg", jpeg_magic)
        
        assert result == DocumentType.IMAGE
    
    def test_png_detection(self) -> None:
        """PNG files detected by magic bytes."""
        from s3 import _detect_document_type
        from base import DocumentType
        
        png_magic = b"\x89PNG\r\n\x1a\n"
        result = _detect_document_type("image.png", png_magic)
        
        assert result == DocumentType.IMAGE
    
    def test_fallback_to_extension(self) -> None:
        """Falls back to extension when magic bytes unknown."""
        from s3 import _detect_document_type
        from base import DocumentType
        
        unknown_content = b"unknown file format"
        result = _detect_document_type("data.csv", unknown_content)
        
        assert result == DocumentType.CSV
    
    def test_markdown_detection(self) -> None:
        """Markdown files detected from content or extension."""
        from s3 import _detect_document_type
        from base import DocumentType
        
        md_content = b"# Heading\n\nContent here"
        result = _detect_document_type("readme.md", md_content)
        
        assert result == DocumentType.MARKDOWN


class TestBlockedExtensions:
    """Tests for blocked file extension enforcement."""
    
    def test_executable_blocked(self) -> None:
        """Executable files are blocked."""
        from s3 import _validate_file_extension
        
        blocked_files = [
            "malware.exe",
            "script.bat",
            "install.msi",
            "shell.sh",
            "powershell.ps1",
        ]
        
        for filename in blocked_files:
            valid, error = _validate_file_extension(filename)
            assert valid is False, f"{filename} should be blocked"
            assert "not allowed" in error.lower()
    
    def test_safe_extensions_allowed(self) -> None:
        """Safe file extensions are allowed."""
        from s3 import _validate_file_extension
        
        safe_files = [
            "document.pdf",
            "image.png",
            "data.csv",
            "config.json",
            "readme.md",
        ]
        
        for filename in safe_files:
            valid, error = _validate_file_extension(filename)
            assert valid is True, f"{filename} should be allowed"
            assert error is None
    
    def test_java_blocked(self) -> None:
        """Java files (JAR, WAR, CLASS) are blocked."""
        from s3 import _validate_file_extension
        
        valid, _ = _validate_file_extension("app.jar")
        assert valid is False


class TestFileSizeValidation:
    """Tests for file size limit enforcement."""
    
    def test_pdf_size_limit(self) -> None:
        """PDF files have 100MB limit."""
        from s3 import _validate_file_size
        from base import DocumentType
        
        # 50MB should pass
        valid, _ = _validate_file_size(50 * 1024 * 1024, DocumentType.PDF)
        assert valid is True
        
        # 150MB should fail
        valid, error = _validate_file_size(150 * 1024 * 1024, DocumentType.PDF)
        assert valid is False
        assert "exceeds" in error.lower()
    
    def test_image_size_limit(self) -> None:
        """Image files have 20MB limit."""
        from s3 import _validate_file_size
        from base import DocumentType
        
        # 10MB should pass
        valid, _ = _validate_file_size(10 * 1024 * 1024, DocumentType.IMAGE)
        assert valid is True
        
        # 30MB should fail
        valid, error = _validate_file_size(30 * 1024 * 1024, DocumentType.IMAGE)
        assert valid is False
    
    def test_csv_size_limit(self) -> None:
        """CSV files have 500MB limit (for data ingestion)."""
        from s3 import _validate_file_size
        from base import DocumentType
        
        # 400MB should pass
        valid, _ = _validate_file_size(400 * 1024 * 1024, DocumentType.CSV)
        assert valid is True


class TestSecureUploadValidator:
    """Tests for secure upload validation."""
    
    @pytest.fixture
    def validator(self):
        """Create validator with virus scanning disabled."""
        from secure_upload import SecureUploadValidator
        return SecureUploadValidator(enable_virus_scan=False)
    
    @pytest.mark.asyncio
    async def test_valid_pdf_upload(self, validator) -> None:
        """Valid PDF upload passes validation."""
        content = b"%PDF-1.4 test content"
        result = await validator.validate_upload("report.pdf", content)
        
        assert result.is_valid is True
        assert result.sanitized_filename == "report.pdf"
        assert result.detected_type == "pdf"
        assert result.content_hash is not None
    
    @pytest.mark.asyncio
    async def test_blocked_extension_rejected(self, validator) -> None:
        """Blocked extensions are rejected."""
        content = b"MZ\x90\x00"  # PE header
        result = await validator.validate_upload("virus.exe", content)
        
        assert result.is_valid is False
        assert "not allowed" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_filename_sanitization(self, validator) -> None:
        """Dangerous characters are removed from filename."""
        content = b"test content"
        result = await validator.validate_upload(
            "../../../etc/passwd.txt",
            content
        )
        
        assert result.is_valid is True
        # Path traversal should be stripped
        assert ".." not in result.sanitized_filename
        assert "/" not in result.sanitized_filename
    
    @pytest.mark.asyncio
    async def test_null_byte_injection_prevented(self, validator) -> None:
        """Null bytes in filename are sanitized."""
        content = b"test content"
        # Null byte injection attempt - should sanitize but still be valid
        result = await validator.validate_upload(
            "safe\x00hidden.txt",
            content
        )
        
        assert result.is_valid is True
        assert "\x00" not in result.sanitized_filename
    
    @pytest.mark.asyncio
    async def test_extension_mismatch_warning(self, validator) -> None:
        """Warning issued when extension doesn't match content."""
        # JPEG magic bytes but .txt extension
        jpeg_content = b"\xff\xd8\xff\xe0test"
        result = await validator.validate_upload("image.txt", jpeg_content)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("mismatch" in w.lower() for w in result.warnings)
    
    @pytest.mark.asyncio
    async def test_size_limit_enforced(self, validator) -> None:
        """File size limits are enforced."""
        # Create oversized content for image category
        oversized = b"x" * (25 * 1024 * 1024)  # 25MB
        result = await validator.validate_upload("big.png", oversized)
        
        assert result.is_valid is False
        assert "exceeds" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_content_hash_computed(self, validator) -> None:
        """SHA256 hash is computed for verified content."""
        content = b"test content for hashing"
        result = await validator.validate_upload("test.txt", content)
        
        assert result.is_valid is True
        assert result.content_hash is not None
        assert len(result.content_hash) == 64  # SHA256 hex
    
    @pytest.mark.asyncio
    async def test_file_like_object_support(self, validator) -> None:
        """Validator accepts file-like objects."""
        import io
        
        content = b"file content"
        file_obj = io.BytesIO(content)
        
        result = await validator.validate_upload("doc.txt", file_obj)
        
        assert result.is_valid is True
        assert result.file_size == len(content)


class TestBlockedCategories:
    """Tests for blocked file categories."""
    
    def test_web_shell_extensions_blocked(self) -> None:
        """Web shell extensions are blocked."""
        from secure_upload import BLOCKED_EXTENSIONS
        
        web_shells = [".php", ".asp", ".aspx", ".jsp", ".cgi"]
        for ext in web_shells:
            assert ext in BLOCKED_EXTENSIONS, f"{ext} should be blocked"
    
    def test_office_macros_blocked(self) -> None:
        """Office macro-enabled formats are blocked."""
        from secure_upload import BLOCKED_EXTENSIONS
        
        macros = [".docm", ".xlsm", ".pptm"]
        for ext in macros:
            assert ext in BLOCKED_EXTENSIONS, f"{ext} should be blocked"
    
    def test_script_extensions_blocked(self) -> None:
        """Script extensions are blocked."""
        from secure_upload import BLOCKED_EXTENSIONS
        
        scripts = [".vbs", ".js", ".ps1", ".py", ".rb"]
        for ext in scripts:
            assert ext in BLOCKED_EXTENSIONS, f"{ext} should be blocked"


class TestClamAVScanner:
    """Tests for ClamAV integration."""
    
    def test_scanner_environment_config(self) -> None:
        """Scanner loads config from environment."""
        from secure_upload import ClamAVScanner
        
        env = {
            "CLAMAV_HOST": "clamav-server",
            "CLAMAV_PORT": "3311",
            "CLAMAV_TIMEOUT": "60",
        }
        
        with patch.dict(os.environ, env):
            scanner = ClamAVScanner.from_environment()
        
        assert scanner.host == "clamav-server"
        assert scanner.port == 3311
        assert scanner.timeout == 60
    
    def test_scanner_unavailable_graceful(self) -> None:
        """Scanner gracefully handles unavailability."""
        from secure_upload import ClamAVScanner
        
        scanner = ClamAVScanner(host="nonexistent", port=9999)
        
        # Should not raise
        is_available = scanner.is_available()
        assert is_available is False
        
        # Scan should pass when scanner unavailable
        is_clean, msg = scanner.scan(b"test content")
        assert is_clean is True
        assert "skipped" in msg.lower() or "unavailable" in msg.lower()


class TestContentHash:
    """Tests for content integrity hashing."""
    
    def test_hash_computation(self) -> None:
        """SHA256 hash is correctly computed."""
        from s3 import _compute_content_hash
        import hashlib
        
        content = b"test content for integrity"
        computed = _compute_content_hash(content)
        expected = hashlib.sha256(content).hexdigest()
        
        assert computed == expected
    
    def test_hash_uniqueness(self) -> None:
        """Different content produces different hashes."""
        from s3 import _compute_content_hash
        
        hash1 = _compute_content_hash(b"content one")
        hash2 = _compute_content_hash(b"content two")
        
        assert hash1 != hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
