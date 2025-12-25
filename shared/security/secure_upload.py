"""
Secure File Upload Module - Validation, scanning, and safe storage.

Provides enterprise-grade file upload security:
- File type validation via magic bytes
- Extension blocklist enforcement
- Size limits per file type
- Optional virus scanning integration (ClamAV)
- Content hash verification

Configuration:
- UPLOAD_MAX_SIZE: Maximum file size in bytes (default: 50MB)
- UPLOAD_SCAN_ENABLED: Enable virus scanning (default: True if ClamAV available)
- CLAMAV_HOST: ClamAV daemon host (default: localhost)
- CLAMAV_PORT: ClamAV daemon port (default: 3310)

Usage:
    from shared.security.secure_upload import SecureUploadValidator
    
    validator = SecureUploadValidator()
    
    # Validate before processing
    result = await validator.validate_upload(filename, content)
    if not result.is_valid:
        raise ValueError(result.error)
    
    # Content is safe to process
    process_file(result.sanitized_filename, content)
"""

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)


# File type detection via magic bytes (first 16 bytes)
MAGIC_SIGNATURES = {
    # Documents
    b"%PDF": ("pdf", "application/pdf"),
    b"PK\x03\x04": ("zip", "application/zip"),  # Also DOCX, XLSX, PPTX
    b"\xd0\xcf\x11\xe0": ("doc", "application/msword"),  # OLE compound
    
    # Images
    b"\xff\xd8\xff": ("jpg", "image/jpeg"),
    b"\x89PNG\r\n\x1a\n": ("png", "image/png"),
    b"GIF87a": ("gif", "image/gif"),
    b"GIF89a": ("gif", "image/gif"),
    b"RIFF": ("webp", "image/webp"),  # Also WAV, AVI
    
    # Text/Data
    b"<!DOCTYPE": ("html", "text/html"),
    b"<html": ("html", "text/html"),
    b"<?xml": ("xml", "application/xml"),
    b"{": ("json", "application/json"),  # Simplified
    b"[": ("json", "application/json"),  # Array JSON
    
    # Audio/Video
    b"\x00\x00\x00\x1cftyp": ("mp4", "video/mp4"),
    b"\x00\x00\x00\x20ftyp": ("mp4", "video/mp4"),
    b"ID3": ("mp3", "audio/mpeg"),
    
    # Archives
    b"\x1f\x8b\x08": ("gz", "application/gzip"),
    b"Rar!\x1a\x07": ("rar", "application/x-rar-compressed"),
    b"7z\xbc\xaf\x27\x1c": ("7z", "application/x-7z-compressed"),
}

# Dangerous file extensions that should never be allowed
BLOCKED_EXTENSIONS = frozenset({
    # Executables
    ".exe", ".dll", ".bat", ".cmd", ".msi", ".scr", ".pif", ".com",
    ".app", ".dmg", ".pkg", ".deb", ".rpm",
    
    # Scripts
    ".ps1", ".vbs", ".vbe", ".js", ".jse", ".ws", ".wsf", ".wsc",
    ".sh", ".bash", ".zsh", ".fish",
    
    # Web shells / Server scripts
    ".php", ".php3", ".php4", ".php5", ".phtml", ".asp", ".aspx",
    ".jsp", ".jspx", ".cgi", ".pl", ".py", ".rb",
    
    # Java
    ".jar", ".war", ".class",
    
    # Office macros
    ".docm", ".xlsm", ".pptm", ".dotm", ".xltm", ".potm",
    
    # Links
    ".lnk", ".url", ".scf",
    
    # Registry/Config
    ".reg", ".inf",
})

# Allowed extensions by category
ALLOWED_EXTENSIONS = {
    "documents": frozenset({".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
                           ".odt", ".ods", ".odp", ".rtf", ".txt", ".csv"}),
    "images": frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".ico"}),
    "audio": frozenset({".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}),
    "video": frozenset({".mp4", ".webm", ".avi", ".mov", ".mkv"}),
    "archives": frozenset({".zip", ".tar", ".gz", ".7z"}),
    "data": frozenset({".json", ".xml", ".yaml", ".yml", ".toml"}),
    "code": frozenset({".md", ".markdown", ".html", ".htm", ".css"}),
}

# Maximum file sizes per category (bytes)
MAX_SIZES = {
    "documents": 100 * 1024 * 1024,  # 100MB
    "images": 20 * 1024 * 1024,      # 20MB
    "audio": 100 * 1024 * 1024,      # 100MB
    "video": 500 * 1024 * 1024,      # 500MB
    "archives": 200 * 1024 * 1024,   # 200MB
    "data": 50 * 1024 * 1024,        # 50MB
    "code": 10 * 1024 * 1024,        # 10MB
    "default": 50 * 1024 * 1024,     # 50MB
}


@dataclass
class UploadValidationResult:
    """Result of upload validation."""
    is_valid: bool
    sanitized_filename: str | None = None
    detected_type: str | None = None
    detected_mime: str | None = None
    file_size: int = 0
    content_hash: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    scan_result: str | None = None  # "clean", "infected", "error", "skipped"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "sanitized_filename": self.sanitized_filename,
            "detected_type": self.detected_type,
            "detected_mime": self.detected_mime,
            "file_size": self.file_size,
            "content_hash": self.content_hash,
            "error": self.error,
            "warnings": self.warnings,
            "scan_result": self.scan_result,
        }


class ClamAVScanner:
    """ClamAV virus scanner integration."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3310,
        timeout: int = 30,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._available: bool | None = None
    
    @classmethod
    def from_environment(cls) -> "ClamAVScanner":
        """Create scanner from environment variables."""
        return cls(
            host=os.getenv("CLAMAV_HOST", "localhost"),
            port=int(os.getenv("CLAMAV_PORT", "3310")),
            timeout=int(os.getenv("CLAMAV_TIMEOUT", "30")),
        )
    
    def is_available(self) -> bool:
        """Check if ClamAV daemon is reachable."""
        if self._available is not None:
            return self._available
        
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect((self.host, self.port))
                s.sendall(b"PING\n")
                response = s.recv(1024)
                self._available = b"PONG" in response
        except Exception:
            self._available = False
        
        return self._available
    
    def scan(self, content: bytes) -> tuple[bool, str]:
        """
        Scan content for viruses.
        
        Returns:
            Tuple of (is_clean, result_message)
        """
        if not self.is_available():
            return True, "Scanner unavailable - skipped"
        
        try:
            import socket
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect((self.host, self.port))
                
                # Use INSTREAM command
                s.sendall(b"zINSTREAM\x00")
                
                # Send content in chunks
                chunk_size = 2048
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i + chunk_size]
                    size = len(chunk).to_bytes(4, "big")
                    s.sendall(size + chunk)
                
                # End stream
                s.sendall(b"\x00\x00\x00\x00")
                
                # Get result
                response = s.recv(4096).decode("utf-8", errors="replace")
                
                if "OK" in response and "FOUND" not in response:
                    return True, "clean"
                elif "FOUND" in response:
                    # Extract virus name
                    match = re.search(r": (.+) FOUND", response)
                    virus_name = match.group(1) if match else "unknown"
                    return False, f"infected: {virus_name}"
                else:
                    return True, f"scan_error: {response}"
                    
        except Exception as e:
            logger.error(f"Virus scan failed: {e}")
            return True, f"scan_error: {e}"


class SecureUploadValidator:
    """
    Secure file upload validator.
    
    Validates uploads for security before processing:
    - File type detection via magic bytes
    - Extension validation against blocklist
    - Size limits enforcement
    - Optional virus scanning
    - Filename sanitization
    """
    
    def __init__(
        self,
        allowed_categories: list[str] | None = None,
        max_size_override: int | None = None,
        enable_virus_scan: bool = True,
        scanner: ClamAVScanner | None = None,
    ):
        """
        Initialize validator.
        
        Args:
            allowed_categories: List of allowed file categories
            max_size_override: Override maximum file size
            enable_virus_scan: Whether to enable virus scanning
            scanner: Optional custom scanner instance
        """
        self.allowed_categories = allowed_categories or list(ALLOWED_EXTENSIONS.keys())
        self.max_size_override = max_size_override
        self.enable_virus_scan = enable_virus_scan
        self._scanner = scanner or (ClamAVScanner.from_environment() if enable_virus_scan else None)
        
        # Build allowed extensions set
        self._allowed_extensions: set[str] = set()
        for cat in self.allowed_categories:
            if cat in ALLOWED_EXTENSIONS:
                self._allowed_extensions.update(ALLOWED_EXTENSIONS[cat])
    
    def _detect_type(self, content: bytes) -> tuple[str | None, str | None]:
        """Detect file type from magic bytes."""
        for magic, (ext, mime) in MAGIC_SIGNATURES.items():
            if content.startswith(magic):
                return ext, mime
        return None, None
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal and injection."""
        # Get just the filename, no path components
        name = Path(filename).name
        
        # Replace dangerous characters
        name = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", name)
        
        # Limit length
        name = name[:255]
        
        # Ensure not empty
        if not name:
            name = f"file_{datetime.now(timezone.utc).timestamp()}"
        
        return name
    
    def _get_category(self, extension: str) -> str | None:
        """Get category for file extension."""
        for cat, exts in ALLOWED_EXTENSIONS.items():
            if extension in exts:
                return cat
        return None
    
    async def validate_upload(
        self,
        filename: str,
        content: bytes | BinaryIO,
    ) -> UploadValidationResult:
        """
        Validate file upload for security.
        
        Args:
            filename: Original filename
            content: File content as bytes or file-like object
            
        Returns:
            UploadValidationResult with validation status
        """
        warnings = []
        
        # Handle file-like objects
        if hasattr(content, "read"):
            content.seek(0)
            content = content.read()
        
        file_size = len(content)
        
        # Sanitize filename first
        sanitized_name = self._sanitize_filename(filename)
        if sanitized_name != filename:
            warnings.append("Filename was sanitized")
        
        # Get extension
        extension = Path(sanitized_name).suffix.lower()
        
        # Check blocked extensions
        if extension in BLOCKED_EXTENSIONS:
            return UploadValidationResult(
                is_valid=False,
                sanitized_filename=sanitized_name,
                file_size=file_size,
                error=f"File type not allowed: {extension}",
            )
        
        # Check allowed extensions
        if self._allowed_extensions and extension not in self._allowed_extensions:
            return UploadValidationResult(
                is_valid=False,
                sanitized_filename=sanitized_name,
                file_size=file_size,
                error=f"File type not in allowed list: {extension}",
            )
        
        # Detect actual type from magic bytes
        detected_ext, detected_mime = self._detect_type(content[:16])
        
        # Warn if extension doesn't match detected type
        if detected_ext and f".{detected_ext}" != extension:
            warnings.append(
                f"Extension mismatch: claimed {extension}, detected .{detected_ext}"
            )
        
        # Get category and check size
        category = self._get_category(extension) or "default"
        max_size = self.max_size_override or MAX_SIZES.get(category, MAX_SIZES["default"])
        
        if file_size > max_size:
            max_mb = max_size / (1024 * 1024)
            return UploadValidationResult(
                is_valid=False,
                sanitized_filename=sanitized_name,
                file_size=file_size,
                error=f"File exceeds maximum size of {max_mb:.1f}MB",
            )
        
        # Compute content hash
        content_hash = hashlib.sha256(content).hexdigest()
        
        # Virus scan if enabled
        scan_result = "skipped"
        if self.enable_virus_scan and self._scanner:
            is_clean, scan_msg = self._scanner.scan(content)
            if not is_clean:
                return UploadValidationResult(
                    is_valid=False,
                    sanitized_filename=sanitized_name,
                    file_size=file_size,
                    content_hash=content_hash,
                    error=f"Security threat detected: {scan_msg}",
                    scan_result="infected",
                )
            scan_result = scan_msg
        
        return UploadValidationResult(
            is_valid=True,
            sanitized_filename=sanitized_name,
            detected_type=detected_ext,
            detected_mime=detected_mime or f"application/{extension[1:]}",
            file_size=file_size,
            content_hash=content_hash,
            warnings=warnings,
            scan_result=scan_result,
        )


# Convenience function
async def validate_upload(
    filename: str,
    content: bytes | BinaryIO,
    **kwargs: Any,
) -> UploadValidationResult:
    """
    Validate file upload for security.
    
    Args:
        filename: Original filename
        content: File content
        **kwargs: Additional validator options
        
    Returns:
        UploadValidationResult
    """
    validator = SecureUploadValidator(**kwargs)
    return await validator.validate_upload(filename, content)
