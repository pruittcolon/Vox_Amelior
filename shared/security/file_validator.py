"""
File Validator Module

OWASP-compliant file type validation using magic bytes.
Prevents file type spoofing attacks by verifying file signatures.

Security features:
- Magic byte signature validation
- Extension vs content type matching
- Executable detection
- Image disguised as data detection
"""

import logging
import os
from dataclasses import dataclass
from typing import BinaryIO, Literal

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of file validation."""
    
    is_valid: bool
    declared_type: str
    detected_type: str | None
    rejection_reason: str | None = None
    confidence: Literal["high", "medium", "low"] = "high"


# Magic byte signatures for supported file types
# Reference: https://en.wikipedia.org/wiki/List_of_file_signatures
MAGIC_SIGNATURES: dict[str, list[tuple[bytes, int]]] = {
    # CSV has no magic bytes - validated by content structure
    "csv": [],
    "tsv": [],
    
    # JSON - starts with { or [
    "json": [
        (b"{", 0),
        (b"[", 0),
        (b"\xef\xbb\xbf{", 0),  # UTF-8 BOM + {
        (b"\xef\xbb\xbf[", 0),  # UTF-8 BOM + [
    ],
    
    # Excel formats
    "xlsx": [(b"PK\x03\x04", 0)],  # ZIP-based (Office Open XML)
    "xls": [(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", 0)],  # OLE2 Compound
    
    # Parquet
    "parquet": [(b"PAR1", 0)],
    
    # SQLite
    "db": [(b"SQLite format 3\x00", 0)],
    "sqlite": [(b"SQLite format 3\x00", 0)],
    "sqlite3": [(b"SQLite format 3\x00", 0)],
    
    # ZIP (for archives)
    "zip": [(b"PK\x03\x04", 0)],
}

# Known dangerous file signatures to REJECT
DANGEROUS_SIGNATURES: list[tuple[bytes, str]] = [
    # Executables
    (b"MZ", "Windows executable (PE)"),
    (b"\x7fELF", "Linux executable (ELF)"),
    (b"\xca\xfe\xba\xbe", "macOS executable (Mach-O)"),
    (b"\xfe\xed\xfa\xce", "macOS executable (Mach-O 32-bit)"),
    (b"\xfe\xed\xfa\xcf", "macOS executable (Mach-O 64-bit)"),
    
    # Scripts with shebang
    (b"#!/", "Shell script"),
    (b"#!", "Script with shebang"),
    
    # Java
    (b"\xca\xfe\xba\xbe", "Java class file"),
    
    # Images (should not be uploaded as data files)
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"GIF87a", "GIF image"),
    (b"GIF89a", "GIF image"),
    (b"BM", "BMP image"),
    (b"RIFF", "RIFF format (possibly image/audio)"),
    
    # Audio/Video
    (b"ID3", "MP3 audio"),
    (b"\xff\xfb", "MP3 audio"),
    (b"\x00\x00\x00\x1c\x66\x74\x79\x70", "MP4 video"),
    (b"\x00\x00\x00\x20\x66\x74\x79\x70", "MP4 video"),
    
    # Archives that might contain executables
    (b"Rar!\x1a\x07", "RAR archive"),
    (b"\x1f\x8b\x08", "GZIP archive"),
    (b"\x37\x7a\xbc\xaf\x27\x1c", "7-Zip archive"),
    
    # PDF (potential for embedded scripts)
    (b"%PDF", "PDF document"),
    
    # Office macro-enabled formats
    (b"\xd0\xcf\x11\xe0", "Legacy Office format (potential macros)"),
]


def read_magic_bytes(file_obj: BinaryIO, num_bytes: int = 32) -> bytes:
    """
    Read magic bytes from file without consuming the stream.
    
    Args:
        file_obj: File-like object (must be seekable)
        num_bytes: Number of bytes to read
        
    Returns:
        First n bytes of the file
    """
    current_pos = file_obj.tell()
    magic = file_obj.read(num_bytes)
    file_obj.seek(current_pos)  # Reset position
    return magic


def detect_dangerous_content(magic_bytes: bytes) -> tuple[bool, str | None]:
    """
    Check if file contains dangerous content signatures.
    
    Args:
        magic_bytes: First bytes of the file
        
    Returns:
        Tuple of (is_dangerous, description)
    """
    for signature, description in DANGEROUS_SIGNATURES:
        if magic_bytes.startswith(signature):
            return True, description
    return False, None


def validate_magic_bytes(
    magic_bytes: bytes, 
    declared_extension: str
) -> tuple[bool, str | None]:
    """
    Validate that magic bytes match declared file extension.
    
    Args:
        magic_bytes: First bytes of the file
        declared_extension: File extension (without dot)
        
    Returns:
        Tuple of (is_valid, detected_type)
    """
    ext = declared_extension.lower()
    
    # Get expected signatures for this extension
    expected_signatures = MAGIC_SIGNATURES.get(ext, [])
    
    # CSV/TSV have no magic bytes - can't validate by signature
    if ext in ("csv", "tsv"):
        return True, ext
    
    # Check if any expected signature matches
    for signature, offset in expected_signatures:
        if len(magic_bytes) >= offset + len(signature):
            if magic_bytes[offset:offset + len(signature)] == signature:
                return True, ext
    
    # No match found - this is suspicious
    return False, None


def validate_csv_structure(file_path: str, sample_lines: int = 10) -> tuple[bool, str | None]:
    """
    Validate CSV file by checking structure.
    
    Since CSV has no magic bytes, we validate by checking:
    - File is readable as text
    - Has consistent column count
    - No binary content
    
    Args:
        file_path: Path to the file
        sample_lines: Number of lines to check
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="strict") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= sample_lines:
                    break
                lines.append(line)
            
            if not lines:
                return False, "File is empty"
            
            # Check for binary content (null bytes)
            for line in lines:
                if "\x00" in line:
                    return False, "File contains binary content"
            
            # Check consistent column count (if comma-separated)
            comma_counts = [line.count(",") for line in lines]
            if len(set(comma_counts)) > 2:  # Allow some variance for headers
                return False, "Inconsistent column count"
            
            return True, None
            
    except UnicodeDecodeError:
        return False, "File is not valid UTF-8 text"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_file(
    file_path: str,
    declared_extension: str,
    check_dangerous: bool = True,
    check_structure: bool = True,
) -> ValidationResult:
    """
    Comprehensive file validation.
    
    Performs OWASP-recommended checks:
    1. Dangerous signature detection
    2. Magic byte vs extension matching
    3. Structure validation (for CSV/JSON)
    
    Args:
        file_path: Path to the file to validate
        declared_extension: File extension (without dot)
        check_dangerous: Whether to check for dangerous content
        check_structure: Whether to validate file structure
        
    Returns:
        ValidationResult with detailed findings
    """
    ext = declared_extension.lower().lstrip(".")
    
    try:
        with open(file_path, "rb") as f:
            magic_bytes = read_magic_bytes(f, 32)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            declared_type=ext,
            detected_type=None,
            rejection_reason=f"Cannot read file: {str(e)}",
            confidence="high",
        )
    
    # Check for dangerous content first
    if check_dangerous:
        is_dangerous, danger_desc = detect_dangerous_content(magic_bytes)
        if is_dangerous:
            logger.warning(f"Dangerous content detected in {file_path}: {danger_desc}")
            return ValidationResult(
                is_valid=False,
                declared_type=ext,
                detected_type=danger_desc,
                rejection_reason=f"File contains dangerous content: {danger_desc}",
                confidence="high",
            )
    
    # Validate magic bytes match extension
    magic_valid, detected_type = validate_magic_bytes(magic_bytes, ext)
    
    # For CSV/TSV, also check structure
    if ext in ("csv", "tsv") and check_structure:
        structure_valid, structure_error = validate_csv_structure(file_path)
        if not structure_valid:
            return ValidationResult(
                is_valid=False,
                declared_type=ext,
                detected_type=ext,
                rejection_reason=f"Invalid {ext.upper()} structure: {structure_error}",
                confidence="medium",
            )
    
    if not magic_valid and ext not in ("csv", "tsv"):
        return ValidationResult(
            is_valid=False,
            declared_type=ext,
            detected_type=detected_type,
            rejection_reason=f"File signature does not match declared type '{ext}'",
            confidence="high",
        )
    
    return ValidationResult(
        is_valid=True,
        declared_type=ext,
        detected_type=detected_type or ext,
        rejection_reason=None,
        confidence="high",
    )


# Allowed extensions for data uploads
ALLOWED_DATA_EXTENSIONS = frozenset({
    "csv", "tsv", "json", "jsonl",
    "xlsx", "xls",
    "parquet",
    "db", "sqlite", "sqlite3",
    "zip",
})


def is_extension_allowed(extension: str) -> bool:
    """Check if file extension is in the allowed list."""
    return extension.lower().lstrip(".") in ALLOWED_DATA_EXTENSIONS
