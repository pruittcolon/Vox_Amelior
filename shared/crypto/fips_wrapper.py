"""
FIPS 140-2 Compliant Cryptography Wrapper.

Provides wrappers around cryptographic operations to ensure
only FIPS-approved algorithms are used.

FIPS 140-2 Approved Algorithms:
- Hash: SHA-256, SHA-384, SHA-512 (NOT MD5, SHA-1)
- Symmetric: AES-128, AES-192, AES-256 (NOT DES, 3DES, RC4)
- Asymmetric: RSA-2048+, ECDSA P-256+ (NOT RSA-1024)
- KDF: PBKDF2, HKDF (NOT bcrypt for new)
- MAC: HMAC-SHA256+

Usage:
    crypto = FIPSCrypto()
    
    # Hash
    digest = crypto.hash(data, algorithm="SHA-256")
    
    # HMAC
    mac = crypto.hmac(data, key, algorithm="SHA-256")
    
    # Encrypt
    ciphertext = crypto.encrypt(plaintext, key)
    
    # Decrypt
    plaintext = crypto.decrypt(ciphertext, key)
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

logger = logging.getLogger(__name__)


class FIPSAlgorithm(Enum):
    """FIPS 140-2 approved algorithms."""
    
    # Hash algorithms
    SHA_256 = "SHA-256"
    SHA_384 = "SHA-384"
    SHA_512 = "SHA-512"
    SHA3_256 = "SHA3-256"
    SHA3_384 = "SHA3-384"
    SHA3_512 = "SHA3-512"
    
    # Symmetric encryption
    AES_128_GCM = "AES-128-GCM"
    AES_192_GCM = "AES-192-GCM"
    AES_256_GCM = "AES-256-GCM"
    AES_128_CBC = "AES-128-CBC"
    AES_256_CBC = "AES-256-CBC"
    
    # Asymmetric
    RSA_2048 = "RSA-2048"
    RSA_3072 = "RSA-3072"
    RSA_4096 = "RSA-4096"
    ECDSA_P256 = "ECDSA-P256"
    ECDSA_P384 = "ECDSA-P384"
    ECDSA_P521 = "ECDSA-P521"
    
    # Key derivation
    PBKDF2 = "PBKDF2"
    HKDF = "HKDF"


# Algorithms that are NOT FIPS approved and should be rejected
BLOCKED_ALGORITHMS = {
    "MD5", "MD4", "SHA1", "SHA-1",
    "DES", "3DES", "RC4", "RC2", "IDEA",
    "RSA-1024", "RSA-512",
    "DSA-1024",
}


# FIPS approved algorithm configuration
FIPS_APPROVED_ALGORITHMS = {
    "hash": [
        FIPSAlgorithm.SHA_256,
        FIPSAlgorithm.SHA_384,
        FIPSAlgorithm.SHA_512,
        FIPSAlgorithm.SHA3_256,
        FIPSAlgorithm.SHA3_384,
        FIPSAlgorithm.SHA3_512,
    ],
    "symmetric": [
        FIPSAlgorithm.AES_128_GCM,
        FIPSAlgorithm.AES_192_GCM,
        FIPSAlgorithm.AES_256_GCM,
        FIPSAlgorithm.AES_128_CBC,
        FIPSAlgorithm.AES_256_CBC,
    ],
    "asymmetric": [
        FIPSAlgorithm.RSA_2048,
        FIPSAlgorithm.RSA_3072,
        FIPSAlgorithm.RSA_4096,
        FIPSAlgorithm.ECDSA_P256,
        FIPSAlgorithm.ECDSA_P384,
        FIPSAlgorithm.ECDSA_P521,
    ],
    "kdf": [
        FIPSAlgorithm.PBKDF2,
        FIPSAlgorithm.HKDF,
    ],
}


@dataclass
class FIPSConfig:
    """FIPS mode configuration."""
    
    # Enforce FIPS mode (reject non-FIPS algorithms)
    enforce: bool = True
    
    # Log warnings for non-FIPS algorithms
    log_warnings: bool = True
    
    # Default algorithms
    default_hash: FIPSAlgorithm = FIPSAlgorithm.SHA_256
    default_symmetric: FIPSAlgorithm = FIPSAlgorithm.AES_256_GCM
    default_asymmetric: FIPSAlgorithm = FIPSAlgorithm.RSA_2048
    
    # Minimum key sizes (bits)
    min_symmetric_key_bits: int = 128
    min_rsa_key_bits: int = 2048
    min_ec_key_bits: int = 256
    
    @classmethod
    def from_env(cls) -> "FIPSConfig":
        """Create config from environment."""
        return cls(
            enforce=os.getenv("FIPS_ENFORCE", "true").lower() == "true",
            log_warnings=os.getenv("FIPS_LOG_WARNINGS", "true").lower() == "true",
        )


class FIPSViolationError(Exception):
    """Raised when a non-FIPS algorithm is requested in FIPS mode."""
    pass


class FIPSCrypto:
    """FIPS 140-2 compliant cryptography wrapper.
    
    All operations use FIPS-approved algorithms.
    Non-FIPS algorithms are rejected in enforce mode.
    """
    
    def __init__(self, config: Optional[FIPSConfig] = None):
        """Initialize FIPS crypto wrapper.
        
        Args:
            config: FIPS configuration (uses env if None)
        """
        self.config = config or FIPSConfig.from_env()
        logger.info(
            "FIPSCrypto initialized (enforce=%s)",
            self.config.enforce,
        )
    
    def _check_algorithm(self, algorithm: str) -> None:
        """Check if algorithm is FIPS approved.
        
        Raises:
            FIPSViolationError: If algorithm not approved and enforce mode
        """
        algo_upper = algorithm.upper().replace("_", "-")
        
        if algo_upper in BLOCKED_ALGORITHMS:
            msg = f"Algorithm '{algorithm}' is not FIPS 140-2 approved"
            if self.config.enforce:
                logger.error(msg)
                raise FIPSViolationError(msg)
            elif self.config.log_warnings:
                logger.warning(msg)
    
    def hash(
        self,
        data: Union[bytes, str],
        algorithm: str = "SHA-256",
    ) -> bytes:
        """Compute cryptographic hash using FIPS-approved algorithm.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm (SHA-256, SHA-384, SHA-512)
            
        Returns:
            Hash digest bytes
        """
        self._check_algorithm(algorithm)
        
        if isinstance(data, str):
            data = data.encode("utf-8")
        
        algo_map = {
            "SHA-256": hashlib.sha256,
            "SHA256": hashlib.sha256,
            "SHA-384": hashlib.sha384,
            "SHA384": hashlib.sha384,
            "SHA-512": hashlib.sha512,
            "SHA512": hashlib.sha512,
            "SHA3-256": hashlib.sha3_256,
            "SHA3-384": hashlib.sha3_384,
            "SHA3-512": hashlib.sha3_512,
        }
        
        hasher = algo_map.get(algorithm.upper().replace("_", "-"))
        if not hasher:
            raise FIPSViolationError(f"Unsupported hash algorithm: {algorithm}")
        
        return hasher(data).digest()
    
    def hash_hex(
        self,
        data: Union[bytes, str],
        algorithm: str = "SHA-256",
    ) -> str:
        """Compute hash and return as hex string."""
        return self.hash(data, algorithm).hex()
    
    def hmac(
        self,
        data: Union[bytes, str],
        key: bytes,
        algorithm: str = "SHA-256",
    ) -> bytes:
        """Compute HMAC using FIPS-approved algorithm.
        
        Args:
            data: Data to authenticate
            key: HMAC key
            algorithm: Hash algorithm for HMAC
            
        Returns:
            HMAC bytes
        """
        self._check_algorithm(algorithm)
        
        if isinstance(data, str):
            data = data.encode("utf-8")
        
        algo_map = {
            "SHA-256": hashlib.sha256,
            "SHA256": hashlib.sha256,
            "SHA-384": hashlib.sha384,
            "SHA384": hashlib.sha384,
            "SHA-512": hashlib.sha512,
            "SHA512": hashlib.sha512,
        }
        
        hasher = algo_map.get(algorithm.upper().replace("_", "-"))
        if not hasher:
            raise FIPSViolationError(f"Unsupported HMAC algorithm: {algorithm}")
        
        return hmac.new(key, data, hasher).digest()
    
    def hmac_verify(
        self,
        data: Union[bytes, str],
        key: bytes,
        expected_hmac: bytes,
        algorithm: str = "SHA-256",
    ) -> bool:
        """Verify HMAC (constant-time comparison)."""
        computed = self.hmac(data, key, algorithm)
        return hmac.compare_digest(computed, expected_hmac)
    
    def encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        algorithm: str = "AES-256-GCM",
    ) -> bytes:
        """Encrypt data using FIPS-approved algorithm.
        
        Args:
            plaintext: Data to encrypt
            key: Encryption key
            algorithm: Encryption algorithm
            
        Returns:
            Ciphertext (nonce + encrypted data + tag)
        """
        self._check_algorithm(algorithm)
        
        # Validate key length
        if "256" in algorithm and len(key) < 32:
            raise FIPSViolationError("AES-256 requires 32-byte key")
        if "128" in algorithm and len(key) < 16:
            raise FIPSViolationError("AES-128 requires 16-byte key")
        
        if "GCM" in algorithm.upper():
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            # Use first 32 bytes for AES-256, 16 for AES-128
            key_size = 32 if "256" in algorithm else 16
            aes_key = key[:key_size]
            
            # 96-bit nonce for GCM (NIST recommended)
            nonce = secrets.token_bytes(12)
            
            aesgcm = AESGCM(aes_key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            
            # Return: nonce (12) + ciphertext + tag (16)
            return nonce + ciphertext
        
        raise FIPSViolationError(f"Unsupported encryption algorithm: {algorithm}")
    
    def decrypt(
        self,
        ciphertext: bytes,
        key: bytes,
        algorithm: str = "AES-256-GCM",
    ) -> bytes:
        """Decrypt data using FIPS-approved algorithm.
        
        Args:
            ciphertext: Data to decrypt (nonce + encrypted + tag)
            key: Decryption key
            algorithm: Encryption algorithm
            
        Returns:
            Plaintext
        """
        self._check_algorithm(algorithm)
        
        if "GCM" in algorithm.upper():
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            if len(ciphertext) < 12:
                raise ValueError("Ciphertext too short")
            
            key_size = 32 if "256" in algorithm else 16
            aes_key = key[:key_size]
            
            nonce = ciphertext[:12]
            actual_ciphertext = ciphertext[12:]
            
            aesgcm = AESGCM(aes_key)
            return aesgcm.decrypt(nonce, actual_ciphertext, None)
        
        raise FIPSViolationError(f"Unsupported decryption algorithm: {algorithm}")
    
    def derive_key(
        self,
        password: Union[bytes, str],
        salt: bytes,
        length: int = 32,
        algorithm: str = "PBKDF2",
        iterations: int = 600000,  # OWASP 2023 recommendation
    ) -> bytes:
        """Derive key from password using FIPS-approved KDF.
        
        Args:
            password: Password to derive from
            salt: Random salt (should be 16+ bytes)
            length: Desired key length
            algorithm: KDF algorithm
            iterations: PBKDF2 iterations
            
        Returns:
            Derived key bytes
        """
        self._check_algorithm(algorithm)
        
        if isinstance(password, str):
            password = password.encode("utf-8")
        
        if len(salt) < 16:
            logger.warning("Salt should be at least 16 bytes for FIPS compliance")
        
        if algorithm.upper() == "PBKDF2":
            return hashlib.pbkdf2_hmac(
                "sha256",
                password,
                salt,
                iterations,
                dklen=length,
            )
        
        if algorithm.upper() == "HKDF":
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
            from cryptography.hazmat.primitives import hashes
            
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=length,
                salt=salt,
                info=b"nemo-key-derivation",
            )
            return hkdf.derive(password)
        
        raise FIPSViolationError(f"Unsupported KDF: {algorithm}")
    
    def generate_key(self, length: int = 32) -> bytes:
        """Generate cryptographically secure random key.
        
        Args:
            length: Key length in bytes
            
        Returns:
            Random key bytes
        """
        if length < 16:
            raise FIPSViolationError("Minimum key length is 128 bits (16 bytes)")
        
        return secrets.token_bytes(length)
    
    def generate_salt(self, length: int = 16) -> bytes:
        """Generate random salt for key derivation.
        
        Args:
            length: Salt length in bytes (minimum 16)
            
        Returns:
            Random salt bytes
        """
        if length < 16:
            logger.warning("Salt should be at least 16 bytes")
        
        return secrets.token_bytes(length)
    
    def constant_time_compare(self, a: bytes, b: bytes) -> bool:
        """Compare two byte strings in constant time.
        
        Prevents timing attacks.
        """
        return hmac.compare_digest(a, b)


def ensure_fips_mode() -> bool:
    """Check if system is in FIPS mode.
    
    On Linux, checks /proc/sys/crypto/fips_enabled
    
    Returns:
        True if FIPS mode is enabled
    """
    fips_file = "/proc/sys/crypto/fips_enabled"
    
    if os.path.exists(fips_file):
        try:
            with open(fips_file) as f:
                return f.read().strip() == "1"
        except Exception:
            pass
    
    # Check OpenSSL FIPS mode
    try:
        import ssl
        # OpenSSL 3.0+ has FIPS provider
        if hasattr(ssl, "FIPS_mode"):
            return ssl.FIPS_mode()
    except Exception:
        pass
    
    return False


def validate_algorithm(algorithm: str, category: str = "hash") -> bool:
    """Validate an algorithm is FIPS approved.
    
    Args:
        algorithm: Algorithm name
        category: Algorithm category (hash, symmetric, asymmetric, kdf)
        
    Returns:
        True if algorithm is FIPS approved
    """
    if algorithm.upper() in BLOCKED_ALGORITHMS:
        return False
    
    approved = FIPS_APPROVED_ALGORITHMS.get(category, [])
    algo_normalized = algorithm.upper().replace("_", "-")
    
    return any(
        algo_normalized == algo.value.upper().replace("_", "-")
        for algo in approved
    )
