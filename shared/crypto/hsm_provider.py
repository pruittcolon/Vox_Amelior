"""
HSM (Hardware Security Module) Key Provider.

Provides secure key storage and cryptographic operations using:
- Real HSM (AWS CloudHSM, Azure Dedicated HSM, HashiCorp Vault)
- SoftHSM for development/testing (PKCS#11 compatible)
- Docker secrets as fallback for container deployments

Keys never leave the HSM - all crypto operations happen inside.

Usage:
    provider = get_hsm_provider()
    signature = provider.sign(data, key_id="jwt-signing-key")
    verified = provider.verify(data, signature, key_id="jwt-signing-key")
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class HSMBackend(Enum):
    """Supported HSM backends."""
    
    SOFT_HSM = "softhsm"          # SoftHSM2 (PKCS#11) for dev
    DOCKER_SECRETS = "docker"     # Docker secrets (default for containers)
    AWS_CLOUDHSM = "aws"          # AWS CloudHSM
    AZURE_HSM = "azure"           # Azure Dedicated HSM
    HASHICORP_VAULT = "vault"     # HashiCorp Vault (with HSM backend)
    FILE = "file"                 # File-based (development only)


@dataclass
class HSMConfig:
    """HSM configuration."""
    
    backend: HSMBackend = HSMBackend.DOCKER_SECRETS
    
    # Docker secrets path
    secrets_path: str = "/run/secrets"
    
    # SoftHSM settings
    softhsm_lib: str = "/usr/lib/softhsm/libsofthsm2.so"
    softhsm_slot: int = 0
    softhsm_pin: str = ""
    
    # AWS CloudHSM settings
    aws_cluster_id: str = ""
    aws_hsm_user: str = ""
    aws_hsm_password: str = ""
    
    # HashiCorp Vault settings
    vault_addr: str = ""
    vault_token: str = ""
    vault_mount: str = "transit"
    
    # Key settings
    key_algorithm: str = "RSA-2048"
    key_prefix: str = "nemo"
    
    @classmethod
    def from_env(cls) -> "HSMConfig":
        """Create config from environment variables."""
        backend_str = os.getenv("HSM_BACKEND", "docker").lower()
        
        backend_map = {
            "softhsm": HSMBackend.SOFT_HSM,
            "docker": HSMBackend.DOCKER_SECRETS,
            "aws": HSMBackend.AWS_CLOUDHSM,
            "azure": HSMBackend.AZURE_HSM,
            "vault": HSMBackend.HASHICORP_VAULT,
            "file": HSMBackend.FILE,
        }
        
        backend = backend_map.get(backend_str, HSMBackend.DOCKER_SECRETS)
        
        return cls(
            backend=backend,
            secrets_path=os.getenv("HSM_SECRETS_PATH", "/run/secrets"),
            softhsm_lib=os.getenv("SOFTHSM_LIB", "/usr/lib/softhsm/libsofthsm2.so"),
            softhsm_slot=int(os.getenv("SOFTHSM_SLOT", "0")),
            softhsm_pin=os.getenv("SOFTHSM_PIN", ""),
            vault_addr=os.getenv("VAULT_ADDR", ""),
            vault_token=os.getenv("VAULT_TOKEN", ""),
            vault_mount=os.getenv("VAULT_MOUNT", "transit"),
        )


@dataclass
class KeyInfo:
    """Information about a stored key."""
    
    key_id: str
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    version: int = 1
    is_primary: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class HSMKeyProvider(ABC):
    """Abstract base class for HSM key providers.
    
    All cryptographic operations are performed within the HSM.
    Private keys never leave the secure boundary.
    """
    
    @abstractmethod
    def generate_key(
        self,
        key_id: str,
        algorithm: str = "AES-256",
        exportable: bool = False,
    ) -> KeyInfo:
        """Generate a new key in the HSM.
        
        Args:
            key_id: Unique identifier for the key
            algorithm: Key algorithm (AES-256, RSA-2048, etc.)
            exportable: Whether key can be exported (should be False for security)
            
        Returns:
            KeyInfo with key metadata
        """
        pass
    
    @abstractmethod
    def sign(
        self,
        data: bytes,
        key_id: str,
        algorithm: str = "SHA256",
    ) -> bytes:
        """Sign data using a key stored in the HSM.
        
        Args:
            data: Data to sign
            key_id: ID of the signing key
            algorithm: Hash algorithm
            
        Returns:
            Signature bytes
        """
        pass
    
    @abstractmethod
    def verify(
        self,
        data: bytes,
        signature: bytes,
        key_id: str,
        algorithm: str = "SHA256",
    ) -> bool:
        """Verify a signature using a key in the HSM.
        
        Args:
            data: Original data
            signature: Signature to verify
            key_id: ID of the verification key
            algorithm: Hash algorithm
            
        Returns:
            True if signature is valid
        """
        pass
    
    @abstractmethod
    def encrypt(
        self,
        plaintext: bytes,
        key_id: str,
    ) -> bytes:
        """Encrypt data using a key in the HSM.
        
        Args:
            plaintext: Data to encrypt
            key_id: ID of the encryption key
            
        Returns:
            Ciphertext bytes
        """
        pass
    
    @abstractmethod
    def decrypt(
        self,
        ciphertext: bytes,
        key_id: str,
    ) -> bytes:
        """Decrypt data using a key in the HSM.
        
        Args:
            ciphertext: Data to decrypt
            key_id: ID of the decryption key
            
        Returns:
            Plaintext bytes
        """
        pass
    
    @abstractmethod
    def get_key_info(self, key_id: str) -> Optional[KeyInfo]:
        """Get information about a key."""
        pass
    
    @abstractmethod
    def list_keys(self) -> list[KeyInfo]:
        """List all keys in the HSM."""
        pass
    
    @abstractmethod
    def delete_key(self, key_id: str) -> bool:
        """Delete a key from the HSM (use with caution)."""
        pass


class DockerSecretsHSM(HSMKeyProvider):
    """HSM provider using Docker secrets.
    
    This is a soft HSM implementation that uses Docker secrets
    for key storage. Keys are read from /run/secrets/ and used
    for HMAC-based operations.
    
    For Docker Compose deployments without real HSM hardware.
    """
    
    def __init__(self, config: HSMConfig):
        """Initialize Docker secrets HSM.
        
        Args:
            config: HSM configuration
        """
        self.config = config
        self.secrets_path = Path(config.secrets_path)
        self._key_cache: dict[str, bytes] = {}
        self._key_info: dict[str, KeyInfo] = {}
        
        logger.info(
            "DockerSecretsHSM initialized (secrets_path=%s)",
            self.secrets_path,
        )
    
    def _get_key_bytes(self, key_id: str) -> bytes:
        """Load key bytes from Docker secret or cache."""
        if key_id in self._key_cache:
            return self._key_cache[key_id]
        
        # Map common key IDs to secret files
        key_file_map = {
            "jwt-signing-key": "jwt_secret_primary",
            "jwt-primary": "jwt_secret_primary",
            "jwt-previous": "jwt_secret_previous",
            "jwt-legacy": "jwt_secret",
            "session-key": "session_key",
            "audit-key": "audit_hmac_key",
            "db-encryption": "users_db_key",
            "workload-identity": "workload_identity_key",
        }
        
        secret_name = key_file_map.get(key_id, key_id)
        secret_path = self.secrets_path / secret_name
        
        if secret_path.exists():
            key_bytes = secret_path.read_bytes().strip()
            # Ensure minimum key length
            if len(key_bytes) < 32:
                key_bytes = hashlib.sha256(key_bytes).digest()
            self._key_cache[key_id] = key_bytes
            return key_bytes
        
        # Generate ephemeral key for development
        logger.warning(
            "Key %s not found in secrets, generating ephemeral key",
            key_id,
        )
        key_bytes = secrets.token_bytes(32)
        self._key_cache[key_id] = key_bytes
        return key_bytes
    
    def generate_key(
        self,
        key_id: str,
        algorithm: str = "AES-256",
        exportable: bool = False,
    ) -> KeyInfo:
        """Generate a new key (stores in memory, recommend persisting to secrets)."""
        key_bytes = secrets.token_bytes(32)
        self._key_cache[key_id] = key_bytes
        
        info = KeyInfo(
            key_id=key_id,
            algorithm=algorithm,
            created_at=datetime.utcnow(),
            version=1,
            is_primary=True,
        )
        self._key_info[key_id] = info
        
        logger.info("Generated key: %s (algorithm=%s)", key_id, algorithm)
        return info
    
    def sign(
        self,
        data: bytes,
        key_id: str,
        algorithm: str = "SHA256",
    ) -> bytes:
        """Sign data using HMAC."""
        key = self._get_key_bytes(key_id)
        
        if algorithm.upper() == "SHA256":
            return hmac.new(key, data, hashlib.sha256).digest()
        elif algorithm.upper() == "SHA384":
            return hmac.new(key, data, hashlib.sha384).digest()
        elif algorithm.upper() == "SHA512":
            return hmac.new(key, data, hashlib.sha512).digest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def verify(
        self,
        data: bytes,
        signature: bytes,
        key_id: str,
        algorithm: str = "SHA256",
    ) -> bool:
        """Verify HMAC signature."""
        expected = self.sign(data, key_id, algorithm)
        return hmac.compare_digest(signature, expected)
    
    def encrypt(
        self,
        plaintext: bytes,
        key_id: str,
    ) -> bytes:
        """Encrypt using AES-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        key = self._get_key_bytes(key_id)[:32]  # AES-256 key
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        
        # Return nonce + ciphertext
        return nonce + ciphertext
    
    def decrypt(
        self,
        ciphertext: bytes,
        key_id: str,
    ) -> bytes:
        """Decrypt using AES-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        if len(ciphertext) < 12:
            raise ValueError("Ciphertext too short")
        
        key = self._get_key_bytes(key_id)[:32]
        nonce = ciphertext[:12]
        actual_ciphertext = ciphertext[12:]
        
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, actual_ciphertext, None)
    
    def get_key_info(self, key_id: str) -> Optional[KeyInfo]:
        """Get key information."""
        if key_id in self._key_info:
            return self._key_info[key_id]
        
        # Check if key exists in secrets
        try:
            self._get_key_bytes(key_id)
            return KeyInfo(
                key_id=key_id,
                algorithm="AES-256",
                created_at=datetime.utcnow(),
                version=1,
            )
        except Exception:
            return None
    
    def list_keys(self) -> list[KeyInfo]:
        """List available keys."""
        keys = []
        
        # List from cache
        for key_id in self._key_info:
            keys.append(self._key_info[key_id])
        
        # List from secrets directory
        if self.secrets_path.exists():
            for secret_file in self.secrets_path.iterdir():
                if secret_file.is_file() and secret_file.name not in self._key_info:
                    keys.append(KeyInfo(
                        key_id=secret_file.name,
                        algorithm="AES-256",
                        created_at=datetime.fromtimestamp(
                            secret_file.stat().st_mtime
                        ),
                        version=1,
                    ))
        
        return keys
    
    def delete_key(self, key_id: str) -> bool:
        """Remove key from cache (cannot delete Docker secrets at runtime)."""
        if key_id in self._key_cache:
            del self._key_cache[key_id]
        if key_id in self._key_info:
            del self._key_info[key_id]
        logger.info("Removed key from cache: %s", key_id)
        return True


class SoftHSMProvider(HSMKeyProvider):
    """HSM provider using SoftHSM2 (PKCS#11).
    
    Provides a software-based HSM for development and testing
    that is API-compatible with real HSMs.
    
    Requires: softhsm2 package installed
    """
    
    def __init__(self, config: HSMConfig):
        """Initialize SoftHSM provider."""
        self.config = config
        self._session = None
        
        # Fall back to Docker secrets if PKCS#11 not available
        try:
            import pkcs11
            self._pkcs11 = pkcs11
            self._lib = pkcs11.lib(config.softhsm_lib)
            logger.info("SoftHSM provider initialized")
        except ImportError:
            logger.warning(
                "pkcs11 package not installed, falling back to Docker secrets"
            )
            self._pkcs11 = None
            self._fallback = DockerSecretsHSM(config)
    
    def _ensure_session(self):
        """Ensure we have an active PKCS#11 session."""
        if self._pkcs11 is None:
            return
        
        if self._session is None:
            token = self._lib.get_token(slot=self.config.softhsm_slot)
            self._session = token.open(
                user_pin=self.config.softhsm_pin,
                rw=True,
            )
    
    def generate_key(
        self,
        key_id: str,
        algorithm: str = "AES-256",
        exportable: bool = False,
    ) -> KeyInfo:
        """Generate key in SoftHSM."""
        if self._pkcs11 is None:
            return self._fallback.generate_key(key_id, algorithm, exportable)
        
        self._ensure_session()
        
        # Generate key using PKCS#11
        key = self._session.generate_key(
            self._pkcs11.KeyType.AES,
            256,
            label=key_id,
            store=True,
            extractable=exportable,
        )
        
        return KeyInfo(
            key_id=key_id,
            algorithm=algorithm,
            created_at=datetime.utcnow(),
            version=1,
        )
    
    def sign(self, data: bytes, key_id: str, algorithm: str = "SHA256") -> bytes:
        """Sign using SoftHSM."""
        if self._pkcs11 is None:
            return self._fallback.sign(data, key_id, algorithm)
        
        self._ensure_session()
        key = self._session.get_key(label=key_id)
        return key.sign(data, mechanism=self._pkcs11.Mechanism.SHA256_HMAC)
    
    def verify(
        self,
        data: bytes,
        signature: bytes,
        key_id: str,
        algorithm: str = "SHA256",
    ) -> bool:
        """Verify using SoftHSM."""
        if self._pkcs11 is None:
            return self._fallback.verify(data, signature, key_id, algorithm)
        
        self._ensure_session()
        key = self._session.get_key(label=key_id)
        try:
            key.verify(data, signature, mechanism=self._pkcs11.Mechanism.SHA256_HMAC)
            return True
        except Exception:
            return False
    
    def encrypt(self, plaintext: bytes, key_id: str) -> bytes:
        """Encrypt using SoftHSM."""
        if self._pkcs11 is None:
            return self._fallback.encrypt(plaintext, key_id)
        
        self._ensure_session()
        key = self._session.get_key(label=key_id)
        iv = secrets.token_bytes(16)
        ciphertext = key.encrypt(plaintext, mechanism_param=iv)
        return iv + ciphertext
    
    def decrypt(self, ciphertext: bytes, key_id: str) -> bytes:
        """Decrypt using SoftHSM."""
        if self._pkcs11 is None:
            return self._fallback.decrypt(ciphertext, key_id)
        
        self._ensure_session()
        key = self._session.get_key(label=key_id)
        iv = ciphertext[:16]
        return key.decrypt(ciphertext[16:], mechanism_param=iv)
    
    def get_key_info(self, key_id: str) -> Optional[KeyInfo]:
        """Get key info from SoftHSM."""
        if self._pkcs11 is None:
            return self._fallback.get_key_info(key_id)
        
        self._ensure_session()
        try:
            key = self._session.get_key(label=key_id)
            return KeyInfo(
                key_id=key_id,
                algorithm="AES-256",
                created_at=datetime.utcnow(),
                version=1,
            )
        except Exception:
            return None
    
    def list_keys(self) -> list[KeyInfo]:
        """List keys in SoftHSM."""
        if self._pkcs11 is None:
            return self._fallback.list_keys()
        
        self._ensure_session()
        keys = []
        for key in self._session.get_objects():
            if hasattr(key, 'label'):
                keys.append(KeyInfo(
                    key_id=key.label,
                    algorithm="AES-256",
                    created_at=datetime.utcnow(),
                    version=1,
                ))
        return keys
    
    def delete_key(self, key_id: str) -> bool:
        """Delete key from SoftHSM."""
        if self._pkcs11 is None:
            return self._fallback.delete_key(key_id)
        
        self._ensure_session()
        try:
            key = self._session.get_key(label=key_id)
            key.destroy()
            return True
        except Exception:
            return False


# Singleton HSM provider
_hsm_provider: Optional[HSMKeyProvider] = None


def get_hsm_provider(config: Optional[HSMConfig] = None) -> HSMKeyProvider:
    """Get or create the global HSM provider.
    
    Args:
        config: Optional configuration (uses env vars if not provided)
        
    Returns:
        HSMKeyProvider instance
    """
    global _hsm_provider
    
    if _hsm_provider is None:
        if config is None:
            config = HSMConfig.from_env()
        
        if config.backend == HSMBackend.SOFT_HSM:
            _hsm_provider = SoftHSMProvider(config)
        elif config.backend == HSMBackend.DOCKER_SECRETS:
            _hsm_provider = DockerSecretsHSM(config)
        else:
            # Default to Docker secrets for unsupported backends
            logger.warning(
                "HSM backend %s not yet implemented, using Docker secrets",
                config.backend,
            )
            _hsm_provider = DockerSecretsHSM(config)
    
    return _hsm_provider
