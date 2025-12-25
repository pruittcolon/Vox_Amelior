"""
Workload Identity Manager for Zero Trust Architecture.

Manages cryptographic identities for service workloads, integrating
with Istio service mesh for mTLS-based authentication.

Features:
- Automatic identity discovery from Istio sidecar
- Certificate-based identity extraction
- Identity token generation for fallback scenarios
- Integration with SPIFFE workload API

Reference: https://istio.io/latest/docs/concepts/security/
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from shared.security.spiffe_identity import (
    SPIFFEIdentity,
    SPIFFEValidator,
    create_nemo_validator,
    create_spiffe_id,
    get_current_identity,
    parse_spiffe_id,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkloadIdentityToken:
    """Token representing a workload's identity.

    Used as fallback when mTLS identity is not directly available,
    or for additional claims-based authorization.
    """

    spiffe_id: str
    service_name: str
    namespace: str
    issued_at: int
    expires_at: int
    claims: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the token has expired."""
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "spiffe_id": self.spiffe_id,
            "service_name": self.service_name,
            "namespace": self.namespace,
            "iat": self.issued_at,
            "exp": self.expires_at,
            "claims": self.claims,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkloadIdentityToken":
        """Create from dictionary."""
        return cls(
            spiffe_id=data["spiffe_id"],
            service_name=data["service_name"],
            namespace=data["namespace"],
            issued_at=data["iat"],
            expires_at=data["exp"],
            claims=data.get("claims", {}),
        )


class WorkloadIdentityManager:
    """Manages workload identities for zero trust authentication.

    This manager provides:
    - Current workload identity discovery
    - Identity token generation and validation
    - Integration with Istio mTLS identities
    - Claims-based access control

    Usage:
        manager = WorkloadIdentityManager(service_name="api-gateway")
        identity = manager.get_current_identity()
        token = manager.create_identity_token()
    """

    def __init__(
        self,
        service_name: str,
        namespace: str = "nemo",
        trust_domain: str = "cluster.local",
        token_ttl_seconds: int = 300,
    ):
        """Initialize the identity manager.

        Args:
            service_name: Name of the current service
            namespace: Kubernetes namespace
            trust_domain: SPIFFE trust domain
            token_ttl_seconds: Token expiration time in seconds
        """
        self.service_name = service_name
        self.namespace = namespace
        self.trust_domain = trust_domain
        self.token_ttl = token_ttl_seconds

        # Create SPIFFE ID for this service
        self._spiffe_id = create_spiffe_id(service_name, namespace, trust_domain)
        self._identity = parse_spiffe_id(self._spiffe_id)

        # Load signing key from secrets
        self._signing_key = self._load_signing_key()

        # Initialize validator
        self._validator = create_nemo_validator()

        logger.info(
            "WorkloadIdentityManager initialized: %s (namespace=%s)",
            service_name,
            namespace,
        )

    def _load_signing_key(self) -> bytes:
        """Load the signing key for identity tokens."""
        # Try Docker secrets first
        key_paths = [
            "/run/secrets/workload_identity_key",
            "/run/secrets/jwt_secret_primary",
        ]
        for path in key_paths:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        key = f.read().strip()
                        if len(key) >= 32:
                            return key
                except Exception as e:
                    logger.debug("Failed to read key from %s: %s", path, e)

        # Fall back to environment variable
        env_key = os.getenv("WORKLOAD_IDENTITY_KEY", "")
        if env_key and len(env_key) >= 32:
            return env_key.encode("utf-8")

        # Generate ephemeral key for development (warn in production)
        if os.getenv("PRODUCTION", "").lower() in ("true", "1"):
            logger.warning(
                "No workload identity key configured in production! "
                "Set WORKLOAD_IDENTITY_KEY or mount /run/secrets/workload_identity_key"
            )

        import secrets

        return secrets.token_bytes(32)

    def get_current_identity(self) -> SPIFFEIdentity:
        """Get the current workload's SPIFFE identity.

        First attempts to get the identity from Istio (mTLS cert),
        then falls back to configured identity.
        """
        # Try to get identity from Istio sidecar
        istio_identity = get_current_identity()
        if istio_identity:
            return istio_identity

        # Fall back to configured identity
        if self._identity:
            return self._identity

        raise RuntimeError("Unable to determine workload identity")

    def create_identity_token(
        self,
        additional_claims: Optional[dict[str, Any]] = None,
        ttl_override: Optional[int] = None,
    ) -> str:
        """Create a signed identity token for this workload.

        This token can be used for fallback authentication when
        mTLS identity is not directly available.

        Args:
            additional_claims: Extra claims to include in the token
            ttl_override: Override default TTL

        Returns:
            Signed identity token string
        """
        now = int(time.time())
        ttl = ttl_override or self.token_ttl

        token_data = WorkloadIdentityToken(
            spiffe_id=self._spiffe_id,
            service_name=self.service_name,
            namespace=self.namespace,
            issued_at=now,
            expires_at=now + ttl,
            claims=additional_claims or {},
        )

        # Serialize and sign
        payload = json.dumps(token_data.to_dict(), sort_keys=True)
        payload_b64 = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")

        signature = hmac.new(
            self._signing_key,
            payload_b64.encode(),
            hashlib.sha256,
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")

        return f"{payload_b64}.{signature_b64}"

    def verify_identity_token(self, token: str) -> Optional[WorkloadIdentityToken]:
        """Verify and decode an identity token.

        Args:
            token: The identity token string

        Returns:
            WorkloadIdentityToken if valid, None otherwise
        """
        try:
            parts = token.split(".")
            if len(parts) != 2:
                logger.warning("Invalid identity token format")
                return None

            payload_b64, signature_b64 = parts

            # Verify signature
            expected_sig = hmac.new(
                self._signing_key,
                payload_b64.encode(),
                hashlib.sha256,
            ).digest()
            expected_b64 = base64.urlsafe_b64encode(expected_sig).decode().rstrip("=")

            if not hmac.compare_digest(signature_b64, expected_b64):
                logger.warning("Identity token signature mismatch")
                return None

            # Decode payload
            # Add padding if needed
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding

            payload_json = base64.urlsafe_b64decode(payload_b64).decode()
            token_data = WorkloadIdentityToken.from_dict(json.loads(payload_json))

            # Check expiration
            if token_data.is_expired():
                logger.warning("Identity token expired")
                return None

            return token_data

        except Exception as e:
            logger.warning("Failed to verify identity token: %s", e)
            return None

    def validate_caller(
        self,
        caller_spiffe_id: str,
        target_service: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Validate if a caller is authorized to access this service.

        Args:
            caller_spiffe_id: SPIFFE ID of the calling service
            target_service: Service being accessed (default: current service)

        Returns:
            Tuple of (is_authorized, reason)
        """
        target = target_service or self.service_name

        caller_identity = parse_spiffe_id(caller_spiffe_id)
        if not caller_identity:
            return False, f"Invalid caller SPIFFE ID: {caller_spiffe_id}"

        return self._validator.validate(caller_identity, target)

    def get_identity_header(self) -> dict[str, str]:
        """Get HTTP headers for outgoing requests.

        Returns headers that identify this workload to other services.
        Used when mTLS identity needs to be supplemented.
        """
        token = self.create_identity_token()
        return {
            "X-Workload-Identity": token,
            "X-SPIFFE-ID": self._spiffe_id,
        }

    def extract_caller_identity(
        self,
        headers: dict[str, str],
    ) -> Optional[SPIFFEIdentity]:
        """Extract caller identity from request headers.

        In Istio mesh, the caller identity comes from mTLS.
        This method handles fallback cases.

        Args:
            headers: Request headers

        Returns:
            Caller's SPIFFE identity if available
        """
        # Try Istio-injected header (set by Envoy proxy)
        envoy_peer = headers.get("X-Forwarded-Client-Cert", "")
        if envoy_peer:
            # Parse XFCC header to extract SPIFFE ID
            # Format: By=spiffe://...;URI=spiffe://...
            for part in envoy_peer.split(";"):
                if part.startswith("URI="):
                    spiffe_id = part[4:]
                    identity = parse_spiffe_id(spiffe_id)
                    if identity:
                        return identity

        # Fallback: Check workload identity token
        identity_token = headers.get("X-Workload-Identity")
        if identity_token:
            token_data = self.verify_identity_token(identity_token)
            if token_data:
                return parse_spiffe_id(token_data.spiffe_id)

        # Fallback: Direct SPIFFE ID header
        spiffe_id = headers.get("X-SPIFFE-ID")
        if spiffe_id:
            return parse_spiffe_id(spiffe_id)

        return None


# Singleton instance cache
_identity_managers: dict[str, WorkloadIdentityManager] = {}


def get_identity_manager(service_name: str, **kwargs) -> WorkloadIdentityManager:
    """Get or create a workload identity manager for a service.

    Args:
        service_name: Name of the service
        **kwargs: Additional arguments for WorkloadIdentityManager

    Returns:
        WorkloadIdentityManager instance
    """
    if service_name not in _identity_managers:
        _identity_managers[service_name] = WorkloadIdentityManager(
            service_name=service_name,
            **kwargs,
        )
    return _identity_managers[service_name]
