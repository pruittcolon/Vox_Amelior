"""
SPIFFE Identity Helpers for Zero Trust Architecture.

SPIFFE (Secure Production Identity Framework for Everyone) provides
cryptographic identity to services in dynamic environments.

This module provides helpers for:
- Extracting SPIFFE IDs from mTLS connections
- Validating service identities
- Trust domain management

Reference: https://spiffe.io/
"""

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TrustDomain(Enum):
    """Supported trust domains for SPIFFE identities."""

    CLUSTER_LOCAL = "cluster.local"
    NEMO_PROD = "nemo.production"
    NEMO_STAGING = "nemo.staging"


@dataclass
class SPIFFEIdentity:
    """Represents a SPIFFE identity.

    Format: spiffe://<trust-domain>/ns/<namespace>/sa/<service-account>

    Attributes:
        trust_domain: The trust domain (e.g., cluster.local)
        namespace: Kubernetes namespace
        service_account: Service account name
        raw_id: Original SPIFFE ID string
    """

    trust_domain: str
    namespace: str
    service_account: str
    raw_id: str

    @property
    def service_name(self) -> str:
        """Extract service name from service account.

        Convention: Service account is typically named after the service.
        e.g., 'api-gateway' or 'gemma-service'
        """
        return self.service_account

    def matches(self, other: "SPIFFEIdentity") -> bool:
        """Check if two identities are equivalent."""
        return (
            self.trust_domain == other.trust_domain
            and self.namespace == other.namespace
            and self.service_account == other.service_account
        )

    def __str__(self) -> str:
        return self.raw_id


# SPIFFE URI pattern: spiffe://trust-domain/ns/namespace/sa/service-account
SPIFFE_PATTERN = re.compile(
    r"^spiffe://(?P<trust_domain>[^/]+)/ns/(?P<namespace>[^/]+)/sa/(?P<service_account>.+)$"
)


def parse_spiffe_id(spiffe_id: str) -> Optional[SPIFFEIdentity]:
    """Parse a SPIFFE ID string into a structured identity.

    Args:
        spiffe_id: SPIFFE URI string (e.g., spiffe://cluster.local/ns/nemo/sa/api-gateway)

    Returns:
        SPIFFEIdentity if valid, None otherwise

    Example:
        >>> identity = parse_spiffe_id("spiffe://cluster.local/ns/nemo/sa/api-gateway")
        >>> identity.service_account
        'api-gateway'
    """
    if not spiffe_id:
        return None

    match = SPIFFE_PATTERN.match(spiffe_id)
    if not match:
        logger.warning("Invalid SPIFFE ID format: %s", spiffe_id)
        return None

    return SPIFFEIdentity(
        trust_domain=match.group("trust_domain"),
        namespace=match.group("namespace"),
        service_account=match.group("service_account"),
        raw_id=spiffe_id,
    )


def create_spiffe_id(
    service_account: str,
    namespace: str = "nemo",
    trust_domain: str = "cluster.local",
) -> str:
    """Create a SPIFFE ID URI.

    Args:
        service_account: Service account name
        namespace: Kubernetes namespace (default: nemo)
        trust_domain: Trust domain (default: cluster.local)

    Returns:
        SPIFFE URI string
    """
    return f"spiffe://{trust_domain}/ns/{namespace}/sa/{service_account}"


def get_current_identity() -> Optional[SPIFFEIdentity]:
    """Get the SPIFFE identity of the current workload.

    In Istio, the identity is provided via the SPIFFE workload API
    or can be extracted from mounted certificates.

    Returns:
        Current workload's SPIFFE identity, or None if not available
    """
    # Check for Istio-injected identity
    # The identity is typically available via the workload's certificate

    # Method 1: Check environment variable (set by sidecar)
    spiffe_id = os.getenv("SPIFFE_ID")
    if spiffe_id:
        return parse_spiffe_id(spiffe_id)

    # Method 2: Read from mounted certificate (Istio < 1.17)
    cert_path = "/var/run/secrets/istio/root-cert.pem"
    if os.path.exists(cert_path):
        try:
            identity = _extract_spiffe_from_cert(cert_path)
            if identity:
                return identity
        except Exception as e:
            logger.debug("Could not extract SPIFFE from cert: %s", e)

    # Method 3: Infer from pod metadata
    namespace = _get_namespace()
    service_account = _get_service_account()
    if namespace and service_account:
        spiffe_id = create_spiffe_id(service_account, namespace)
        return parse_spiffe_id(spiffe_id)

    return None


def _get_namespace() -> Optional[str]:
    """Get the current Kubernetes namespace."""
    # Check environment variable first
    ns = os.getenv("POD_NAMESPACE")
    if ns:
        return ns

    # Read from mounted file
    ns_file = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    if os.path.exists(ns_file):
        try:
            with open(ns_file) as f:
                return f.read().strip()
        except Exception:
            pass

    return os.getenv("NAMESPACE", "nemo")


def _get_service_account() -> Optional[str]:
    """Get the current Kubernetes service account."""
    # Check environment variable
    sa = os.getenv("SERVICE_ACCOUNT")
    if sa:
        return sa

    # Infer from pod name or service name
    pod_name = os.getenv("POD_NAME", "")
    if pod_name:
        # Strip replica suffix (e.g., api-gateway-5d4c7f8b9c-x2j4k -> api-gateway)
        parts = pod_name.rsplit("-", 2)
        if len(parts) >= 2:
            return parts[0]

    return os.getenv("SERVICE_NAME")


def _extract_spiffe_from_cert(cert_path: str) -> Optional[SPIFFEIdentity]:
    """Extract SPIFFE ID from X.509 certificate SAN extension."""
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend

        with open(cert_path, "rb") as f:
            cert_data = f.read()

        cert = x509.load_pem_x509_certificate(cert_data, default_backend())

        # Look for SPIFFE ID in Subject Alternative Name extension
        try:
            san_ext = cert.extensions.get_extension_for_oid(
                x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            )
            for name in san_ext.value:
                if isinstance(name, x509.UniformResourceIdentifier):
                    uri = name.value
                    if uri.startswith("spiffe://"):
                        return parse_spiffe_id(uri)
        except x509.ExtensionNotFound:
            pass

    except ImportError:
        logger.debug("cryptography package not available for cert parsing")
    except Exception as e:
        logger.debug("Failed to parse certificate: %s", e)

    return None


class SPIFFEValidator:
    """Validates SPIFFE identities against access policies.

    This validator enforces service-to-service authorization based
    on SPIFFE identities, implementing the "never trust, always verify"
    principle of zero trust architecture.
    """

    def __init__(
        self,
        allowed_trust_domains: Optional[list[str]] = None,
        allowed_namespaces: Optional[list[str]] = None,
    ):
        """Initialize the validator.

        Args:
            allowed_trust_domains: List of trusted domains (default: cluster.local)
            allowed_namespaces: List of allowed namespaces (default: nemo)
        """
        self.allowed_trust_domains = allowed_trust_domains or ["cluster.local"]
        self.allowed_namespaces = allowed_namespaces or ["nemo"]
        self._access_rules: dict[str, set[str]] = {}

    def add_access_rule(self, target_service: str, allowed_callers: list[str]) -> None:
        """Add an access rule for a service.

        Args:
            target_service: The service being protected
            allowed_callers: List of service accounts allowed to call it
        """
        if target_service not in self._access_rules:
            self._access_rules[target_service] = set()
        self._access_rules[target_service].update(allowed_callers)

    def validate(
        self,
        caller_identity: SPIFFEIdentity,
        target_service: str,
    ) -> tuple[bool, str]:
        """Validate if a caller is allowed to access a target service.

        Args:
            caller_identity: The SPIFFE identity of the caller
            target_service: The service being accessed

        Returns:
            Tuple of (is_allowed, reason)
        """
        # Validate trust domain
        if caller_identity.trust_domain not in self.allowed_trust_domains:
            return False, f"Untrusted domain: {caller_identity.trust_domain}"

        # Validate namespace
        if caller_identity.namespace not in self.allowed_namespaces:
            return False, f"Untrusted namespace: {caller_identity.namespace}"

        # Check access rules
        if target_service in self._access_rules:
            allowed = self._access_rules[target_service]
            if caller_identity.service_account not in allowed:
                return False, f"Service {caller_identity.service_account} not authorized"

        logger.debug(
            "Access granted: %s -> %s",
            caller_identity.service_account,
            target_service,
        )
        return True, "Access granted"


# Pre-configured Nemo access rules based on architecture
def get_nemo_access_rules() -> dict[str, list[str]]:
    """Get the default access rules for Nemo services.

    These rules mirror the Istio authorization policies.
    """
    return {
        "gemma-service": ["api-gateway", "ml-service", "gpu-coordinator"],
        "rag-service": ["api-gateway", "gemma-service", "transcription-service", "insights-service"],
        "ml-service": ["api-gateway"],
        "transcription-service": ["api-gateway", "gpu-coordinator"],
        "emotion-service": ["api-gateway", "transcription-service"],
        "insights-service": ["api-gateway"],
        "fiserv-service": ["api-gateway"],
        "redis": ["api-gateway", "rag-service", "transcription-service", "gpu-coordinator"],
        "postgres": ["api-gateway", "gpu-coordinator", "rag-service"],
    }


def create_nemo_validator() -> SPIFFEValidator:
    """Create a validator pre-configured with Nemo access rules."""
    validator = SPIFFEValidator()
    for target, callers in get_nemo_access_rules().items():
        validator.add_access_rule(target, callers)
    return validator
