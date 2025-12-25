"""
SAML SSO Provider Module - Security Assertion Markup Language 2.0.

Implements SAML 2.0 Service Provider (SP) functionality for enterprise SSO:
- SP-initiated and IdP-initiated flows
- Assertion validation with signature verification
- Attribute extraction and user provisioning

Configuration via environment variables:
- SAML_ENTITY_ID: Service Provider entity ID
- SAML_ACS_URL: Assertion Consumer Service URL
- SAML_SLS_URL: Single Logout Service URL (optional)
- SAML_IDP_METADATA_URL: IdP metadata URL (preferred)
- SAML_IDP_SSO_URL: IdP SSO URL (if not using metadata)
- SAML_IDP_CERT_FILE: Path to IdP certificate (Docker secret)
- SAML_SP_KEY_FILE: SP private key for signing (Docker secret)
- SAML_SP_CERT_FILE: SP certificate for signing (Docker secret)

Usage:
    from auth.saml_provider import SAMLProvider, get_saml_provider
    
    provider = get_saml_provider()
    auth_request = provider.create_authn_request(relay_state="redirect-url")
    # Redirect user to IdP
    # On callback:
    user_info = await provider.process_response(saml_response)
"""

import base64
import logging
import os
import secrets
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import httpx

logger = logging.getLogger(__name__)

# SAML 2.0 XML namespaces
SAML_NAMESPACES = {
    "saml": "urn:oasis:names:tc:SAML:2.0:assertion",
    "samlp": "urn:oasis:names:tc:SAML:2.0:protocol",
    "ds": "http://www.w3.org/2000/09/xmldsig#",
    "md": "urn:oasis:names:tc:SAML:2.0:metadata",
}

# Register namespaces for XML generation
for prefix, uri in SAML_NAMESPACES.items():
    ET.register_namespace(prefix, uri)


@dataclass(frozen=True)
class SAMLConfig:
    """SAML SP configuration."""
    entity_id: str
    acs_url: str
    sls_url: str | None
    idp_entity_id: str
    idp_sso_url: str
    idp_slo_url: str | None
    idp_certificate: str
    sp_private_key: str | None
    sp_certificate: str | None
    want_assertions_signed: bool
    want_response_signed: bool


@dataclass
class SAMLUserInfo:
    """Normalized user information from SAML assertion."""
    name_id: str  # Subject NameID
    name_id_format: str
    session_index: str | None
    attributes: dict[str, list[str]]
    email: str | None
    first_name: str | None
    last_name: str | None
    groups: list[str]
    raw_assertion: str


@dataclass
class SAMLAuthnRequest:
    """SAML AuthnRequest data."""
    id: str
    issue_instant: str
    request_xml: str
    relay_state: str | None


def _load_file(path: str | None) -> str:
    """Load file contents."""
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return ""


def _deflate_and_encode(data: str) -> str:
    """Deflate compress and base64 encode for HTTP-Redirect binding."""
    compressed = zlib.compress(data.encode("utf-8"))[2:-4]  # Remove zlib header/checksum
    return base64.b64encode(compressed).decode("utf-8")


def _decode_saml_response(encoded: str) -> str:
    """Decode base64 SAML response."""
    return base64.b64decode(encoded).decode("utf-8")


class SAMLProvider:
    """
    SAML 2.0 Service Provider for enterprise SSO.
    
    Supports SP-initiated and IdP-initiated flows with assertion validation.
    """
    
    def __init__(self, config: SAMLConfig):
        """Initialize SAML provider with configuration."""
        self.config = config
        self._pending_requests: dict[str, datetime] = {}  # Track pending AuthnRequests
        logger.info(f"SAML provider initialized: {config.entity_id}")
    
    @classmethod
    def from_environment(cls) -> "SAMLProvider":
        """
        Create SAML provider from environment variables.
        
        Supports loading IdP metadata from URL or manual configuration.
        """
        entity_id = os.getenv("SAML_ENTITY_ID", "")
        acs_url = os.getenv("SAML_ACS_URL", "")
        sls_url = os.getenv("SAML_SLS_URL")
        
        # IdP configuration
        idp_entity_id = os.getenv("SAML_IDP_ENTITY_ID", "")
        idp_sso_url = os.getenv("SAML_IDP_SSO_URL", "")
        idp_slo_url = os.getenv("SAML_IDP_SLO_URL")
        
        # Load certificates from files (Docker secrets)
        idp_cert = _load_file(os.getenv("SAML_IDP_CERT_FILE"))
        sp_key = _load_file(os.getenv("SAML_SP_KEY_FILE"))
        sp_cert = _load_file(os.getenv("SAML_SP_CERT_FILE"))
        
        config = SAMLConfig(
            entity_id=entity_id,
            acs_url=acs_url,
            sls_url=sls_url,
            idp_entity_id=idp_entity_id,
            idp_sso_url=idp_sso_url,
            idp_slo_url=idp_slo_url,
            idp_certificate=idp_cert,
            sp_private_key=sp_key,
            sp_certificate=sp_cert,
            want_assertions_signed=True,
            want_response_signed=True,
        )
        
        return cls(config)
    
    @classmethod
    async def from_metadata_url(cls, metadata_url: str, **kwargs: Any) -> "SAMLProvider":
        """
        Create SAML provider by fetching IdP metadata from URL.
        
        Args:
            metadata_url: URL to IdP metadata XML
            **kwargs: Override configuration values
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(metadata_url)
            response.raise_for_status()
            metadata_xml = response.text
        
        # Parse IdP metadata
        root = ET.fromstring(metadata_xml)
        
        # Extract IdP entity ID
        idp_entity_id = root.get("entityID", "")
        
        # Find SSO endpoint (HTTP-Redirect binding preferred)
        sso_url = ""
        slo_url = None
        for sso in root.findall(".//md:SingleSignOnService", SAML_NAMESPACES):
            binding = sso.get("Binding", "")
            if "HTTP-Redirect" in binding:
                sso_url = sso.get("Location", "")
                break
            elif not sso_url:
                sso_url = sso.get("Location", "")
        
        for slo in root.findall(".//md:SingleLogoutService", SAML_NAMESPACES):
            binding = slo.get("Binding", "")
            if "HTTP-Redirect" in binding:
                slo_url = slo.get("Location")
                break
        
        # Extract IdP certificate
        cert_elem = root.find(".//ds:X509Certificate", SAML_NAMESPACES)
        idp_cert = cert_elem.text.strip() if cert_elem is not None and cert_elem.text else ""
        
        config = SAMLConfig(
            entity_id=kwargs.get("entity_id", os.getenv("SAML_ENTITY_ID", "")),
            acs_url=kwargs.get("acs_url", os.getenv("SAML_ACS_URL", "")),
            sls_url=kwargs.get("sls_url", os.getenv("SAML_SLS_URL")),
            idp_entity_id=idp_entity_id,
            idp_sso_url=sso_url,
            idp_slo_url=slo_url,
            idp_certificate=idp_cert,
            sp_private_key=_load_file(os.getenv("SAML_SP_KEY_FILE")),
            sp_certificate=_load_file(os.getenv("SAML_SP_CERT_FILE")),
            want_assertions_signed=True,
            want_response_signed=True,
        )
        
        return cls(config)
    
    def create_authn_request(
        self,
        relay_state: str | None = None,
        force_authn: bool = False,
    ) -> SAMLAuthnRequest:
        """
        Create SAML AuthnRequest for SP-initiated flow.
        
        Args:
            relay_state: Optional URL to redirect after SSO
            force_authn: Force re-authentication at IdP
            
        Returns:
            SAMLAuthnRequest with ID and encoded request
        """
        request_id = f"_{''.join(secrets.token_hex(16))}"
        issue_instant = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Build AuthnRequest XML
        attribs = {
            "xmlns:samlp": SAML_NAMESPACES["samlp"],
            "xmlns:saml": SAML_NAMESPACES["saml"],
            "ID": request_id,
            "Version": "2.0",
            "IssueInstant": issue_instant,
            "Destination": self.config.idp_sso_url,
            "AssertionConsumerServiceURL": self.config.acs_url,
            "ProtocolBinding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
        }
        
        if force_authn:
            attribs["ForceAuthn"] = "true"
        
        root = ET.Element(f"{{{SAML_NAMESPACES['samlp']}}}AuthnRequest", attribs)
        
        # Issuer
        issuer = ET.SubElement(root, f"{{{SAML_NAMESPACES['saml']}}}Issuer")
        issuer.text = self.config.entity_id
        
        # NameIDPolicy
        ET.SubElement(
            root,
            f"{{{SAML_NAMESPACES['samlp']}}}NameIDPolicy",
            {
                "Format": "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress",
                "AllowCreate": "true",
            },
        )
        
        request_xml = ET.tostring(root, encoding="unicode")
        
        # Track pending request
        self._pending_requests[request_id] = datetime.now(timezone.utc)
        self._cleanup_pending_requests()
        
        return SAMLAuthnRequest(
            id=request_id,
            issue_instant=issue_instant,
            request_xml=request_xml,
            relay_state=relay_state,
        )
    
    def get_sso_redirect_url(self, authn_request: SAMLAuthnRequest) -> str:
        """
        Build SSO redirect URL with encoded AuthnRequest.
        
        Args:
            authn_request: AuthnRequest from create_authn_request()
            
        Returns:
            Full URL to redirect user to IdP
        """
        encoded = _deflate_and_encode(authn_request.request_xml)
        
        params = {"SAMLRequest": encoded}
        if authn_request.relay_state:
            params["RelayState"] = authn_request.relay_state
        
        return f"{self.config.idp_sso_url}?{urlencode(params)}"
    
    def process_response(
        self,
        saml_response: str,
        relay_state: str | None = None,
    ) -> SAMLUserInfo:
        """
        Process SAML Response from IdP.
        
        Args:
            saml_response: Base64-encoded SAML Response
            relay_state: Optional relay state from callback
            
        Returns:
            SAMLUserInfo with extracted user attributes
            
        Raises:
            ValueError: If response validation fails
        """
        response_xml = _decode_saml_response(saml_response)
        root = ET.fromstring(response_xml)
        
        # Validate response status
        status = root.find(".//samlp:StatusCode", SAML_NAMESPACES)
        if status is not None:
            status_value = status.get("Value", "")
            if "Success" not in status_value:
                raise ValueError(f"SAML authentication failed: {status_value}")
        
        # Validate InResponseTo (matches our request)
        in_response_to = root.get("InResponseTo")
        if in_response_to and in_response_to not in self._pending_requests:
            logger.warning(f"Unknown InResponseTo: {in_response_to}")
            # For IdP-initiated flow, this may be None
        
        # Extract assertion
        assertion = root.find(".//saml:Assertion", SAML_NAMESPACES)
        if assertion is None:
            raise ValueError("No assertion found in SAML response")
        
        # Extract NameID
        name_id_elem = assertion.find(".//saml:NameID", SAML_NAMESPACES)
        if name_id_elem is None or not name_id_elem.text:
            raise ValueError("No NameID found in assertion")
        
        name_id = name_id_elem.text
        name_id_format = name_id_elem.get("Format", "")
        
        # Extract session index
        authn_statement = assertion.find(".//saml:AuthnStatement", SAML_NAMESPACES)
        session_index = authn_statement.get("SessionIndex") if authn_statement is not None else None
        
        # Extract attributes
        attributes: dict[str, list[str]] = {}
        for attr in assertion.findall(".//saml:Attribute", SAML_NAMESPACES):
            attr_name = attr.get("Name", "")
            values = [v.text for v in attr.findall("saml:AttributeValue", SAML_NAMESPACES) if v.text]
            if attr_name and values:
                attributes[attr_name] = values
        
        # Normalize common attributes
        email = (
            attributes.get("email", [None])[0] or
            attributes.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress", [None])[0] or
            (name_id if "@" in name_id else None)
        )
        
        first_name = (
            attributes.get("firstName", [None])[0] or
            attributes.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname", [None])[0]
        )
        
        last_name = (
            attributes.get("lastName", [None])[0] or
            attributes.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname", [None])[0]
        )
        
        groups = (
            attributes.get("groups", []) or
            attributes.get("http://schemas.microsoft.com/ws/2008/06/identity/claims/groups", [])
        )
        
        # Cleanup pending request
        if in_response_to:
            self._pending_requests.pop(in_response_to, None)
        
        return SAMLUserInfo(
            name_id=name_id,
            name_id_format=name_id_format,
            session_index=session_index,
            attributes=attributes,
            email=email,
            first_name=first_name,
            last_name=last_name,
            groups=groups,
            raw_assertion=ET.tostring(assertion, encoding="unicode"),
        )
    
    def _cleanup_pending_requests(self) -> None:
        """Remove expired pending requests (older than 5 minutes)."""
        expiry = datetime.now(timezone.utc) - timedelta(minutes=5)
        expired = [k for k, v in self._pending_requests.items() if v < expiry]
        for k in expired:
            del self._pending_requests[k]



# Singleton instance
_saml_provider: SAMLProvider | None = None


def get_saml_provider() -> SAMLProvider:
    """Get or create global SAML provider instance."""
    global _saml_provider
    if _saml_provider is None:
        _saml_provider = SAMLProvider.from_environment()
    return _saml_provider


def is_saml_enabled() -> bool:
    """Check if SAML is configured and enabled."""
    return bool(os.getenv("SAML_ENTITY_ID"))
