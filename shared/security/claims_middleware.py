"""
Claims-Based Authorization Middleware.

Implements fine-grained access control based on identity claims
extracted from mTLS certificates and workload identity tokens.

This is the application-layer enforcement of zero trust policies,
complementing the Istio authorization policies.

Usage:
    # FastAPI middleware
    app.add_middleware(ClaimsAuthMiddleware, service_name="api-gateway")

    # Or as a dependency
    @app.get("/protected")
    async def protected_endpoint(claims: Claims = Depends(require_claims("admin"))):
        return {"user": claims.service_name}
"""

import logging
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional

from fastapi import Depends, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from shared.security.spiffe_identity import SPIFFEIdentity, parse_spiffe_id
from shared.security.workload_identity import (
    WorkloadIdentityManager,
    get_identity_manager,
)

logger = logging.getLogger(__name__)


@dataclass
class Claims:
    """Represents claims extracted from a request.

    Claims provide information about the caller's identity
    and permissions for fine-grained access control.
    """

    # Identity claims
    spiffe_id: str
    service_name: str
    namespace: str
    trust_domain: str

    # Authorization claims
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    tenant_id: Optional[str] = None

    # Request context
    request_id: Optional[str] = None
    source_ip: Optional[str] = None

    # Additional claims
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spiffe_identity(
        cls,
        identity: SPIFFEIdentity,
        roles: Optional[list[str]] = None,
        permissions: Optional[list[str]] = None,
        **extra,
    ) -> "Claims":
        """Create claims from a SPIFFE identity."""
        return cls(
            spiffe_id=identity.raw_id,
            service_name=identity.service_account,
            namespace=identity.namespace,
            trust_domain=identity.trust_domain,
            roles=roles or [],
            permissions=permissions or [],
            extra=extra,
        )


# Service-to-role mappings
# These define what roles each service has
SERVICE_ROLES: dict[str, list[str]] = {
    "api-gateway": ["gateway", "auth", "routing"],
    "gemma-service": ["llm", "inference"],
    "ml-service": ["ml", "prediction", "analysis"],
    "rag-service": ["rag", "search", "memory"],
    "transcription-service": ["transcription", "audio", "streaming"],
    "emotion-service": ["emotion", "sentiment", "analysis"],
    "insights-service": ["insights", "analytics"],
    "fiserv-service": ["banking", "fiserv", "financial"],
    "gpu-coordinator": ["gpu", "scheduling", "coordination"],
}

# Role-to-permission mappings
ROLE_PERMISSIONS: dict[str, list[str]] = {
    "gateway": ["route", "authenticate", "rate_limit"],
    "llm": ["generate", "chat", "complete"],
    "ml": ["predict", "train", "analyze"],
    "rag": ["search", "ingest", "retrieve"],
    "transcription": ["transcribe", "stream", "diarize"],
    "emotion": ["analyze_sentiment", "detect_emotion"],
    "insights": ["query_insights", "aggregate"],
    "banking": ["account_lookup", "transaction_query"],
    "gpu": ["request_gpu", "release_gpu", "schedule"],
    "auth": ["issue_token", "validate_token"],
}


def get_service_permissions(service_name: str) -> list[str]:
    """Get all permissions for a service based on its roles."""
    roles = SERVICE_ROLES.get(service_name, [])
    permissions = []
    for role in roles:
        permissions.extend(ROLE_PERMISSIONS.get(role, []))
    return list(set(permissions))


class ClaimsExtractor:
    """Extracts claims from incoming requests.

    Handles multiple identity sources:
    1. Istio mTLS (X-Forwarded-Client-Cert header)
    2. Workload identity token (X-Workload-Identity header)
    3. Direct SPIFFE ID (X-SPIFFE-ID header)
    """

    def __init__(self, identity_manager: WorkloadIdentityManager):
        """Initialize the claims extractor.

        Args:
            identity_manager: Workload identity manager for token verification
        """
        self._identity_manager = identity_manager

    def extract(self, request: Request) -> Optional[Claims]:
        """Extract claims from a request.

        Args:
            request: The incoming request

        Returns:
            Claims if identity found, None otherwise
        """
        # Get caller identity from request
        headers = dict(request.headers)
        identity = self._identity_manager.extract_caller_identity(headers)

        if not identity:
            logger.debug("No caller identity found in request")
            return None

        # Build claims from identity
        roles = SERVICE_ROLES.get(identity.service_account, [])
        permissions = get_service_permissions(identity.service_account)

        claims = Claims.from_spiffe_identity(
            identity=identity,
            roles=roles,
            permissions=permissions,
            request_id=request.headers.get("X-Request-ID"),
            source_ip=request.client.host if request.client else None,
        )

        # Extract tenant ID if present
        claims.tenant_id = request.headers.get("X-Tenant-ID")

        logger.debug(
            "Extracted claims: service=%s, roles=%s, permissions=%d",
            claims.service_name,
            claims.roles,
            len(claims.permissions),
        )

        return claims


class ClaimsAuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for claims-based authorization.

    Extracts identity claims from requests and makes them
    available via request.state.claims.
    """

    # Paths that don't require identity verification
    EXEMPT_PATHS = [
        "/health",
        "/health/ready",
        "/health/live",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    ]

    def __init__(
        self,
        app,
        service_name: str,
        require_identity: bool = True,
        exempt_paths: Optional[list[str]] = None,
    ):
        """Initialize the middleware.

        Args:
            app: FastAPI application
            service_name: Name of the current service
            require_identity: If True, reject requests without identity
            exempt_paths: Additional paths to exempt from identity check
        """
        super().__init__(app)
        self.service_name = service_name
        self.require_identity = require_identity
        self.exempt_paths = self.EXEMPT_PATHS + (exempt_paths or [])

        self._identity_manager = get_identity_manager(service_name)
        self._extractor = ClaimsExtractor(self._identity_manager)

    async def dispatch(self, request: Request, call_next):
        """Process the request."""
        path = request.url.path

        # Check exempt paths
        if any(path.startswith(p) for p in self.exempt_paths):
            return await call_next(request)

        # Extract claims
        claims = self._extractor.extract(request)

        if claims:
            # Validate caller is authorized for this service
            is_authorized, reason = self._identity_manager.validate_caller(
                claims.spiffe_id,
                self.service_name,
            )

            if not is_authorized:
                logger.warning(
                    "Unauthorized access attempt: %s -> %s (%s)",
                    claims.service_name,
                    self.service_name,
                    reason,
                )
                return JSONResponse(
                    {"error": "Unauthorized", "detail": reason},
                    status_code=403,
                )

            # Store claims in request state
            request.state.claims = claims

        elif self.require_identity:
            logger.warning("No identity found for request to %s", path)
            return JSONResponse(
                {"error": "Identity required", "detail": "No valid workload identity"},
                status_code=401,
            )

        return await call_next(request)


def require_permission(permission: str) -> Callable:
    """Dependency that requires a specific permission.

    Usage:
        @app.get("/predict")
        async def predict(
            request: Request,
            _: None = Depends(require_permission("predict"))
        ):
            ...
    """

    async def checker(request: Request) -> None:
        claims: Optional[Claims] = getattr(request.state, "claims", None)

        if not claims:
            raise HTTPException(status_code=401, detail="Identity required")

        if permission not in claims.permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required",
            )

    return Depends(checker)


def require_role(role: str) -> Callable:
    """Dependency that requires a specific role.

    Usage:
        @app.post("/admin/config")
        async def admin_config(
            request: Request,
            _: None = Depends(require_role("admin"))
        ):
            ...
    """

    async def checker(request: Request) -> None:
        claims: Optional[Claims] = getattr(request.state, "claims", None)

        if not claims:
            raise HTTPException(status_code=401, detail="Identity required")

        if role not in claims.roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' required",
            )

    return Depends(checker)


def require_service(allowed_services: list[str]) -> Callable:
    """Dependency that requires caller to be one of the allowed services.

    Usage:
        @app.post("/internal/sync")
        async def sync(
            request: Request,
            _: None = Depends(require_service(["api-gateway", "ml-service"]))
        ):
            ...
    """

    async def checker(request: Request) -> None:
        claims: Optional[Claims] = getattr(request.state, "claims", None)

        if not claims:
            raise HTTPException(status_code=401, detail="Identity required")

        if claims.service_name not in allowed_services:
            raise HTTPException(
                status_code=403,
                detail=f"Service '{claims.service_name}' not authorized",
            )

    return Depends(checker)


def get_claims(request: Request) -> Claims:
    """Get claims from request, raising if not present.

    Usage:
        @app.get("/whoami")
        async def whoami(claims: Claims = Depends(get_claims)):
            return {"service": claims.service_name}
    """
    claims: Optional[Claims] = getattr(request.state, "claims", None)
    if not claims:
        raise HTTPException(status_code=401, detail="Identity required")
    return claims
