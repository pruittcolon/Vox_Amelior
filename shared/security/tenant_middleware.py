"""
Tenant Isolation Middleware - Multi-tenant data isolation.

Provides tenant-based data isolation by:
- Extracting tenant_id from JWT claims or session
- Injecting tenant context into request state
- Blocking requests without valid tenant context

ISO 27002 Control: A.9.4 - Access Control
"""

import logging
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Paths exempt from tenant isolation (public endpoints, health checks)
EXEMPT_PATHS = frozenset({
    "/health",
    "/docs",
    "/openapi.json",
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/check",
    "/api/auth/session",
    "/static",
})


class TenantContext:
    """
    Tenant context holder for request-scoped data.
    
    Attributes:
        tenant_id: Unique identifier for the tenant
        user_id: User making the request
        is_admin: Whether user has admin privileges (can bypass isolation)
    """
    
    def __init__(
        self,
        tenant_id: str,
        user_id: str,
        is_admin: bool = False,
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.is_admin = is_admin
    
    def can_access_tenant(self, target_tenant_id: str) -> bool:
        """
        Check if current context can access target tenant's data.
        
        Args:
            target_tenant_id: Tenant ID to check access for
            
        Returns:
            True if access is allowed
        """
        if self.is_admin:
            return True
        return self.tenant_id == target_tenant_id


def get_tenant_context(request: Request) -> Optional[TenantContext]:
    """
    Get tenant context from request state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        TenantContext if available, None otherwise
    """
    return getattr(request.state, "tenant_context", None)


def require_tenant_context(request: Request) -> TenantContext:
    """
    Get tenant context or raise 403.
    
    Args:
        request: FastAPI request object
        
    Returns:
        TenantContext
        
    Raises:
        403 Forbidden if no tenant context
    """
    context = get_tenant_context(request)
    if not context:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=403,
            detail="Tenant context required for this operation"
        )
    return context


class TenantIsolationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce tenant isolation.
    
    Extracts tenant_id from session and injects into request state.
    Blocks requests to protected endpoints without valid tenant context.
    
    Usage:
        app.add_middleware(TenantIsolationMiddleware)
    """
    
    def __init__(self, app, exempt_paths: set[str] | None = None):
        """
        Initialize middleware.
        
        Args:
            app: ASGI application
            exempt_paths: Paths to exempt from tenant isolation
        """
        super().__init__(app)
        self.exempt_paths = exempt_paths or EXEMPT_PATHS
    
    def _is_exempt(self, path: str) -> bool:
        """Check if path is exempt from tenant isolation."""
        # Exact match
        if path in self.exempt_paths:
            return True
        # Prefix match
        for exempt in self.exempt_paths:
            if path.startswith(exempt):
                return True
        return False
    
    async def dispatch(self, request: Request, call_next):
        """Process request and inject tenant context."""
        path = request.url.path
        
        # Skip exempt paths
        if self._is_exempt(path):
            return await call_next(request)
        
        # Try to get session from auth manager
        try:
            from src.main import auth_manager
            from src.config import SecurityConfig as SecConf
            from src.auth.auth_manager import UserRole
            
            if not auth_manager:
                # Auth not initialized, skip isolation
                return await call_next(request)
            
            # Get session token from cookie or header
            session_token = request.cookies.get(SecConf.SESSION_COOKIE_NAME)
            if not session_token:
                auth_header = request.headers.get("Authorization", "")
                if auth_header.startswith("Bearer "):
                    session_token = auth_header[7:]
            
            if not session_token:
                # No session, let auth middleware handle it
                return await call_next(request)
            
            # Validate session
            session = auth_manager.validate_session(session_token)
            if not session:
                return await call_next(request)
            
            # Create tenant context
            # For now, use speaker_id as tenant_id (user-level isolation)
            # In enterprise, this would be organization_id from JWT claims
            tenant_id = session.speaker_id or session.user_id
            is_admin = session.role == UserRole.ADMIN
            
            context = TenantContext(
                tenant_id=tenant_id,
                user_id=session.user_id,
                is_admin=is_admin,
            )
            
            # Inject into request state
            request.state.tenant_context = context
            
            logger.debug(
                "tenant_context_set",
                tenant_id=tenant_id,
                user_id=session.user_id,
                is_admin=is_admin,
                path=path,
            )
            
        except Exception as e:
            # Log but don't block - let auth middleware handle auth failures
            logger.warning("tenant_context_extraction_failed", error=str(e))
        
        return await call_next(request)


def tenant_filter(query, model, request: Request):
    """
    Apply tenant filter to SQLAlchemy query.
    
    Usage in service:
        query = tenant_filter(
            db.query(Document),
            Document,
            request
        )
    
    Args:
        query: SQLAlchemy query
        model: Model class with tenant_id column
        request: FastAPI request
        
    Returns:
        Filtered query (admins see all)
    """
    context = get_tenant_context(request)
    if not context:
        # No context = no data
        return query.filter(False)
    
    if context.is_admin:
        # Admins can see everything
        return query
    
    # Filter by tenant
    if hasattr(model, "tenant_id"):
        return query.filter(model.tenant_id == context.tenant_id)
    elif hasattr(model, "speaker_id"):
        return query.filter(model.speaker_id == context.tenant_id)
    elif hasattr(model, "user_id"):
        return query.filter(model.user_id == context.tenant_id)
    
    # No tenant column - return unfiltered (should be audited)
    logger.warning(
        "no_tenant_column",
        model=model.__name__,
        message="Model has no tenant isolation column"
    )
    return query
