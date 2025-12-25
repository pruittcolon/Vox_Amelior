"""
Core __init__ Module - Public API for core package.

Exports dependency factories, middleware, and lifespan management
for use by main.py and router modules.
"""

from .dependencies import (
    AuthManagerDep,
    ServiceAuthDep,
    analysis_jobs,
    get_app_instance_dir,
    get_auth_manager,
    get_service_auth,
    get_session_key,
    personality_jobs,
)
from .lifespan import lifespan
from .middleware import (
    RATE_LIMIT_ENABLED,
    CanonicalHostMiddleware,
    CSRFMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    create_audit_middleware,
)

__all__ = [
    # Dependencies
    "get_auth_manager",
    "get_service_auth",
    "get_session_key",
    "get_app_instance_dir",
    "AuthManagerDep",
    "ServiceAuthDep",
    "personality_jobs",
    "analysis_jobs",
    # Middleware
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware",
    "CSRFMiddleware",
    "CanonicalHostMiddleware",
    "create_audit_middleware",
    "RATE_LIMIT_ENABLED",
    # Lifespan
    "lifespan",
]
