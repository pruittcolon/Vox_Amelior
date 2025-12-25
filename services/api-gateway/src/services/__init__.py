"""
Services Package.

Business logic layer - all services should be imported from here.

Phase 3 of API Restructure.
"""

from .auth_service import AuthResult, AuthService

__all__ = [
    "AuthService",
    "AuthResult",
]
