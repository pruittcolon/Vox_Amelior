"""
RBAC Engine - Role-Based Access Control.

Implements enterprise-grade RBAC following 2024 best practices:
- Hierarchical roles with permission inheritance
- Resource-level permissions
- Tenant-scoped authorization

Usage:
    @app.get("/admin")
    async def admin_endpoint(
        user: User = Depends(require_permission(Permission.ADMIN))
    ):
        return {"message": "Admin access granted"}
"""

from enum import Enum
from functools import wraps
from typing import Callable, Optional, Set

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel


# =============================================================================
# ROLE AND PERMISSION DEFINITIONS
# =============================================================================


class Role(str, Enum):
    """
    System roles with hierarchical permissions.
    
    Hierarchy: SUPER_ADMIN > ADMIN > MANAGER > USER > VIEWER
    """

    SUPER_ADMIN = "super_admin"  # Platform-level admin
    ADMIN = "admin"  # Tenant admin
    MANAGER = "manager"  # Department/team manager
    USER = "user"  # Standard user
    VIEWER = "viewer"  # Read-only access
    SERVICE = "service"  # Service account


class Permission(str, Enum):
    """Fine-grained permissions."""

    # Resource permissions
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    
    # Admin permissions
    ADMIN = "admin"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    MANAGE_SETTINGS = "manage_settings"
    
    # Feature permissions
    USE_AI = "use_ai"
    USE_ANALYTICS = "use_analytics"
    USE_TRANSCRIPTION = "use_transcription"
    EXPORT_DATA = "export_data"
    
    # Audit permissions
    VIEW_AUDIT_LOG = "view_audit_log"
    
    # API permissions
    API_ACCESS = "api_access"
    WEBHOOK_MANAGE = "webhook_manage"


# =============================================================================
# PERMISSION MAPPINGS
# =============================================================================

# Role to permissions mapping
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.SUPER_ADMIN: {
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
        Permission.ADMIN,
        Permission.MANAGE_USERS,
        Permission.MANAGE_ROLES,
        Permission.MANAGE_SETTINGS,
        Permission.USE_AI,
        Permission.USE_ANALYTICS,
        Permission.USE_TRANSCRIPTION,
        Permission.EXPORT_DATA,
        Permission.VIEW_AUDIT_LOG,
        Permission.API_ACCESS,
        Permission.WEBHOOK_MANAGE,
    },
    Role.ADMIN: {
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
        Permission.ADMIN,
        Permission.MANAGE_USERS,
        Permission.MANAGE_SETTINGS,
        Permission.USE_AI,
        Permission.USE_ANALYTICS,
        Permission.USE_TRANSCRIPTION,
        Permission.EXPORT_DATA,
        Permission.VIEW_AUDIT_LOG,
        Permission.API_ACCESS,
    },
    Role.MANAGER: {
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
        Permission.USE_AI,
        Permission.USE_ANALYTICS,
        Permission.USE_TRANSCRIPTION,
        Permission.EXPORT_DATA,
        Permission.API_ACCESS,
    },
    Role.USER: {
        Permission.READ,
        Permission.WRITE,
        Permission.USE_AI,
        Permission.USE_ANALYTICS,
        Permission.USE_TRANSCRIPTION,
        Permission.API_ACCESS,
    },
    Role.VIEWER: {
        Permission.READ,
        Permission.API_ACCESS,
    },
    Role.SERVICE: {
        Permission.READ,
        Permission.WRITE,
        Permission.API_ACCESS,
    },
}  # type: ignore


# =============================================================================
# PERMISSION CHECKING
# =============================================================================


def check_permission(role: Role, required: Permission) -> bool:
    """
    Check if a role has a specific permission.
    
    Args:
        role: User's role
        required: Required permission
        
    Returns:
        True if role has permission, False otherwise
    """
    permissions = ROLE_PERMISSIONS.get(role, set())
    return required in permissions


def check_any_permission(role: Role, required: Set[Permission]) -> bool:
    """
    Check if a role has any of the required permissions.
    
    Args:
        role: User's role
        required: Set of permissions (any one is sufficient)
        
    Returns:
        True if role has at least one permission
    """
    permissions = ROLE_PERMISSIONS.get(role, set())
    return bool(permissions & required)


def check_all_permissions(role: Role, required: Set[Permission]) -> bool:
    """
    Check if a role has all required permissions.
    
    Args:
        role: User's role
        required: Set of permissions (all required)
        
    Returns:
        True if role has all permissions
    """
    permissions = ROLE_PERMISSIONS.get(role, set())
    return required.issubset(permissions)


def get_role_permissions(role: Role) -> Set[Permission]:
    """
    Get all permissions for a role.
    
    Args:
        role: User's role
        
    Returns:
        Set of permissions
    """
    return ROLE_PERMISSIONS.get(role, set())


# =============================================================================
# FASTAPI DEPENDENCIES
# =============================================================================


class UserContext(BaseModel):
    """User context for authorization."""

    user_id: str
    role: Role
    tenant_id: Optional[str] = None
    permissions: Set[Permission] = set()


async def get_current_user_role(request: Request) -> Role:
    """
    Get current user's role from request state.
    
    Args:
        request: FastAPI request
        
    Returns:
        User's role
        
    Raises:
        HTTPException: If not authenticated
    """
    if not hasattr(request.state, "user") or not request.state.user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    role_str = request.state.user.get("role", "user")
    try:
        return Role(role_str)
    except ValueError:
        return Role.USER


def require_permission(permission: Permission) -> Callable:
    """
    Dependency factory for permission checking.
    
    Usage:
        @app.get("/admin")
        async def admin_endpoint(
            user: User = Depends(require_permission(Permission.ADMIN))
        ):
            ...
    
    Args:
        permission: Required permission
        
    Returns:
        FastAPI dependency function
    """

    async def dependency(
        request: Request,
        role: Role = Depends(get_current_user_role),
    ):
        if not check_permission(role, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {permission.value} required"
            )
        return request.state.user

    return dependency


def require_any_permission(*permissions: Permission) -> Callable:
    """
    Dependency factory requiring any of the specified permissions.
    
    Args:
        permissions: One or more permissions (any is sufficient)
        
    Returns:
        FastAPI dependency function
    """

    async def dependency(
        request: Request,
        role: Role = Depends(get_current_user_role),
    ):
        if not check_any_permission(role, set(permissions)):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: one of {[p.value for p in permissions]} required"
            )
        return request.state.user

    return dependency


def require_role(*roles: Role) -> Callable:
    """
    Dependency factory for role checking.
    
    Args:
        roles: One or more allowed roles
        
    Returns:
        FastAPI dependency function
    """

    async def dependency(
        request: Request,
        user_role: Role = Depends(get_current_user_role),
    ):
        if user_role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role denied: one of {[r.value for r in roles]} required"
            )
        return request.state.user

    return dependency


# =============================================================================
# DECORATOR (Alternative to Depends)
# =============================================================================


def permission_required(permission: Permission):
    """
    Decorator for permission-based access control.
    
    Usage:
        @permission_required(Permission.ADMIN)
        async def admin_function(request: Request):
            ...
    
    Args:
        permission: Required permission
        
    Returns:
        Decorated function
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            if not hasattr(request.state, "user") or not request.state.user:
                raise HTTPException(status_code=401, detail="Not authenticated")
            
            role_str = request.state.user.get("role", "user")
            try:
                role = Role(role_str)
            except ValueError:
                role = Role.USER
            
            if not check_permission(role, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {permission.value} required"
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    
    return decorator
