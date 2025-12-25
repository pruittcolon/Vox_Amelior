"""
Authentication router.

Handles all authentication endpoints including:
- Login/logout with session management
- Public self-registration (if enabled)
- Session validation
- User management (admin-only)

Phase 5.1: Consolidated from main.py for proper layered architecture.
"""

import logging
import os
import re
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# Rate limiting state (module-level, reset on restart)
_LOGIN_WINDOW = int(os.getenv("LOGIN_RATE_LIMIT_WINDOW", "60"))
_LOGIN_LIMIT = int(os.getenv("LOGIN_RATE_LIMIT_LIMIT", "5"))
login_attempts: dict[str, dict[str, Any]] = {"window": _LOGIN_WINDOW, "limit": _LOGIN_LIMIT, "ips": {}}

_REGISTER_WINDOW = int(os.getenv("REGISTER_RATE_LIMIT_WINDOW", "60"))
_REGISTER_LIMIT = int(os.getenv("REGISTER_RATE_LIMIT_LIMIT", "5"))
register_attempts: dict[str, dict[str, Any]] = {"window": _REGISTER_WINDOW, "limit": _REGISTER_LIMIT, "ips": {}}


class CreateUserRequest(BaseModel):
    """Admin-only user creation request with NIST 2024 password validation."""

    username: str
    password: str
    email: str | None = None
    role: str = "user"  # "admin" or "user"
    speaker_id: str | None = None

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        if not re.match(r"^[a-zA-Z0-9_]{3,50}$", v):
            raise ValueError("Username must be 3-50 alphanumeric characters or underscores")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        """NIST 2024 recommends minimum 8 chars, prioritize length over complexity."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if len(v) > 128:
            raise ValueError("Password must be at most 128 characters")
        # Require basic complexity: at least one of each category
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one digit")
        return v

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v.lower() not in ("admin", "user"):
            raise ValueError('Role must be "admin" or "user"')
        return v.lower()

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if v and not re.match(r"^[^@]+@[^@]+\.[^@]+$", v):
            raise ValueError("Invalid email format")
        return v


class CreateUserResponse(BaseModel):
    success: bool
    message: str
    user: dict[str, Any] | None = None


class UserListResponse(BaseModel):
    users: list[dict[str, Any]]


def get_auth_manager():
    """Dependency to get the global auth manager."""
    from src.main import auth_manager

    if not auth_manager:
        raise HTTPException(status_code=503, detail="Auth not available")
    return auth_manager


def _client_ip(request: Request) -> str:
    """Extract client IP from request, respecting X-Forwarded-For header."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_admin(request: Request, auth_manager=Depends(get_auth_manager)):
    """Dependency that requires admin role."""
    from src.auth.auth_manager import UserRole
    from src.config import SecurityConfig as SecConf

    ws_session = request.cookies.get(SecConf.SESSION_COOKIE_NAME)
    if not ws_session:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session = auth_manager.validate_session(ws_session)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    if session.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")

    return session


@router.get("/users", response_model=UserListResponse)
async def list_users(admin_session=Depends(require_admin), auth_manager=Depends(get_auth_manager)):
    """List all users (admin only)."""
    users = auth_manager.list_users()
    return UserListResponse(users=users)


@router.post("/users/create", response_model=CreateUserResponse)
async def create_user(
    request: CreateUserRequest,
    raw_request: Request,
    admin_session=Depends(require_admin),
    auth_manager=Depends(get_auth_manager),
):
    """
    Create a new user (admin only).

    Implements NIST 2024 password requirements:
    - Minimum 8 characters
    - At least 1 uppercase, 1 lowercase, 1 digit
    - Maximum 128 characters
    """
    from src.auth.auth_manager import UserRole

    ip_address = _client_ip(raw_request)

    # Check if user already exists
    if auth_manager.get_user(request.username):
        logger.warning(
            "user_creation_failed: username '%s' already exists (admin: %s, ip: %s)",
            request.username,
            admin_session.user_id,
            ip_address,
        )
        raise HTTPException(status_code=409, detail=f"User '{request.username}' already exists")

    try:
        # Create the user
        role = UserRole.ADMIN if request.role == "admin" else UserRole.USER
        user = auth_manager.create_user(
            username=request.username,
            password=request.password,
            role=role,
            speaker_id=request.speaker_id,
            email=request.email,
        )

        # Log successful creation
        logger.info(
            "user_created: '%s' (role: %s) by admin '%s' from %s",
            user.username,
            user.role.value,
            admin_session.user_id,
            ip_address,
        )

        return CreateUserResponse(
            success=True,
            message=f"User '{user.username}' created successfully",
            user={
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role.value,
                "speaker_id": user.speaker_id,
                "email": user.email,
            },
        )

    except ValueError as e:
        logger.warning(
            "user_creation_validation_failed: %s (admin: %s, ip: %s)", str(e), admin_session.user_id, ip_address
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("user_creation_error: %s (admin: %s, ip: %s)", str(e), admin_session.user_id, ip_address)
        raise HTTPException(status_code=500, detail="Failed to create user")


# =============================================================================
# Request Models for Core Auth Routes
# =============================================================================


class LoginRequest(BaseModel):
    """Login request with username and password."""

    username: str
    password: str


class RegisterRequest(BaseModel):
    """Public self-registration request with NIST 2024 password validation."""

    username: str
    password: str
    email: str | None = None
    speaker_id: str | None = None

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_]{3,50}$", v):
            raise ValueError("Username must be 3-50 alphanumeric characters or underscores")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if len(v) > 128:
            raise ValueError("Password must be at most 128 characters")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one digit")
        return v


# =============================================================================
# Core Auth Routes (migrated from main.py - Phase 5.1)
# =============================================================================


def get_security_config():
    """Get security configuration for cookie settings."""
    from src.config import SecurityConfig as SecConf

    return SecConf


@router.post("/register")
async def register(
    request: RegisterRequest, response: Response, raw_request: Request, auth_manager=Depends(get_auth_manager)
):
    """Public account creation. Creates a USER role and logs them in."""
    sec_conf = get_security_config()

    if os.getenv("ALLOW_SELF_REGISTRATION", "true").lower() not in {"1", "true", "yes"}:
        raise HTTPException(status_code=403, detail="Self-registration is disabled")

    # Rate limiting
    ip = _client_ip(raw_request)
    now = time.time()
    window = register_attempts["window"]
    lim = register_attempts["limit"]
    ip_state = register_attempts["ips"].get(ip, {"count": 0, "start": now})
    if now - ip_state["start"] > window:
        ip_state = {"count": 0, "start": now}
    ip_state["count"] += 1
    register_attempts["ips"][ip] = ip_state
    if ip_state["count"] > lim:
        raise HTTPException(status_code=429, detail="Too many registration attempts. Please try again later.")

    # Check username uniqueness
    try:
        existing_users = auth_manager.list_users()
        if any(u.get("username", "").lower() == request.username.lower() for u in existing_users):
            raise HTTPException(status_code=409, detail="Username already exists")
    except HTTPException:
        raise
    except Exception:
        pass

    from src.auth.auth_manager import UserRole

    # Generate unique speaker ID
    speaker_id = request.speaker_id or request.username.lower().replace(" ", "_")

    try:
        auth_manager.create_user(
            username=request.username,
            password=request.password,
            role=UserRole.USER,
            speaker_id=speaker_id,
            email=request.email,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Auto-login after registration
    session_token, lockout_status = auth_manager.authenticate(request.username, request.password, ip_address=ip)
    if lockout_status and lockout_status.is_locked:
        raise HTTPException(
            status_code=423,
            detail=f"Account locked. Try again in {lockout_status.remaining_seconds // 60} minutes.",
            headers={"Retry-After": str(lockout_status.remaining_seconds)},
        )
    if not session_token:
        raise HTTPException(status_code=500, detail="Login after registration failed")
    session = auth_manager.validate_session(session_token)
    if not session:
        raise HTTPException(status_code=500, detail="Session creation failed")

    response.set_cookie(
        key=sec_conf.SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        max_age=86400,
        samesite=getattr(sec_conf, "SESSION_COOKIE_SAMESITE", "lax"),
        secure=sec_conf.SESSION_COOKIE_SECURE,
    )
    response.set_cookie(
        key=sec_conf.CSRF_COOKIE_NAME,
        value=session.csrf_token or "",
        httponly=False,
        max_age=86400,
        samesite=getattr(sec_conf, "SESSION_COOKIE_SAMESITE", "lax"),
        secure=sec_conf.SESSION_COOKIE_SECURE,
    )

    return {
        "success": True,
        "session_token": session_token,
        "csrf_token": session.csrf_token or "",
        "user": {
            "user_id": session.user_id,
            "role": session.role.value,
            "speaker_id": speaker_id,
        },
    }


@router.post("/login")
async def login(
    request: LoginRequest, response: Response, raw_request: Request, auth_manager=Depends(get_auth_manager)
):
    """Authenticate user and create session with cookies."""
    sec_conf = get_security_config()

    # Rate limiting
    ip = _client_ip(raw_request)
    now = time.time()
    window = login_attempts["window"]
    lim = login_attempts["limit"]
    ip_state = login_attempts["ips"].get(ip, {"count": 0, "start": now})
    if now - ip_state["start"] > window:
        ip_state = {"count": 0, "start": now}
    ip_state["count"] += 1
    login_attempts["ips"][ip] = ip_state
    if ip_state["count"] > lim:
        raise HTTPException(status_code=429, detail="Too many login attempts. Please try again later.")

    # Authenticate
    session_token, lockout_status = auth_manager.authenticate(request.username, request.password, ip_address=ip)
    
    # Check if account is locked
    if lockout_status and lockout_status.is_locked:
        logger.warning("login_blocked: account locked for %s (ip: %s)", request.username, ip)
        raise HTTPException(
            status_code=423,
            detail=f"Account locked. Try again in {lockout_status.remaining_seconds // 60} minutes.",
            headers={"Retry-After": str(lockout_status.remaining_seconds)},
        )
    
    if not session_token:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    session = auth_manager.validate_session(session_token)
    if not session:
        raise HTTPException(status_code=500, detail="Session creation failed")

    # Set cookies
    response.set_cookie(
        key=sec_conf.SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        max_age=86400,
        samesite=getattr(sec_conf, "SESSION_COOKIE_SAMESITE", "lax"),
        secure=sec_conf.SESSION_COOKIE_SECURE,
    )
    response.set_cookie(
        key=sec_conf.CSRF_COOKIE_NAME,
        value=session.csrf_token or "",
        httponly=False,
        max_age=86400,
        samesite=getattr(sec_conf, "SESSION_COOKIE_SAMESITE", "lax"),
        secure=sec_conf.SESSION_COOKIE_SECURE,
    )

    return {
        "success": True,
        "session_token": session_token,
        "csrf_token": session.csrf_token or "",
        "user": {"user_id": session.user_id, "role": session.role.value},
    }


@router.post("/logout")
async def logout(request: Request, response: Response, auth_manager=Depends(get_auth_manager)):
    """End session and clear cookies."""
    sec_conf = get_security_config()

    response.delete_cookie(sec_conf.SESSION_COOKIE_NAME)
    response.delete_cookie(sec_conf.CSRF_COOKIE_NAME)
    return {"success": True}


@router.get("/session")
async def check_session(request: Request, auth_manager=Depends(get_auth_manager)):
    """Check if current session is valid."""
    sec_conf = get_security_config()
    ws_session = request.cookies.get(sec_conf.SESSION_COOKIE_NAME)

    if not ws_session:
        return {"valid": False}

    session = auth_manager.validate_session(ws_session)
    if not session:
        return {"valid": False}

    return {"valid": True, "user": {"user_id": session.user_id, "role": session.role.value}}


@router.get("/check")
async def check_auth(request: Request, auth_manager=Depends(get_auth_manager)):
    """
    Check if user is authenticated.
    Supports both cookie-based and Bearer token authentication.
    """
    sec_conf = get_security_config()

    # Try cookie first
    ws_session = request.cookies.get(sec_conf.SESSION_COOKIE_NAME)

    # Fall back to Bearer token
    if not ws_session:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            ws_session = auth_header[7:]

    if not ws_session:
        return {"valid": False}

    session = auth_manager.validate_session(ws_session)
    if not session:
        return {"valid": False}

    return {
        "valid": True,
        "user": {"user_id": session.user_id, "role": session.role.value, "speaker_id": session.speaker_id},
    }
