"""
Enhanced Authentication API Routes
With audit logging, rate limiting, and encrypted sessions
"""

import re
import secrets

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from pydantic import BaseModel, field_validator
from src.config import SecurityConfig as SecConf

from shared.security.audit_logger import get_audit_logger

from .auth_manager import UserRole, get_auth_manager
from .permissions import get_current_user, require_admin

router = APIRouter(prefix="/api/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    username: str
    password: str
    remember_me: bool = False


class LoginResponse(BaseModel):
    success: bool
    session_token: str | None = None
    user: dict | None = None
    message: str | None = None
    csrf_token: str | None = None


class SessionResponse(BaseModel):
    valid: bool
    user: dict | None = None


class PasswordChangeRequest(BaseModel):
    old_password: str
    new_password: str


class UserListResponse(BaseModel):
    users: list[dict]


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
    user: dict | None = None


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, req: Request, response: Response):
    """
    Authenticate user and create encrypted session
    Sets httpOnly cookie with session token
    """
    auth_manager = get_auth_manager()
    audit_logger = get_audit_logger()

    ip_address = req.client.host if req.client else "unknown"

    # Authenticate
    session_token = auth_manager.authenticate(
        username=request.username, password=request.password, ip_address=ip_address
    )

    if not session_token:
        # Log failed login
        audit_logger.log_login(
            username=request.username, ip_address=ip_address, success=False, details="Invalid credentials"
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_info = auth_manager.get_user_info(session_token)
    session = auth_manager.validate_session(session_token)
    csrf_token = session.csrf_token if session and session.csrf_token else secrets.token_hex(32)

    # Log successful login
    audit_logger.log_login(
        username=request.username,
        ip_address=ip_address,
        success=True,
        details=f"Role: {user_info.get('role')}, Speaker: {user_info.get('speaker_id')}",
    )

    # Set httpOnly cookie
    max_age = 2592000 if request.remember_me else 86400  # 30 days vs 24 hours
    response.set_cookie(
        key=SecConf.SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        secure=SecConf.SESSION_COOKIE_SECURE,
        samesite=getattr(SecConf, "SESSION_COOKIE_SAMESITE", "lax"),
        max_age=max_age,
    )
    response.set_cookie(
        key=SecConf.CSRF_COOKIE_NAME,
        value=csrf_token,
        httponly=False,
        secure=SecConf.SESSION_COOKIE_SECURE,
        samesite=getattr(SecConf, "SESSION_COOKIE_SAMESITE", "lax"),
        max_age=max_age,
    )

    return LoginResponse(
        success=True, session_token=session_token, user=user_info, message="Login successful", csrf_token=csrf_token
    )


@router.get("/check", response_model=SessionResponse)
async def check_session(ws_session: str | None = Cookie(None)):
    """
    Validate current session
    Called by frontend on page load
    """
    if not ws_session:
        return SessionResponse(valid=False, user=None)

    auth_manager = get_auth_manager()
    session = auth_manager.validate_session(ws_session)

    if not session:
        return SessionResponse(valid=False, user=None)

    user_info = auth_manager.get_user_info(ws_session)
    return SessionResponse(valid=True, user=user_info)


@router.post("/logout")
async def logout(req: Request, response: Response, ws_session: str | None = Cookie(None)):
    """
    End session and clear cookie
    """
    auth_manager = get_auth_manager()
    audit_logger = get_audit_logger()

    if ws_session:
        session = auth_manager.validate_session(ws_session)
        if session:
            ip_address = req.client.host if req.client else "unknown"
            audit_logger.log_logout(user_id=session.user_id, ip_address=ip_address)
        auth_manager.logout(ws_session)

    response.delete_cookie(key=SecConf.SESSION_COOKIE_NAME)
    response.delete_cookie(key=SecConf.CSRF_COOKIE_NAME)
    return {"success": True, "message": "Logged out"}


@router.get("/user")
async def get_current_user_info(user=Depends(get_current_user)):
    """
    Get current user information
    """
    return {
        "user_id": user.user_id,
        "username": user.username,
        "role": user.role.value,
        "speaker_id": user.speaker_id,
        "email": user.email,
    }


@router.post("/change-password")
async def change_password(request: PasswordChangeRequest, req: Request, user=Depends(get_current_user)):
    """
    Change user password
    Requires current password verification
    """
    auth_manager = get_auth_manager()
    audit_logger = get_audit_logger()

    success = auth_manager.change_password(
        username=user.username, old_password=request.old_password, new_password=request.new_password
    )

    ip_address = req.client.host if req.client else "unknown"

    if success:
        audit_logger.log_password_change(user_id=user.user_id, ip_address=ip_address, success=True)
        return {"success": True, "message": "Password changed successfully"}
    else:
        audit_logger.log_password_change(user_id=user.user_id, ip_address=ip_address, success=False)
        raise HTTPException(status_code=401, detail="Current password is incorrect")


@router.get("/users", response_model=UserListResponse)
async def list_users(admin_session=Depends(require_admin)):
    """
    List all users (admin only)
    """
    auth_manager = get_auth_manager()
    users = auth_manager.list_users()
    return UserListResponse(users=users)


@router.get("/rate-limit/status")
async def get_rate_limit_status(req: Request, user=Depends(get_current_user)):
    """
    Get current rate limit status for user
    """
    from src.middleware.rate_limiter import get_rate_limiter

    rate_limiter = get_rate_limiter()
    ip_address = req.client.host if req.client else "unknown"

    status = rate_limiter.get_status(ip_address)

    return {"ip_address": ip_address, "user_id": user.user_id, "limits": status}


# Helper dependency for role checking
def require_role(role: UserRole):
    """Dependency factory that requires specific role"""

    def role_checker(ws_session: str | None = Cookie(None)):
        if not ws_session:
            raise HTTPException(status_code=401, detail="Not authenticated")

        auth_manager = get_auth_manager()
        if not auth_manager.check_permission(ws_session, role):
            session = auth_manager.validate_session(ws_session)
            user_role = session.role.value if session else "unknown"
            raise HTTPException(
                status_code=403, detail=f"Insufficient permissions. Required: {role.value}, Your role: {user_role}"
            )

        return ws_session

    return role_checker


@router.get("/admin/test")
async def admin_only_route(session=Depends(require_admin)):
    """Example: Admin-only endpoint"""
    return {"message": "Welcome, admin!", "user_id": session.user_id}


@router.post("/users/create", response_model=CreateUserResponse)
async def create_user(request: CreateUserRequest, req: Request, admin_session=Depends(require_admin)):
    """
    Create a new user (admin only).

    Implements NIST 2024 password requirements:
    - Minimum 8 characters
    - At least 1 uppercase, 1 lowercase, 1 digit
    - Maximum 128 characters
    """
    auth_manager = get_auth_manager()
    audit_logger = get_audit_logger()

    ip_address = req.client.host if req.client else "unknown"

    # Check if user already exists
    if auth_manager.get_user(request.username):
        audit_logger.log_security_event(
            event_description=f"User creation failed: username '{request.username}' already exists",
            ip_address=ip_address,
            details={"admin_user": admin_session.user_id, "attempted_username": request.username},
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
        audit_logger.log_data_access(
            user_id=admin_session.user_id, resource_type="user", resource_id=user.username, action="create"
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
        audit_logger.log_security_event(
            event_description=f"User creation validation failed: {str(e)}",
            ip_address=ip_address,
            details={"admin_user": admin_session.user_id, "username": request.username},
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        audit_logger.log_security_event(
            event_description=f"User creation error: {str(e)}",
            ip_address=ip_address,
            details={"admin_user": admin_session.user_id, "username": request.username},
        )
        raise HTTPException(status_code=500, detail="Failed to create user")
