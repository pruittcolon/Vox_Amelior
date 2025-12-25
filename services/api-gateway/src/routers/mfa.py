"""
MFA Router - Multi-Factor Authentication endpoints.

Handles MFA setup, verification, and management:
- POST /api/auth/mfa/setup - Generate TOTP secret and QR code
- POST /api/auth/mfa/verify - Verify TOTP and enable MFA
- POST /api/auth/mfa/disable - Disable MFA
- GET /api/auth/mfa/status - Get MFA status

Phase 2 Security: TOTP-based MFA implementation.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth/mfa", tags=["MFA"])


# =============================================================================
# Request/Response Models
# =============================================================================


class MFASetupResponse(BaseModel):
    """MFA setup response with secret and provisioning URI."""
    success: bool
    secret: str
    provisioning_uri: str
    backup_codes: list[str]
    message: str = "Scan QR code in authenticator app, then verify with a code."


class MFAVerifyRequest(BaseModel):
    """MFA verification request."""
    code: str
    
    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        # Normalize: remove spaces and dashes
        v = v.strip().replace(" ", "").replace("-", "")
        if len(v) not in (6, 8):  # 6 for TOTP, 8 for backup code
            raise ValueError("Code must be 6 digits or 8-character backup code")
        return v


class MFAVerifyResponse(BaseModel):
    """MFA verification response."""
    success: bool
    message: str


class MFAStatusResponse(BaseModel):
    """MFA status response."""
    enabled: bool
    backup_codes_remaining: int


class MFADisableRequest(BaseModel):
    """MFA disable request - requires current code for security."""
    code: str
    password: str  # Require password for extra security


# =============================================================================
# Dependencies
# =============================================================================


def get_auth_manager():
    """Dependency to get the global auth manager."""
    from src.main import auth_manager
    if not auth_manager:
        raise HTTPException(status_code=503, detail="Auth not available")
    return auth_manager


def get_mfa_manager():
    """Dependency to get MFA manager."""
    from shared.security.mfa import get_mfa_manager as _get_mfa
    return _get_mfa()


def get_security_config():
    """Get security configuration."""
    from src.config import SecurityConfig as SecConf
    return SecConf


def require_session(request: Request, auth_manager=Depends(get_auth_manager)):
    """Dependency that requires a valid session."""
    sec_conf = get_security_config()
    session_token = request.cookies.get(sec_conf.SESSION_COOKIE_NAME)
    
    if not session_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            session_token = auth_header[7:]
    
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    session = auth_manager.validate_session(session_token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    return session


# =============================================================================
# Routes
# =============================================================================


@router.post("/setup", response_model=MFASetupResponse)
async def setup_mfa(
    session=Depends(require_session),
    mfa_manager=Depends(get_mfa_manager),
):
    """
    Generate MFA secret for user.
    
    Returns provisioning URI for QR code and backup codes.
    MFA is not enabled until verified with /verify endpoint.
    """
    try:
        setup = mfa_manager.setup_mfa(session.user_id, session.user_id)
        
        logger.info("mfa_setup_initiated", user_id=session.user_id)
        
        return MFASetupResponse(
            success=True,
            secret=setup.secret,
            provisioning_uri=setup.provisioning_uri,
            backup_codes=setup.backup_codes,
        )
    except Exception as e:
        logger.error("mfa_setup_failed", user_id=session.user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to setup MFA")


@router.post("/verify", response_model=MFAVerifyResponse)
async def verify_mfa(
    request: MFAVerifyRequest,
    session=Depends(require_session),
    mfa_manager=Depends(get_mfa_manager),
):
    """
    Verify TOTP code and enable MFA.
    
    Must be called after /setup to activate MFA.
    """
    if mfa_manager.verify_and_enable(session.user_id, request.code):
        logger.info("mfa_enabled", user_id=session.user_id)
        return MFAVerifyResponse(
            success=True,
            message="MFA enabled successfully. Keep your backup codes safe!"
        )
    
    logger.warning("mfa_enable_failed", user_id=session.user_id)
    raise HTTPException(status_code=400, detail="Invalid code. Please try again.")


@router.get("/status", response_model=MFAStatusResponse)
async def get_mfa_status(
    session=Depends(require_session),
    mfa_manager=Depends(get_mfa_manager),
):
    """Get current MFA status for user."""
    status = mfa_manager.get_status(session.user_id)
    return MFAStatusResponse(
        enabled=status.enabled,
        backup_codes_remaining=status.backup_codes_remaining,
    )


@router.post("/disable", response_model=MFAVerifyResponse)
async def disable_mfa(
    request: MFADisableRequest,
    session=Depends(require_session),
    auth_manager=Depends(get_auth_manager),
    mfa_manager=Depends(get_mfa_manager),
):
    """
    Disable MFA for user.
    
    Requires current TOTP code or backup code AND password.
    """
    # Verify password
    user = auth_manager.get_user_by_id(session.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not auth_manager._verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid password")
    
    # Verify MFA code
    if not mfa_manager.verify(session.user_id, request.code):
        raise HTTPException(status_code=400, detail="Invalid MFA code")
    
    # Disable MFA
    mfa_manager.disable(session.user_id)
    
    logger.info("mfa_disabled_by_user", user_id=session.user_id)
    return MFAVerifyResponse(
        success=True,
        message="MFA disabled successfully."
    )
