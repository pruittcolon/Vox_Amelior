"""
Auth Service Layer.

Wraps AuthManager with a clean service interface for dependency injection.

Phase 3 of API Restructure.
"""

from dataclasses import dataclass
from typing import Any

from src.auth.auth_manager import AuthManager, Session, User, UserRole


@dataclass
class AuthResult:
    """Result of authentication attempt."""

    success: bool
    session_token: str | None = None
    user_id: str | None = None
    role: str | None = None
    csrf_token: str | None = None
    error: str | None = None


class AuthService:
    """Authentication and authorization service.

    Provides a clean interface for auth operations, wrapping the underlying
    AuthManager. Designed for dependency injection via FastAPI's Depends().

    Example:
        @router.post("/login")
        async def login(
            request: LoginRequest,
            auth_service: AuthService = Depends(get_auth_service)
        ):
            result = auth_service.authenticate(request.username, request.password)
            if not result.success:
                raise HTTPException(401, result.error)
            return LoginResponse(**result.__dict__)
    """

    def __init__(self, auth_manager: AuthManager):
        """Initialize with AuthManager instance.

        Args:
            auth_manager: The underlying AuthManager for auth operations.
        """
        self._manager = auth_manager

    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: str | None = None,
    ) -> AuthResult:
        """Authenticate user and create session.

        Args:
            username: User's username.
            password: User's password.
            ip_address: Client IP for audit logging.

        Returns:
            AuthResult with session info on success, error on failure.
        """
        session_token = self._manager.authenticate(
            username=username,
            password=password,
            ip_address=ip_address,
        )

        if not session_token:
            return AuthResult(
                success=False,
                error="Invalid credentials",
            )

        session = self._manager.validate_session(session_token)
        if not session:
            return AuthResult(
                success=False,
                error="Session creation failed",
            )

        return AuthResult(
            success=True,
            session_token=session_token,
            user_id=session.user_id,
            role=session.role.value,
            csrf_token=session.csrf_token,
        )

    def validate_session(self, session_token: str) -> Session | None:
        """Validate session token.

        Args:
            session_token: Encrypted session token.

        Returns:
            Session object if valid, None otherwise.
        """
        return self._manager.validate_session(session_token)

    def logout(self, session_token: str) -> bool:
        """End session and invalidate token.

        Args:
            session_token: Session to invalidate.

        Returns:
            True if session was found and removed.
        """
        return self._manager.logout(session_token)

    def get_user_info(self, session_token: str) -> dict[str, Any] | None:
        """Get user info from session.

        Args:
            session_token: Valid session token.

        Returns:
            Dict with user_id, username, role, speaker_id, email.
        """
        return self._manager.get_user_info(session_token)

    def create_user(
        self,
        username: str,
        password: str,
        role: UserRole = UserRole.USER,
        speaker_id: str | None = None,
        email: str | None = None,
    ) -> User:
        """Create a new user.

        Args:
            username: Unique username.
            password: Plain text password (will be hashed).
            role: User role (default: USER).
            speaker_id: Optional speaker ID for data isolation.
            email: Optional email address.

        Returns:
            Created User object.

        Raises:
            ValueError: If username already exists.
        """
        return self._manager.create_user(
            username=username,
            password=password,
            role=role,
            speaker_id=speaker_id,
            email=email,
        )

    def check_permission(self, session_token: str, required_role: UserRole) -> bool:
        """Check if session has required role.

        Args:
            session_token: Session to check.
            required_role: Minimum required role.

        Returns:
            True if user has required role or higher.
        """
        return self._manager.check_permission(session_token, required_role)

    def change_password(
        self,
        username: str,
        old_password: str,
        new_password: str,
    ) -> bool:
        """Change user password.

        Args:
            username: Username to update.
            old_password: Current password for verification.
            new_password: New password to set.

        Returns:
            True if password was changed successfully.
        """
        return self._manager.change_password(username, old_password, new_password)

    def list_users(self) -> list:
        """List all users (admin only).

        Returns:
            List of user dicts (without password hashes).
        """
        return self._manager.list_users()
