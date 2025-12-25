"""
OIDC Provider Module - OpenID Connect Single Sign-On.

Implements OAuth 2.0 Authorization Code Flow with PKCE for enterprise SSO.
Supports multiple identity providers: Google, Azure AD, Okta, Auth0.

Configuration via environment variables:
- OIDC_PROVIDER: Provider name (google, azure, okta, custom)
- OIDC_CLIENT_ID: OAuth client ID
- OIDC_CLIENT_SECRET_FILE: Path to client secret (Docker secret)
- OIDC_REDIRECT_URI: Callback URL after authentication
- OIDC_ISSUER: Provider issuer URL (for custom providers)

Usage:
    from auth.oidc_provider import OIDCProvider, get_oidc_provider
    
    provider = get_oidc_provider()
    auth_url = provider.get_authorization_url(state="random-state")
    # Redirect user to auth_url
    # On callback:
    tokens = await provider.exchange_code(code, code_verifier)
    user_info = await provider.get_user_info(tokens.access_token)
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)


class OIDCProviderType(Enum):
    """Supported OIDC identity providers."""
    GOOGLE = "google"
    AZURE = "azure"
    OKTA = "okta"
    AUTH0 = "auth0"
    CUSTOM = "custom"


@dataclass(frozen=True)
class OIDCConfig:
    """OIDC provider configuration."""
    provider_type: OIDCProviderType
    client_id: str
    client_secret: str
    redirect_uri: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    jwks_uri: str
    issuer: str
    scopes: list[str]


@dataclass
class OIDCTokens:
    """OAuth tokens from token exchange."""
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: str | None = None
    id_token: str | None = None
    scope: str | None = None


@dataclass
class OIDCUserInfo:
    """Normalized user information from OIDC provider."""
    sub: str  # Subject (unique user ID from provider)
    email: str | None
    email_verified: bool
    name: str | None
    given_name: str | None
    family_name: str | None
    picture: str | None
    locale: str | None
    raw: dict[str, Any]  # Original response


# Well-known OIDC configurations
PROVIDER_CONFIGS = {
    OIDCProviderType.GOOGLE: {
        "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_endpoint": "https://oauth2.googleapis.com/token",
        "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
        "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
        "issuer": "https://accounts.google.com",
        "scopes": ["openid", "email", "profile"],
    },
    OIDCProviderType.AZURE: {
        "authorization_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "userinfo_endpoint": "https://graph.microsoft.com/oidc/userinfo",
        "jwks_uri": "https://login.microsoftonline.com/common/discovery/v2.0/keys",
        "issuer": "https://login.microsoftonline.com/{tenant}/v2.0",
        "scopes": ["openid", "email", "profile", "User.Read"],
    },
    OIDCProviderType.OKTA: {
        # Domain-based - requires OIDC_ISSUER env var
        "scopes": ["openid", "email", "profile"],
    },
    OIDCProviderType.AUTH0: {
        # Domain-based - requires OIDC_ISSUER env var
        "scopes": ["openid", "email", "profile"],
    },
}


def _generate_code_verifier() -> str:
    """Generate PKCE code verifier (RFC 7636)."""
    return secrets.token_urlsafe(64)[:128]


def _generate_code_challenge(verifier: str) -> str:
    """Generate PKCE code challenge (S256 method)."""
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _load_secret(path: str | None, env_var: str | None = None) -> str:
    """Load secret from file or environment variable."""
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    if env_var:
        return os.getenv(env_var, "")
    return ""


class OIDCProvider:
    """
    OIDC provider for enterprise SSO.
    
    Implements OAuth 2.0 Authorization Code Flow with PKCE.
    """
    
    def __init__(self, config: OIDCConfig):
        """Initialize OIDC provider with configuration."""
        self.config = config
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"OIDC provider initialized: {config.provider_type.value}")
    
    @classmethod
    def from_environment(cls) -> "OIDCProvider":
        """
        Create OIDC provider from environment variables.
        
        Environment variables:
        - OIDC_PROVIDER: google, azure, okta, auth0, custom
        - OIDC_CLIENT_ID: OAuth client ID
        - OIDC_CLIENT_SECRET_FILE: Path to secret file (preferred)
        - OIDC_CLIENT_SECRET: Secret as env var (fallback)
        - OIDC_REDIRECT_URI: Callback URL
        - OIDC_ISSUER: Provider issuer URL (required for okta/auth0/custom)
        - OIDC_SCOPES: Space-separated scopes (optional)
        """
        provider_name = os.getenv("OIDC_PROVIDER", "google").lower()
        try:
            provider_type = OIDCProviderType(provider_name)
        except ValueError:
            provider_type = OIDCProviderType.CUSTOM
        
        client_id = os.getenv("OIDC_CLIENT_ID", "")
        client_secret = _load_secret(
            os.getenv("OIDC_CLIENT_SECRET_FILE"),
            "OIDC_CLIENT_SECRET"
        )
        redirect_uri = os.getenv("OIDC_REDIRECT_URI", "")
        issuer = os.getenv("OIDC_ISSUER", "")
        custom_scopes = os.getenv("OIDC_SCOPES", "").split()
        
        # Get provider-specific endpoints
        provider_config = PROVIDER_CONFIGS.get(provider_type, {})
        
        if provider_type in (OIDCProviderType.OKTA, OIDCProviderType.AUTH0, OIDCProviderType.CUSTOM):
            # These require issuer to build endpoints
            if not issuer:
                raise ValueError(f"OIDC_ISSUER required for {provider_type.value}")
            authorization_endpoint = f"{issuer}/v1/authorize"
            token_endpoint = f"{issuer}/v1/token"
            userinfo_endpoint = f"{issuer}/v1/userinfo"
            jwks_uri = f"{issuer}/v1/keys"
        else:
            authorization_endpoint = provider_config.get("authorization_endpoint", "")
            token_endpoint = provider_config.get("token_endpoint", "")
            userinfo_endpoint = provider_config.get("userinfo_endpoint", "")
            jwks_uri = provider_config.get("jwks_uri", "")
            issuer = issuer or provider_config.get("issuer", "")
        
        scopes = custom_scopes or provider_config.get("scopes", ["openid", "email", "profile"])
        
        config = OIDCConfig(
            provider_type=provider_type,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint,
            userinfo_endpoint=userinfo_endpoint,
            jwks_uri=jwks_uri,
            issuer=issuer,
            scopes=scopes,
        )
        
        return cls(config)
    
    def generate_state(self) -> str:
        """Generate cryptographically secure state parameter."""
        return secrets.token_urlsafe(32)
    
    def generate_pkce(self) -> tuple[str, str]:
        """
        Generate PKCE code verifier and challenge.
        
        Returns:
            Tuple of (code_verifier, code_challenge)
        """
        verifier = _generate_code_verifier()
        challenge = _generate_code_challenge(verifier)
        return verifier, challenge
    
    def get_authorization_url(
        self,
        state: str,
        code_challenge: str,
        nonce: str | None = None,
        login_hint: str | None = None,
    ) -> str:
        """
        Build authorization URL for redirect.
        
        Args:
            state: CSRF protection state parameter
            code_challenge: PKCE code challenge (S256)
            nonce: Optional nonce for ID token validation
            login_hint: Optional pre-filled email
            
        Returns:
            Full authorization URL to redirect user to.
        """
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.config.scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        
        if nonce:
            params["nonce"] = nonce
        if login_hint:
            params["login_hint"] = login_hint
        
        # Azure-specific: prompt for account selection
        if self.config.provider_type == OIDCProviderType.AZURE:
            params["prompt"] = "select_account"
        
        return f"{self.config.authorization_endpoint}?{urlencode(params)}"
    
    async def exchange_code(
        self,
        code: str,
        code_verifier: str,
    ) -> OIDCTokens:
        """
        Exchange authorization code for tokens.
        
        Args:
            code: Authorization code from callback
            code_verifier: PKCE code verifier
            
        Returns:
            OIDCTokens with access_token, refresh_token, id_token
            
        Raises:
            httpx.HTTPStatusError: If token exchange fails
        """
        data = {
            "grant_type": "authorization_code",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "code": code,
            "redirect_uri": self.config.redirect_uri,
            "code_verifier": code_verifier,
        }
        
        response = await self._client.post(
            self.config.token_endpoint,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        
        token_data = response.json()
        
        return OIDCTokens(
            access_token=token_data["access_token"],
            token_type=token_data.get("token_type", "Bearer"),
            expires_in=token_data.get("expires_in", 3600),
            refresh_token=token_data.get("refresh_token"),
            id_token=token_data.get("id_token"),
            scope=token_data.get("scope"),
        )
    
    async def refresh_tokens(self, refresh_token: str) -> OIDCTokens:
        """
        Refresh tokens using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New OIDCTokens
        """
        data = {
            "grant_type": "refresh_token",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "refresh_token": refresh_token,
        }
        
        response = await self._client.post(
            self.config.token_endpoint,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        
        token_data = response.json()
        
        return OIDCTokens(
            access_token=token_data["access_token"],
            token_type=token_data.get("token_type", "Bearer"),
            expires_in=token_data.get("expires_in", 3600),
            refresh_token=token_data.get("refresh_token", refresh_token),
            id_token=token_data.get("id_token"),
            scope=token_data.get("scope"),
        )
    
    async def get_user_info(self, access_token: str) -> OIDCUserInfo:
        """
        Fetch user information from provider.
        
        Args:
            access_token: Valid access token
            
        Returns:
            Normalized OIDCUserInfo
        """
        response = await self._client.get(
            self.config.userinfo_endpoint,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        response.raise_for_status()
        
        data = response.json()
        
        return OIDCUserInfo(
            sub=data.get("sub", ""),
            email=data.get("email"),
            email_verified=data.get("email_verified", False),
            name=data.get("name"),
            given_name=data.get("given_name"),
            family_name=data.get("family_name"),
            picture=data.get("picture"),
            locale=data.get("locale"),
            raw=data,
        )
    
    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()


# Singleton instance
_oidc_provider: OIDCProvider | None = None


def get_oidc_provider() -> OIDCProvider:
    """Get or create global OIDC provider instance."""
    global _oidc_provider
    if _oidc_provider is None:
        _oidc_provider = OIDCProvider.from_environment()
    return _oidc_provider


def is_oidc_enabled() -> bool:
    """Check if OIDC is configured and enabled."""
    return bool(os.getenv("OIDC_CLIENT_ID"))
