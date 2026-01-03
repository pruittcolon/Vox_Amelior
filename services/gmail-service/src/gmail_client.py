"""
Gmail API Client - OAuth and email operations.

Handles Google OAuth 2.0 flow, token management, and Gmail API interactions.
All tokens are encrypted at rest using AES-256-GCM.
"""

import base64
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .models import EmailFetchRequest, EmailMessage, TimeframePreset

logger = logging.getLogger(__name__)

# Gmail API scopes - read-only access to emails
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.labels",
]


def _load_secret(filename: str, env_fallback: str = "") -> str:
    """Load secret from Docker secrets file or environment variable.
    
    Args:
        filename: Name of the secret file in /run/secrets/.
        env_fallback: Environment variable name for fallback.
    
    Returns:
        Secret value or empty string if not found.
    """
    secret_path = Path(f"/run/secrets/{filename}")
    if secret_path.exists():
        content = secret_path.read_text().strip()
        # Skip comment lines (placeholder files)
        lines = [l for l in content.split('\n') if l and not l.startswith('#')]
        if lines:
            return lines[0].strip()
    return os.getenv(env_fallback, "")


# Configuration - load from Docker secrets first
CLIENT_ID = _load_secret("gmail_client_id", "GMAIL_CLIENT_ID")
CLIENT_SECRET = _load_secret("gmail_client_secret", "GMAIL_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GMAIL_REDIRECT_URI", "https://localhost/api/gmail/oauth/callback")
TOKEN_STORAGE_PATH = Path(os.getenv("TOKEN_STORAGE_PATH", "/app/tokens"))


def _load_encryption_key() -> bytes:
    """Load or generate the token encryption key.

    Returns:
        32-byte Fernet-compatible encryption key.
    """
    key_path = Path("/run/secrets/gmail_token_key")
    if key_path.exists():
        return key_path.read_bytes().strip()

    # Fallback: derive from service JWT secret
    jwt_secret_path = Path("/run/secrets/service_jwt_secret")
    if jwt_secret_path.exists():
        secret = jwt_secret_path.read_bytes()
        # Use first 32 bytes, base64 encoded for Fernet
        return base64.urlsafe_b64encode(secret[:32])

    # Development fallback (not secure for production)
    logger.warning("[GMAIL] Using development encryption key - NOT SECURE FOR PRODUCTION")
    return Fernet.generate_key()


ENCRYPTION_KEY = _load_encryption_key()
_fernet = Fernet(ENCRYPTION_KEY)


class GmailClient:
    """Gmail API client with OAuth token management.

    This client handles the complete OAuth 2.0 flow for Gmail access,
    including token encryption, refresh, and revocation.
    """

    def __init__(self, user_id: str) -> None:
        """Initialize Gmail client for a specific user.

        Args:
            user_id: Unique identifier for the user (used for token storage).
        """
        self.user_id = user_id
        self._credentials: Credentials | None = None
        self._service: Any = None

    @property
    def token_path(self) -> Path:
        """Path to the encrypted token file for this user."""
        TOKEN_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        return TOKEN_STORAGE_PATH / f"{self.user_id}.token.enc"

    def get_authorization_url(self, state: str | None = None) -> tuple[str, str]:
        """Generate Google OAuth authorization URL.

        Args:
            state: Optional CSRF state parameter.

        Returns:
            Tuple of (authorization_url, state).

        Raises:
            ValueError: If OAuth credentials are not configured.
        """
        if not CLIENT_ID or not CLIENT_SECRET:
            raise ValueError("Gmail OAuth credentials not configured")

        client_config = {
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI],
            }
        }

        flow = Flow.from_client_config(
            client_config,
            scopes=GMAIL_SCOPES,
            redirect_uri=REDIRECT_URI,
        )

        authorization_url, flow_state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            state=state,
            prompt="consent",  # Force consent to get refresh token
        )

        return authorization_url, flow_state

    def handle_callback(self, code: str, state: str) -> dict[str, Any]:
        """Handle OAuth callback and store tokens.

        Args:
            code: Authorization code from Google.
            state: State parameter for verification.

        Returns:
            Dict with user email and token expiry.

        Raises:
            ValueError: If OAuth exchange fails.
        """
        if not CLIENT_ID or not CLIENT_SECRET:
            raise ValueError("Gmail OAuth credentials not configured")

        client_config = {
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI],
            }
        }

        flow = Flow.from_client_config(
            client_config,
            scopes=GMAIL_SCOPES,
            redirect_uri=REDIRECT_URI,
            state=state,
        )

        flow.fetch_token(code=code)
        self._credentials = flow.credentials

        # Store encrypted tokens
        self._save_tokens()

        # Get user email
        service = build("gmail", "v1", credentials=self._credentials)
        profile = service.users().getProfile(userId="me").execute()
        email = profile.get("emailAddress", "")

        logger.info(f"[GMAIL] OAuth complete for user {self.user_id}, email: {email}")

        return {
            "email": email,
            "expires_at": self._credentials.expiry.isoformat() if self._credentials.expiry else None,
            "scopes": list(self._credentials.scopes) if self._credentials.scopes else [],
        }

    def _save_tokens(self) -> None:
        """Save encrypted credentials to disk."""
        if not self._credentials:
            return

        token_data = {
            "token": self._credentials.token,
            "refresh_token": self._credentials.refresh_token,
            "token_uri": self._credentials.token_uri,
            "client_id": self._credentials.client_id,
            "client_secret": self._credentials.client_secret,
            "scopes": list(self._credentials.scopes) if self._credentials.scopes else [],
            "expiry": self._credentials.expiry.isoformat() if self._credentials.expiry else None,
        }

        encrypted = _fernet.encrypt(json.dumps(token_data).encode())
        self.token_path.write_bytes(encrypted)
        logger.debug(f"[GMAIL] Tokens saved for user {self.user_id}")

    def _load_tokens(self) -> bool:
        """Load and decrypt credentials from disk.

        Returns:
            True if tokens were loaded successfully.
        """
        if not self.token_path.exists():
            return False

        try:
            encrypted = self.token_path.read_bytes()
            decrypted = _fernet.decrypt(encrypted)
            token_data = json.loads(decrypted)

            expiry = None
            if token_data.get("expiry"):
                expiry = datetime.fromisoformat(token_data["expiry"])

            self._credentials = Credentials(
                token=token_data["token"],
                refresh_token=token_data.get("refresh_token"),
                token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=token_data.get("client_id", CLIENT_ID),
                client_secret=token_data.get("client_secret", CLIENT_SECRET),
                scopes=token_data.get("scopes", GMAIL_SCOPES),
                expiry=expiry,
            )

            logger.debug(f"[GMAIL] Tokens loaded for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"[GMAIL] Failed to load tokens: {e}")
            return False

    def _refresh_if_needed(self) -> bool:
        """Refresh credentials if expired.

        Returns:
            True if credentials are valid (or were refreshed successfully).
        """
        if not self._credentials:
            return False

        if not self._credentials.expired:
            return True

        if not self._credentials.refresh_token:
            logger.warning("[GMAIL] Token expired and no refresh token available")
            return False

        try:
            self._credentials.refresh(Request())
            self._save_tokens()
            logger.info(f"[GMAIL] Tokens refreshed for user {self.user_id}")
            return True
        except Exception as e:
            logger.error(f"[GMAIL] Token refresh failed: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if Gmail is connected and tokens are valid.

        Returns:
            True if connected with valid tokens.
        """
        if not self._credentials:
            if not self._load_tokens():
                return False

        return self._refresh_if_needed()

    def get_status(self) -> dict[str, Any]:
        """Get OAuth connection status.

        Returns:
            Dict with connection status details.
        """
        if not self.is_connected():
            return {
                "connected": False,
                "email": None,
                "expires_at": None,
                "scopes": [],
            }

        try:
            service = build("gmail", "v1", credentials=self._credentials)
            profile = service.users().getProfile(userId="me").execute()
            email = profile.get("emailAddress", "")

            return {
                "connected": True,
                "email": email,
                "expires_at": self._credentials.expiry.isoformat() if self._credentials.expiry else None,
                "scopes": list(self._credentials.scopes) if self._credentials.scopes else [],
            }
        except Exception as e:
            logger.error(f"[GMAIL] Failed to get status: {e}")
            return {
                "connected": False,
                "email": None,
                "expires_at": None,
                "scopes": [],
                "error": str(e),
            }

    def disconnect(self) -> bool:
        """Revoke tokens and disconnect Gmail.

        Returns:
            True if disconnection was successful.
        """
        try:
            if self.token_path.exists():
                self.token_path.unlink()

            # Attempt to revoke the token with Google
            if self._credentials and self._credentials.token:
                import httpx

                httpx.post(
                    "https://oauth2.googleapis.com/revoke",
                    params={"token": self._credentials.token},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

            self._credentials = None
            self._service = None
            logger.info(f"[GMAIL] Disconnected user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"[GMAIL] Disconnect error: {e}")
            return False

    def _get_service(self) -> Any:
        """Get or create Gmail API service.

        Returns:
            Gmail API service object.

        Raises:
            ValueError: If not connected.
        """
        if not self.is_connected():
            raise ValueError("Gmail not connected")

        if not self._service:
            self._service = build("gmail", "v1", credentials=self._credentials)

        return self._service

    def _calculate_date_range(
        self, request: EmailFetchRequest
    ) -> tuple[datetime, datetime]:
        """Calculate date range from fetch request.

        Args:
            request: Email fetch request with timeframe.

        Returns:
            Tuple of (start_date, end_date).
        """
        now = datetime.now(timezone.utc)

        if request.timeframe == TimeframePreset.CUSTOM:
            start = request.start_date or (now - timedelta(days=7))
            end = request.end_date or now
            return start, end

        timeframe_days = {
            TimeframePreset.LAST_24H: 1,
            TimeframePreset.LAST_7D: 7,
            TimeframePreset.LAST_30D: 30,
            TimeframePreset.LAST_90D: 90,
        }

        days = timeframe_days.get(request.timeframe, 7)
        return now - timedelta(days=days), now

    def fetch_emails(self, request: EmailFetchRequest) -> list[EmailMessage]:
        """Fetch emails from Gmail based on request parameters.

        Args:
            request: Email fetch request with filters.

        Returns:
            List of EmailMessage objects.

        Raises:
            ValueError: If not connected.
            HttpError: If Gmail API request fails.
        """
        service = self._get_service()
        start_date, end_date = self._calculate_date_range(request)

        # Build Gmail search query
        query_parts = []

        # Date filters (Gmail uses epoch seconds)
        start_epoch = int(start_date.timestamp())
        end_epoch = int(end_date.timestamp())
        query_parts.append(f"after:{start_epoch}")
        query_parts.append(f"before:{end_epoch}")

        # Label filters
        if request.labels:
            for label in request.labels:
                query_parts.append(f"label:{label}")

        # Custom query
        if request.query:
            query_parts.append(request.query)

        query = " ".join(query_parts)
        logger.info(f"[GMAIL] Fetching emails with query: {query}")

        try:
            # List message IDs
            results = (
                service.users()
                .messages()
                .list(userId="me", q=query, maxResults=request.max_results)
                .execute()
            )

            messages = results.get("messages", [])
            emails: list[EmailMessage] = []

            for msg_info in messages:
                try:
                    # Get full message
                    msg = (
                        service.users()
                        .messages()
                        .get(
                            userId="me",
                            id=msg_info["id"],
                            format="full" if request.include_body else "metadata",
                        )
                        .execute()
                    )

                    email = self._parse_message(msg, include_body=request.include_body)
                    emails.append(email)

                except HttpError as e:
                    logger.warning(f"[GMAIL] Failed to fetch message {msg_info['id']}: {e}")
                    continue

            logger.info(f"[GMAIL] Fetched {len(emails)} emails")
            return emails

        except HttpError as e:
            logger.error(f"[GMAIL] API error: {e}")
            raise

    def _parse_message(self, msg: dict[str, Any], include_body: bool = True) -> EmailMessage:
        """Parse Gmail API message into EmailMessage model.

        Args:
            msg: Raw Gmail API message dict.
            include_body: Whether to extract body content.

        Returns:
            EmailMessage model.
        """
        headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}

        # Parse sender
        sender_raw = headers.get("from", "")
        sender_name, sender_email = parseaddr(sender_raw)

        # Parse recipients
        to_raw = headers.get("to", "")
        recipients = [addr.strip() for addr in to_raw.split(",") if addr.strip()]

        # Parse date
        date_str = headers.get("date", "")
        try:
            from email.utils import parsedate_to_datetime

            date = parsedate_to_datetime(date_str)
        except Exception:
            # Fallback to internal timestamp
            internal_date = msg.get("internalDate", "0")
            date = datetime.fromtimestamp(int(internal_date) / 1000, tz=timezone.utc)

        # Extract body
        body = None
        if include_body:
            body = self._extract_body(msg.get("payload", {}))

        # Check for attachments
        has_attachments = self._has_attachments(msg.get("payload", {}))

        # Check labels
        labels = msg.get("labelIds", [])
        is_unread = "UNREAD" in labels

        return EmailMessage(
            id=msg["id"],
            thread_id=msg.get("threadId", msg["id"]),
            subject=headers.get("subject", "(No Subject)"),
            sender=sender_email or sender_raw,
            sender_name=sender_name,
            recipients=recipients,
            date=date,
            snippet=msg.get("snippet", ""),
            body=body,
            labels=labels,
            is_unread=is_unread,
            has_attachments=has_attachments,
        )

    def _extract_body(self, payload: dict[str, Any]) -> str:
        """Extract email body from message payload.

        Args:
            payload: Gmail message payload.

        Returns:
            Decoded body text (prefers plain text over HTML).
        """
        body_data = None

        # Check for direct body
        if payload.get("body", {}).get("data"):
            body_data = payload["body"]["data"]
            mime_type = payload.get("mimeType", "")
        else:
            # Check parts for text/plain or text/html
            parts = payload.get("parts", [])
            for part in parts:
                mime_type = part.get("mimeType", "")
                if mime_type == "text/plain" and part.get("body", {}).get("data"):
                    body_data = part["body"]["data"]
                    break
                elif mime_type == "text/html" and part.get("body", {}).get("data"):
                    body_data = part["body"]["data"]
                    # Continue looking for text/plain
                elif part.get("parts"):
                    # Nested multipart
                    nested_body = self._extract_body(part)
                    if nested_body:
                        return nested_body

        if not body_data:
            return ""

        try:
            decoded = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
            # Strip HTML tags if HTML content
            if "<html" in decoded.lower() or "<body" in decoded.lower():
                decoded = re.sub(r"<[^>]+>", " ", decoded)
                decoded = re.sub(r"\s+", " ", decoded).strip()
            return decoded[:10000]  # Limit body size
        except Exception as e:
            logger.warning(f"[GMAIL] Body decode error: {e}")
            return ""

    def _has_attachments(self, payload: dict[str, Any]) -> bool:
        """Check if message has attachments.

        Args:
            payload: Gmail message payload.

        Returns:
            True if message has attachments.
        """
        parts = payload.get("parts", [])
        for part in parts:
            if part.get("filename"):
                return True
            if part.get("parts"):
                if self._has_attachments(part):
                    return True
        return False

    def get_email_by_id(self, email_id: str) -> EmailMessage | None:
        """Fetch a single email by ID.

        Args:
            email_id: Gmail message ID.

        Returns:
            EmailMessage or None if not found.
        """
        try:
            service = self._get_service()
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=email_id, format="full")
                .execute()
            )
            return self._parse_message(msg, include_body=True)
        except HttpError as e:
            logger.error(f"[GMAIL] Failed to get email {email_id}: {e}")
            return None
