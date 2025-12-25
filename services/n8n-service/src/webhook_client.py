"""
n8n Webhook Client

Async HTTP client for sending events to n8n webhooks.
"""

import logging
from datetime import datetime
from typing import Any

import httpx

from .command_registry import VoiceCommand
from .config import N8N_EMOTION_ALERT_PATH, N8N_VOICE_COMMAND_PATH, N8N_WEBHOOK_BASE_URL
from .emotion_tracker import EmotionAlert

logger = logging.getLogger(__name__)


class N8nWebhookClient:
    """
    Async client for sending events to n8n webhooks.

    Usage:
        client = N8nWebhookClient()
        await client.send_voice_command(command, speaker="pruitt", text="turn off lights")
    """

    def __init__(
        self,
        base_url: str = N8N_WEBHOOK_BASE_URL,
        voice_command_path: str = N8N_VOICE_COMMAND_PATH,
        emotion_alert_path: str = N8N_EMOTION_ALERT_PATH,
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.voice_command_path = voice_command_path
        self.emotion_alert_path = emotion_alert_path
        self.timeout = timeout

    def _build_url(self, path: str) -> str:
        """Build full webhook URL"""
        return f"{self.base_url}{path}"

    async def send_voice_command(
        self, command: VoiceCommand, speaker: str, text: str, segment_metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        Send a voice command event to n8n.

        Args:
            command: The matched VoiceCommand
            speaker: Who said the command
            text: Original transcribed text
            segment_metadata: Optional additional segment data

        Returns:
            True if webhook was successful, False otherwise
        """
        url = self._build_url(self.voice_command_path)

        payload = {
            "event_type": "voice_command",
            "command_id": command.command_id,
            "n8n_action": command.n8n_action,
            "speaker": speaker,
            "original_text": text,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": segment_metadata or {},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)

                if response.status_code in (200, 201, 204):
                    logger.info(f"✅ Voice command webhook sent: {command.command_id} -> {url}")
                    return True
                else:
                    logger.warning(f"⚠️ Voice command webhook failed: {response.status_code} - {response.text}")
                    return False

        except httpx.ConnectError:
            logger.warning(f"⚠️ Cannot connect to n8n webhook at {url} - is n8n running?")
            return False
        except Exception as e:
            logger.error(f"❌ Voice command webhook error: {e}")
            return False

    async def send_emotion_alert(self, alert: EmotionAlert, segment_metadata: dict[str, Any] | None = None) -> bool:
        """
        Send an emotion alert event to n8n.

        Args:
            alert: The EmotionAlert that was triggered
            segment_metadata: Optional additional segment data

        Returns:
            True if webhook was successful, False otherwise
        """
        url = self._build_url(self.emotion_alert_path)

        payload = {
            "event_type": "emotion_alert",
            "speaker": alert.speaker,
            "emotion": alert.emotion,
            "consecutive_count": alert.consecutive_count,
            "timestamp": alert.timestamp.isoformat(),
            "metadata": segment_metadata or {},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)

                if response.status_code in (200, 201, 204):
                    logger.info(
                        f"✅ Emotion alert webhook sent: {alert.speaker} ({alert.consecutive_count}x {alert.emotion})"
                    )
                    return True
                else:
                    logger.warning(f"⚠️ Emotion alert webhook failed: {response.status_code} - {response.text}")
                    return False

        except httpx.ConnectError:
            logger.warning(f"⚠️ Cannot connect to n8n webhook at {url} - is n8n running?")
            return False
        except Exception as e:
            logger.error(f"❌ Emotion alert webhook error: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if n8n is reachable"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try to reach n8n base (adjust if n8n has different health endpoint)
                response = await client.get(f"{self.base_url.replace('/webhook', '')}/healthz")
                return response.status_code == 200
        except Exception:
            return False


# Global client instance
_client: N8nWebhookClient | None = None


def get_webhook_client() -> N8nWebhookClient:
    """Get the global webhook client singleton"""
    global _client
    if _client is None:
        _client = N8nWebhookClient()
    return _client
