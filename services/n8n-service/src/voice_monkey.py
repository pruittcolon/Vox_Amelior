"""
Voice Monkey Direct Integration

Direct HTTP calls to Alexa Voice Monkey API without going through n8n.
"""

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Voice Monkey API endpoint (v2)
VOICE_MONKEY_API_URL = "https://api-v2.voicemonkey.io/trigger"


@dataclass
class VoiceMonkeyConfig:
    """Configuration for Voice Monkey integration"""

    token: str
    device_id: str
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "VoiceMonkeyConfig":
        """Load configuration from Docker secrets or environment variables"""
        import os

        # Read from Docker secret files first (production)
        token = ""
        device_id = ""

        token_file = os.getenv("VOICE_MONKEY_TOKEN_FILE", "/run/secrets/voice_monkey_token")
        device_file = os.getenv("VOICE_MONKEY_DEVICE_FILE", "/run/secrets/voice_monkey_device_id")

        # Try reading from secret files
        if os.path.exists(token_file):
            try:
                with open(token_file) as f:
                    token = f.read().strip()
            except Exception:
                pass

        if os.path.exists(device_file):
            try:
                with open(device_file) as f:
                    device_id = f.read().strip()
            except Exception:
                pass

        # Fallback to environment variables (development)
        if not token:
            token = os.getenv("VOICE_MONKEY_TOKEN", "")
        if not device_id:
            device_id = os.getenv("VOICE_MONKEY_DEVICE_ID", "")

        return cls(
            token=token, device_id=device_id, enabled=os.getenv("VOICE_MONKEY_ENABLED", "true").lower() == "true"
        )

    def is_configured(self) -> bool:
        """Check if Voice Monkey is properly configured (only token required)"""
        return bool(self.token and self.enabled)


class VoiceMonkeyClient:
    """
    Direct HTTP client for Alexa Voice Monkey API.
    
    This makes the actual curl call:
    curl -X POST "https://api.voicemonkey.io/trigger" \
      -H "Content-Type: application/json" \
      -d '{"token": "...", "device": "...", "preset": "..."}'
    """

    def __init__(self, config: VoiceMonkeyConfig | None = None):
        self.config = config or VoiceMonkeyConfig.from_env()
        self.timeout = 10.0

    async def trigger_preset(self, preset: str, announcement: str | None = None) -> dict[str, Any]:
        """
        Trigger a Voice Monkey device using the same GET-based approach as the mobile app.

        Mobile app equivalent (from evenai.dart):
        final String fullUrl = "$voicemonkeyUrl&device=$encodedDevice";
        final response = await dio.get(fullUrl);

        For lights_off, we use device name "alllights" to match mobile app behavior.

        Args:
            preset: The preset name (maps to device name, e.g., "lights_off" -> "alllights")
            announcement: Optional text (not used in GET mode)

        Returns:
            Response data from Voice Monkey API
        """
        if not self.config.token:
            logger.warning("Voice Monkey not configured - skipping trigger")
            return {"success": False, "error": "Voice Monkey not configured (missing VOICE_MONKEY_TOKEN)"}

        # Map preset names to Voice Monkey device names (matching mobile app AlexaService)
        device_mapping = {
            # All lights (toggle)
            "lights_off": "alllights",
            "lights_on": "alllights",
            "alexa_voice_monkey_lights_off": "alllights",
            "alllights": "alllights",
            "all_lights_off": "alllights",
            # Bedroom light (toggle)
            "bedroom_light": "bedroomlight",
            "bedroomlight": "bedroomlight",
            "bedroom_light_off": "bedroomlight",
            "bedroom_light_on": "bedroomlight",
            # TV
            "tv_off": "tvoff",
            "tvoff": "tvoff",
            # Kitchen light - all map to same toggle device "kitchenlights"
            "kitchenlightoff": "kitchenlights",
            "kitchenlighton": "kitchenlights",
            "kitchenlighttoggle": "kitchenlights",
            "kitchen_light_off": "kitchenlights",
            "kitchen_light_on": "kitchenlights",
            "kitchen_lights": "kitchenlights",
            "kitchenlights": "kitchenlights",
            "kitchen_light": "kitchenlights",
            "turn_kitchen_light": "kitchenlights",
            # Living room light - all map to same toggle device "livingroomlight"
            "livingroomlightoff": "livingroomlight",
            "livingroomlighton": "livingroomlight",
            "livingroomlighttoggle": "livingroomlight",
            "living_room_light_off": "livingroomlight",
            "living_room_light_on": "livingroomlight",
            "turn_living_room_light": "livingroomlight",
            "livingroomlight": "livingroomlight",
        }

        # Get the device name (use mapping or preset directly)
        device_name = device_mapping.get(preset, preset)

        # Build URL like mobile app: base_url + &device=devicename
        # Base URL already has token, e.g.: https://api-v2.voicemonkey.io/trigger?token=xxx
        trigger_url = f"{VOICE_MONKEY_API_URL}?token={self.config.token}&device={device_name}"

        logger.info(f"🔊 Triggering Voice Monkey device: {device_name} (from preset: {preset})")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Use GET like the mobile app
                response = await client.get(trigger_url)

                if response.status_code == 200:
                    logger.info(f"✅ Voice Monkey triggered successfully: {device_name}")
                    return {"success": True, "device": device_name, "preset": preset, "response": response.text}
                else:
                    logger.warning(f"⚠️ Voice Monkey API error: {response.status_code} - {response.text}")
                    return {
                        "success": False,
                        "error": f"API returned {response.status_code}",
                        "response": response.text,
                    }

        except httpx.ConnectError as e:
            logger.error(f"❌ Cannot connect to Voice Monkey API: {e}")
            return {"success": False, "error": f"Connection error: {e}"}
        except Exception as e:
            logger.error(f"❌ Voice Monkey request failed: {e}")
            return {"success": False, "error": str(e)}

    async def trigger_lights_off(self) -> dict[str, Any]:
        """Convenience method to turn off lights"""
        return await self.trigger_preset("lights_off")

    async def trigger_lights_on(self) -> dict[str, Any]:
        """Convenience method to turn on lights"""
        return await self.trigger_preset("lights_on")

    async def trigger_with_announcement(self, text: str) -> dict[str, Any]:
        """Make Alexa speak a custom announcement"""
        return await self.trigger_preset("announce", announcement=text)


# Global client instance
_client: VoiceMonkeyClient | None = None


def get_voice_monkey_client() -> VoiceMonkeyClient:
    """Get the global Voice Monkey client singleton"""
    global _client
    if _client is None:
        _client = VoiceMonkeyClient()
    return _client
