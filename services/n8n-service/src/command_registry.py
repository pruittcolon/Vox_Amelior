"""
Voice Command Registry

Extensible pattern-based command matching system.
Add new commands by calling CommandRegistry.register() or via the /commands API.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VoiceCommand:
    """A registered voice command pattern"""

    command_id: str
    pattern: str  # Regex pattern
    description: str
    n8n_action: str  # Action identifier sent to n8n
    compiled_pattern: re.Pattern = field(init=False, repr=False)
    created_at: datetime = field(default_factory=datetime.utcnow)
    enabled: bool = True

    def __post_init__(self):
        self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, text: str) -> bool:
        """Check if text matches this command pattern"""
        return bool(self.compiled_pattern.search(text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "pattern": self.pattern,
            "description": self.description,
            "n8n_action": self.n8n_action,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
        }


class CommandRegistry:
    """
    Registry for voice commands with pattern matching.

    Usage:
        registry = CommandRegistry()
        registry.register("lights_off", r"turn.*off.*light", "Turn off lights", "alexa_lights_off")

        match = registry.match("Honey, can you turn off the lights")
        if match:
            print(f"Matched: {match.command_id}")
    """

    def __init__(self):
        self._commands: dict[str, VoiceCommand] = {}
        self._load_default_commands()

    def _load_default_commands(self):
        """Load the default voice command patterns"""

        # === BEDROOM LIGHT COMMANDS ===

        # Command 1: "honey can I turn off the light" - generic light command with "honey"
        self.register(
            command_id="honey_light",
            pattern=r"honey\s+(?:can\s+(?:i|you)\s+)?(?:please\s+)?turn\s+(?:on|off)\s+(?:the\s+)?light",
            description="Control bedroom light via 'honey turn off the light'",
            n8n_action="bedroomlight",
        )

        # Command 2: "bedroom light off/on" - direct bedroom light command
        self.register(
            command_id="bedroom_light_direct",
            pattern=r"(?:turn\s+)?(?:the\s+)?bedroom\s*lights?\s+(?:on|off)",
            description="Bedroom light on/off command",
            n8n_action="bedroomlight",
        )

        # === KITCHEN LIGHT COMMANDS ===
        # Common ASR errors: "kitchen" -> "chicken", "kicking", "kit chin", "kitchin", "kitten"
        # "light" -> "lite", "lights", "life", "like", "lied"
        # "off" -> "of", "a", "ah" | "on" -> "in", "gone", "own", "and"
        # VoiceMonkey is a TOGGLE - both on/off commands trigger the same device

        # "kitchen light off" and variants - all go to toggle
        self.register(
            command_id="kitchen_light_off",
            pattern=r"\b(?:kitchen|chicken|kicking|kit\s*chin|kitchin|kitten)\s*(?:light|lite|lights|life|like|lied)s?\s*(?:off|of|a|ah)\b",
            description="Turn off the kitchen light (toggle)",
            n8n_action="kitchenlights",  # Toggle device
        )

        # "kitchen light on" and variants - all go to toggle
        self.register(
            command_id="kitchen_light_on",
            pattern=r"\b(?:kitchen|chicken|kicking|kit\s*chin|kitchin|kitten)\s*(?:light|lite|lights|life|like|lied)s?\s*(?:on|in|own|and|gone)\b",
            description="Turn on the kitchen light (toggle)",
            n8n_action="kitchenlights",  # Toggle device
        )

        # "turn off/on the kitchen light" alternate phrasing
        self.register(
            command_id="turn_kitchen_light",
            pattern=r"\bturn\s*(?:off|on|of|in)\s*(?:the\s*)?(?:kitchen|chicken|kicking)\s*(?:light|lite|lights)s?\b",
            description="Turn off/on the kitchen light (alternate phrasing)",
            n8n_action="kitchenlights",  # Toggle device
        )

        # === LIVING ROOM LIGHT COMMANDS ===
        # Common ASR errors: "living" -> "leaving", "livid", "livin", "live in"
        # "room" -> "rim", "rum", "roam", "rome"
        # ALSO support "livingroom" as one word (no space)
        # VoiceMonkey is a TOGGLE - both on/off commands trigger the same device

        # "living room light off" and "livingroom light off" variants
        self.register(
            command_id="living_room_light_off",
            pattern=r"\b(?:living\s*room|livingroom|leaving\s*room|livid\s*room|livin\s*room|live\s*in\s*room|living\s*rim|living\s*rum|living\s*roam)\s*(?:light|lite|lights|life|like|lied)s?\s*(?:off|of|a|ah)\b",
            description="Turn off the living room light (toggle)",
            n8n_action="livingroomlight",  # Toggle device
        )

        # "living room light on" and "livingroom light on" variants
        self.register(
            command_id="living_room_light_on",
            pattern=r"\b(?:living\s*room|livingroom|leaving\s*room|livid\s*room|livin\s*room|live\s*in\s*room|living\s*rim|living\s*rum|living\s*roam)\s*(?:light|lite|lights|life|like|lied)s?\s*(?:on|in|own|and|gone)\b",
            description="Turn on the living room light (toggle)",
            n8n_action="livingroomlight",  # Toggle device
        )

        # "turn off/on the living room light" alternate phrasing
        self.register(
            command_id="turn_living_room_light",
            pattern=r"\bturn\s*(?:off|on|of|in)\s*(?:the\s*)?(?:living\s*room|livingroom|leaving\s*room|livid\s*room)\s*(?:light|lite|lights)s?\b",
            description="Turn off/on the living room light (alternate phrasing)",
            n8n_action="livingroomlight",  # Toggle device
        )

        # === GOOGLE AI COMMANDS ===
        # These are for logging/tracking - actual execution happens on mobile

        # Timer commands: "set timer for 5 minutes", "timer 30 seconds"
        self.register(
            command_id="google_ai_timer",
            pattern=r"(?:set\s+)?(?:a\s+)?timer\s+(?:for\s+)?\d+\s*(?:second|seconds|sec|minute|minutes|min|hour|hours|hr)",
            description="Set a countdown timer (Google AI)",
            n8n_action="google_ai_timer",
        )

        # Alarm commands: "set alarm for 7 AM", "wake me up at 6:30"
        self.register(
            command_id="google_ai_alarm",
            pattern=r"(?:set\s+)?(?:an?\s+)?alarm\s+(?:for\s+)?\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?",
            description="Set an alarm (Google AI)",
            n8n_action="google_ai_alarm",
        )

        # Wake up commands: "wake me up at 7"
        self.register(
            command_id="google_ai_wake_up",
            pattern=r"wake\s+(?:me\s+)?up\s+(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?",
            description="Wake up alarm (Google AI)",
            n8n_action="google_ai_alarm",
        )

    def register(
        self, command_id: str, pattern: str, description: str, n8n_action: str, enabled: bool = True
    ) -> VoiceCommand:
        """
        Register a new voice command.

        Args:
            command_id: Unique identifier for this command
            pattern: Regex pattern to match against transcribed text
            description: Human-readable description
            n8n_action: Action identifier sent to n8n webhook
            enabled: Whether this command is active

        Returns:
            The registered VoiceCommand
        """
        if command_id in self._commands:
            logger.warning(f"Overwriting existing command: {command_id}")

        command = VoiceCommand(
            command_id=command_id, pattern=pattern, description=description, n8n_action=n8n_action, enabled=enabled
        )
        self._commands[command_id] = command
        logger.info(f"Registered voice command: {command_id}")
        return command

    def unregister(self, command_id: str) -> bool:
        """Remove a command from the registry"""
        if command_id in self._commands:
            del self._commands[command_id]
            logger.info(f"Unregistered voice command: {command_id}")
            return True
        return False

    def match(self, text: str) -> VoiceCommand | None:
        """
        Match text against all registered commands.
        Returns the first matching command or None.
        """
        if not text or not text.strip():
            return None

        for command in self._commands.values():
            if command.enabled and command.matches(text):
                logger.info(f"Voice command matched: {command.command_id} for text: '{text[:50]}...'")
                return command
        return None

    def match_all(self, text: str) -> list[VoiceCommand]:
        """Match text against all commands, returning all matches"""
        if not text or not text.strip():
            return []

        matches = []
        for command in self._commands.values():
            if command.enabled and command.matches(text):
                matches.append(command)
        return matches

    def get(self, command_id: str) -> VoiceCommand | None:
        """Get a command by ID"""
        return self._commands.get(command_id)

    def list_all(self) -> list[VoiceCommand]:
        """List all registered commands"""
        return list(self._commands.values())

    def set_enabled(self, command_id: str, enabled: bool) -> bool:
        """Enable or disable a command"""
        if command_id in self._commands:
            self._commands[command_id].enabled = enabled
            return True
        return False


# Global registry instance
_registry: CommandRegistry | None = None


def get_command_registry() -> CommandRegistry:
    """Get the global command registry singleton"""
    global _registry
    if _registry is None:
        _registry = CommandRegistry()
    return _registry
