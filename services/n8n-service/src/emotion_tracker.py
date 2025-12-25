"""
Emotion Alert Tracker

Tracks consecutive emotions per speaker and fires alerts when threshold is met.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import EMOTION_ALERT_SPEAKERS, EMOTION_ALERT_THRESHOLD, EMOTION_ALERT_TYPE

logger = logging.getLogger(__name__)


@dataclass
class EmotionAlert:
    """Represents an emotion alert that should be fired"""

    speaker: str
    emotion: str
    consecutive_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "emotion": self.emotion,
            "consecutive_count": self.consecutive_count,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SpeakerEmotionState:
    """Tracks emotion state for a single speaker"""

    speaker: str
    emotions: deque = field(default_factory=lambda: deque(maxlen=100))
    consecutive_target_count: int = 0
    last_alert_at: datetime | None = None
    alert_fired: bool = False  # Prevents duplicate alerts until reset

    def add_emotion(self, emotion: str, target_emotion: str, threshold: int) -> EmotionAlert | None:
        """
        Add an emotion and check if alert should fire.

        Returns EmotionAlert if threshold is met, None otherwise.
        """
        self.emotions.append(emotion)

        # Check if this is the target emotion
        if emotion.lower() == target_emotion.lower():
            self.consecutive_target_count += 1
        else:
            # Reset consecutive count on non-target emotion
            self.consecutive_target_count = 0
            self.alert_fired = False  # Allow new alert after reset

        # Check threshold
        if self.consecutive_target_count >= threshold and not self.alert_fired:
            self.alert_fired = True
            self.last_alert_at = datetime.utcnow()
            logger.warning(
                f"EMOTION ALERT: {self.speaker} has {self.consecutive_target_count} "
                f"consecutive '{target_emotion}' emotions"
            )
            return EmotionAlert(
                speaker=self.speaker, emotion=target_emotion, consecutive_count=self.consecutive_target_count
            )

        return None

    def get_status(self) -> dict[str, Any]:
        """Get current tracking status"""
        return {
            "speaker": self.speaker,
            "current_consecutive_count": self.consecutive_target_count,
            "last_emotions": list(self.emotions)[-10:],  # Last 10
            "alert_fired": self.alert_fired,
            "last_alert_at": self.last_alert_at.isoformat() if self.last_alert_at else None,
        }


class EmotionAlertTracker:
    """
    Tracks emotions per speaker and fires alerts at threshold.

    Usage:
        tracker = EmotionAlertTracker(threshold=20, target_emotion="anger")

        alert = tracker.add_emotion("pruitt", "anger")
        if alert:
            print(f"Alert fired for {alert.speaker}!")
    """

    def __init__(
        self,
        threshold: int = EMOTION_ALERT_THRESHOLD,
        target_emotion: str = EMOTION_ALERT_TYPE,
        tracked_speakers: list[str] | None = None,
    ):
        self.threshold = threshold
        self.target_emotion = target_emotion.lower()
        self.tracked_speakers = [s.lower() for s in (tracked_speakers or EMOTION_ALERT_SPEAKERS)]
        self._speaker_states: dict[str, SpeakerEmotionState] = {}
        self._alert_history: list[EmotionAlert] = []

        logger.info(
            f"EmotionAlertTracker initialized: threshold={threshold}, "
            f"target={target_emotion}, speakers={self.tracked_speakers}"
        )

    def add_emotion(self, speaker: str, emotion: str) -> EmotionAlert | None:
        """
        Track an emotion for a speaker.

        Args:
            speaker: Speaker identifier (e.g., "pruitt", "ericah")
            emotion: Detected emotion (e.g., "anger", "joy", "neutral")

        Returns:
            EmotionAlert if threshold is reached, None otherwise
        """
        speaker_lower = speaker.lower()

        # Only track configured speakers
        if speaker_lower not in self.tracked_speakers:
            return None

        # Get or create speaker state
        if speaker_lower not in self._speaker_states:
            self._speaker_states[speaker_lower] = SpeakerEmotionState(speaker=speaker_lower)

        state = self._speaker_states[speaker_lower]
        alert = state.add_emotion(emotion, self.target_emotion, self.threshold)

        if alert:
            self._alert_history.append(alert)

        return alert

    def get_speaker_status(self, speaker: str) -> dict[str, Any] | None:
        """Get tracking status for a specific speaker"""
        speaker_lower = speaker.lower()
        if speaker_lower in self._speaker_states:
            return self._speaker_states[speaker_lower].get_status()
        return None

    def get_all_status(self) -> dict[str, Any]:
        """Get tracking status for all speakers"""
        return {
            "threshold": self.threshold,
            "target_emotion": self.target_emotion,
            "tracked_speakers": self.tracked_speakers,
            "speaker_states": {speaker: state.get_status() for speaker, state in self._speaker_states.items()},
            "total_alerts_fired": len(self._alert_history),
        }

    def get_alert_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent alert history"""
        return [a.to_dict() for a in self._alert_history[-limit:]]

    def reset_speaker(self, speaker: str) -> bool:
        """Reset tracking state for a speaker"""
        speaker_lower = speaker.lower()
        if speaker_lower in self._speaker_states:
            self._speaker_states[speaker_lower] = SpeakerEmotionState(speaker=speaker_lower)
            logger.info(f"Reset emotion tracking for speaker: {speaker}")
            return True
        return False

    def reset_all(self):
        """Reset all tracking state"""
        self._speaker_states.clear()
        logger.info("Reset all emotion tracking state")


# Global tracker instance
_tracker: EmotionAlertTracker | None = None


def get_emotion_tracker() -> EmotionAlertTracker:
    """Get the global emotion tracker singleton"""
    global _tracker
    if _tracker is None:
        _tracker = EmotionAlertTracker()
    return _tracker
