"""
Speaker Training Data Collector

Automatically saves audio segments from identified speakers (pruitt, ericah)
for future model training and fine-tuning.

Storage structure:
    /gateway_instance/training_data/
    ├── pruitt/
    │   ├── segments/
    │   │   ├── 2025-12-12_11-30-45_5.2s.wav
    │   │   └── 2025-12-12_11-32-10_3.1s.wav
    │   └── metadata.json
    └── ericah/
        ├── segments/
        └── metadata.json
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Speakers to collect training data for
ALLOWED_SPEAKERS = {"pruitt", "ericah"}

# Collection settings
MIN_CONFIDENCE = 0.7  # Minimum speaker identification confidence
MIN_DURATION_SEC = 1.0  # Minimum segment duration
MAX_DURATION_SEC = 30.0  # Maximum segment duration (will split longer)
SAMPLE_RATE = 16000  # Standard sample rate for training


class SpeakerTrainingCollector:
    """
    Collects audio segments from identified speakers for training.

    Only saves high-confidence segments from allowed speakers
    to build up training data over time.
    """

    def __init__(self, base_dir: str = "/gateway_instance/training_data"):
        """
        Initialize the training collector.

        Args:
            base_dir: Base directory for storing training data
        """
        self.base_dir = Path(base_dir)
        self._recent_hashes: dict[str, list[str]] = {}  # speaker -> list of audio hashes
        self._max_cached_hashes = 100  # Per speaker

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[TRAINING] SpeakerTrainingCollector initialized at {self.base_dir}")

    def _get_speaker_dir(self, speaker: str) -> Path:
        """Get or create the directory for a speaker's segments."""
        speaker_dir = self.base_dir / speaker.lower() / "segments"
        speaker_dir.mkdir(parents=True, exist_ok=True)
        return speaker_dir

    def _get_metadata_path(self, speaker: str) -> Path:
        """Get the metadata JSON path for a speaker."""
        return self.base_dir / speaker.lower() / "metadata.json"

    def _load_metadata(self, speaker: str) -> dict[str, Any]:
        """Load metadata for a speaker."""
        meta_path = self._get_metadata_path(speaker)
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[TRAINING] Failed to load metadata for {speaker}: {e}")

        return {
            "speaker": speaker.lower(),
            "total_segments": 0,
            "total_duration_sec": 0.0,
            "avg_confidence": 0.0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "segments": [],
        }

    def _save_metadata(self, speaker: str, metadata: dict[str, Any]) -> None:
        """Save metadata for a speaker."""
        meta_path = self._get_metadata_path(speaker)
        metadata["updated_at"] = datetime.now().isoformat()

        try:
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"[TRAINING] Failed to save metadata for {speaker}: {e}")

    def _compute_audio_hash(self, audio_data: np.ndarray) -> str:
        """Compute a hash of audio data for deduplication."""
        # Use a subset of the audio for faster hashing
        sample = audio_data[::100] if len(audio_data) > 1000 else audio_data
        return hashlib.md5(sample.tobytes()).hexdigest()[:16]

    def _is_duplicate(self, speaker: str, audio_hash: str) -> bool:
        """Check if we've recently saved similar audio."""
        if speaker not in self._recent_hashes:
            self._recent_hashes[speaker] = []
        return audio_hash in self._recent_hashes[speaker]

    def _add_hash(self, speaker: str, audio_hash: str) -> None:
        """Add a hash to the recent cache."""
        if speaker not in self._recent_hashes:
            self._recent_hashes[speaker] = []

        self._recent_hashes[speaker].append(audio_hash)

        # Trim cache if too large
        if len(self._recent_hashes[speaker]) > self._max_cached_hashes:
            self._recent_hashes[speaker] = self._recent_hashes[speaker][-self._max_cached_hashes :]

    async def save_segment(
        self,
        speaker: str,
        audio_data: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        confidence: float = 0.0,
        text: str = "",
        session_id: str = "",
    ) -> str | None:
        """
        Save audio segment for identified speaker.

        Args:
            speaker: Speaker name (must be in ALLOWED_SPEAKERS)
            audio_data: Audio waveform (numpy array)
            sample_rate: Audio sample rate
            confidence: Speaker identification confidence (0-1)
            text: Transcribed text for this segment
            session_id: Session identifier

        Returns:
            Path to saved file, or None if not saved
        """
        speaker_lower = speaker.lower()

        # Validate speaker
        if speaker_lower not in ALLOWED_SPEAKERS:
            logger.debug(f"[TRAINING] Skipping segment - speaker '{speaker}' not in allowed list")
            return None

        # Validate confidence
        if confidence < MIN_CONFIDENCE:
            logger.debug(f"[TRAINING] Skipping segment - confidence {confidence:.2f} < {MIN_CONFIDENCE}")
            return None

        # Calculate duration
        duration = len(audio_data) / sample_rate

        # Validate duration
        if duration < MIN_DURATION_SEC:
            logger.debug(f"[TRAINING] Skipping segment - duration {duration:.2f}s < {MIN_DURATION_SEC}s")
            return None

        # Check for duplicates
        audio_hash = self._compute_audio_hash(audio_data)
        if self._is_duplicate(speaker_lower, audio_hash):
            logger.debug(f"[TRAINING] Skipping duplicate segment for {speaker}")
            return None

        # Split if too long
        if duration > MAX_DURATION_SEC:
            logger.info(f"[TRAINING] Splitting long segment ({duration:.1f}s) for {speaker}")
            # Save first MAX_DURATION_SEC
            max_samples = int(MAX_DURATION_SEC * sample_rate)
            audio_data = audio_data[:max_samples]
            duration = MAX_DURATION_SEC

        # Ensure correct sample rate
        if sample_rate != SAMPLE_RATE:
            # Resample would go here, but for simplicity we'll accept as-is
            logger.debug(f"[TRAINING] Audio sample rate {sample_rate} != target {SAMPLE_RATE}")

        # Generate filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{duration:.1f}s_{confidence:.2f}conf.wav"

        # Save audio file
        speaker_dir = self._get_speaker_dir(speaker_lower)
        filepath = speaker_dir / filename

        try:
            # Run file I/O in thread pool to not block
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: sf.write(str(filepath), audio_data, sample_rate))

            logger.info(
                f"[TRAINING] ✅ Saved segment for {speaker}: {filename} ({duration:.1f}s, conf={confidence:.2f})"
            )

            # Update metadata
            metadata = self._load_metadata(speaker_lower)
            metadata["total_segments"] += 1
            metadata["total_duration_sec"] += duration

            # Update rolling average confidence
            n = metadata["total_segments"]
            old_avg = metadata["avg_confidence"]
            metadata["avg_confidence"] = ((old_avg * (n - 1)) + confidence) / n

            # Add segment entry
            metadata["segments"].append(
                {
                    "filename": filename,
                    "duration_sec": duration,
                    "confidence": confidence,
                    "text": text[:200] if text else "",  # Truncate long text
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                }
            )

            # Keep only last 1000 segment entries in metadata
            if len(metadata["segments"]) > 1000:
                metadata["segments"] = metadata["segments"][-1000:]

            self._save_metadata(speaker_lower, metadata)

            # Save transcription text to readable log file
            self._save_transcription_to_log(speaker_lower, text, confidence, duration)

            # Add to hash cache
            self._add_hash(speaker_lower, audio_hash)

            return str(filepath)

        except Exception as e:
            logger.error(f"[TRAINING] ❌ Failed to save segment for {speaker}: {e}")
            return None

    def _save_transcription_to_log(
        self,
        speaker: str,
        text: str,
        confidence: float,
        duration: float,
    ) -> None:
        """
        Save transcription text to a readable log file.

        Creates a file at /gateway_instance/training_data/{speaker}/transcriptions.txt
        that contains all transcriptions in human-readable format.
        """
        if not text or not text.strip():
            return

        try:
            speaker_dir = self.base_dir / speaker.lower()
            speaker_dir.mkdir(parents=True, exist_ok=True)

            log_path = speaker_dir / "transcriptions.txt"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            entry = f"[{timestamp}] (conf={confidence:.2f}, dur={duration:.1f}s)\n{text.strip()}\n---\n"

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(entry)

            logger.debug(f"[TRAINING] Logged transcription for {speaker}")

        except Exception as e:
            logger.error(f"[TRAINING] Failed to log transcription for {speaker}: {e}")

    def get_stats(self, speaker: str = None) -> dict[str, Any]:
        """
        Get collection statistics for a speaker or all speakers.

        Args:
            speaker: Optional speaker name. If None, returns stats for all.

        Returns:
            Dictionary with collection statistics
        """
        if speaker:
            metadata = self._load_metadata(speaker.lower())
            return {
                "speaker": speaker.lower(),
                "total_segments": metadata["total_segments"],
                "total_duration_sec": metadata["total_duration_sec"],
                "avg_confidence": metadata["avg_confidence"],
                "updated_at": metadata["updated_at"],
            }

        # All speakers
        stats = {}
        for sp in ALLOWED_SPEAKERS:
            metadata = self._load_metadata(sp)
            stats[sp] = {
                "total_segments": metadata["total_segments"],
                "total_duration_sec": metadata["total_duration_sec"],
                "avg_confidence": metadata["avg_confidence"],
            }

        return {"speakers": stats, "base_dir": str(self.base_dir)}


# Singleton instance
_collector: SpeakerTrainingCollector | None = None


def get_training_collector() -> SpeakerTrainingCollector:
    """Get the singleton training collector instance."""
    global _collector
    if _collector is None:
        _collector = SpeakerTrainingCollector()
    return _collector


def init_training_collector(base_dir: str = "/gateway_instance/training_data") -> SpeakerTrainingCollector:
    """Initialize the training collector with a specific base directory."""
    global _collector
    _collector = SpeakerTrainingCollector(base_dir=base_dir)
    return _collector
