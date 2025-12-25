"""
Fiserv API Call Tracker

Tracks API calls to stay within the 1000 call sandbox limit.
Persists count to file so it survives restarts.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# Persistent storage path (inside container or local)
TRACKER_FILE = os.getenv("FISERV_TRACKER_FILE", "/tmp/fiserv_api_calls.json")
API_CALL_LIMIT = 1000


@dataclass
class APICallStats:
    """API call statistics."""

    total_calls: int = 0
    calls_remaining: int = API_CALL_LIMIT
    token_refreshes: int = 0
    party_calls: int = 0
    account_calls: int = 0
    transaction_calls: int = 0
    other_calls: int = 0
    last_call_at: str = ""
    first_call_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class APICallTracker:
    """
    Tracks Fiserv API calls.

    IMPORTANT: Sandbox limit is 1000 calls total.
    """

    def __init__(self, tracker_file: str = TRACKER_FILE):
        self.tracker_file = Path(tracker_file)
        self._lock = Lock()
        self._stats = self._load_stats()

    def _load_stats(self) -> APICallStats:
        """Load stats from file or create new."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file) as f:
                    data = json.load(f)
                    return APICallStats(**data)
            except Exception as e:
                logger.warning(f"Failed to load tracker file: {e}")
        return APICallStats()

    def _save_stats(self) -> None:
        """Persist stats to file."""
        try:
            self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tracker_file, "w") as f:
                json.dump(self._stats.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tracker: {e}")

    def record_call(self, call_type: str = "other") -> bool:
        """
        Record an API call.

        Args:
            call_type: Type of call (token, party, account, transaction, other)

        Returns:
            True if call was allowed, False if limit exceeded
        """
        with self._lock:
            if self._stats.total_calls >= API_CALL_LIMIT:
                logger.error(f"API LIMIT REACHED! {self._stats.total_calls}/{API_CALL_LIMIT}")
                return False

            now = datetime.now().isoformat()

            self._stats.total_calls += 1
            self._stats.calls_remaining = API_CALL_LIMIT - self._stats.total_calls
            self._stats.last_call_at = now

            if not self._stats.first_call_at:
                self._stats.first_call_at = now

            # Track by type
            if call_type == "token":
                self._stats.token_refreshes += 1
            elif call_type == "party":
                self._stats.party_calls += 1
            elif call_type == "account":
                self._stats.account_calls += 1
            elif call_type == "transaction":
                self._stats.transaction_calls += 1
            else:
                self._stats.other_calls += 1

            self._save_stats()

            # Log warning at thresholds
            remaining = self._stats.calls_remaining
            if remaining <= 100:
                logger.warning(f"API CALLS LOW: {remaining} remaining!")
            elif remaining <= 500:
                logger.info(f"API calls: {self._stats.total_calls}/{API_CALL_LIMIT} ({remaining} remaining)")

            return True

    def get_stats(self) -> dict[str, Any]:
        """Get current stats."""
        with self._lock:
            return {
                **self._stats.to_dict(),
                "limit": API_CALL_LIMIT,
                "usage_percent": round(self._stats.total_calls / API_CALL_LIMIT * 100, 1),
            }

    def can_make_call(self) -> bool:
        """Check if we can make another API call."""
        return self._stats.total_calls < API_CALL_LIMIT

    def reset(self) -> None:
        """Reset the tracker (use carefully!)."""
        with self._lock:
            self._stats = APICallStats()
            self._save_stats()
            logger.info("API call tracker reset!")


# Singleton
_tracker: APICallTracker = None


def get_tracker() -> APICallTracker:
    """Get tracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = APICallTracker()
    return _tracker
