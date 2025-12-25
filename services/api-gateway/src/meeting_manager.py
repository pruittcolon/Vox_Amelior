"""
Meeting Manager Module - Meeting Intelligence
Handles meeting summaries, action items, and search from transcripts.
"""

import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MeetingStatus(str, Enum):
    PENDING = "pending"
    SUMMARIZED = "summarized"
    ARCHIVED = "archived"


class ActionItemStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class MeetingManager:
    """
    Manages meeting intelligence features.

    Features:
    - Generate summaries from transcripts
    - Extract action items
    - Search meetings by content/participant
    - Track meeting metadata
    """

    def __init__(self, db_path: str = "/app/instance/meetings.db"):
        """Initialize meeting manager with SQLite database."""
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
        logger.info(f"[MeetingManager] Initialized with db: {db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                transcript_id TEXT,
                transcript_text TEXT,
                summary TEXT,
                participants TEXT,
                start_time TEXT,
                end_time TEXT,
                duration_minutes INTEGER,
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                description TEXT NOT NULL,
                assignee TEXT,
                due_date TEXT,
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'open',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_meetings_status ON meetings(status);
            CREATE INDEX IF NOT EXISTS idx_meetings_created ON meetings(created_at);
            CREATE INDEX IF NOT EXISTS idx_action_items_meeting ON action_items(meeting_id);
            CREATE INDEX IF NOT EXISTS idx_action_items_status ON action_items(status);
        """)
        conn.commit()
        logger.info("[MeetingManager] Database initialized")

    def create_meeting(
        self,
        title: str,
        transcript_id: str | None = None,
        transcript_text: str | None = None,
        participants: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new meeting record.

        Args:
            title: Meeting title
            transcript_id: Reference to transcript job
            transcript_text: Raw transcript text
            participants: List of participant names
            start_time: Meeting start time (ISO format)
            end_time: Meeting end time (ISO format)

        Returns:
            Created meeting dict with id
        """
        now = datetime.utcnow().isoformat() + "Z"
        duration = None
        if start_time and end_time:
            try:
                start = datetime.fromisoformat(start_time.replace("Z", ""))
                end = datetime.fromisoformat(end_time.replace("Z", ""))
                duration = int((end - start).total_seconds() / 60)
            except Exception:
                pass

        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO meetings 
            (title, transcript_id, transcript_text, participants, start_time, end_time, 
             duration_minutes, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                title,
                transcript_id,
                transcript_text,
                json.dumps(participants or []),
                start_time,
                end_time,
                duration,
                MeetingStatus.PENDING.value,
                now,
                now,
            ),
        )
        conn.commit()
        meeting_id = cursor.lastrowid

        logger.info(f"[MeetingManager] Created meeting {meeting_id}: {title}")
        return {"id": meeting_id, "title": title, "status": MeetingStatus.PENDING.value, "created_at": now}

    def summarize_meeting(
        self,
        meeting_id: int,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate or set a meeting summary.

        If no summary provided, auto-generates from transcript using simple extraction.

        Args:
            meeting_id: ID of meeting to summarize
            summary: Optional pre-generated summary

        Returns:
            Updated meeting with summary
        """
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
        row = cursor.fetchone()

        if not row:
            raise ValueError(f"Meeting {meeting_id} not found")

        meeting = dict(row)

        if not summary:
            # Auto-generate summary from transcript
            transcript = meeting.get("transcript_text") or ""
            summary = self._generate_summary(transcript)

        now = datetime.utcnow().isoformat() + "Z"
        conn.execute(
            """
            UPDATE meetings 
            SET summary = ?, status = ?, updated_at = ?
            WHERE id = ?
        """,
            (summary, MeetingStatus.SUMMARIZED.value, now, meeting_id),
        )
        conn.commit()

        # Auto-extract action items
        action_items = self._extract_action_items(meeting.get("transcript_text") or "")
        for item in action_items:
            self.add_action_item(meeting_id, item["description"], item.get("assignee"))

        logger.info(f"[MeetingManager] Summarized meeting {meeting_id}, extracted {len(action_items)} action items")

        return {
            "id": meeting_id,
            "title": meeting["title"],
            "summary": summary,
            "action_items_extracted": len(action_items),
            "status": MeetingStatus.SUMMARIZED.value,
        }

    def _generate_summary(self, transcript: str) -> str:
        """Generate a simple summary from transcript text."""
        if not transcript:
            return "No transcript available for summarization."

        # Simple extractive summary - take key sentences
        sentences = re.split(r"[.!?]+", transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return "Meeting transcript too short for automatic summary."

        # Take first 3 and last 2 meaningful sentences
        summary_parts = []
        if len(sentences) >= 5:
            summary_parts = sentences[:3] + sentences[-2:]
        else:
            summary_parts = sentences[:5]

        summary = ". ".join(summary_parts)
        if summary and not summary.endswith("."):
            summary += "."

        return f"Meeting Summary: {summary}"

    def _extract_action_items(self, transcript: str) -> list[dict[str, Any]]:
        """Extract action items from transcript using pattern matching."""
        action_items = []

        # Patterns that indicate action items
        patterns = [
            r"(?:I will|I'll|we will|we'll|should|need to|have to|must)\s+(.{10,80})",
            r"(?:action item|todo|task):\s*(.{10,80})",
            r"(?:please|kindly)\s+(.{10,80})",
            r"(?:follow up on|follow-up on)\s+(.{10,80})",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            for match in matches[:5]:  # Limit to 5 per pattern
                clean = match.strip()
                if clean and len(clean) > 10:
                    action_items.append({"description": clean[:200], "assignee": None})

        # Deduplicate
        seen = set()
        unique_items = []
        for item in action_items:
            key = item["description"].lower()[:50]
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        return unique_items[:10]  # Max 10 action items

    def add_action_item(
        self,
        meeting_id: int,
        description: str,
        assignee: str | None = None,
        due_date: str | None = None,
        priority: str = "medium",
    ) -> dict[str, Any]:
        """Add an action item to a meeting."""
        now = datetime.utcnow().isoformat() + "Z"
        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO action_items 
            (meeting_id, description, assignee, due_date, priority, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (meeting_id, description, assignee, due_date, priority, ActionItemStatus.OPEN.value, now, now),
        )
        conn.commit()

        return {
            "id": cursor.lastrowid,
            "meeting_id": meeting_id,
            "description": description,
            "assignee": assignee,
            "status": ActionItemStatus.OPEN.value,
        }

    def get_meeting(self, meeting_id: int) -> dict[str, Any] | None:
        """Get a meeting by ID with its action items."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
        row = cursor.fetchone()

        if not row:
            return None

        meeting = dict(row)
        meeting["participants"] = json.loads(meeting.get("participants") or "[]")

        # Get action items
        cursor = conn.execute("SELECT * FROM action_items WHERE meeting_id = ? ORDER BY created_at", (meeting_id,))
        meeting["action_items"] = [dict(r) for r in cursor.fetchall()]

        return meeting

    def get_action_items(self, meeting_id: int) -> list[dict[str, Any]]:
        """Get all action items for a meeting."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM action_items WHERE meeting_id = ? ORDER BY created_at", (meeting_id,))
        return [dict(r) for r in cursor.fetchall()]

    def list_meetings(
        self, days: int = 30, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """
        List meetings with optional filters.

        Args:
            days: Number of days to look back
            status: Filter by status
            limit: Max results
            offset: Pagination offset

        Returns:
            Dict with meetings list and count
        """
        conn = self._get_conn()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

        query = "SELECT * FROM meetings WHERE created_at >= ?"
        params = [cutoff]

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.execute(query, params)
        meetings = []
        for row in cursor.fetchall():
            m = dict(row)
            m["participants"] = json.loads(m.get("participants") or "[]")
            meetings.append(m)

        # Get total count
        count_query = "SELECT COUNT(*) FROM meetings WHERE created_at >= ?"
        count_params = [cutoff]
        if status:
            count_query += " AND status = ?"
            count_params.append(status)

        total = conn.execute(count_query, count_params).fetchone()[0]

        return {"meetings": meetings, "total": total, "limit": limit, "offset": offset}

    def search_meetings(self, query: str, limit: int = 20) -> dict[str, Any]:
        """
        Search meetings by title, summary, or transcript content.

        Args:
            query: Search query
            limit: Max results

        Returns:
            Dict with matching meetings
        """
        conn = self._get_conn()
        search_term = f"%{query}%"

        cursor = conn.execute(
            """
            SELECT * FROM meetings 
            WHERE title LIKE ? OR summary LIKE ? OR transcript_text LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (search_term, search_term, search_term, limit),
        )

        meetings = []
        for row in cursor.fetchall():
            m = dict(row)
            m["participants"] = json.loads(m.get("participants") or "[]")
            meetings.append(m)

        return {"query": query, "results": meetings, "count": len(meetings)}

    def update_action_item_status(self, action_id: int, status: str) -> dict[str, Any]:
        """Update action item status."""
        now = datetime.utcnow().isoformat() + "Z"
        conn = self._get_conn()
        conn.execute(
            """
            UPDATE action_items SET status = ?, updated_at = ? WHERE id = ?
        """,
            (status, now, action_id),
        )
        conn.commit()

        return {"id": action_id, "status": status, "updated_at": now}

    def get_stats(self) -> dict[str, Any]:
        """Get meeting intelligence statistics."""
        conn = self._get_conn()

        total_meetings = conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0]
        summarized = conn.execute(
            "SELECT COUNT(*) FROM meetings WHERE status = ?", (MeetingStatus.SUMMARIZED.value,)
        ).fetchone()[0]

        total_actions = conn.execute("SELECT COUNT(*) FROM action_items").fetchone()[0]
        open_actions = conn.execute(
            "SELECT COUNT(*) FROM action_items WHERE status = ?", (ActionItemStatus.OPEN.value,)
        ).fetchone()[0]
        completed_actions = conn.execute(
            "SELECT COUNT(*) FROM action_items WHERE status = ?", (ActionItemStatus.COMPLETED.value,)
        ).fetchone()[0]

        return {
            "total_meetings": total_meetings,
            "summarized_meetings": summarized,
            "pending_meetings": total_meetings - summarized,
            "total_action_items": total_actions,
            "open_action_items": open_actions,
            "completed_action_items": completed_actions,
            "completion_rate": round(completed_actions / max(total_actions, 1) * 100, 1),
        }


# Singleton instance
_meeting_manager: MeetingManager | None = None


def get_meeting_manager(db_path: str | None = None) -> MeetingManager:
    """Get or create the singleton MeetingManager instance."""
    global _meeting_manager
    if _meeting_manager is None:
        _meeting_manager = MeetingManager(db_path or "/app/instance/meetings.db")
    return _meeting_manager
