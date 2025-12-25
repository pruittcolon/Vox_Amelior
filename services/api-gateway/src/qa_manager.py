"""
QA Manager Module - Enterprise AI Assistant
Handles feedback storage, review queue, and golden answer caching.
"""

import hashlib
import logging
import sqlite3
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackRating(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class QAManager:
    """
    Manages Q&A feedback, review queue, and golden answer cache.

    Features:
    - Submit user feedback (ratings + corrections)
    - Admin review queue for flagged Q&A pairs
    - Golden Answer cache for verified responses
    - Query similarity matching for cache hits
    """

    def __init__(self, db_path: str = "/app/instance/qa_store.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
        logger.info(f"QAManager initialized with db: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript("""
            -- Feedback table: stores all user feedback on Q&A interactions
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT NOT NULL,
                query_text TEXT NOT NULL,
                ai_answer TEXT NOT NULL,
                rating TEXT NOT NULL,
                correction TEXT,
                user_id TEXT,
                session_id TEXT,
                review_status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT
            );
            
            -- Golden answers table: verified Q&A pairs that override AI
            CREATE TABLE IF NOT EXISTS golden_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE NOT NULL,
                query_text TEXT NOT NULL,
                golden_answer TEXT NOT NULL,
                approved_by TEXT,
                source_feedback_id INTEGER,
                confidence REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                FOREIGN KEY (source_feedback_id) REFERENCES feedback(id)
            );
            
            -- Indexes for fast lookups
            CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(review_status);
            CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
            CREATE INDEX IF NOT EXISTS idx_feedback_query_hash ON feedback(query_hash);
            CREATE INDEX IF NOT EXISTS idx_golden_query_hash ON golden_answers(query_hash);
        """)
        conn.commit()

    @staticmethod
    def _hash_query(query: str) -> str:
        """Generate consistent hash for query matching."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]

    # =========================================================================
    # Feedback Operations
    # =========================================================================

    def submit_feedback(
        self,
        query: str,
        ai_answer: str,
        rating: str,
        correction: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Submit user feedback for a Q&A interaction.

        Args:
            query: The original user question
            ai_answer: The AI's response
            rating: 'positive' or 'negative'
            correction: Optional user-provided correct answer
            user_id: ID of the user providing feedback
            session_id: Session ID for tracking

        Returns:
            Dict with feedback_id and status
        """
        conn = self._get_conn()
        query_hash = self._hash_query(query)
        now = datetime.utcnow().isoformat() + "Z"

        # Auto-flag negative feedback for review
        review_status = (
            ReviewStatus.PENDING.value if rating == FeedbackRating.NEGATIVE.value else ReviewStatus.APPROVED.value
        )

        cursor = conn.execute(
            """
            INSERT INTO feedback (query_hash, query_text, ai_answer, rating, correction, user_id, session_id, review_status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (query_hash, query, ai_answer, rating, correction, user_id, session_id, review_status, now),
        )
        conn.commit()
        feedback_id = cursor.lastrowid

        logger.info(f"Feedback submitted: id={feedback_id}, rating={rating}, review_status={review_status}")

        return {
            "success": True,
            "feedback_id": feedback_id,
            "review_status": review_status,
            "message": "Feedback recorded"
            + (" and flagged for review" if review_status == ReviewStatus.PENDING.value else ""),
        }

    def get_review_queue(
        self,
        status: str = "pending",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get feedback items pending review.

        Args:
            status: Filter by review status ('pending', 'approved', 'rejected', 'all')
            limit: Max items to return
            offset: Pagination offset

        Returns:
            Dict with items list and pagination info
        """
        conn = self._get_conn()

        if status == "all":
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        else:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE review_status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM feedback WHERE review_status = ?", (status,)).fetchone()[0]

        items = [dict(row) for row in rows]

        return {
            "success": True,
            "items": items,
            "count": len(items),
            "total": total,
            "has_more": offset + len(items) < total,
            "next_offset": offset + len(items),
            "filter_status": status,
        }

    def get_feedback_by_id(self, feedback_id: int) -> dict[str, Any] | None:
        """Get a single feedback item by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,)).fetchone()
        return dict(row) if row else None

    # =========================================================================
    # Admin Review Operations
    # =========================================================================

    def approve_answer(
        self,
        feedback_id: int,
        golden_answer: str,
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        """
        Approve a feedback item and create/update a golden answer.

        Args:
            feedback_id: ID of the feedback to approve
            golden_answer: The verified correct answer
            approved_by: User ID of the approving admin

        Returns:
            Dict with status and golden_answer_id
        """
        conn = self._get_conn()

        # Get the feedback
        feedback = self.get_feedback_by_id(feedback_id)
        if not feedback:
            return {"success": False, "error": "Feedback not found"}

        query_hash = feedback["query_hash"]
        query_text = feedback["query_text"]
        now = datetime.utcnow().isoformat() + "Z"

        # Update feedback status
        conn.execute(
            "UPDATE feedback SET review_status = ?, updated_at = ? WHERE id = ?",
            (ReviewStatus.APPROVED.value, now, feedback_id),
        )

        # Upsert golden answer
        existing = conn.execute("SELECT id FROM golden_answers WHERE query_hash = ?", (query_hash,)).fetchone()

        if existing:
            conn.execute(
                """
                UPDATE golden_answers 
                SET golden_answer = ?, approved_by = ?, source_feedback_id = ?, updated_at = ?
                WHERE query_hash = ?
                """,
                (golden_answer, approved_by, feedback_id, now, query_hash),
            )
            golden_id = existing["id"]
            action = "updated"
        else:
            cursor = conn.execute(
                """
                INSERT INTO golden_answers (query_hash, query_text, golden_answer, approved_by, source_feedback_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (query_hash, query_text, golden_answer, approved_by, feedback_id, now),
            )
            golden_id = cursor.lastrowid
            action = "created"

        conn.commit()

        logger.info(f"Golden answer {action}: id={golden_id}, feedback_id={feedback_id}")

        return {
            "success": True,
            "golden_answer_id": golden_id,
            "action": action,
            "message": f"Answer approved and golden answer {action}",
        }

    def reject_feedback(self, feedback_id: int, rejected_by: str | None = None) -> dict[str, Any]:
        """Mark feedback as rejected (not suitable for golden answer)."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        result = conn.execute(
            "UPDATE feedback SET review_status = ?, updated_at = ? WHERE id = ?",
            (ReviewStatus.REJECTED.value, now, feedback_id),
        )
        conn.commit()

        if result.rowcount == 0:
            return {"success": False, "error": "Feedback not found"}

        return {"success": True, "message": "Feedback rejected"}

    # =========================================================================
    # Golden Answer Cache
    # =========================================================================

    def check_golden_answer(self, query: str) -> dict[str, Any] | None:
        """
        Check if there's a golden answer for the given query.

        Uses exact hash matching for now. Future: add vector similarity.

        Args:
            query: The user's question

        Returns:
            Golden answer dict if found, None otherwise
        """
        conn = self._get_conn()
        query_hash = self._hash_query(query)

        row = conn.execute("SELECT * FROM golden_answers WHERE query_hash = ?", (query_hash,)).fetchone()

        if row:
            logger.info(f"Golden answer cache HIT for query_hash={query_hash[:8]}...")
            return dict(row)

        return None

    def list_golden_answers(self, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """List all golden answers."""
        conn = self._get_conn()

        rows = conn.execute(
            "SELECT * FROM golden_answers ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM golden_answers").fetchone()[0]

        items = [dict(row) for row in rows]

        return {
            "success": True,
            "items": items,
            "count": len(items),
            "total": total,
        }

    def delete_golden_answer(self, golden_id: int) -> dict[str, Any]:
        """Delete a golden answer."""
        conn = self._get_conn()
        result = conn.execute("DELETE FROM golden_answers WHERE id = ?", (golden_id,))
        conn.commit()

        if result.rowcount == 0:
            return {"success": False, "error": "Golden answer not found"}

        return {"success": True, "message": "Golden answer deleted"}

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get Q&A system statistics."""
        conn = self._get_conn()

        total_feedback = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        positive = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating = 'positive'").fetchone()[0]
        negative = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating = 'negative'").fetchone()[0]
        pending_review = conn.execute("SELECT COUNT(*) FROM feedback WHERE review_status = 'pending'").fetchone()[0]
        golden_count = conn.execute("SELECT COUNT(*) FROM golden_answers").fetchone()[0]

        return {
            "total_feedback": total_feedback,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "pending_review": pending_review,
            "golden_answers": golden_count,
            "approval_rate": round(positive / total_feedback * 100, 1) if total_feedback > 0 else 0,
        }


# Singleton instance
_qa_manager: QAManager | None = None


def get_qa_manager(db_path: str | None = None) -> QAManager:
    """Get or create the singleton QAManager instance."""
    global _qa_manager
    if _qa_manager is None:
        _qa_manager = QAManager(db_path or "/app/instance/qa_store.db")
    return _qa_manager
