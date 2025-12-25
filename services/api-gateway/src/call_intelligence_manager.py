"""
Call Intelligence Manager Module - Service Credit Union

Enterprise Call Intelligence for member call transcription analysis,
AI-powered summarization, and common problems detection.

Features:
- Ingest call transcriptions from Parakeet service
- Generate AI summaries via Gemma
- Categorize common problems using ML patterns
- Track sentiment and emotion per call segment
- PII redaction for compliance
- Role-based analytics (MSR, Loan Officer, Executive)
"""

import json
import logging
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CallChannel(str, Enum):
    PHONE = "phone"
    CHAT = "chat"
    VIDEO = "video"


class CallDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class SummaryType(str, Enum):
    NARRATIVE = "narrative"
    BULLET = "bullet"
    EXECUTIVE = "executive"


class ProblemSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Pre-defined problem taxonomy for credit unions
PROBLEM_TAXONOMY = {
    "account_access": {
        "name": "Account Access",
        "subcategories": {
            "login_issues": {"name": "Login Issues", "severity": "medium"},
            "password_reset": {"name": "Password Reset", "severity": "low"},
            "mfa_problems": {"name": "MFA Problems", "severity": "medium"},
            "locked_account": {"name": "Locked Account", "severity": "high"},
        },
    },
    "transactions": {
        "name": "Transactions",
        "subcategories": {
            "failed_transfer": {"name": "Failed Transfer", "severity": "high"},
            "missing_deposit": {"name": "Missing Deposit", "severity": "high"},
            "dispute": {"name": "Transaction Dispute", "severity": "medium"},
            "pending_hold": {"name": "Pending/Hold", "severity": "medium"},
            "wire_transfer": {"name": "Wire Transfer", "severity": "medium"},
        },
    },
    "loans": {
        "name": "Loans",
        "subcategories": {
            "payment_question": {"name": "Payment Question", "severity": "low"},
            "rate_inquiry": {"name": "Rate Inquiry", "severity": "low"},
            "application_status": {"name": "Application Status", "severity": "medium"},
            "payoff_request": {"name": "Payoff Request", "severity": "medium"},
            "refinance": {"name": "Refinance Inquiry", "severity": "low"},
        },
    },
    "cards": {
        "name": "Cards",
        "subcategories": {
            "lost_stolen": {"name": "Lost/Stolen Card", "severity": "high"},
            "fraud_alert": {"name": "Fraud Alert", "severity": "high"},
            "limit_increase": {"name": "Limit Increase", "severity": "low"},
            "pin_reset": {"name": "PIN Reset", "severity": "medium"},
            "rewards": {"name": "Rewards Question", "severity": "low"},
        },
    },
    "digital_banking": {
        "name": "Digital Banking",
        "subcategories": {
            "app_issues": {"name": "Mobile App Issues", "severity": "medium"},
            "online_banking": {"name": "Online Banking", "severity": "medium"},
            "bill_pay": {"name": "Bill Pay", "severity": "medium"},
            "zelle": {"name": "Zelle Issues", "severity": "medium"},
        },
    },
    "member_services": {
        "name": "Member Services",
        "subcategories": {
            "address_change": {"name": "Address Change", "severity": "low"},
            "joint_account": {"name": "Joint Account", "severity": "low"},
            "beneficiaries": {"name": "Beneficiaries", "severity": "low"},
            "account_closure": {"name": "Account Closure", "severity": "medium"},
        },
    },
    "fees": {
        "name": "Fees",
        "subcategories": {
            "fee_dispute": {"name": "Fee Dispute", "severity": "medium"},
            "fee_waiver": {"name": "Fee Waiver Request", "severity": "medium"},
            "fee_explanation": {"name": "Fee Explanation", "severity": "low"},
        },
    },
    "fraud": {
        "name": "Fraud",
        "subcategories": {
            "unauthorized_activity": {"name": "Unauthorized Activity", "severity": "critical"},
            "identity_theft": {"name": "Identity Theft", "severity": "critical"},
            "scam_report": {"name": "Scam Report", "severity": "high"},
            "card_compromised": {"name": "Card Compromised", "severity": "critical"},
        },
    },
}

# PII patterns for redaction
PII_PATTERNS = [
    (r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", "[SSN_REDACTED]"),  # SSN
    (r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b", "[CARD_REDACTED]"),  # Credit card
    (r"\b\d{9,12}\b", "[ACCOUNT_REDACTED]"),  # Account numbers
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]"),  # Phone numbers
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),  # Email
    (r"\b\d{5}(?:-\d{4})?\b", "[ZIP_REDACTED]"),  # ZIP codes
]


class CallIntelligenceManager:
    """
    Manages call intelligence features for Service Credit Union.

    Features:
    - Ingest and store call transcriptions
    - Generate AI summaries
    - Detect and categorize common problems
    - Track sentiment and emotion
    - PII redaction for compliance
    - Role-based analytics
    """

    def __init__(self, db_path: str = "/app/instance/calls.db"):
        """Initialize call intelligence manager with SQLite database."""
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
        logger.info(f"[CallIntelligenceManager] Initialized with db: {db_path}")

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
            -- Call Records
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT UNIQUE NOT NULL,
                member_id TEXT,
                agent_id TEXT,
                channel TEXT DEFAULT 'phone',
                direction TEXT DEFAULT 'inbound',
                duration_seconds INTEGER,
                transcript_raw TEXT,
                transcript_redacted TEXT,
                summary TEXT,
                summary_type TEXT,
                intent_category TEXT,
                sentiment_score REAL,
                emotion_dominant TEXT,
                problem_categories TEXT,
                action_items TEXT,
                fiserv_context TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            
            -- Call Segments (speaker turns)
            CREATE TABLE IF NOT EXISTS call_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id INTEGER NOT NULL,
                speaker TEXT,
                start_time REAL,
                end_time REAL,
                text TEXT,
                sentiment_score REAL,
                emotion TEXT,
                keywords TEXT,
                FOREIGN KEY (call_id) REFERENCES calls(id)
            );
            
            -- Problem Categories (taxonomy)
            CREATE TABLE IF NOT EXISTS problem_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                parent_category TEXT,
                description TEXT,
                severity TEXT DEFAULT 'medium',
                resolution_template TEXT,
                keywords TEXT,
                created_at TEXT NOT NULL
            );
            
            -- Call Problems (many-to-many)
            CREATE TABLE IF NOT EXISTS call_problems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id INTEGER NOT NULL,
                category_id TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_notes TEXT,
                detected_at TEXT NOT NULL,
                FOREIGN KEY (call_id) REFERENCES calls(id)
            );
            
            -- Action Items
            CREATE TABLE IF NOT EXISTS call_action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id INTEGER NOT NULL,
                description TEXT NOT NULL,
                assignee TEXT,
                due_date TEXT,
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'open',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (call_id) REFERENCES calls(id)
            );
            
            -- Analytics Snapshots
            CREATE TABLE IF NOT EXISTS call_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_date TEXT NOT NULL,
                total_calls INTEGER,
                avg_duration_seconds REAL,
                avg_sentiment REAL,
                top_problems TEXT,
                role_breakdown TEXT,
                created_at TEXT NOT NULL
            );
            
            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_calls_member ON calls(member_id);
            CREATE INDEX IF NOT EXISTS idx_calls_created ON calls(created_at);
            CREATE INDEX IF NOT EXISTS idx_calls_intent ON calls(intent_category);
            CREATE INDEX IF NOT EXISTS idx_calls_call_id ON calls(call_id);
            CREATE INDEX IF NOT EXISTS idx_segments_call ON call_segments(call_id);
            CREATE INDEX IF NOT EXISTS idx_problems_call ON call_problems(call_id);
            CREATE INDEX IF NOT EXISTS idx_action_items_call ON call_action_items(call_id);
        """)
        conn.commit()

        # Seed problem categories
        self._seed_problem_categories()
        logger.info("[CallIntelligenceManager] Database initialized")

    def _seed_problem_categories(self):
        """Seed the problem categories from taxonomy."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        for cat_id, cat_data in PROBLEM_TAXONOMY.items():
            # Insert parent category
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO problem_categories 
                    (category_id, name, parent_category, severity, created_at)
                    VALUES (?, ?, NULL, 'medium', ?)
                """,
                    (cat_id, cat_data["name"], now),
                )
            except sqlite3.IntegrityError:
                pass

            # Insert subcategories
            for sub_id, sub_data in cat_data.get("subcategories", {}).items():
                full_id = f"{cat_id}.{sub_id}"
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO problem_categories 
                        (category_id, name, parent_category, severity, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (full_id, sub_data["name"], cat_id, sub_data["severity"], now),
                    )
                except sqlite3.IntegrityError:
                    pass

        conn.commit()

    # ================================================================
    # Call Ingestion
    # ================================================================

    def ingest_call(
        self,
        transcript: str,
        member_id: str | None = None,
        agent_id: str | None = None,
        duration_seconds: int | None = None,
        channel: str = "phone",
        direction: str = "inbound",
        segments: list[dict[str, Any]] | None = None,
        fiserv_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a new call transcription.

        Args:
            transcript: Full transcript text
            member_id: Optional Fiserv member ID
            agent_id: Optional agent/MSR ID
            duration_seconds: Call duration
            channel: phone, chat, or video
            direction: inbound or outbound
            segments: Optional list of speaker segments
            fiserv_context: Optional member context from Fiserv

        Returns:
            Created call record with ID
        """
        call_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"

        # Redact PII
        transcript_redacted = self.redact_pii(transcript)

        # Basic sentiment analysis (placeholder - can be enhanced with ML)
        sentiment_score = self._basic_sentiment_analysis(transcript)

        # Detect dominant emotion (placeholder)
        emotion_dominant = self._detect_dominant_emotion(transcript)

        # Auto-detect problems
        detected_problems = self._detect_problems(transcript)

        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO calls 
            (call_id, member_id, agent_id, channel, direction, duration_seconds,
             transcript_raw, transcript_redacted, sentiment_score, emotion_dominant,
             problem_categories, fiserv_context, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                call_id,
                member_id,
                agent_id,
                channel,
                direction,
                duration_seconds,
                transcript,
                transcript_redacted,
                sentiment_score,
                emotion_dominant,
                json.dumps(detected_problems),
                json.dumps(fiserv_context) if fiserv_context else None,
                now,
                now,
            ),
        )
        db_id = cursor.lastrowid

        # Store segments if provided
        if segments:
            for seg in segments:
                conn.execute(
                    """
                    INSERT INTO call_segments 
                    (call_id, speaker, start_time, end_time, text, sentiment_score, emotion, keywords)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        db_id,
                        seg.get("speaker"),
                        seg.get("start_time"),
                        seg.get("end_time"),
                        seg.get("text"),
                        seg.get("sentiment_score"),
                        seg.get("emotion"),
                        json.dumps(seg.get("keywords", [])),
                    ),
                )

        # Record detected problems
        for problem in detected_problems:
            conn.execute(
                """
                INSERT INTO call_problems (call_id, category_id, confidence, detected_at)
                VALUES (?, ?, ?, ?)
            """,
                (db_id, problem["category_id"], problem.get("confidence", 0.8), now),
            )

        conn.commit()

        logger.info(f"[CallIntelligenceManager] Ingested call {call_id} for member {member_id}")

        return {
            "id": db_id,
            "call_id": call_id,
            "member_id": member_id,
            "duration_seconds": duration_seconds,
            "sentiment_score": sentiment_score,
            "problems_detected": len(detected_problems),
            "created_at": now,
        }

    def get_call(self, call_id: str) -> dict[str, Any] | None:
        """Get a call by its UUID with segments and problems."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,))
        row = cursor.fetchone()

        if not row:
            return None

        call = dict(row)
        call["problem_categories"] = json.loads(call.get("problem_categories") or "[]")
        call["action_items"] = json.loads(call.get("action_items") or "[]")
        call["fiserv_context"] = json.loads(call.get("fiserv_context") or "{}")

        # Get segments
        cursor = conn.execute("SELECT * FROM call_segments WHERE call_id = ? ORDER BY start_time", (call["id"],))
        call["segments"] = [dict(r) for r in cursor.fetchall()]

        # Get problems
        cursor = conn.execute(
            """
            SELECT cp.*, pc.name as category_name, pc.severity
            FROM call_problems cp
            JOIN problem_categories pc ON cp.category_id = pc.category_id
            WHERE cp.call_id = ?
        """,
            (call["id"],),
        )
        call["problems"] = [dict(r) for r in cursor.fetchall()]

        # Get action items
        cursor = conn.execute("SELECT * FROM call_action_items WHERE call_id = ? ORDER BY created_at", (call["id"],))
        call["action_item_records"] = [dict(r) for r in cursor.fetchall()]

        return call

    def list_calls(
        self,
        member_id: str | None = None,
        agent_id: str | None = None,
        days: int = 30,
        problem_category: str | None = None,
        min_sentiment: float | None = None,
        max_sentiment: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List calls with optional filters.

        Args:
            member_id: Filter by member
            agent_id: Filter by agent
            days: Number of days to look back
            problem_category: Filter by problem category
            min_sentiment: Minimum sentiment score
            max_sentiment: Maximum sentiment score
            limit: Max results
            offset: Pagination offset

        Returns:
            Dict with calls list and count
        """
        conn = self._get_conn()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

        query = "SELECT * FROM calls WHERE created_at >= ?"
        params: list[Any] = [cutoff]

        if member_id:
            query += " AND member_id = ?"
            params.append(member_id)

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)

        if min_sentiment is not None:
            query += " AND sentiment_score >= ?"
            params.append(min_sentiment)

        if max_sentiment is not None:
            query += " AND sentiment_score <= ?"
            params.append(max_sentiment)

        if problem_category:
            query += " AND problem_categories LIKE ?"
            params.append(f"%{problem_category}%")

        # Get total count first
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        total = conn.execute(count_query, params).fetchone()[0]

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.execute(query, params)
        calls = []
        for row in cursor.fetchall():
            call = dict(row)
            call["problem_categories"] = json.loads(call.get("problem_categories") or "[]")
            # Don't include raw transcript in list view
            call.pop("transcript_raw", None)
            calls.append(call)

        return {"calls": calls, "total": total, "limit": limit, "offset": offset}

    def search_calls(self, query: str, limit: int = 20) -> dict[str, Any]:
        """
        Full-text search in call transcripts and summaries.

        Args:
            query: Search query
            limit: Max results

        Returns:
            Dict with matching calls
        """
        conn = self._get_conn()
        search_term = f"%{query}%"

        cursor = conn.execute(
            """
            SELECT * FROM calls 
            WHERE transcript_redacted LIKE ? 
               OR summary LIKE ? 
               OR intent_category LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (search_term, search_term, search_term, limit),
        )

        calls = []
        for row in cursor.fetchall():
            call = dict(row)
            call["problem_categories"] = json.loads(call.get("problem_categories") or "[]")
            call.pop("transcript_raw", None)
            calls.append(call)

        return {"query": query, "results": calls, "count": len(calls)}

    # ================================================================
    # AI Summarization
    # ================================================================

    def set_summary(self, call_id: str, summary: str, summary_type: str = "narrative") -> dict[str, Any]:
        """
        Set or update the summary for a call.

        This is typically called after Gemma generates a summary.

        Args:
            call_id: Call UUID
            summary: Generated summary text
            summary_type: narrative, bullet, or executive

        Returns:
            Updated call info
        """
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        cursor = conn.execute("SELECT id FROM calls WHERE call_id = ?", (call_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Call {call_id} not found")

        conn.execute(
            """
            UPDATE calls 
            SET summary = ?, summary_type = ?, updated_at = ?
            WHERE call_id = ?
        """,
            (summary, summary_type, now, call_id),
        )
        conn.commit()

        logger.info(f"[CallIntelligenceManager] Set summary for call {call_id}")

        return {"call_id": call_id, "summary_type": summary_type, "summary_length": len(summary), "updated_at": now}

    def get_summary_prompt(self, call_id: str, summary_type: str = "narrative") -> dict[str, Any]:
        """
        Generate a prompt for Gemma to summarize a call.

        Args:
            call_id: Call UUID
            summary_type: narrative, bullet, or executive

        Returns:
            Dict with prompt and context
        """
        call = self.get_call(call_id)
        if not call:
            raise ValueError(f"Call {call_id} not found")

        transcript = call.get("transcript_redacted") or call.get("transcript_raw", "")
        member_id = call.get("member_id", "Unknown")
        duration = call.get("duration_seconds", 0)
        problems = call.get("problems", [])

        prompts = {
            "narrative": f"""You are a credit union call analyst for Service Credit Union. Summarize this member call concisely.

Call Transcript:
{transcript[:6000]}

Member ID: {member_id}
Call Duration: {duration} seconds
Detected Issues: {", ".join(p.get("category_name", "") for p in problems) or "None"}

Provide:
1. Brief summary (2-3 sentences)
2. Member's primary concern
3. Resolution provided (if any)
4. Follow-up needed (yes/no, and what)
""",
            "bullet": f"""Summarize this Service Credit Union call as bullet points.

Call Transcript:
{transcript[:6000]}

Format your response as:
• Primary Issue: [issue]
• Member Sentiment: [positive/neutral/negative]
• Resolution: [what was resolved]
• Action Items: [follow-ups needed]
• Risk Level: [low/medium/high]
""",
            "executive": f"""Provide an executive summary of this member interaction for Service Credit Union leadership.

Call Transcript:
{transcript[:6000]}

Member ID: {member_id}
Duration: {duration // 60} minutes

Focus on:
- Business impact (revenue risk, operational cost)
- Member retention risk assessment
- Upsell/cross-sell opportunity
- Compliance concerns (if any)
- Recommended next action
""",
        }

        return {
            "call_id": call_id,
            "summary_type": summary_type,
            "prompt": prompts.get(summary_type, prompts["narrative"]),
            "transcript_length": len(transcript),
            "member_id": member_id,
        }

    def extract_action_items(self, call_id: str, action_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Store extracted action items for a call.

        Args:
            call_id: Call UUID
            action_items: List of action item dicts with description, assignee, priority

        Returns:
            List of created action items
        """
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        cursor = conn.execute("SELECT id FROM calls WHERE call_id = ?", (call_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Call {call_id} not found")

        db_id = row[0]
        created = []

        for item in action_items:
            cursor = conn.execute(
                """
                INSERT INTO call_action_items 
                (call_id, description, assignee, due_date, priority, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'open', ?, ?)
            """,
                (
                    db_id,
                    item.get("description", ""),
                    item.get("assignee"),
                    item.get("due_date"),
                    item.get("priority", "medium"),
                    now,
                    now,
                ),
            )
            created.append(
                {
                    "id": cursor.lastrowid,
                    "description": item.get("description"),
                    "assignee": item.get("assignee"),
                    "status": "open",
                }
            )

        # Update call's action_items JSON
        conn.execute(
            """
            UPDATE calls SET action_items = ?, updated_at = ? WHERE id = ?
        """,
            (json.dumps([i.get("description") for i in action_items]), now, db_id),
        )

        conn.commit()

        logger.info(f"[CallIntelligenceManager] Extracted {len(created)} action items for call {call_id}")

        return created

    # ================================================================
    # Common Problems Detection
    # ================================================================

    def _detect_problems(self, transcript: str) -> list[dict[str, Any]]:
        """
        Detect problems in transcript using keyword matching.

        This is a basic implementation - can be enhanced with ML.
        """
        transcript_lower = transcript.lower()
        detected = []

        # Keyword mappings for each category
        keyword_map = {
            "account_access.login_issues": ["can't log in", "login problem", "won't let me in", "login failed"],
            "account_access.password_reset": ["forgot password", "reset password", "password doesn't work"],
            "account_access.locked_account": ["locked out", "account locked", "frozen account"],
            "transactions.failed_transfer": ["transfer failed", "didn't go through", "transfer didn't work"],
            "transactions.missing_deposit": ["missing deposit", "where's my deposit", "deposit not showing"],
            "transactions.dispute": ["dispute", "didn't authorize", "fraudulent charge", "wrong charge"],
            "loans.payment_question": ["loan payment", "when is my payment due", "payment amount"],
            "loans.rate_inquiry": ["interest rate", "what's the rate", "lower rate"],
            "cards.lost_stolen": ["lost my card", "card was stolen", "can't find my card"],
            "cards.fraud_alert": ["fraud alert", "suspicious activity", "didn't make this purchase"],
            "digital_banking.app_issues": ["app not working", "app crashed", "mobile app problem"],
            "digital_banking.bill_pay": ["bill pay", "pay my bill", "scheduled payment"],
            "fees.fee_dispute": ["fee", "charged me", "waive the fee", "overdraft"],
            "fraud.unauthorized_activity": ["didn't authorize", "someone used my", "stolen identity"],
        }

        for category_id, keywords in keyword_map.items():
            for keyword in keywords:
                if keyword in transcript_lower:
                    detected.append({"category_id": category_id, "keyword_matched": keyword, "confidence": 0.85})
                    break  # Only add each category once

        return detected

    def get_problem_trends(self, days: int = 30) -> dict[str, Any]:
        """
        Get problem category trends over time.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with trend data by category
        """
        conn = self._get_conn()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

        cursor = conn.execute(
            """
            SELECT 
                cp.category_id,
                pc.name as category_name,
                pc.severity,
                COUNT(*) as count,
                DATE(cp.detected_at) as date
            FROM call_problems cp
            JOIN problem_categories pc ON cp.category_id = pc.category_id
            WHERE cp.detected_at >= ?
            GROUP BY cp.category_id, DATE(cp.detected_at)
            ORDER BY date DESC, count DESC
        """,
            (cutoff,),
        )

        trends = {}
        for row in cursor.fetchall():
            cat_id = row["category_id"]
            if cat_id not in trends:
                trends[cat_id] = {
                    "category_id": cat_id,
                    "name": row["category_name"],
                    "severity": row["severity"],
                    "daily_counts": [],
                    "total": 0,
                }
            trends[cat_id]["daily_counts"].append({"date": row["date"], "count": row["count"]})
            trends[cat_id]["total"] += row["count"]

        # Sort by total count
        sorted_trends = sorted(trends.values(), key=lambda x: x["total"], reverse=True)

        return {"days": days, "trends": sorted_trends, "cutoff": cutoff}

    def get_top_problems(self, days: int = 7, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get top problem categories by volume.

        Args:
            days: Number of days to analyze
            limit: Max categories to return

        Returns:
            List of top problems with counts
        """
        conn = self._get_conn()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

        cursor = conn.execute(
            """
            SELECT 
                cp.category_id,
                pc.name as category_name,
                pc.parent_category,
                pc.severity,
                COUNT(*) as count,
                AVG(cp.confidence) as avg_confidence
            FROM call_problems cp
            JOIN problem_categories pc ON cp.category_id = pc.category_id
            WHERE cp.detected_at >= ?
            GROUP BY cp.category_id
            ORDER BY count DESC
            LIMIT ?
        """,
            (cutoff, limit),
        )

        return [dict(r) for r in cursor.fetchall()]

    # ================================================================
    # Sentiment & Emotion
    # ================================================================

    def _basic_sentiment_analysis(self, text: str) -> float:
        """
        Basic rule-based sentiment analysis.

        Returns score from -1.0 (very negative) to 1.0 (very positive).
        This is a placeholder - should be replaced with ML model.
        """
        text_lower = text.lower()

        positive_words = [
            "thank you",
            "thanks",
            "great",
            "excellent",
            "wonderful",
            "helpful",
            "appreciate",
            "perfect",
            "amazing",
            "love",
            "happy",
            "satisfied",
        ]
        negative_words = [
            "frustrated",
            "angry",
            "upset",
            "terrible",
            "awful",
            "horrible",
            "disappointed",
            "unacceptable",
            "ridiculous",
            "hate",
            "worst",
            "never",
        ]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return round((positive_count - negative_count) / max(total, 1), 2)

    def _detect_dominant_emotion(self, text: str) -> str:
        """
        Detect dominant emotion from text.

        This is a placeholder - should be enhanced with emotion model.
        """
        text_lower = text.lower()

        emotion_keywords = {
            "angry": ["angry", "furious", "outraged", "mad", "frustrated"],
            "anxious": ["worried", "concerned", "anxious", "nervous", "scared"],
            "happy": ["happy", "glad", "pleased", "delighted", "excited"],
            "sad": ["sad", "disappointed", "upset", "unhappy"],
            "neutral": [],
        }

        emotion_counts = {}
        for emotion, keywords in emotion_keywords.items():
            emotion_counts[emotion] = sum(1 for kw in keywords if kw in text_lower)

        max_emotion = max(emotion_counts, key=emotion_counts.get)
        if emotion_counts[max_emotion] == 0:
            return "neutral"
        return max_emotion

    def get_sentiment_trends(self, days: int = 30) -> dict[str, Any]:
        """
        Get sentiment trends over time.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with daily sentiment averages
        """
        conn = self._get_conn()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

        cursor = conn.execute(
            """
            SELECT 
                DATE(created_at) as date,
                AVG(sentiment_score) as avg_sentiment,
                COUNT(*) as call_count,
                SUM(CASE WHEN sentiment_score >= 0.3 THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN sentiment_score <= -0.3 THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN sentiment_score > -0.3 AND sentiment_score < 0.3 THEN 1 ELSE 0 END) as neutral
            FROM calls
            WHERE created_at >= ?
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """,
            (cutoff,),
        )

        daily = [dict(r) for r in cursor.fetchall()]

        # Calculate overall stats
        total_calls = sum(d["call_count"] for d in daily)
        avg_sentiment = sum(d["avg_sentiment"] * d["call_count"] for d in daily) / max(total_calls, 1)

        return {
            "days": days,
            "daily": daily,
            "total_calls": total_calls,
            "avg_sentiment": round(avg_sentiment, 3),
            "cutoff": cutoff,
        }

    # ================================================================
    # PII Redaction
    # ================================================================

    def redact_pii(self, text: str) -> str:
        """
        Redact PII from text using regex patterns.

        Args:
            text: Raw text to redact

        Returns:
            Text with PII replaced by placeholders
        """
        redacted = text
        for pattern, replacement in PII_PATTERNS:
            redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
        return redacted

    # ================================================================
    # Analytics & Stats
    # ================================================================

    def get_dashboard_stats(self) -> dict[str, Any]:
        """Get call intelligence dashboard KPIs."""
        conn = self._get_conn()

        # Total calls
        total_calls = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

        # Calls today
        today = datetime.utcnow().date().isoformat()
        calls_today = conn.execute("SELECT COUNT(*) FROM calls WHERE DATE(created_at) = ?", (today,)).fetchone()[0]

        # Average sentiment
        avg_sentiment = conn.execute("SELECT AVG(sentiment_score) FROM calls").fetchone()[0] or 0

        # Average duration
        avg_duration = (
            conn.execute("SELECT AVG(duration_seconds) FROM calls WHERE duration_seconds IS NOT NULL").fetchone()[0]
            or 0
        )

        # Open action items
        open_actions = conn.execute("SELECT COUNT(*) FROM call_action_items WHERE status = 'open'").fetchone()[0]

        # Critical issues (last 24 hours)
        yesterday = (datetime.utcnow() - timedelta(hours=24)).isoformat() + "Z"
        critical_issues = conn.execute(
            """
            SELECT COUNT(*) FROM call_problems cp
            JOIN problem_categories pc ON cp.category_id = pc.category_id
            WHERE cp.detected_at >= ? AND pc.severity = 'critical'
        """,
            (yesterday,),
        ).fetchone()[0]

        # Top problem this week
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat() + "Z"
        top_problem_row = conn.execute(
            """
            SELECT pc.name, COUNT(*) as count
            FROM call_problems cp
            JOIN problem_categories pc ON cp.category_id = pc.category_id
            WHERE cp.detected_at >= ?
            GROUP BY cp.category_id
            ORDER BY count DESC
            LIMIT 1
        """,
            (week_ago,),
        ).fetchone()
        top_problem = dict(top_problem_row) if top_problem_row else {"name": "N/A", "count": 0}

        return {
            "total_calls": total_calls,
            "calls_today": calls_today,
            "avg_sentiment": round(avg_sentiment, 3),
            "avg_duration_minutes": round(avg_duration / 60, 1) if avg_duration else 0,
            "open_action_items": open_actions,
            "critical_issues_24h": critical_issues,
            "top_problem_this_week": top_problem,
        }

    def get_member_call_summary(self, member_id: str) -> dict[str, Any]:
        """
        Get a summary of all calls for a member.

        Args:
            member_id: Fiserv member ID

        Returns:
            Dict with member call history summary
        """
        conn = self._get_conn()

        # Get call count and dates
        cursor = conn.execute(
            """
            SELECT 
                COUNT(*) as total_calls,
                MIN(created_at) as first_call,
                MAX(created_at) as last_call,
                AVG(sentiment_score) as avg_sentiment,
                AVG(duration_seconds) as avg_duration
            FROM calls
            WHERE member_id = ?
        """,
            (member_id,),
        )
        stats = dict(cursor.fetchone())

        # Get recent calls
        cursor = conn.execute(
            """
            SELECT call_id, created_at, intent_category, sentiment_score, summary
            FROM calls
            WHERE member_id = ?
            ORDER BY created_at DESC
            LIMIT 5
        """,
            (member_id,),
        )
        recent_calls = [dict(r) for r in cursor.fetchall()]

        # Get common problems for this member
        cursor = conn.execute(
            """
            SELECT pc.name, COUNT(*) as count
            FROM calls c
            JOIN call_problems cp ON c.id = cp.call_id
            JOIN problem_categories pc ON cp.category_id = pc.category_id
            WHERE c.member_id = ?
            GROUP BY cp.category_id
            ORDER BY count DESC
            LIMIT 5
        """,
            (member_id,),
        )
        common_problems = [dict(r) for r in cursor.fetchall()]

        return {
            "member_id": member_id,
            "stats": stats,
            "recent_calls": recent_calls,
            "common_problems": common_problems,
        }


# Singleton instance
_call_intelligence_manager: CallIntelligenceManager | None = None


def get_call_intelligence_manager(db_path: str | None = None) -> CallIntelligenceManager:
    """Get or create the singleton CallIntelligenceManager instance."""
    global _call_intelligence_manager
    if _call_intelligence_manager is None:
        _call_intelligence_manager = CallIntelligenceManager(db_path or "/app/instance/calls.db")
    return _call_intelligence_manager
