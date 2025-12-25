"""
Analytics Manager Module - Enterprise Analytics Dashboard
Aggregates metrics from all enterprise phases with ROI, sentiment, and efficiency tracking.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetricCategory(str, Enum):
    """Metric categories."""

    USAGE = "usage"
    ENGAGEMENT = "engagement"
    EFFICIENCY = "efficiency"
    SENTIMENT = "sentiment"
    ROI = "roi"


class TimeGranularity(str, Enum):
    """Time granularity for trends."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class AnalyticsManager:
    """
    Enterprise analytics engine that aggregates data across all phases.

    Features:
    - Cross-phase metric aggregation
    - ROI calculations
    - Sentiment trend analysis
    - Efficiency KPIs
    - Time-series trends
    - Report generation
    """

    def __init__(self, db_path: str = "/app/instance/analytics_store.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

        # References to other managers (set externally)
        self._qa_manager = None
        self._automation_manager = None
        self._knowledge_manager = None

        logger.info(f"AnalyticsManager initialized with db: {self.db_path}")

    def set_managers(self, qa=None, automation=None, knowledge=None):
        """Set references to other managers for cross-phase aggregation."""
        self._qa_manager = qa
        self._automation_manager = automation
        self._knowledge_manager = knowledge

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
            -- Metric snapshots for trend analysis
            CREATE TABLE IF NOT EXISTS metric_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT  -- JSON for extra context
            );
            
            -- Reports table
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                report_type TEXT DEFAULT 'summary',
                date_range TEXT,  -- JSON with start/end
                data TEXT NOT NULL,  -- JSON report data
                created_at TEXT NOT NULL
            );
            
            -- Event log for detailed tracking
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                source TEXT,  -- qa, automation, knowledge, etc.
                user_id TEXT,
                data TEXT,  -- JSON event data
                created_at TEXT NOT NULL
            );
            
            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON metric_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_snapshots_cat ON metric_snapshots(category);
            CREATE INDEX IF NOT EXISTS idx_events_type ON analytics_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_ts ON analytics_events(created_at);
        """)
        conn.commit()

    # =========================================================================
    # Event Logging
    # =========================================================================

    def log_event(
        self,
        event_type: str,
        source: str,
        data: dict[str, Any] | None = None,
        user_id: str | None = None,
    ):
        """Log an analytics event."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"
        conn.execute(
            "INSERT INTO analytics_events (event_type, source, user_id, data, created_at) VALUES (?, ?, ?, ?, ?)",
            (event_type, source, user_id, json.dumps(data or {}), now),
        )
        conn.commit()

    def get_event_counts(self, hours: int = 24) -> dict[str, int]:
        """Get event counts by type for the specified time period."""
        conn = self._get_conn()
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        rows = conn.execute(
            """
            SELECT event_type, COUNT(*) as count 
            FROM analytics_events 
            WHERE created_at > ?
            GROUP BY event_type
            """,
            (cutoff,),
        ).fetchall()
        return {r["event_type"]: r["count"] for r in rows}

    # =========================================================================
    # Metric Snapshots
    # =========================================================================

    def record_snapshot(
        self,
        category: str,
        metric_name: str,
        value: float,
        metadata: dict[str, Any] | None = None,
    ):
        """Record a metric snapshot."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"
        conn.execute(
            "INSERT INTO metric_snapshots (category, metric_name, value, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
            (category, metric_name, value, now, json.dumps(metadata or {})),
        )
        conn.commit()

    def get_trend_data(
        self,
        category: str,
        metric_name: str,
        granularity: str = "daily",
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """Get trend data for a specific metric."""
        conn = self._get_conn()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        if granularity == "hourly":
            group_format = "%Y-%m-%d %H:00"
        elif granularity == "daily":
            group_format = "%Y-%m-%d"
        elif granularity == "weekly":
            group_format = "%Y-W%W"
        else:
            group_format = "%Y-%m"

        rows = conn.execute(
            f"""
            SELECT strftime('{group_format}', timestamp) as period,
                   AVG(value) as avg_value,
                   MAX(value) as max_value,
                   MIN(value) as min_value,
                   COUNT(*) as count
            FROM metric_snapshots
            WHERE category = ? AND metric_name = ? AND timestamp > ?
            GROUP BY period
            ORDER BY period
            """,
            (category, metric_name, cutoff),
        ).fetchall()

        return [
            {
                "period": r["period"],
                "avg": round(r["avg_value"], 2),
                "max": round(r["max_value"], 2),
                "min": round(r["min_value"], 2),
                "count": r["count"],
            }
            for r in rows
        ]

    # =========================================================================
    # Cross-Phase Aggregation
    # =========================================================================

    def get_usage_metrics(self) -> dict[str, Any]:
        """Get usage metrics from all phases."""
        metrics = {
            "queries": 0,
            "kb_searches": 0,
            "kb_views": 0,
            "rules_fired": 0,
            "expert_searches": 0,
            "golden_matches": 0,
        }

        # Get stats from managers
        if self._qa_manager:
            try:
                qa_stats = self._qa_manager.get_stats()
                metrics["queries"] = qa_stats.get("total_feedback", 0)
                metrics["golden_matches"] = qa_stats.get("golden_answers", 0)
            except Exception as e:
                logger.warning(f"Failed to get QA stats: {e}")

        if self._automation_manager:
            try:
                auto_stats = self._automation_manager.get_stats()
                metrics["rules_fired"] = auto_stats.get("rules_fired_24h", 0)
            except Exception as e:
                logger.warning(f"Failed to get automation stats: {e}")

        if self._knowledge_manager:
            try:
                kb_stats = self._knowledge_manager.get_stats()
                metrics["kb_searches"] = kb_stats.get("searches_24h", 0)
                metrics["kb_views"] = kb_stats.get("articles", {}).get("total", 0)
                metrics["expert_searches"] = kb_stats.get("experts", 0)
            except Exception as e:
                logger.warning(f"Failed to get knowledge stats: {e}")

        # Also check event log
        event_counts = self.get_event_counts(24)
        metrics["api_calls"] = sum(event_counts.values())

        return metrics

    def get_efficiency_metrics(self) -> dict[str, Any]:
        """Calculate efficiency KPIs."""
        metrics = {
            "auto_resolution_rate": 0.0,
            "golden_hit_rate": 0.0,
            "webhook_success_rate": 0.0,
            "avg_response_time_ms": 0,
        }

        if self._qa_manager:
            try:
                qa_stats = self._qa_manager.get_stats()
                total_fb = qa_stats.get("total_feedback", 0)
                golden = qa_stats.get("golden_answers", 0)
                if total_fb > 0:
                    metrics["golden_hit_rate"] = round((golden / max(total_fb, 1)) * 100, 1)
            except Exception:
                pass

        if self._automation_manager:
            try:
                auto_stats = self._automation_manager.get_stats()
                wh_stats = auto_stats.get("webhook_deliveries_24h", {})
                metrics["webhook_success_rate"] = wh_stats.get("success_rate", 0)
            except Exception:
                pass

        # Calculate auto-resolution rate from golden answers
        if metrics["golden_hit_rate"] > 0:
            metrics["auto_resolution_rate"] = metrics["golden_hit_rate"]

        return metrics

    def get_sentiment_metrics(self) -> dict[str, Any]:
        """Get sentiment distribution from emotion data."""
        # Default distribution - would be populated from emotion service
        sentiment = {
            "positive": 25,
            "neutral": 60,
            "negative": 15,
            "distribution": {
                "happy": 15,
                "neutral": 60,
                "sad": 8,
                "angry": 5,
                "surprised": 7,
                "fearful": 3,
                "disgusted": 2,
            },
            "trend": "stable",
        }

        # Try to get real emotion data from events
        conn = self._get_conn()
        cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        rows = conn.execute(
            """
            SELECT data FROM analytics_events 
            WHERE event_type = 'emotion_detected' AND created_at > ?
            """,
            (cutoff,),
        ).fetchall()

        if rows:
            emotions = {}
            for r in rows:
                data = json.loads(r["data"] or "{}")
                emotion = data.get("emotion", "neutral")
                emotions[emotion] = emotions.get(emotion, 0) + 1

            if emotions:
                sentiment["distribution"] = emotions
                total = sum(emotions.values())
                positive = emotions.get("happy", 0) + emotions.get("surprised", 0)
                negative = emotions.get("sad", 0) + emotions.get("angry", 0) + emotions.get("fearful", 0)
                neutral = emotions.get("neutral", 0)

                sentiment["positive"] = round((positive / max(total, 1)) * 100, 1)
                sentiment["negative"] = round((negative / max(total, 1)) * 100, 1)
                sentiment["neutral"] = round((neutral / max(total, 1)) * 100, 1)

        return sentiment

    def get_roi_metrics(self) -> dict[str, Any]:
        """Calculate ROI metrics."""
        hourly_rate = 50  # Assumed hourly rate in USD

        usage = self.get_usage_metrics()
        efficiency = self.get_efficiency_metrics()

        # Time saved estimates
        auto_answers = usage.get("golden_matches", 0)
        rules_fired = usage.get("rules_fired", 0)

        time_saved_queries_min = auto_answers * 5  # 5 min per auto-answered query
        time_saved_automation_min = rules_fired * 2  # 2 min per automated task
        total_time_saved_min = time_saved_queries_min + time_saved_automation_min

        time_saved_hours = total_time_saved_min / 60
        cost_savings = time_saved_hours * hourly_rate

        # Productivity gain
        productivity_gain = efficiency.get("auto_resolution_rate", 0)

        return {
            "time_saved_minutes": total_time_saved_min,
            "time_saved_hours": round(time_saved_hours, 1),
            "estimated_cost_savings_usd": round(cost_savings, 2),
            "productivity_gain_percent": round(productivity_gain, 1),
            "breakdown": {
                "auto_answers_time_min": time_saved_queries_min,
                "automation_time_min": time_saved_automation_min,
            },
            "assumptions": {
                "hourly_rate_usd": hourly_rate,
                "minutes_per_query": 5,
                "minutes_per_automation": 2,
            },
        }

    # =========================================================================
    # Dashboard Overview
    # =========================================================================

    def get_overview(self) -> dict[str, Any]:
        """Get dashboard overview with all key metrics."""
        usage = self.get_usage_metrics()
        efficiency = self.get_efficiency_metrics()
        sentiment = self.get_sentiment_metrics()
        roi = self.get_roi_metrics()

        # Get counts from managers
        active_rules = 0
        active_webhooks = 0
        kb_articles = 0
        experts = 0

        if self._automation_manager:
            try:
                auto_stats = self._automation_manager.get_stats()
                active_rules = auto_stats.get("rules", {}).get("enabled", 0)
                active_webhooks = auto_stats.get("webhooks", {}).get("enabled", 0)
            except Exception:
                pass

        if self._knowledge_manager:
            try:
                kb_stats = self._knowledge_manager.get_stats()
                kb_articles = kb_stats.get("articles", {}).get("published", 0)
                experts = kb_stats.get("experts", 0)
            except Exception:
                pass

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "queries_24h": usage.get("queries", 0) + usage.get("kb_searches", 0),
                "automation_events_24h": usage.get("rules_fired", 0),
                "time_saved_min": roi.get("time_saved_minutes", 0),
                "sentiment_positive": sentiment.get("positive", 0),
            },
            "kpis": {
                "auto_resolution_rate": efficiency.get("auto_resolution_rate", 0),
                "webhook_success_rate": efficiency.get("webhook_success_rate", 0),
                "golden_hit_rate": efficiency.get("golden_hit_rate", 0),
            },
            "counts": {
                "active_rules": active_rules,
                "active_webhooks": active_webhooks,
                "kb_articles": kb_articles,
                "experts": experts,
            },
            "usage": usage,
            "efficiency": efficiency,
            "sentiment": sentiment,
            "roi": roi,
        }

    # =========================================================================
    # Trends
    # =========================================================================

    def get_trends(self, granularity: str = "daily", days: int = 7) -> dict[str, Any]:
        """Get trend data for key metrics."""
        trends = {}

        metrics_to_track = [
            ("usage", "queries"),
            ("usage", "kb_searches"),
            ("efficiency", "auto_resolution_rate"),
            ("sentiment", "positive"),
        ]

        for category, metric in metrics_to_track:
            key = f"{category}_{metric}"
            trends[key] = self.get_trend_data(category, metric, granularity, days)

        return {
            "granularity": granularity,
            "days": days,
            "trends": trends,
        }

    # =========================================================================
    # Reports
    # =========================================================================

    def generate_report(
        self,
        title: str | None = None,
        report_type: str = "summary",
    ) -> dict[str, Any]:
        """Generate an analytics report."""
        conn = self._get_conn()
        now = datetime.utcnow()

        if not title:
            title = f"Analytics Report - {now.strftime('%Y-%m-%d')}"

        # Collect all data
        overview = self.get_overview()
        trends = self.get_trends("daily", 7)

        report_data = {
            "generated_at": now.isoformat() + "Z",
            "report_type": report_type,
            "overview": overview,
            "trends": trends,
            "period": {
                "start": (now - timedelta(days=7)).isoformat(),
                "end": now.isoformat(),
            },
        }

        # Save report
        cursor = conn.execute(
            """
            INSERT INTO reports (title, report_type, date_range, data, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (title, report_type, json.dumps(report_data["period"]), json.dumps(report_data), now.isoformat() + "Z"),
        )
        conn.commit()

        report_data["id"] = cursor.lastrowid
        report_data["title"] = title

        logger.info(f"Report generated: id={cursor.lastrowid} title={title}")
        return report_data

    def list_reports(self, limit: int = 20) -> list[dict[str, Any]]:
        """List saved reports."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, title, report_type, date_range, created_at FROM reports ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        return [
            {
                "id": r["id"],
                "title": r["title"],
                "report_type": r["report_type"],
                "date_range": json.loads(r["date_range"] or "{}"),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def get_report(self, report_id: int) -> dict[str, Any] | None:
        """Get a specific report."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if not row:
            return None

        data = json.loads(row["data"] or "{}")
        data["id"] = row["id"]
        data["title"] = row["title"]
        return data

    # =========================================================================
    # Export
    # =========================================================================

    def export_metrics(self, format: str = "json") -> dict[str, Any]:
        """Export all metrics for download."""
        overview = self.get_overview()
        trends = self.get_trends("daily", 30)

        export_data = {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "format": format,
            "overview": overview,
            "trends": trends,
        }

        return export_data

    # =========================================================================
    # Aggregate Stats (for other managers to call)
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get quick stats for use by other systems."""
        conn = self._get_conn()

        events_24h = conn.execute(
            "SELECT COUNT(*) FROM analytics_events WHERE created_at > datetime('now', '-1 day')"
        ).fetchone()[0]

        reports_count = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]

        snapshots_24h = conn.execute(
            "SELECT COUNT(*) FROM metric_snapshots WHERE timestamp > datetime('now', '-1 day')"
        ).fetchone()[0]

        return {
            "events_24h": events_24h,
            "reports_count": reports_count,
            "snapshots_24h": snapshots_24h,
        }


# Singleton instance
_analytics_manager: AnalyticsManager | None = None


def get_analytics_manager(db_path: str | None = None) -> AnalyticsManager:
    """Get or create the singleton AnalyticsManager instance."""
    global _analytics_manager
    if _analytics_manager is None:
        _analytics_manager = AnalyticsManager(db_path or "/app/instance/analytics_store.db")
    return _analytics_manager
