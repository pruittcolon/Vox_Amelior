"""
Automation Manager Module - Enterprise Business Automation
Handles rules engine, triggers, webhooks, and workflow automation.
"""

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, time as dtime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ConditionType(str, Enum):
    """Types of conditions that can trigger a rule."""

    TRANSCRIPT_CONTAINS = "transcript_contains"
    SPEAKER_IS = "speaker_is"
    EMOTION_IS = "emotion_is"
    TIME_BETWEEN = "time_between"
    KEYWORD_MATCH = "keyword_match"
    REGEX_MATCH = "regex_match"


class ActionType(str, Enum):
    """Types of actions a rule can perform."""

    SEND_WEBHOOK = "send_webhook"
    LOG_EVENT = "log_event"
    SEND_NOTIFICATION = "send_notification"
    CALL_API = "call_api"


class TriggerType(str, Enum):
    """Types of events that can trigger rules."""

    TRANSCRIPT = "transcript"
    EMOTION = "emotion"
    MANUAL = "manual"
    SCHEDULE = "schedule"


class WebhookStatus(str, Enum):
    """Webhook delivery status."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Condition:
    """A condition that must be met for a rule to fire."""

    type: str
    value: Any
    operator: str = "equals"  # equals, contains, regex, gt, lt

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Condition":
        return cls(
            type=data.get("type", ""),
            value=data.get("value", ""),
            operator=data.get("operator", "equals"),
        )


@dataclass
class Action:
    """An action to perform when a rule fires."""

    type: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        return cls(
            type=data.get("type", ""),
            config=data.get("config", {}),
        )


class AutomationManager:
    """
    Enterprise automation engine.

    Features:
    - Rules engine with conditions and actions
    - Webhook management with retry logic
    - Event triggers and logging
    - Statistics and monitoring
    """

    def __init__(self, db_path: str = "/app/instance/automation_store.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
        logger.info(f"AutomationManager initialized with db: {self.db_path}")

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
            -- Rules table
            CREATE TABLE IF NOT EXISTS rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                conditions TEXT NOT NULL,  -- JSON array of conditions
                actions TEXT NOT NULL,     -- JSON array of actions
                enabled INTEGER DEFAULT 1,
                priority INTEGER DEFAULT 100,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                created_by TEXT
            );
            
            -- Webhooks table
            CREATE TABLE IF NOT EXISTS webhooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                secret TEXT,
                events TEXT,  -- JSON array of event types
                headers TEXT, -- JSON object of custom headers
                enabled INTEGER DEFAULT 1,
                retry_count INTEGER DEFAULT 3,
                timeout_sec INTEGER DEFAULT 30,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                created_by TEXT
            );
            
            -- Webhook delivery logs
            CREATE TABLE IF NOT EXISTS webhook_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                webhook_id INTEGER NOT NULL,
                event_type TEXT,
                payload TEXT,
                status TEXT NOT NULL,
                status_code INTEGER,
                response_body TEXT,
                attempt INTEGER DEFAULT 1,
                duration_ms INTEGER,
                error TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (webhook_id) REFERENCES webhooks(id) ON DELETE CASCADE
            );
            
            -- Trigger execution logs
            CREATE TABLE IF NOT EXISTS trigger_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_type TEXT NOT NULL,
                event_data TEXT,
                rules_evaluated INTEGER DEFAULT 0,
                rules_fired INTEGER DEFAULT 0,
                actions_executed INTEGER DEFAULT 0,
                duration_ms INTEGER,
                created_at TEXT NOT NULL
            );
            
            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_rules_enabled ON rules(enabled);
            CREATE INDEX IF NOT EXISTS idx_webhooks_enabled ON webhooks(enabled);
            CREATE INDEX IF NOT EXISTS idx_webhook_logs_webhook ON webhook_logs(webhook_id);
            CREATE INDEX IF NOT EXISTS idx_webhook_logs_status ON webhook_logs(status);
            CREATE INDEX IF NOT EXISTS idx_trigger_logs_type ON trigger_logs(trigger_type);
        """)
        conn.commit()

    # =========================================================================
    # Rules CRUD
    # =========================================================================

    def create_rule(
        self,
        name: str,
        conditions: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        description: str | None = None,
        priority: int = 100,
        enabled: bool = True,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        """Create a new automation rule."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        cursor = conn.execute(
            """
            INSERT INTO rules (name, description, conditions, actions, enabled, priority, created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                description,
                json.dumps(conditions),
                json.dumps(actions),
                1 if enabled else 0,
                priority,
                now,
                created_by,
            ),
        )
        conn.commit()
        rule_id = cursor.lastrowid

        logger.info(f"Rule created: id={rule_id}, name={name}")
        return self.get_rule(rule_id)

    def get_rule(self, rule_id: int) -> dict[str, Any] | None:
        """Get a rule by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM rules WHERE id = ?", (rule_id,)).fetchone()
        if not row:
            return None
        return self._parse_rule_row(row)

    def _parse_rule_row(self, row: sqlite3.Row) -> dict[str, Any]:
        """Parse a rule row into a dict."""
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "conditions": json.loads(row["conditions"]),
            "actions": json.loads(row["actions"]),
            "enabled": bool(row["enabled"]),
            "priority": row["priority"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "created_by": row["created_by"],
        }

    def list_rules(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        """List all rules."""
        conn = self._get_conn()
        if enabled_only:
            rows = conn.execute("SELECT * FROM rules WHERE enabled = 1 ORDER BY priority ASC, id ASC").fetchall()
        else:
            rows = conn.execute("SELECT * FROM rules ORDER BY priority ASC, id ASC").fetchall()
        return [self._parse_rule_row(row) for row in rows]

    def update_rule(
        self,
        rule_id: int,
        name: str | None = None,
        conditions: list[dict[str, Any]] | None = None,
        actions: list[dict[str, Any]] | None = None,
        description: str | None = None,
        priority: int | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        """Update a rule."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if conditions is not None:
            updates.append("conditions = ?")
            params.append(json.dumps(conditions))
        if actions is not None:
            updates.append("actions = ?")
            params.append(json.dumps(actions))
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if priority is not None:
            updates.append("priority = ?")
            params.append(priority)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(1 if enabled else 0)

        updates.append("updated_at = ?")
        params.append(now)
        params.append(rule_id)

        conn.execute(f"UPDATE rules SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()

        return self.get_rule(rule_id)

    def delete_rule(self, rule_id: int) -> dict[str, Any]:
        """Delete a rule."""
        conn = self._get_conn()
        result = conn.execute("DELETE FROM rules WHERE id = ?", (rule_id,))
        conn.commit()

        if result.rowcount == 0:
            return {"success": False, "error": "Rule not found"}
        return {"success": True, "message": f"Rule {rule_id} deleted"}

    def toggle_rule(self, rule_id: int, enabled: bool) -> dict[str, Any]:
        """Enable or disable a rule."""
        return self.update_rule(rule_id, enabled=enabled)

    # =========================================================================
    # Webhooks CRUD
    # =========================================================================

    def create_webhook(
        self,
        name: str,
        url: str,
        events: list[str] | None = None,
        secret: str | None = None,
        headers: dict[str, str] | None = None,
        retry_count: int = 3,
        timeout_sec: int = 30,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        """Create a new webhook endpoint."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        cursor = conn.execute(
            """
            INSERT INTO webhooks (name, url, secret, events, headers, retry_count, timeout_sec, created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                url,
                secret,
                json.dumps(events or []),
                json.dumps(headers or {}),
                retry_count,
                timeout_sec,
                now,
                created_by,
            ),
        )
        conn.commit()
        webhook_id = cursor.lastrowid

        logger.info(f"Webhook created: id={webhook_id}, name={name}, url={url}")
        return self.get_webhook(webhook_id)

    def get_webhook(self, webhook_id: int) -> dict[str, Any] | None:
        """Get a webhook by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM webhooks WHERE id = ?", (webhook_id,)).fetchone()
        if not row:
            return None
        return self._parse_webhook_row(row)

    def _parse_webhook_row(self, row: sqlite3.Row) -> dict[str, Any]:
        """Parse a webhook row into a dict."""
        return {
            "id": row["id"],
            "name": row["name"],
            "url": row["url"],
            "secret": row["secret"],
            "events": json.loads(row["events"] or "[]"),
            "headers": json.loads(row["headers"] or "{}"),
            "enabled": bool(row["enabled"]),
            "retry_count": row["retry_count"],
            "timeout_sec": row["timeout_sec"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "created_by": row["created_by"],
        }

    def list_webhooks(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        """List all webhooks."""
        conn = self._get_conn()
        if enabled_only:
            rows = conn.execute("SELECT * FROM webhooks WHERE enabled = 1 ORDER BY id").fetchall()
        else:
            rows = conn.execute("SELECT * FROM webhooks ORDER BY id").fetchall()
        return [self._parse_webhook_row(row) for row in rows]

    def delete_webhook(self, webhook_id: int) -> dict[str, Any]:
        """Delete a webhook."""
        conn = self._get_conn()
        result = conn.execute("DELETE FROM webhooks WHERE id = ?", (webhook_id,))
        conn.commit()

        if result.rowcount == 0:
            return {"success": False, "error": "Webhook not found"}
        return {"success": True, "message": f"Webhook {webhook_id} deleted"}

    def get_webhook_logs(self, webhook_id: int, limit: int = 50) -> list[dict[str, Any]]:
        """Get delivery logs for a webhook."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM webhook_logs WHERE webhook_id = ? ORDER BY created_at DESC LIMIT ?", (webhook_id, limit)
        ).fetchall()

        return [
            {
                "id": r["id"],
                "webhook_id": r["webhook_id"],
                "event_type": r["event_type"],
                "status": r["status"],
                "status_code": r["status_code"],
                "attempt": r["attempt"],
                "duration_ms": r["duration_ms"],
                "error": r["error"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # =========================================================================
    # Rule Evaluation Engine
    # =========================================================================

    def evaluate_conditions(
        self,
        conditions: list[dict[str, Any]],
        event_data: dict[str, Any],
    ) -> bool:
        """
        Evaluate if all conditions are met for the given event data.

        Args:
            conditions: List of condition dicts
            event_data: Event data to evaluate against

        Returns:
            True if ALL conditions are met
        """
        for cond in conditions:
            cond_type = cond.get("type", "")
            cond_value = cond.get("value", "")
            operator = cond.get("operator", "equals")

            if cond_type == ConditionType.TRANSCRIPT_CONTAINS.value:
                text = (event_data.get("text") or "").lower()
                if cond_value.lower() not in text:
                    return False

            elif cond_type == ConditionType.SPEAKER_IS.value:
                speaker = (event_data.get("speaker") or "").lower()
                if speaker != cond_value.lower():
                    return False

            elif cond_type == ConditionType.EMOTION_IS.value:
                emotion = (event_data.get("emotion") or "").lower()
                if emotion != cond_value.lower():
                    return False

            elif cond_type == ConditionType.KEYWORD_MATCH.value:
                text = (event_data.get("text") or "").lower()
                keywords = [k.strip().lower() for k in cond_value.split(",")]
                if not any(kw in text for kw in keywords):
                    return False

            elif cond_type == ConditionType.REGEX_MATCH.value:
                text = event_data.get("text") or ""
                try:
                    if not re.search(cond_value, text, re.IGNORECASE):
                        return False
                except re.error:
                    return False

            elif cond_type == ConditionType.TIME_BETWEEN.value:
                # value should be "HH:MM-HH:MM"
                try:
                    start_str, end_str = cond_value.split("-")
                    start_h, start_m = map(int, start_str.split(":"))
                    end_h, end_m = map(int, end_str.split(":"))
                    now = datetime.utcnow().time()
                    start_time = dtime(start_h, start_m)
                    end_time = dtime(end_h, end_m)
                    if not (start_time <= now <= end_time):
                        return False
                except (ValueError, TypeError, AttributeError):
                    logger.debug("Invalid time range format in condition: %s", cond_value)
                    return False

        return True

    async def execute_actions(
        self,
        actions: list[dict[str, Any]],
        event_data: dict[str, Any],
        rule_id: int,
    ) -> list[dict[str, Any]]:
        """Execute all actions for a fired rule."""
        results = []

        for action in actions:
            action_type = action.get("type", "")
            config = action.get("config", {})
            result = {"type": action_type, "success": False}

            try:
                if action_type == ActionType.SEND_WEBHOOK.value:
                    webhook_id = config.get("webhook_id")
                    if webhook_id:
                        delivery = await self.deliver_webhook(
                            webhook_id=webhook_id,
                            event_type="rule_fired",
                            payload={
                                "rule_id": rule_id,
                                "event_data": event_data,
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                            },
                        )
                        result["success"] = delivery.get("success", False)
                        result["delivery"] = delivery

                elif action_type == ActionType.LOG_EVENT.value:
                    message = config.get("message", "Rule fired")
                    logger.info(f"[AUTOMATION] Rule {rule_id}: {message} | Data: {event_data}")
                    result["success"] = True
                    result["message"] = message

                elif action_type == ActionType.SEND_NOTIFICATION.value:
                    # Placeholder for notification system
                    result["success"] = True
                    result["message"] = "Notification queued (not implemented)"

                elif action_type == ActionType.CALL_API.value:
                    endpoint = config.get("endpoint", "")
                    method = config.get("method", "POST")
                    # Would call internal API here
                    result["success"] = True
                    result["endpoint"] = endpoint

            except Exception as e:
                result["error"] = str(e)
                logger.error(f"[AUTOMATION] Action failed: {e}")

            results.append(result)

        return results

    # =========================================================================
    # Trigger System
    # =========================================================================

    async def fire_trigger(
        self,
        trigger_type: str,
        event_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Fire a trigger and evaluate all matching rules.

        Args:
            trigger_type: Type of trigger (transcript, emotion, manual)
            event_data: Data associated with the event

        Returns:
            Summary of rule evaluations and actions
        """
        start_time = datetime.utcnow()
        conn = self._get_conn()

        # Get all enabled rules sorted by priority
        rules = self.list_rules(enabled_only=True)

        rules_evaluated = 0
        rules_fired = 0
        actions_executed = 0
        fired_rules = []

        for rule in rules:
            rules_evaluated += 1

            # Evaluate conditions
            if self.evaluate_conditions(rule["conditions"], event_data):
                rules_fired += 1

                # Execute actions
                action_results = await self.execute_actions(rule["actions"], event_data, rule["id"])

                actions_executed += len(action_results)
                fired_rules.append(
                    {
                        "rule_id": rule["id"],
                        "rule_name": rule["name"],
                        "actions": action_results,
                    }
                )

        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        # Log trigger execution
        now = datetime.utcnow().isoformat() + "Z"
        conn.execute(
            """
            INSERT INTO trigger_logs (trigger_type, event_data, rules_evaluated, rules_fired, actions_executed, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (trigger_type, json.dumps(event_data), rules_evaluated, rules_fired, actions_executed, duration_ms, now),
        )
        conn.commit()

        logger.info(
            f"[AUTOMATION] Trigger fired: type={trigger_type} rules_eval={rules_evaluated} "
            f"rules_fired={rules_fired} actions={actions_executed} duration={duration_ms}ms"
        )

        return {
            "success": True,
            "trigger_type": trigger_type,
            "rules_evaluated": rules_evaluated,
            "rules_fired": rules_fired,
            "actions_executed": actions_executed,
            "duration_ms": duration_ms,
            "fired_rules": fired_rules,
        }

    def test_rule(
        self,
        rule_id: int,
        sample_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Test a rule against sample data without executing actions.

        Returns whether conditions would match.
        """
        rule = self.get_rule(rule_id)
        if not rule:
            return {"success": False, "error": "Rule not found"}

        matches = self.evaluate_conditions(rule["conditions"], sample_data)

        return {
            "success": True,
            "rule_id": rule_id,
            "rule_name": rule["name"],
            "matches": matches,
            "conditions_count": len(rule["conditions"]),
            "actions_count": len(rule["actions"]),
            "sample_data": sample_data,
        }

    # =========================================================================
    # Webhook Delivery
    # =========================================================================

    async def deliver_webhook(
        self,
        webhook_id: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Deliver a webhook with retry logic."""
        webhook = self.get_webhook(webhook_id)
        if not webhook:
            return {"success": False, "error": "Webhook not found"}

        if not webhook["enabled"]:
            return {"success": False, "error": "Webhook is disabled"}

        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        # Build headers
        headers = {"Content-Type": "application/json"}
        headers.update(webhook.get("headers", {}))

        # Add signature if secret is set
        if webhook.get("secret"):
            payload_str = json.dumps(payload)
            signature = hashlib.sha256((payload_str + webhook["secret"]).encode()).hexdigest()
            headers["X-Webhook-Signature"] = signature

        # Attempt delivery with retries
        max_retries = webhook.get("retry_count", 3)
        timeout = webhook.get("timeout_sec", 30)

        for attempt in range(1, max_retries + 1):
            start = datetime.utcnow()
            status = WebhookStatus.PENDING.value
            status_code = None
            response_body = None
            error = None

            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        webhook["url"],
                        json=payload,
                        headers=headers,
                    )
                    status_code = response.status_code
                    response_body = response.text[:1000]  # Truncate

                    if 200 <= status_code < 300:
                        status = WebhookStatus.SUCCESS.value
                    else:
                        status = WebhookStatus.FAILED.value
                        error = f"HTTP {status_code}"

            except httpx.TimeoutException:
                status = WebhookStatus.FAILED.value
                error = "Timeout"
            except Exception as e:
                status = WebhookStatus.FAILED.value
                error = str(e)

            duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)

            # Log attempt
            conn.execute(
                """
                INSERT INTO webhook_logs (webhook_id, event_type, payload, status, status_code, response_body, attempt, duration_ms, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    webhook_id,
                    event_type,
                    json.dumps(payload),
                    status,
                    status_code,
                    response_body,
                    attempt,
                    duration_ms,
                    error,
                    now,
                ),
            )
            conn.commit()

            if status == WebhookStatus.SUCCESS.value:
                logger.info(f"[WEBHOOK] Delivered: id={webhook_id} status={status_code} duration={duration_ms}ms")
                return {
                    "success": True,
                    "webhook_id": webhook_id,
                    "status_code": status_code,
                    "attempt": attempt,
                    "duration_ms": duration_ms,
                }

            # Exponential backoff before retry
            if attempt < max_retries:
                await asyncio.sleep(2**attempt)

        logger.warning(f"[WEBHOOK] Failed after {max_retries} attempts: id={webhook_id} error={error}")
        return {
            "success": False,
            "webhook_id": webhook_id,
            "error": error,
            "attempts": max_retries,
        }

    async def test_webhook(self, webhook_id: int) -> dict[str, Any]:
        """Send a test payload to a webhook."""
        return await self.deliver_webhook(
            webhook_id=webhook_id,
            event_type="test",
            payload={
                "event": "test",
                "message": "This is a test webhook delivery",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get automation statistics."""
        conn = self._get_conn()

        rules_total = conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
        rules_enabled = conn.execute("SELECT COUNT(*) FROM rules WHERE enabled = 1").fetchone()[0]
        webhooks_total = conn.execute("SELECT COUNT(*) FROM webhooks").fetchone()[0]
        webhooks_enabled = conn.execute("SELECT COUNT(*) FROM webhooks WHERE enabled = 1").fetchone()[0]

        # Recent trigger stats (last 24h)
        triggers_24h = conn.execute(
            "SELECT COUNT(*) FROM trigger_logs WHERE created_at > datetime('now', '-1 day')"
        ).fetchone()[0]
        rules_fired_24h = conn.execute(
            "SELECT COALESCE(SUM(rules_fired), 0) FROM trigger_logs WHERE created_at > datetime('now', '-1 day')"
        ).fetchone()[0]

        # Webhook delivery stats (last 24h)
        webhook_success = conn.execute(
            "SELECT COUNT(*) FROM webhook_logs WHERE status = 'success' AND created_at > datetime('now', '-1 day')"
        ).fetchone()[0]
        webhook_failed = conn.execute(
            "SELECT COUNT(*) FROM webhook_logs WHERE status = 'failed' AND created_at > datetime('now', '-1 day')"
        ).fetchone()[0]

        return {
            "rules": {
                "total": rules_total,
                "enabled": rules_enabled,
            },
            "webhooks": {
                "total": webhooks_total,
                "enabled": webhooks_enabled,
            },
            "triggers_24h": triggers_24h,
            "rules_fired_24h": rules_fired_24h,
            "webhook_deliveries_24h": {
                "success": webhook_success,
                "failed": webhook_failed,
                "success_rate": round(webhook_success / max(webhook_success + webhook_failed, 1) * 100, 1),
            },
        }

    def get_trigger_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent trigger execution logs."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM trigger_logs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()

        return [
            {
                "id": r["id"],
                "trigger_type": r["trigger_type"],
                "rules_evaluated": r["rules_evaluated"],
                "rules_fired": r["rules_fired"],
                "actions_executed": r["actions_executed"],
                "duration_ms": r["duration_ms"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]


# Singleton instance
_automation_manager: AutomationManager | None = None


def get_automation_manager(db_path: str | None = None) -> AutomationManager:
    """Get or create the singleton AutomationManager instance."""
    global _automation_manager
    if _automation_manager is None:
        _automation_manager = AutomationManager(db_path or "/app/instance/automation_store.db")
    return _automation_manager
