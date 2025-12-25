"""
Request Audit Logger - Enterprise-Grade Request Tracking
=========================================================
Logs all API requests with timestamps, latencies, and metadata for anomaly detection.
Designed to be testable via CLI.

2025 Best Practices:
- Structured JSON logging (structlog)
- Correlation IDs for request tracing
- Anomaly detection thresholds
- Non-blocking async logging
"""

import asyncio
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

# Try structlog, fall back to standard logging
try:
    import structlog

    logger = structlog.get_logger()
    HAS_STRUCTLOG = True
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    HAS_STRUCTLOG = False


@dataclass
class RequestLog:
    """Single request log entry"""

    timestamp: str
    request_id: str
    method: str
    path: str
    status_code: int
    latency_ms: float
    user_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyReport:
    """Detected anomaly in request patterns"""

    anomaly_type: str
    path: str
    description: str
    severity: str  # low, medium, high, critical
    timestamp: str
    metrics: dict[str, Any] = field(default_factory=dict)


class RequestAuditLogger:
    """
    Enterprise request audit logger with anomaly detection.

    Features:
    - Per-endpoint latency tracking
    - Error rate monitoring
    - Request rate tracking
    - Automatic anomaly detection
    - JSON log file output
    """

    # Lax thresholds for development (10x normal limits)
    DEFAULT_THRESHOLDS = {
        "max_requests_per_minute": 1000,  # Very lax for dev
        "max_error_rate": 0.50,  # 50% error rate before alert
        "latency_stddev_multiplier": 5.0,  # 5 stddev before anomaly
        "min_samples_for_analysis": 5,
    }

    def __init__(
        self,
        log_dir: str | None = None,
        thresholds: dict[str, Any] | None = None,
        enable_anomaly_detection: bool = True,
    ):
        # Use env var or default to /app/instance/logs/audit (writable in container)
        if log_dir is None:
            log_dir = os.getenv("AUDIT_LOG_DIR", "/app/instance/logs/audit")
        self.log_dir = Path(log_dir)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Fallback to /tmp if instance dir not writable
            self.log_dir = Path("/tmp/audit_logs")
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.enable_anomaly_detection = enable_anomaly_detection

        # In-memory stats (last 10 minutes)
        self._request_logs: list[RequestLog] = []
        self._latencies_by_path: dict[str, list[float]] = defaultdict(list)
        self._errors_by_path: dict[str, int] = defaultdict(int)
        self._requests_by_path: dict[str, int] = defaultdict(int)
        self._requests_by_ip: dict[str, list[float]] = defaultdict(list)

        # Anomaly history
        self._anomalies: list[AnomalyReport] = []

        # Current log file
        self._current_log_file = self._get_log_file()

    def _get_log_file(self) -> Path:
        """Get current log file path (daily rotation)"""
        date_str = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"requests_{date_str}.jsonl"

    def _now_iso(self) -> str:
        """Current timestamp in ISO format"""
        return datetime.now().isoformat()

    async def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
        user_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        error: str | None = None,
        **extra,
    ) -> None:
        """Log a single request"""
        log_entry = RequestLog(
            timestamp=self._now_iso(),
            request_id=request_id,
            method=method,
            path=path,
            status_code=status_code,
            latency_ms=latency_ms,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            error=error,
            extra=extra,
        )

        # Add to in-memory stats
        self._request_logs.append(log_entry)
        self._latencies_by_path[path].append(latency_ms)
        self._requests_by_path[path] += 1

        if status_code >= 400:
            self._errors_by_path[path] += 1

        if ip_address:
            self._requests_by_ip[ip_address].append(time.time())

        # Write to file (non-blocking)
        await self._write_log(log_entry)

        # Run anomaly detection periodically
        if self.enable_anomaly_detection and len(self._request_logs) % 100 == 0:
            await self._detect_anomalies()

        # Prune old data (keep last 10 minutes)
        await self._prune_old_data()

    async def _write_log(self, log_entry: RequestLog) -> None:
        """
        Write log entry to file with HMAC chain for tamper detection.
        
        Each entry includes:
        - The log data itself
        - An HMAC seal computed from: log_data + previous_seal
        - This creates a verifiable chain where any modification is detectable
        """
        import hashlib
        import hmac
        
        log_file = self._get_log_file()
        
        # Get the HMAC key from secrets (fallback to env var or default for dev)
        hmac_key = os.getenv("AUDIT_HMAC_KEY", "")
        if not hmac_key:
            try:
                key_file = Path("/run/secrets/audit_hmac_key")
                if key_file.exists():
                    hmac_key = key_file.read_text().strip()
            except Exception:
                pass
        
        # Use a default dev key if none configured (warn in production)
        if not hmac_key:
            hmac_key = "dev-audit-key-not-for-production"
            if os.getenv("PRODUCTION", "").lower() in ("true", "1", "yes"):
                if HAS_STRUCTLOG:
                    logger.warning("audit_hmac_key_missing", msg="Using default HMAC key in production!")
                else:
                    logger.warning("Using default HMAC key in production - configure AUDIT_HMAC_KEY!")
        
        hmac_key_bytes = hmac_key.encode("utf-8")
        
        try:
            # Get the previous seal (from the last line of the log file)
            previous_seal = self._get_previous_seal(log_file)
            
            # Serialize log entry
            log_data = json.dumps(asdict(log_entry), default=str, sort_keys=True)
            
            # Compute HMAC seal: hash(log_data + previous_seal)
            message = f"{log_data}:{previous_seal}".encode("utf-8")
            current_seal = hmac.new(hmac_key_bytes, message, hashlib.sha256).hexdigest()
            
            # Create sealed log entry
            sealed_entry = {
                "data": asdict(log_entry),
                "seal": current_seal,
                "prev_seal": previous_seal[:16] + "..." if previous_seal else None,  # Truncated for readability
            }
            
            # Write atomically (append mode)
            with open(log_file, "a") as f:
                f.write(json.dumps(sealed_entry, default=str) + "\n")
            
            # Store current seal for next entry (in-memory cache)
            self._last_seal = current_seal
            
        except Exception as e:
            if HAS_STRUCTLOG:
                logger.error("log_write_failed", error=str(e))
            else:
                logger.error(f"Log write failed: {e}")
    
    def _get_previous_seal(self, log_file: Path) -> str:
        """Get the seal from the last log entry (for chain verification)."""
        if hasattr(self, "_last_seal") and self._last_seal:
            return self._last_seal
        
        if not log_file.exists() or log_file.stat().st_size == 0:
            return ""  # Genesis entry
        
        try:
            # Read the last line efficiently
            with open(log_file, "rb") as f:
                # Seek to end
                f.seek(0, 2)
                file_size = f.tell()
                
                if file_size == 0:
                    return ""
                
                # Read last 8KB to find last complete line
                chunk_size = min(8192, file_size)
                f.seek(file_size - chunk_size)
                last_chunk = f.read(chunk_size).decode("utf-8", errors="ignore")
                
                lines = last_chunk.strip().split("\n")
                if lines:
                    last_line = lines[-1]
                    try:
                        last_entry = json.loads(last_line)
                        return last_entry.get("seal", "")
                    except json.JSONDecodeError:
                        return ""
        except Exception:
            return ""
        
        return ""
    
    def verify_log_integrity(self, log_file: Path | None = None) -> tuple[bool, list[str]]:
        """
        Verify the integrity of audit log chain.
        
        Returns:
            Tuple of (is_valid, list of issues found)
        """
        import hashlib
        import hmac as hmac_module
        
        if log_file is None:
            log_file = self._get_log_file()
        
        issues = []
        
        # Get HMAC key
        hmac_key = os.getenv("AUDIT_HMAC_KEY", "dev-audit-key-not-for-production")
        hmac_key_bytes = hmac_key.encode("utf-8")
        
        if not log_file.exists():
            return True, ["No log file found"]
        
        try:
            previous_seal = ""
            line_number = 0
            
            with open(log_file, "r") as f:
                for line in f:
                    line_number += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        issues.append(f"Line {line_number}: Invalid JSON")
                        continue
                    
                    # Get stored seal
                    stored_seal = entry.get("seal", "")
                    entry_data = entry.get("data", {})
                    
                    # Recompute seal
                    log_data = json.dumps(entry_data, default=str, sort_keys=True)
                    message = f"{log_data}:{previous_seal}".encode("utf-8")
                    expected_seal = hmac_module.new(hmac_key_bytes, message, hashlib.sha256).hexdigest()
                    
                    if stored_seal != expected_seal:
                        issues.append(f"Line {line_number}: Seal mismatch - possible tampering!")
                    
                    previous_seal = stored_seal
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Verification error: {e}"]

    async def _prune_old_data(self) -> None:
        """Remove data older than 10 minutes"""
        cutoff = datetime.now() - timedelta(minutes=10)
        cutoff_str = cutoff.isoformat()

        self._request_logs = [log for log in self._request_logs if log.timestamp > cutoff_str]

        cutoff_ts = time.time() - 600
        for ip in list(self._requests_by_ip.keys()):
            self._requests_by_ip[ip] = [ts for ts in self._requests_by_ip[ip] if ts > cutoff_ts]

    async def _detect_anomalies(self) -> list[AnomalyReport]:
        """Detect anomalies in request patterns"""
        new_anomalies = []
        now = self._now_iso()

        for path, latencies in self._latencies_by_path.items():
            if len(latencies) < self.thresholds["min_samples_for_analysis"]:
                continue

            avg = mean(latencies)
            if len(latencies) >= 2:
                std = stdev(latencies)
                threshold = avg + (std * self.thresholds["latency_stddev_multiplier"])

                # Check for latency spike
                if latencies[-1] > threshold and latencies[-1] > 1000:
                    anomaly = AnomalyReport(
                        anomaly_type="latency_spike",
                        path=path,
                        description=f"Latency {latencies[-1]:.0f}ms exceeds threshold {threshold:.0f}ms",
                        severity="medium",
                        timestamp=now,
                        metrics={"latency": latencies[-1], "threshold": threshold, "avg": avg},
                    )
                    new_anomalies.append(anomaly)

            # Check error rate
            total = self._requests_by_path.get(path, 0)
            errors = self._errors_by_path.get(path, 0)
            if total > 10:
                error_rate = errors / total
                if error_rate > self.thresholds["max_error_rate"]:
                    anomaly = AnomalyReport(
                        anomaly_type="high_error_rate",
                        path=path,
                        description=f"Error rate {error_rate:.1%} exceeds threshold",
                        severity="high",
                        timestamp=now,
                        metrics={"error_rate": error_rate, "total": total, "errors": errors},
                    )
                    new_anomalies.append(anomaly)

        # Check per-IP rate limits
        for ip, timestamps in self._requests_by_ip.items():
            recent = [ts for ts in timestamps if ts > time.time() - 60]
            if len(recent) > self.thresholds["max_requests_per_minute"]:
                anomaly = AnomalyReport(
                    anomaly_type="rate_limit_warning",
                    path="*",
                    description=f"IP {ip} made {len(recent)} requests/min",
                    severity="low",
                    timestamp=now,
                    metrics={"ip": ip, "count": len(recent)},
                )
                new_anomalies.append(anomaly)

        self._anomalies.extend(new_anomalies)
        return new_anomalies

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics for CLI display"""
        stats = {
            "total_requests": len(self._request_logs),
            "requests_by_path": dict(self._requests_by_path),
            "errors_by_path": dict(self._errors_by_path),
            "anomalies_detected": len(self._anomalies),
            "latency_stats": {},
        }

        for path, latencies in self._latencies_by_path.items():
            if latencies:
                stats["latency_stats"][path] = {
                    "avg_ms": round(mean(latencies), 1),
                    "min_ms": round(min(latencies), 1),
                    "max_ms": round(max(latencies), 1),
                    "count": len(latencies),
                }

        return stats

    def get_anomalies(self, since: datetime | None = None) -> list[AnomalyReport]:
        """Get detected anomalies"""
        if since is None:
            return self._anomalies.copy()

        since_str = since.isoformat()
        return [a for a in self._anomalies if a.timestamp >= since_str]

    def clear_stats(self) -> None:
        """Clear in-memory stats (for testing)"""
        self._request_logs.clear()
        self._latencies_by_path.clear()
        self._errors_by_path.clear()
        self._requests_by_path.clear()
        self._requests_by_ip.clear()
        self._anomalies.clear()


# Singleton instance
_audit_logger: RequestAuditLogger | None = None


def get_audit_logger() -> RequestAuditLogger:
    """Get or create the global audit logger"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = RequestAuditLogger()
    return _audit_logger


# FastAPI middleware integration
async def audit_middleware(request, call_next):
    """FastAPI middleware for automatic request logging"""
    import uuid

    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()

    try:
        response = await call_next(request)
        status_code = response.status_code
        error = None
    except Exception as e:
        status_code = 500
        error = str(e)
        raise
    finally:
        latency_ms = (time.time() - start_time) * 1000

        audit = get_audit_logger()
        await audit.log_request(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            latency_ms=latency_ms,
            user_id=getattr(request.state, "user_id", None),
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
            error=error,
        )

    return response
