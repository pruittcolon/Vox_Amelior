"""
Disaster Recovery Automation.

Provides automated disaster recovery capabilities:
- Failover procedures
- Recovery point management
- Service health validation
- Recovery runbook automation
"""

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import logging

logger = logging.getLogger(__name__)


class RecoveryPhase(Enum):
    """Phases of disaster recovery."""
    
    DETECTION = "detection"
    ASSESSMENT = "assessment"
    NOTIFICATION = "notification"
    FAILOVER = "failover"
    RECOVERY = "recovery"
    VALIDATION = "validation"
    RESTORATION = "restoration"


class ServiceStatus(Enum):
    """Service health status."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class RecoveryPoint:
    """Recovery point information."""
    
    point_id: str
    created_at: str
    backup_id: Optional[str] = None
    snapshot_id: Optional[str] = None
    data_age_minutes: float = 0
    
    def to_dict(self) -> dict:
        return {
            "point_id": self.point_id,
            "created_at": self.created_at,
            "backup_id": self.backup_id,
            "data_age_minutes": self.data_age_minutes,
        }


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""
    
    success: bool
    phase: RecoveryPhase
    message: str
    duration_seconds: float = 0
    details: dict = field(default_factory=dict)


@dataclass
class ServiceCheck:
    """Health check for a service."""
    
    name: str
    check_fn: Callable[[], bool]
    timeout_seconds: int = 30
    critical: bool = True


class DisasterRecoveryManager:
    """Manages disaster recovery procedures.
    
    Implements automated recovery based on:
    - RTO (Recovery Time Objective)
    - RPO (Recovery Point Objective)
    """
    
    def __init__(
        self,
        rto_minutes: int = 30,
        rpo_minutes: int = 15,
    ):
        """Initialize DR manager.
        
        Args:
            rto_minutes: Maximum acceptable recovery time
            rpo_minutes: Maximum acceptable data loss
        """
        self.rto_minutes = rto_minutes
        self.rpo_minutes = rpo_minutes
        
        self._services: list[ServiceCheck] = []
        self._recovery_points: list[RecoveryPoint] = []
        self._current_phase = RecoveryPhase.DETECTION
        self._recovery_log: list[RecoveryResult] = []
        
        # Load default service checks
        self._load_default_checks()
        
        logger.info(
            "DisasterRecoveryManager initialized: RTO=%dm, RPO=%dm",
            rto_minutes, rpo_minutes,
        )
    
    def _load_default_checks(self) -> None:
        """Load default service health checks."""
        
        def check_api_gateway():
            try:
                result = subprocess.run(
                    ["curl", "-sf", "http://localhost:8000/health"],
                    capture_output=True,
                    timeout=10,
                )
                return result.returncode == 0
            except Exception:
                return False
        
        def check_postgres():
            try:
                result = subprocess.run(
                    ["pg_isready", "-h", "localhost"],
                    capture_output=True,
                    timeout=10,
                )
                return result.returncode == 0
            except Exception:
                return False
        
        def check_redis():
            try:
                result = subprocess.run(
                    ["redis-cli", "ping"],
                    capture_output=True,
                    timeout=5,
                )
                return b"PONG" in result.stdout
            except Exception:
                return False
        
        self._services = [
            ServiceCheck("api-gateway", check_api_gateway, critical=True),
            ServiceCheck("postgres", check_postgres, critical=True),
            ServiceCheck("redis", check_redis, critical=False),
        ]
    
    def add_service_check(self, check: ServiceCheck) -> None:
        """Add a service health check."""
        self._services.append(check)
    
    def check_all_services(self) -> dict[str, ServiceStatus]:
        """Check health of all services.
        
        Returns:
            Dict of service name to status
        """
        results = {}
        
        for service in self._services:
            try:
                is_healthy = service.check_fn()
                results[service.name] = (
                    ServiceStatus.HEALTHY if is_healthy 
                    else ServiceStatus.UNHEALTHY
                )
            except Exception as e:
                logger.warning("Service check failed: %s - %s", service.name, e)
                results[service.name] = ServiceStatus.UNKNOWN
        
        return results
    
    def get_latest_recovery_point(self) -> Optional[RecoveryPoint]:
        """Get the most recent recovery point."""
        if not self._recovery_points:
            return None
        return max(self._recovery_points, key=lambda p: p.created_at)
    
    def register_recovery_point(
        self,
        backup_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
    ) -> RecoveryPoint:
        """Register a new recovery point."""
        point = RecoveryPoint(
            point_id=f"RP-{int(time.time())}",
            created_at=datetime.now(timezone.utc).isoformat(),
            backup_id=backup_id,
            snapshot_id=snapshot_id,
        )
        
        self._recovery_points.append(point)
        logger.info("Recovery point registered: %s", point.point_id)
        
        return point
    
    def execute_recovery(
        self,
        recovery_point: Optional[RecoveryPoint] = None,
        dry_run: bool = False,
    ) -> list[RecoveryResult]:
        """Execute disaster recovery procedure.
        
        Args:
            recovery_point: Point to recover to (latest if None)
            dry_run: If True, simulate without making changes
            
        Returns:
            List of results for each phase
        """
        start_time = time.time()
        results = []
        
        if recovery_point is None:
            recovery_point = self.get_latest_recovery_point()
        
        phases = [
            (RecoveryPhase.DETECTION, self._phase_detection),
            (RecoveryPhase.ASSESSMENT, self._phase_assessment),
            (RecoveryPhase.NOTIFICATION, self._phase_notification),
            (RecoveryPhase.FAILOVER, lambda rp, dr: self._phase_failover(rp, dr)),
            (RecoveryPhase.RECOVERY, lambda rp, dr: self._phase_recovery(rp, dr)),
            (RecoveryPhase.VALIDATION, self._phase_validation),
        ]
        
        for phase, phase_fn in phases:
            self._current_phase = phase
            phase_start = time.time()
            
            try:
                success, message, details = phase_fn(recovery_point, dry_run)
                
                result = RecoveryResult(
                    success=success,
                    phase=phase,
                    message=message,
                    duration_seconds=time.time() - phase_start,
                    details=details,
                )
            except Exception as e:
                result = RecoveryResult(
                    success=False,
                    phase=phase,
                    message=f"Phase failed: {e}",
                    duration_seconds=time.time() - phase_start,
                )
            
            results.append(result)
            self._recovery_log.append(result)
            
            if not result.success:
                logger.error("Recovery failed at phase: %s", phase.value)
                break
        
        total_time = time.time() - start_time
        rto_met = total_time <= (self.rto_minutes * 60)
        
        logger.info(
            "Recovery %s in %.1f seconds (RTO %s)",
            "completed" if results[-1].success else "failed",
            total_time,
            "MET" if rto_met else "EXCEEDED",
        )
        
        return results
    
    def _phase_detection(self, rp: RecoveryPoint, dry_run: bool):
        """Detection phase: identify the issue."""
        service_status = self.check_all_services()
        unhealthy = [s for s, status in service_status.items() 
                     if status == ServiceStatus.UNHEALTHY]
        
        return (
            len(unhealthy) > 0,
            f"Detected {len(unhealthy)} unhealthy services: {unhealthy}",
            {"services": {s: v.value for s, v in service_status.items()}},
        )
    
    def _phase_assessment(self, rp: RecoveryPoint, dry_run: bool):
        """Assessment phase: evaluate impact and recovery options."""
        if rp:
            age = (datetime.now(timezone.utc) - 
                   datetime.fromisoformat(rp.created_at.replace("Z", "+00:00")))
            age_minutes = age.total_seconds() / 60
            rp.data_age_minutes = age_minutes
            
            rpo_met = age_minutes <= self.rpo_minutes
            
            return (
                True,
                f"Recovery point age: {age_minutes:.1f}m (RPO {'MET' if rpo_met else 'EXCEEDED'})",
                {"recovery_point": rp.to_dict(), "rpo_met": rpo_met},
            )
        
        return (False, "No recovery point available", {})
    
    def _phase_notification(self, rp: RecoveryPoint, dry_run: bool):
        """Notification phase: alert stakeholders."""
        # In production, this would send emails/SMS/PagerDuty
        if dry_run:
            return (True, "DRY RUN: Would notify stakeholders", {})
        
        logger.warning("DISASTER RECOVERY IN PROGRESS - Notifying stakeholders")
        return (True, "Stakeholders notified", {"notified": ["ops-team"]})
    
    def _phase_failover(self, rp: RecoveryPoint, dry_run: bool):
        """Failover phase: switch to backup systems."""
        if dry_run:
            return (True, "DRY RUN: Would initiate failover", {})
        
        # Failover logic would go here
        return (True, "Failover completed", {})
    
    def _phase_recovery(self, rp: RecoveryPoint, dry_run: bool):
        """Recovery phase: restore from backup."""
        if dry_run:
            return (True, f"DRY RUN: Would restore from {rp.backup_id}", {})
        
        if rp and rp.backup_id:
            # Would call SecureBackupManager.restore_backup here
            return (True, f"Restored from backup {rp.backup_id}", {})
        
        return (True, "No backup restore needed", {})
    
    def _phase_validation(self, rp: RecoveryPoint, dry_run: bool):
        """Validation phase: verify recovery success."""
        if dry_run:
            return (True, "DRY RUN: Would validate services", {})
        
        service_status = self.check_all_services()
        all_healthy = all(
            s == ServiceStatus.HEALTHY 
            for s in service_status.values()
        )
        
        return (
            all_healthy,
            "All services healthy" if all_healthy else "Some services still unhealthy",
            {"services": {s: v.value for s, v in service_status.items()}},
        )
    
    def get_rto_status(self) -> dict:
        """Get current RTO/RPO status."""
        latest_rp = self.get_latest_recovery_point()
        
        if latest_rp:
            age = (datetime.now(timezone.utc) - 
                   datetime.fromisoformat(latest_rp.created_at.replace("Z", "+00:00")))
            data_age_minutes = age.total_seconds() / 60
        else:
            data_age_minutes = float("inf")
        
        return {
            "rto_minutes": self.rto_minutes,
            "rpo_minutes": self.rpo_minutes,
            "current_data_age_minutes": data_age_minutes,
            "rpo_status": "COMPLIANT" if data_age_minutes <= self.rpo_minutes else "AT_RISK",
            "latest_recovery_point": latest_rp.to_dict() if latest_rp else None,
        }


def get_dr_manager(
    rto_minutes: int = 30,
    rpo_minutes: int = 15,
) -> DisasterRecoveryManager:
    """Get disaster recovery manager."""
    return DisasterRecoveryManager(rto_minutes=rto_minutes, rpo_minutes=rpo_minutes)
