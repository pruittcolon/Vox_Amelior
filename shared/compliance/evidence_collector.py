"""
Evidence Collector for Compliance Audits.

Automatically collects and organizes evidence for:
- SOC 2 Type II audits
- Security assessments
- Compliance reviews

Evidence types:
- Configuration snapshots
- Access logs
- Security scan results
- Policy enforcement records
"""

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import logging

logger = logging.getLogger(__name__)


class EvidenceType(Enum):
    """Types of compliance evidence."""
    
    CONFIGURATION = "configuration"
    ACCESS_LOG = "access_log"
    SECURITY_SCAN = "security_scan"
    POLICY_RECORD = "policy_record"
    CHANGE_RECORD = "change_record"
    TEST_RESULT = "test_result"
    CERTIFICATE = "certificate"
    SCREENSHOT = "screenshot"


class SOC2Control(Enum):
    """SOC 2 Common Criteria controls."""
    
    CC1_1 = "CC1.1"  # COSO Principle 1
    CC2_1 = "CC2.1"  # Communication
    CC3_1 = "CC3.1"  # Risk Assessment
    CC4_1 = "CC4.1"  # Monitoring
    CC5_1 = "CC5.1"  # Control Activities
    CC6_1 = "CC6.1"  # Logical Access
    CC6_2 = "CC6.2"  # System Access
    CC6_3 = "CC6.3"  # Access Removal
    CC6_6 = "CC6.6"  # Access Restrictions
    CC6_7 = "CC6.7"  # Data Transmission
    CC7_1 = "CC7.1"  # Configuration Management
    CC7_2 = "CC7.2"  # Change Management
    CC7_3 = "CC7.3"  # Security Event Detection
    CC7_4 = "CC7.4"  # Incident Response
    CC8_1 = "CC8.1"  # Change Management
    CC9_1 = "CC9.1"  # Risk Mitigation


@dataclass
class Evidence:
    """A piece of compliance evidence."""
    
    evidence_id: str
    evidence_type: EvidenceType
    title: str
    description: str
    collected_at: str
    collected_by: str = "system"
    
    # SOC 2 mapping
    controls: list[str] = field(default_factory=list)
    
    # Content
    content: Optional[str] = None
    content_hash: Optional[str] = None
    file_path: Optional[str] = None
    
    # Metadata
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["evidence_type"] = self.evidence_type.value
        return data


class EvidenceCollector:
    """Collects compliance evidence automatically."""
    
    def __init__(
        self,
        evidence_path: Optional[str] = None,
        organization: str = "Nemo Platform",
    ):
        """Initialize the collector.
        
        Args:
            evidence_path: Directory to store evidence
            organization: Organization name for records
        """
        self.evidence_path = Path(
            evidence_path or os.getenv("EVIDENCE_PATH", "/var/log/nemo/evidence")
        )
        self.organization = organization
        self._evidence_counter = 0
        
        # Create evidence directory if it doesn't exist
        self.evidence_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("EvidenceCollector initialized: path=%s", self.evidence_path)
    
    def _generate_id(self) -> str:
        """Generate unique evidence ID."""
        self._evidence_counter += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"EVD-{timestamp}-{self._evidence_counter:04d}"
    
    def _hash_content(self, content: str) -> str:
        """Generate SHA-256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def collect(
        self,
        evidence_type: EvidenceType,
        title: str,
        description: str,
        content: str,
        controls: Optional[list[SOC2Control]] = None,
        metadata: Optional[dict] = None,
    ) -> Evidence:
        """Collect a piece of evidence.
        
        Args:
            evidence_type: Type of evidence
            title: Evidence title
            description: What this evidence demonstrates
            content: The evidence content
            controls: SOC 2 controls this maps to
            metadata: Additional metadata
            
        Returns:
            The collected Evidence record
        """
        evidence_id = self._generate_id()
        
        evidence = Evidence(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            title=title,
            description=description,
            collected_at=datetime.now(timezone.utc).isoformat(),
            controls=[c.value for c in (controls or [])],
            content=content[:10000] if len(content) > 10000 else content,  # Truncate
            content_hash=self._hash_content(content),
            metadata=metadata or {},
        )
        
        # Save to file
        self._save_evidence(evidence)
        
        logger.info("Collected evidence: %s - %s", evidence_id, title)
        return evidence
    
    def _save_evidence(self, evidence: Evidence) -> None:
        """Save evidence to file."""
        # Create dated directory
        date_dir = self.evidence_path / datetime.now().strftime("%Y/%m")
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON record
        file_path = date_dir / f"{evidence.evidence_id}.json"
        file_path.write_text(json.dumps(evidence.to_dict(), indent=2))
        evidence.file_path = str(file_path)
    
    def collect_docker_config(self) -> Evidence:
        """Collect Docker container configuration."""
        try:
            result = subprocess.run(
                ["docker", "compose", "config"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            content = result.stdout if result.returncode == 0 else "Failed to collect"
        except Exception as e:
            content = f"Error: {e}"
        
        return self.collect(
            evidence_type=EvidenceType.CONFIGURATION,
            title="Docker Compose Configuration",
            description="Current Docker container configuration showing security settings",
            content=content,
            controls=[SOC2Control.CC7_1, SOC2Control.CC6_1],
        )
    
    def collect_security_headers(self, url: str = "http://localhost:8000") -> Evidence:
        """Collect security headers from API."""
        try:
            result = subprocess.run(
                ["curl", "-sI", url],
                capture_output=True,
                text=True,
                timeout=10,
            )
            content = result.stdout
        except Exception as e:
            content = f"Error: {e}"
        
        return self.collect(
            evidence_type=EvidenceType.SECURITY_SCAN,
            title="HTTP Security Headers",
            description="Security headers returned by the API",
            content=content,
            controls=[SOC2Control.CC6_7],
        )
    
    def collect_tls_config(self) -> Evidence:
        """Collect TLS configuration."""
        try:
            nginx_conf = Path("/etc/nginx/nginx.conf")
            if nginx_conf.exists():
                content = nginx_conf.read_text()
            else:
                content = "Nginx config not found at standard path"
        except Exception as e:
            content = f"Error: {e}"
        
        return self.collect(
            evidence_type=EvidenceType.CONFIGURATION,
            title="TLS/SSL Configuration",
            description="TLS configuration showing cipher suites and protocol versions",
            content=content,
            controls=[SOC2Control.CC6_7, SOC2Control.CC6_1],
        )
    
    def collect_test_results(self, test_output: str, test_name: str) -> Evidence:
        """Collect security test results."""
        return self.collect(
            evidence_type=EvidenceType.TEST_RESULT,
            title=f"Security Tests: {test_name}",
            description="Automated security test execution results",
            content=test_output,
            controls=[SOC2Control.CC4_1, SOC2Control.CC7_3],
        )
    
    def collect_access_review(self, users: list[dict]) -> Evidence:
        """Collect user access review."""
        content = json.dumps(users, indent=2)
        return self.collect(
            evidence_type=EvidenceType.ACCESS_LOG,
            title="User Access Review",
            description="Current user access levels and permissions",
            content=content,
            controls=[SOC2Control.CC6_1, SOC2Control.CC6_2, SOC2Control.CC6_3],
        )
    
    def generate_evidence_index(self) -> dict:
        """Generate an index of all collected evidence."""
        evidence_files = list(self.evidence_path.rglob("EVD-*.json"))
        
        index = {
            "organization": self.organization,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_evidence": len(evidence_files),
            "by_control": {},
            "by_type": {},
            "evidence": [],
        }
        
        for file_path in evidence_files:
            try:
                data = json.loads(file_path.read_text())
                index["evidence"].append({
                    "id": data["evidence_id"],
                    "title": data["title"],
                    "type": data["evidence_type"],
                    "controls": data["controls"],
                    "collected_at": data["collected_at"],
                })
                
                # Group by control
                for control in data.get("controls", []):
                    index["by_control"].setdefault(control, []).append(data["evidence_id"])
                
                # Group by type
                etype = data["evidence_type"]
                index["by_type"].setdefault(etype, []).append(data["evidence_id"])
                
            except Exception as e:
                logger.warning("Failed to index %s: %s", file_path, e)
        
        return index


# Singleton
_collector: Optional[EvidenceCollector] = None


def get_evidence_collector() -> EvidenceCollector:
    """Get or create global evidence collector."""
    global _collector
    
    if _collector is None:
        _collector = EvidenceCollector()
    
    return _collector
