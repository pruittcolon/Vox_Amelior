#!/usr/bin/env python3
"""
Security Assessment Tool.

Performs comprehensive security assessment for certification:
- Configuration review
- Vulnerability scanning
- Policy compliance
- Best practices check

This is the final security gate before certification.
Exit codes:
- 0: Ready for certification
- 1: Issues found
- 2: Configuration error
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).parent.parent


@dataclass
class AssessmentFinding:
    """A security assessment finding."""
    
    category: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    title: str
    description: str
    recommendation: str
    evidence: Optional[str] = None


@dataclass
class AssessmentResult:
    """Complete assessment result."""
    
    timestamp: str
    overall_score: float  # 0-10
    ready_for_cert: bool
    findings: list[AssessmentFinding] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_score": self.overall_score,
            "ready_for_cert": self.ready_for_cert,
            "finding_count": len(self.findings),
            "findings_by_severity": self._count_by_severity(),
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity,
                    "title": f.title,
                    "description": f.description,
                    "recommendation": f.recommendation,
                }
                for f in self.findings
            ],
        }
    
    def _count_by_severity(self) -> dict:
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts


class SecurityAssessment:
    """Comprehensive security assessment."""
    
    def __init__(self):
        """Initialize assessment."""
        self.findings: list[AssessmentFinding] = []
        self._checks_run = 0
        self._checks_passed = 0
    
    def _add_finding(
        self,
        category: str,
        severity: str,
        title: str,
        description: str,
        recommendation: str,
        evidence: Optional[str] = None,
    ) -> None:
        """Add a finding."""
        self.findings.append(AssessmentFinding(
            category=category,
            severity=severity,
            title=title,
            description=description,
            recommendation=recommendation,
            evidence=evidence,
        ))
    
    def check_tls_configuration(self) -> None:
        """Check TLS/SSL configuration."""
        self._checks_run += 1
        
        # Check nginx config
        nginx_configs = list(REPO_ROOT.glob("**/nginx*.conf"))
        
        tls_1_0 = False
        tls_1_1 = False
        weak_ciphers = False
        
        for config in nginx_configs:
            content = config.read_text()
            
            if "TLSv1 " in content or "TLSv1.0" in content:
                tls_1_0 = True
            if "TLSv1.1" in content:
                tls_1_1 = True
            if any(c in content for c in ["DES", "RC4", "MD5", "NULL"]):
                weak_ciphers = True
        
        if tls_1_0:
            self._add_finding(
                "TLS", "HIGH", "TLS 1.0 Enabled",
                "TLS 1.0 is deprecated and insecure",
                "Disable TLS 1.0 in nginx configuration",
            )
        else:
            self._checks_passed += 1
        
        if tls_1_1:
            self._add_finding(
                "TLS", "MEDIUM", "TLS 1.1 Enabled",
                "TLS 1.1 is deprecated",
                "Disable TLS 1.1 in nginx configuration",
            )
        
        if weak_ciphers:
            self._add_finding(
                "TLS", "HIGH", "Weak Cipher Suites",
                "Insecure cipher suites detected",
                "Remove DES, RC4, MD5, NULL ciphers",
            )
    
    def check_security_headers(self) -> None:
        """Check security headers are configured."""
        self._checks_run += 1
        
        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY|SAMEORIGIN",
            "Strict-Transport-Security": "max-age",
            "Content-Security-Policy": "default-src",
            "X-XSS-Protection": "1",
        }
        
        found_headers = set()
        
        # Search in gateway and nginx configs
        search_paths = [
            REPO_ROOT / "services/api-gateway",
            REPO_ROOT / "docker",
        ]
        
        for path in search_paths:
            if path.exists():
                for file in path.rglob("*.py"):
                    content = file.read_text()
                    for header in required_headers:
                        if header.lower() in content.lower():
                            found_headers.add(header)
                
                for file in path.rglob("*.conf"):
                    content = file.read_text()
                    for header in required_headers:
                        if header.lower() in content.lower():
                            found_headers.add(header)
        
        missing = set(required_headers.keys()) - found_headers
        
        if missing:
            self._add_finding(
                "Headers", "MEDIUM", "Missing Security Headers",
                f"Headers not configured: {', '.join(missing)}",
                "Add all security headers to HTTP responses",
            )
        else:
            self._checks_passed += 1
    
    def check_secrets_management(self) -> None:
        """Check secrets are not hardcoded."""
        self._checks_run += 1
        
        secret_patterns = [
            r"password\s*=\s*['\"][^'\"]{8,}['\"]",
            r"secret\s*=\s*['\"][^'\"]{8,}['\"]",
            r"api_key\s*=\s*['\"][^'\"]{8,}['\"]",
            r"-----BEGIN.*PRIVATE KEY-----",
        ]
        
        issues = []
        
        for py_file in REPO_ROOT.rglob("*.py"):
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
            
            try:
                content = py_file.read_text()
                for pattern in secret_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Skip if it's a test or example
                        if "test" in str(py_file).lower() or "example" in match.lower():
                            continue
                        issues.append(f"{py_file.name}: {match[:50]}...")
            except Exception:
                continue
        
        if issues:
            self._add_finding(
                "Secrets", "CRITICAL", "Hardcoded Secrets Found",
                f"Found {len(issues)} potential hardcoded secrets",
                "Move all secrets to environment variables or Docker secrets",
                evidence="\n".join(issues[:5]),
            )
        else:
            self._checks_passed += 1
    
    def check_authentication(self) -> None:
        """Check authentication configuration."""
        self._checks_run += 1
        
        # Check for JWT usage
        jwt_found = False
        mfa_found = False
        lockout_found = False
        
        security_path = REPO_ROOT / "shared/security"
        if security_path.exists():
            for py_file in security_path.glob("*.py"):
                content = py_file.read_text()
                if "jwt" in content.lower():
                    jwt_found = True
                if "mfa" in py_file.name.lower() or "totp" in content.lower():
                    mfa_found = True
                if "lockout" in py_file.name.lower():
                    lockout_found = True
        
        if not jwt_found:
            self._add_finding(
                "Auth", "HIGH", "JWT Not Implemented",
                "Token-based authentication not found",
                "Implement JWT-based authentication",
            )
        
        if not mfa_found:
            self._add_finding(
                "Auth", "MEDIUM", "MFA Not Implemented",
                "Multi-factor authentication not found",
                "Implement TOTP-based MFA",
            )
        
        if not lockout_found:
            self._add_finding(
                "Auth", "MEDIUM", "Account Lockout Missing",
                "Brute force protection not found",
                "Implement account lockout after failed attempts",
            )
        
        if jwt_found and mfa_found and lockout_found:
            self._checks_passed += 1
    
    def check_container_security(self) -> None:
        """Check Docker container security."""
        self._checks_run += 1
        
        compose_files = list(REPO_ROOT.glob("docker-compose*.yml"))
        
        issues = []
        
        for compose in compose_files:
            content = compose.read_text()
            
            if "privileged: true" in content:
                issues.append("Container running in privileged mode")
            
            if "network_mode: host" in content:
                issues.append("Container using host network")
            
            if "cap_add:" in content and "SYS_ADMIN" in content:
                issues.append("Container has SYS_ADMIN capability")
        
        if issues:
            self._add_finding(
                "Container", "HIGH", "Insecure Container Configuration",
                "; ".join(issues),
                "Apply container hardening: non-root, read-only, cap_drop ALL",
            )
        else:
            self._checks_passed += 1
    
    def check_input_validation(self) -> None:
        """Check input validation is implemented."""
        self._checks_run += 1
        
        has_pydantic = False
        has_xss_protection = False
        has_sql_protection = False
        
        for py_file in REPO_ROOT.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            try:
                content = py_file.read_text()
                if "from pydantic" in content or "BaseModel" in content:
                    has_pydantic = True
                if "xss" in content.lower() or "sanitize" in content.lower():
                    has_xss_protection = True
                if "sqlalchemy" in content.lower() or "orm" in content.lower():
                    has_sql_protection = True
            except Exception:
                continue
        
        if not has_pydantic:
            self._add_finding(
                "Validation", "MEDIUM", "No Input Validation Framework",
                "Pydantic or similar validation not found",
                "Implement Pydantic models for all API inputs",
            )
        
        if not has_xss_protection:
            self._add_finding(
                "Validation", "HIGH", "XSS Protection Missing",
                "No XSS sanitization found",
                "Implement HTML/JS sanitization for user inputs",
            )
        
        if has_pydantic and has_xss_protection:
            self._checks_passed += 1
    
    def check_security_tests(self) -> None:
        """Verify security tests exist."""
        self._checks_run += 1
        
        test_path = REPO_ROOT / "tests/security"
        
        if not test_path.exists():
            self._add_finding(
                "Testing", "HIGH", "No Security Tests",
                "Security test directory not found",
                "Create comprehensive security test suite",
            )
            return
        
        test_files = list(test_path.glob("test_*.py"))
        
        if len(test_files) < 3:
            self._add_finding(
                "Testing", "MEDIUM", "Insufficient Security Tests",
                f"Only {len(test_files)} security test files found",
                "Expand security test coverage",
            )
        else:
            self._checks_passed += 1
    
    def run_assessment(self) -> AssessmentResult:
        """Run complete security assessment."""
        print("=" * 60)
        print("🔒 Security Assessment for Certification")
        print("=" * 60)
        
        checks = [
            ("TLS Configuration", self.check_tls_configuration),
            ("Security Headers", self.check_security_headers),
            ("Secrets Management", self.check_secrets_management),
            ("Authentication", self.check_authentication),
            ("Container Security", self.check_container_security),
            ("Input Validation", self.check_input_validation),
            ("Security Tests", self.check_security_tests),
        ]
        
        for name, check in checks:
            print(f"\n⏳ Checking: {name}...")
            check()
        
        # Calculate score
        if self._checks_run > 0:
            base_score = (self._checks_passed / self._checks_run) * 10
        else:
            base_score = 0
        
        # Penalty for critical findings
        critical_count = sum(1 for f in self.findings if f.severity == "CRITICAL")
        high_count = sum(1 for f in self.findings if f.severity == "HIGH")
        
        score = max(0, base_score - (critical_count * 2) - (high_count * 0.5))
        
        ready_for_cert = critical_count == 0 and high_count <= 1 and score >= 8.0
        
        result = AssessmentResult(
            timestamp=datetime.now().isoformat(),
            overall_score=round(score, 1),
            ready_for_cert=ready_for_cert,
            findings=self.findings,
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Assessment Summary")
        print("=" * 60)
        print(f"Score: {score:.1f}/10")
        print(f"Checks: {self._checks_passed}/{self._checks_run} passed")
        print(f"Findings: {len(self.findings)}")
        print(f"  - Critical: {critical_count}")
        print(f"  - High: {high_count}")
        print(f"  - Medium: {sum(1 for f in self.findings if f.severity == 'MEDIUM')}")
        print(f"  - Low: {sum(1 for f in self.findings if f.severity == 'LOW')}")
        print(f"\n{'✅ READY FOR CERTIFICATION' if ready_for_cert else '❌ NOT READY FOR CERTIFICATION'}")
        print("=" * 60)
        
        return result


def main():
    """Run security assessment."""
    assessment = SecurityAssessment()
    result = assessment.run_assessment()
    
    # Save report
    report_path = Path("security-assessment-report.json")
    report_path.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"\n📄 Report saved to: {report_path}")
    
    sys.exit(0 if result.ready_for_cert else 1)


if __name__ == "__main__":
    main()
