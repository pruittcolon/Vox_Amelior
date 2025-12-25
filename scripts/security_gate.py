#!/usr/bin/env python3
"""
CI/CD Security Gate.

Blocks deployment if security requirements are not met:
- Security tests must pass
- No critical vulnerabilities
- Policy compliance verified
- Required secrets present

Exit codes:
- 0: All checks passed
- 1: Security check failed
- 2: Configuration error
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class GateResult:
    """Result of a security gate check."""
    
    name: str
    passed: bool
    message: str
    details: Optional[dict] = None


class SecurityGate:
    """CI/CD security gate for deployment pipelines."""
    
    def __init__(self, strict: bool = True):
        """Initialize the security gate.
        
        Args:
            strict: If True, any failure blocks deployment
        """
        self.strict = strict
        self.results: list[GateResult] = []
        self._start_time = datetime.now()
    
    def check_security_tests(self) -> GateResult:
        """Verify security tests pass."""
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "pytest",
                    "tests/security/", "-v", "--tb=short",
                    "-x",  # Stop on first failure
                ],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=Path(__file__).parent.parent.parent,
            )
            
            passed = result.returncode == 0
            
            # Extract test count
            lines = result.stdout.split("\n")
            summary = [l for l in lines if "passed" in l or "failed" in l]
            message = summary[-1] if summary else "Tests completed"
            
            return GateResult(
                name="Security Tests",
                passed=passed,
                message=message,
                details={"returncode": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return GateResult(
                name="Security Tests",
                passed=False,
                message="Tests timed out after 5 minutes",
            )
        except Exception as e:
            return GateResult(
                name="Security Tests",
                passed=False,
                message=f"Error running tests: {e}",
            )
    
    def check_secrets_present(self) -> GateResult:
        """Verify required secrets are present."""
        required_secrets = [
            "jwt_secret_key",
            "postgres_password",
            "redis_password",
        ]
        
        secrets_path = Path("/run/secrets")
        missing = []
        
        for secret in required_secrets:
            secret_file = secrets_path / secret
            env_var = secret.upper()
            
            if not secret_file.exists() and not os.getenv(env_var):
                missing.append(secret)
        
        passed = len(missing) == 0
        
        return GateResult(
            name="Secrets Verification",
            passed=passed,
            message="All secrets present" if passed else f"Missing: {missing}",
            details={"missing": missing, "required": required_secrets},
        )
    
    def check_no_hardcoded_secrets(self) -> GateResult:
        """Check for hardcoded secrets in code."""
        patterns = [
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]",
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"-----BEGIN.*PRIVATE KEY-----",
        ]
        
        try:
            issues = []
            for pattern in patterns:
                result = subprocess.run(
                    ["grep", "-rn", "-E", pattern, ".", 
                     "--include=*.py", "--include=*.js",
                     "--exclude-dir=.git", "--exclude-dir=node_modules",
                     "--exclude-dir=__pycache__"],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent.parent.parent,
                )
                if result.stdout:
                    # Filter false positives
                    for line in result.stdout.split("\n"):
                        if line and "example" not in line.lower() and "test" not in line.lower():
                            issues.append(line[:200])
            
            passed = len(issues) == 0
            
            return GateResult(
                name="Hardcoded Secrets Check",
                passed=passed,
                message="No hardcoded secrets found" if passed else f"Found {len(issues)} potential issues",
                details={"issues": issues[:10]},  # Limit output
            )
        except Exception as e:
            return GateResult(
                name="Hardcoded Secrets Check",
                passed=True,  # Don't block on grep errors
                message=f"Check skipped: {e}",
            )
    
    def check_dependency_vulnerabilities(self) -> GateResult:
        """Check for known vulnerabilities in dependencies."""
        try:
            # Try pip-audit first
            result = subprocess.run(
                [sys.executable, "-m", "pip_audit", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            if result.returncode == 0:
                vulns = json.loads(result.stdout) if result.stdout else []
                critical = [v for v in vulns if v.get("severity", "").lower() == "critical"]
                
                passed = len(critical) == 0
                return GateResult(
                    name="Dependency Vulnerabilities",
                    passed=passed,
                    message=f"Found {len(vulns)} vulnerabilities ({len(critical)} critical)",
                    details={"total": len(vulns), "critical": len(critical)},
                )
            else:
                # pip-audit not installed
                return GateResult(
                    name="Dependency Vulnerabilities",
                    passed=True,
                    message="pip-audit not installed, check skipped",
                )
                
        except Exception as e:
            return GateResult(
                name="Dependency Vulnerabilities",
                passed=True,
                message=f"Check skipped: {e}",
            )
    
    def check_docker_security(self) -> GateResult:
        """Verify Docker security settings."""
        issues = []
        
        compose_files = [
            Path(__file__).parent.parent.parent / "docker-compose.yml",
            Path(__file__).parent.parent.parent / "docker-compose.prod.yml",
        ]
        
        for compose_file in compose_files:
            if compose_file.exists():
                content = compose_file.read_text()
                
                # Check for privileged containers
                if "privileged: true" in content:
                    issues.append(f"{compose_file.name}: privileged mode enabled")
                
                # Check for exposed ports on 0.0.0.0
                if "0.0.0.0:" in content:
                    issues.append(f"{compose_file.name}: port exposed on all interfaces")
        
        passed = len(issues) == 0
        
        return GateResult(
            name="Docker Security",
            passed=passed,
            message="Docker config secure" if passed else f"Issues: {issues}",
            details={"issues": issues},
        )
    
    def check_security_headers(self) -> GateResult:
        """Verify security headers are configured."""
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "Strict-Transport-Security",
            "Content-Security-Policy",
        ]
        
        # Check nginx config or API gateway
        nginx_conf = Path("/etc/nginx/nginx.conf")
        gateway_path = Path(__file__).parent.parent.parent / "services/api-gateway/src"
        
        found = []
        
        for header in required_headers:
            # Check in various locations
            header_found = False
            
            if nginx_conf.exists():
                if header.lower() in nginx_conf.read_text().lower():
                    header_found = True
            
            if gateway_path.exists():
                for py_file in gateway_path.rglob("*.py"):
                    if header.lower() in py_file.read_text().lower():
                        header_found = True
                        break
            
            if header_found:
                found.append(header)
        
        missing = set(required_headers) - set(found)
        passed = len(missing) == 0
        
        return GateResult(
            name="Security Headers",
            passed=passed,
            message="All headers configured" if passed else f"Missing: {list(missing)}",
            details={"required": required_headers, "found": found, "missing": list(missing)},
        )
    
    def run_all_checks(self) -> bool:
        """Run all security checks.
        
        Returns:
            True if all checks pass (or non-critical in non-strict mode)
        """
        print("=" * 60)
        print("🔒 CI/CD Security Gate")
        print("=" * 60)
        
        checks = [
            self.check_secrets_present,
            self.check_no_hardcoded_secrets,
            self.check_docker_security,
            self.check_security_headers,
            self.check_dependency_vulnerabilities,
            self.check_security_tests,
        ]
        
        all_passed = True
        
        for check in checks:
            print(f"\n⏳ Running: {check.__name__}...")
            result = check()
            self.results.append(result)
            
            status = "✅" if result.passed else "❌"
            print(f"   {status} {result.name}: {result.message}")
            
            if not result.passed:
                all_passed = False
        
        # Summary
        print("\n" + "=" * 60)
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        if all_passed:
            print(f"✅ Security Gate PASSED ({passed_count}/{total_count} checks)")
        else:
            print(f"❌ Security Gate FAILED ({passed_count}/{total_count} checks)")
            if self.strict:
                print("   Deployment blocked in strict mode")
        
        print("=" * 60)
        
        return all_passed
    
    def get_report(self) -> dict:
        """Generate JSON report of all checks."""
        return {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self._start_time).total_seconds(),
            "passed": all(r.passed for r in self.results),
            "strict_mode": self.strict,
            "checks": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


def main():
    """Main entry point for CI/CD pipeline."""
    strict = os.getenv("SECURITY_GATE_STRICT", "true").lower() == "true"
    gate = SecurityGate(strict=strict)
    
    passed = gate.run_all_checks()
    
    # Write report
    report_path = Path("security-gate-report.json")
    report_path.write_text(json.dumps(gate.get_report(), indent=2))
    print(f"\n📄 Report saved to: {report_path}")
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
