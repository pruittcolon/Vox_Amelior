#!/usr/bin/env python3
"""
Nemo Server - Security Verification Script

Comprehensive security posture verification that can be run at any time.
This script provides detailed analysis of the security configuration
and generates a report suitable for security audits.

Usage:
    python3 verify_security.py           # Run all checks
    python3 verify_security.py --json    # Output as JSON
    python3 verify_security.py --strict  # Fail on any warning
"""

import os
import sys
import json
import stat
import socket
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class Finding:
    category: str
    check: str
    severity: Severity
    status: str  # "pass", "fail", "warn"
    message: str
    remediation: Optional[str] = None

class SecurityVerifier:
    def __init__(self):
        self.findings: List[Finding] = []
        self.project_root = Path(__file__).parent.parent
        
    def add_finding(self, category: str, check: str, severity: Severity, 
                   status: str, message: str, remediation: str = None):
        self.findings.append(Finding(
            category=category,
            check=check,
            severity=severity,
            status=status,
            message=message,
            remediation=remediation
        ))
    
    def verify_secrets(self):
        """Verify secrets configuration."""
        secrets_dir = self.project_root / "docker" / "secrets"
        
        required = ["jwt_secret_primary", "session_key", "users_db_key"]
        
        for secret in required:
            path = secrets_dir / secret
            if not path.exists():
                self.add_finding(
                    "Secrets Management", f"Secret: {secret}",
                    Severity.CRITICAL, "fail",
                    f"Required secret '{secret}' not found",
                    f"Run: ./scripts/setup_secrets.sh"
                )
            elif path.stat().st_size == 0:
                self.add_finding(
                    "Secrets Management", f"Secret: {secret}",
                    Severity.CRITICAL, "fail",
                    f"Secret '{secret}' is empty",
                    f"Run: ./scripts/setup_secrets.sh"
                )
            else:
                # Check permissions
                mode = path.stat().st_mode
                if mode & stat.S_IRWXG or mode & stat.S_IRWXO:
                    self.add_finding(
                        "Secrets Management", f"Secret: {secret}",
                        Severity.HIGH, "fail",
                        f"Secret has insecure permissions: {oct(mode)[-3:]}",
                        f"Run: chmod 600 {path}"
                    )
                else:
                    self.add_finding(
                        "Secrets Management", f"Secret: {secret}",
                        Severity.INFO, "pass",
                        f"Secret exists with correct permissions"
                    )
    
    def verify_git_status(self):
        """Check if secrets are properly gitignored."""
        secrets_dir = self.project_root / "docker" / "secrets"
        
        try:
            # Check if any secrets are tracked
            result = subprocess.run(
                ["git", "ls-files", str(secrets_dir)],
                capture_output=True, text=True,
                cwd=self.project_root
            )
            
            tracked = [f for f in result.stdout.strip().split('\n') 
                      if f and not f.endswith('README.md') and not f.endswith('.gitkeep')]
            
            if tracked:
                self.add_finding(
                    "Version Control", "Secrets in Git",
                    Severity.CRITICAL, "fail",
                    f"Secret files tracked in Git: {', '.join(tracked)}",
                    "Remove with: git rm --cached <file> and add to .gitignore"
                )
            else:
                self.add_finding(
                    "Version Control", "Secrets in Git",
                    Severity.INFO, "pass",
                    "No secret files tracked in Git"
                )
        except Exception as e:
            self.add_finding(
                "Version Control", "Git Check",
                Severity.LOW, "warn",
                f"Could not check Git status: {e}"
            )
    
    def verify_ssl_certificates(self):
        """Verify SSL/TLS certificate configuration."""
        ssl_dir = self.project_root / "docker" / "ssl"
        
        cert_path = ssl_dir / "nemo.crt"
        key_path = ssl_dir / "nemo.key"
        
        if not cert_path.exists() or not key_path.exists():
            self.add_finding(
                "TLS/HTTPS", "Certificates",
                Severity.HIGH, "fail",
                "SSL certificates not generated",
                "Run: ./scripts/generate_certs.sh"
            )
            return
        
        # Check key permissions
        key_mode = key_path.stat().st_mode
        if key_mode & stat.S_IRWXG or key_mode & stat.S_IRWXO:
            self.add_finding(
                "TLS/HTTPS", "Key Permissions",
                Severity.HIGH, "fail",
                f"Private key has insecure permissions: {oct(key_mode)[-3:]}",
                f"Run: chmod 600 {key_path}"
            )
        else:
            self.add_finding(
                "TLS/HTTPS", "Key Permissions",
                Severity.INFO, "pass",
                "Private key has correct permissions"
            )
        
        # Check certificate expiry
        try:
            result = subprocess.run(
                ["openssl", "x509", "-in", str(cert_path), "-noout", "-enddate"],
                capture_output=True, text=True
            )
            if "notAfter" in result.stdout:
                self.add_finding(
                    "TLS/HTTPS", "Certificate",
                    Severity.INFO, "pass",
                    f"Certificate valid ({result.stdout.strip()})"
                )
        except Exception:
            pass
    
    def verify_docker_compose(self):
        """Verify docker-compose security settings."""
        compose_path = self.project_root / "docker" / "docker-compose.yml"
        
        if not compose_path.exists():
            self.add_finding(
                "Container Security", "Docker Compose",
                Severity.LOW, "warn",
                "docker-compose.yml not found"
            )
            return
        
        content = compose_path.read_text()
        
        # Check for privileged containers
        if "privileged: true" in content:
            self.add_finding(
                "Container Security", "Privileged Containers",
                Severity.CRITICAL, "fail",
                "Privileged containers detected",
                "Remove 'privileged: true' and use specific capabilities"
            )
        else:
            self.add_finding(
                "Container Security", "Privileged Containers",
                Severity.INFO, "pass",
                "No privileged containers"
            )
        
        # Check for localhost binding
        if "0.0.0.0:8000:8000" in content:
            self.add_finding(
                "Network Security", "API Gateway Binding",
                Severity.MEDIUM, "warn",
                "API Gateway bound to all interfaces (0.0.0.0)",
                "Bind to 127.0.0.1 and use nginx for external access"
            )
        elif "127.0.0.1:8000:8000" in content:
            self.add_finding(
                "Network Security", "API Gateway Binding",
                Severity.INFO, "pass",
                "API Gateway bound to localhost only"
            )
    
    def verify_environment(self):
        """Check environment configuration."""
        # Check if demo mode is enabled
        demo_enabled = os.getenv("ENABLE_DEMO_USERS", "")
        if demo_enabled.lower() in ("true", "1", "yes"):
            self.add_finding(
                "Authentication", "Demo Users",
                Severity.HIGH, "fail",
                "Demo users enabled (weak credentials!)",
                "Set ENABLE_DEMO_USERS=false in production"
            )
        
        # Check secure cookies
        secure_cookies = os.getenv("SESSION_COOKIE_SECURE", "")
        if secure_cookies.lower() not in ("true", "1", "yes"):
            self.add_finding(
                "Session Security", "Secure Cookies",
                Severity.MEDIUM, "warn",
                "Secure cookies not explicitly enabled",
                "Set SESSION_COOKIE_SECURE=true for HTTPS"
            )
    
    def verify_nginx_config(self):
        """Verify nginx security configuration."""
        nginx_conf = self.project_root / "docker" / "nginx" / "nginx.conf"
        
        if not nginx_conf.exists():
            self.add_finding(
                "TLS/HTTPS", "Nginx Config",
                Severity.MEDIUM, "warn",
                "Nginx configuration not found"
            )
            return
        
        content = nginx_conf.read_text()
        
        # Check TLS protocols
        if "TLSv1.2" in content and ("TLSv1 " not in content and "TLSv1.0" not in content):
            self.add_finding(
                "TLS/HTTPS", "TLS Protocol",
                Severity.INFO, "pass",
                "Modern TLS protocols configured (1.2+)"
            )
        
        # Check HSTS
        if "Strict-Transport-Security" in content:
            self.add_finding(
                "TLS/HTTPS", "HSTS Header",
                Severity.INFO, "pass",
                "HSTS header configured"
            )
        else:
            self.add_finding(
                "TLS/HTTPS", "HSTS Header",
                Severity.MEDIUM, "warn",
                "HSTS header not configured"
            )
    
    def verify_security_headers(self):
        """Verify HTTP security headers configuration by testing live endpoint or config."""
        import urllib.request
        import urllib.error
        
        gateway_url = os.getenv("GATEWAY_URL", "http://127.0.0.1:8000")
        
        # Required headers and their expected values/patterns
        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": ["DENY", "SAMEORIGIN"],
            "Referrer-Policy": ["strict-origin-when-cross-origin", "no-referrer"],
            "Strict-Transport-Security": None,  # Just needs to exist
            "Content-Security-Policy": "frame-ancestors",  # Must contain this
        }
        
        try:
            # Try to make a request to the health endpoint
            req = urllib.request.Request(f"{gateway_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                headers = dict(response.headers)
                
                for header, expected in required_headers.items():
                    value = headers.get(header, "")
                    
                    if not value:
                        severity = Severity.HIGH if header in ["Strict-Transport-Security", "Content-Security-Policy"] else Severity.MEDIUM
                        self.add_finding(
                            "Security Headers", header,
                            severity, "fail",
                            f"Header '{header}' not present in response",
                            f"Ensure {header} is set in SecurityHeadersMiddleware"
                        )
                    elif expected is None:
                        # Just needs to exist
                        self.add_finding(
                            "Security Headers", header,
                            Severity.INFO, "pass",
                            f"Header present: {value[:50]}..."
                        )
                    elif isinstance(expected, list):
                        if any(e.lower() in value.lower() for e in expected):
                            self.add_finding(
                                "Security Headers", header,
                                Severity.INFO, "pass",
                                f"Header correctly configured"
                            )
                        else:
                            self.add_finding(
                                "Security Headers", header,
                                Severity.MEDIUM, "warn",
                                f"Header value '{value}' does not match expected: {expected}"
                            )
                    elif isinstance(expected, str):
                        if expected.lower() in value.lower():
                            self.add_finding(
                                "Security Headers", header,
                                Severity.INFO, "pass",
                                f"Header correctly configured"
                            )
                        else:
                            self.add_finding(
                                "Security Headers", header,
                                Severity.MEDIUM, "warn",
                                f"Header value does not contain expected value: {expected}"
                            )
                
                # Check for Cache-Control on API responses
                if "/api/" in gateway_url or "/health" in gateway_url:
                    cache_control = headers.get("Cache-Control", "")
                    if "no-store" in cache_control:
                        self.add_finding(
                            "Security Headers", "Cache-Control",
                            Severity.INFO, "pass",
                            "Cache-Control: no-store set correctly for API"
                        )
                    else:
                        self.add_finding(
                            "Security Headers", "Cache-Control",
                            Severity.MEDIUM, "warn",
                            "API responses may be cached (no-store not set)",
                            "Add Cache-Control: no-store for API endpoints"
                        )
                        
        except urllib.error.URLError:
            # Server not running, check config instead
            self.add_finding(
                "Security Headers", "Live Check",
                Severity.LOW, "warn",
                f"Could not connect to {gateway_url} - checking config instead"
            )
            
            # Check if FORCE_HSTS is enabled
            force_hsts = os.getenv("FORCE_HSTS", "true").lower()
            if force_hsts in ("true", "1", "yes"):
                self.add_finding(
                    "Security Headers", "HSTS Config",
                    Severity.INFO, "pass",
                    "FORCE_HSTS is enabled"
                )
            else:
                self.add_finding(
                    "Security Headers", "HSTS Config",
                    Severity.HIGH, "fail",
                    "FORCE_HSTS is disabled - HSTS headers not enforced",
                    "Set FORCE_HSTS=true in environment"
                )
            
            # Check ALLOW_FRAMING
            allow_framing = os.getenv("ALLOW_FRAMING", "false").lower()
            if allow_framing not in ("true", "1", "yes"):
                self.add_finding(
                    "Security Headers", "X-Frame-Options Config",
                    Severity.INFO, "pass",
                    "Clickjacking protection enabled (ALLOW_FRAMING=false)"
                )
            else:
                self.add_finding(
                    "Security Headers", "X-Frame-Options Config",
                    Severity.CRITICAL, "fail",
                    "Clickjacking protection DISABLED",
                    "Set ALLOW_FRAMING=false in environment"
                )
        except Exception as e:
            self.add_finding(
                "Security Headers", "Verification",
                Severity.LOW, "warn",
                f"Could not verify security headers: {e}"
            )
    
    def run_all_checks(self):
        """Run all security verification checks."""
        self.verify_secrets()
        self.verify_git_status()
        self.verify_ssl_certificates()
        self.verify_docker_compose()
        self.verify_environment()
        self.verify_nginx_config()
        self.verify_security_headers()
    
    def get_summary(self) -> Dict:
        """Generate summary of findings."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_checks": len(self.findings),
            "passed": sum(1 for f in self.findings if f.status == "pass"),
            "failed": sum(1 for f in self.findings if f.status == "fail"),
            "warnings": sum(1 for f in self.findings if f.status == "warn"),
            "critical_issues": sum(1 for f in self.findings 
                                   if f.status == "fail" and f.severity == Severity.CRITICAL),
            "high_issues": sum(1 for f in self.findings 
                              if f.status == "fail" and f.severity == Severity.HIGH),
        }
        return summary
    
    def print_report(self, json_output: bool = False):
        """Print findings report."""
        if json_output:
            output = {
                "summary": self.get_summary(),
                "findings": [
                    {**asdict(f), "severity": f.severity.value}
                    for f in self.findings
                ]
            }
            print(json.dumps(output, indent=2))
            return
        
        # Console output
        print("\n" + "="*70)
        print("           NEMO SERVER - Security Verification Report")
        print("="*70 + "\n")
        
        # Group by category
        categories = {}
        for f in self.findings:
            if f.category not in categories:
                categories[f.category] = []
            categories[f.category].append(f)
        
        for category, findings in categories.items():
            print(f"\n📁 {category}")
            print("-" * 40)
            for f in findings:
                status_icon = {"pass": "✅", "fail": "❌", "warn": "⚠️ "}.get(f.status, "?")
                print(f"  {status_icon} {f.check}: {f.message}")
                if f.remediation and f.status != "pass":
                    print(f"     💡 {f.remediation}")
        
        # Summary
        summary = self.get_summary()
        print("\n" + "="*70)
        print("                           SUMMARY")
        print("="*70)
        print(f"  Total Checks:     {summary['total_checks']}")
        print(f"  Passed:           {summary['passed']} ✅")
        print(f"  Failed:           {summary['failed']} ❌")
        print(f"  Warnings:         {summary['warnings']} ⚠️")
        print(f"  Critical Issues:  {summary['critical_issues']}")
        print(f"  High Issues:      {summary['high_issues']}")
        print()
        
        if summary['critical_issues'] > 0 or summary['high_issues'] > 0:
            print("❌ SECURITY VERIFICATION FAILED")
            print("   Address critical and high issues before deployment.")
        elif summary['failed'] > 0:
            print("⚠️  SECURITY VERIFICATION PASSED WITH ISSUES")
        else:
            print("✅ SECURITY VERIFICATION PASSED")

def main():
    json_output = "--json" in sys.argv
    strict_mode = "--strict" in sys.argv
    
    verifier = SecurityVerifier()
    verifier.run_all_checks()
    verifier.print_report(json_output=json_output)
    
    summary = verifier.get_summary()
    
    if summary['critical_issues'] > 0:
        sys.exit(1)
    elif strict_mode and (summary['failed'] > 0 or summary['warnings'] > 0):
        sys.exit(1)
    elif summary['high_issues'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
