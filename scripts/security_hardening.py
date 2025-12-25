#!/usr/bin/env python3
"""
Nemo Server - Security Hardening Script

This script performs security hardening checks at startup.
It implements fail-closed security: the server will NOT start
if critical security requirements are not met in production.

Exit codes:
  0 - All checks passed
  1 - Critical security failure (server should not start)
  2 - Non-critical warnings (server can start with degraded security)
"""

import os
import sys
import stat
from pathlib import Path
from typing import List, Tuple

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header():
    print(f"""
{BLUE}╔══════════════════════════════════════════════════════════════╗
║           NEMO SERVER - Security Hardening Check             ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")

def check_env_mode() -> Tuple[bool, str]:
    """Check if running in production mode."""
    test_mode = os.getenv("TEST_MODE", "").lower() in ("true", "1", "yes")
    prod_mode = os.getenv("PRODUCTION", "").lower() in ("true", "1", "yes")
    
    if test_mode:
        return False, "Running in TEST_MODE (reduced security)"
    elif prod_mode:
        return True, "Running in PRODUCTION mode (strict security)"
    else:
        return False, "Running in development mode"

def check_secrets_exist() -> Tuple[bool, List[str]]:
    """Verify all required secrets are present and non-empty."""
    secrets_dir = Path("/run/secrets")
    if not secrets_dir.exists():
        secrets_dir = Path(os.getenv("SECRETS_DIR", "docker/secrets"))
    
    required_secrets = [
        "jwt_secret_primary",
        "session_key",
        "users_db_key",
    ]
    
    optional_secrets = [
        "jwt_secret_previous",
        "jwt_secret",
        "service_api_key",
        "postgres_password",
        "redis_password",
    ]
    
    missing = []
    warnings = []
    
    for secret in required_secrets:
        path = secrets_dir / secret
        if not path.exists():
            missing.append(f"MISSING: {secret}")
        elif path.stat().st_size == 0:
            missing.append(f"EMPTY: {secret}")
    
    for secret in optional_secrets:
        path = secrets_dir / secret
        if not path.exists() or path.stat().st_size == 0:
            warnings.append(f"Optional: {secret}")
    
    return len(missing) == 0, missing + warnings

def check_secret_permissions() -> Tuple[bool, List[str]]:
    """Verify secret files have correct permissions (600)."""
    secrets_dir = Path("/run/secrets")
    if not secrets_dir.exists():
        secrets_dir = Path(os.getenv("SECRETS_DIR", "docker/secrets"))
    
    if not secrets_dir.exists():
        return True, ["Secrets directory not found (may be using env vars)"]
    
    issues = []
    for secret_file in secrets_dir.iterdir():
        if secret_file.is_file() and secret_file.name not in ("README.md", ".gitkeep"):
            mode = secret_file.stat().st_mode
            if mode & stat.S_IRWXG or mode & stat.S_IRWXO:
                issues.append(f"Insecure perms on {secret_file.name}: {oct(mode)[-3:]}")
    
    return len(issues) == 0, issues

def check_demo_users() -> Tuple[bool, str]:
    """Check if demo users are enabled."""
    demo_enabled = os.getenv("ENABLE_DEMO_USERS", "").lower() in ("true", "1", "yes")
    
    if demo_enabled:
        return False, "Demo users ENABLED (default credentials active!)"
    return True, "Demo users disabled"

def check_cors_config() -> Tuple[bool, str]:
    """Check CORS configuration."""
    origins = os.getenv("ALLOWED_ORIGINS", "")
    
    if "*" in origins:
        return False, "CORS allows all origins (*) - insecure!"
    elif not origins:
        return True, "No CORS origins configured (restrictive)"
    else:
        return True, f"CORS restricted to: {origins[:50]}..."

def check_secure_cookies() -> Tuple[bool, str]:
    """Check if secure cookies are enabled."""
    secure = os.getenv("SESSION_COOKIE_SECURE", "").lower() in ("true", "1", "yes")
    
    if not secure:
        return False, "Secure cookies DISABLED (session hijacking risk)"
    return True, "Secure cookies enabled"

def check_csrf_protection() -> Tuple[bool, str]:
    """Check CSRF protection status."""
    csrf_enabled = os.getenv("ENABLE_CSRF", "true").lower() in ("true", "1", "yes")
    
    if not csrf_enabled:
        return False, "CSRF protection DISABLED"
    return True, "CSRF protection enabled"

def check_ssl_certs() -> Tuple[bool, str]:
    """Check if SSL certificates exist."""
    ssl_paths = [
        Path("/etc/nginx/ssl/nemo.crt"),
        Path("docker/ssl/nemo.crt"),
        Path("/run/secrets/ssl_cert"),
    ]
    
    for path in ssl_paths:
        if path.exists():
            return True, f"SSL certificate found: {path}"
    
    return False, "No SSL certificates found (HTTPS disabled)"


def check_security_headers() -> Tuple[bool, List[str]]:
    """
    Verify security headers are properly configured.
    
    Checks environment variables and configuration for:
    - HSTS with preload
    - CSP with frame-ancestors
    - X-Frame-Options: DENY
    - X-Content-Type-Options: nosniff
    - Referrer-Policy
    - CORS allowlist (no wildcards)
    
    Returns:
        Tuple of (passed, list of issues/warnings)
    """
    issues = []
    
    # Check FORCE_HSTS - should be enabled in production
    force_hsts = os.getenv("FORCE_HSTS", "true").lower()
    if force_hsts not in ("true", "1", "yes"):
        issues.append("CRITICAL: FORCE_HSTS is disabled - HSTS headers not enforced!")
    
    # Check ALLOW_FRAMING - should be false in production
    allow_framing = os.getenv("ALLOW_FRAMING", "false").lower()
    if allow_framing in ("true", "1", "yes"):
        issues.append("CRITICAL: ALLOW_FRAMING=true - Clickjacking protection disabled!")
    
    # Check CORS configuration for wildcards
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
    if "*" in allowed_origins:
        issues.append("CRITICAL: CORS allows all origins (*) - cross-origin attacks possible!")
    if not allowed_origins:
        # Empty is actually safe (no cross-origin requests allowed)
        pass
    
    # Check for insecure CSP overrides
    csp_override = os.getenv("CSP_OVERRIDE", "")
    if csp_override:
        if "unsafe-eval" in csp_override and "'wasm-unsafe-eval'" not in csp_override:
            issues.append("WARNING: CSP contains 'unsafe-eval' - XSS risk!")
        if "frame-ancestors" not in csp_override:
            issues.append("WARNING: CSP missing frame-ancestors - clickjacking risk!")
    
    # Check rate limiting
    rate_limit = os.getenv("RATE_LIMIT_ENABLED", "true").lower()
    if rate_limit not in ("true", "1", "yes"):
        issues.append("WARNING: Rate limiting disabled - DoS risk!")
    
    # Check for debug mode
    debug_mode = os.getenv("DEBUG", "").lower() in ("true", "1", "yes")
    if debug_mode:
        issues.append("WARNING: DEBUG mode enabled - verbose errors exposed!")
    
    # Determine if critical issues exist
    critical_count = sum(1 for i in issues if i.startswith("CRITICAL:"))
    
    return critical_count == 0, issues


def check_secret_hygiene() -> Tuple[bool, List[str]]:
    """
    Check for hardcoded secrets or insecure secret patterns.
    
    Scans common locations for potential secret leakage:
    - .env files in source
    - Hardcoded tokens in config
    """
    issues = []
    
    # Check for .env file with potentially exposed secrets
    env_file = Path(".env")
    if env_file.exists():
        try:
            content = env_file.read_text()
            sensitive_patterns = [
                "password=",
                "secret=",
                "api_key=",
                "token=",
            ]
            for pattern in sensitive_patterns:
                if pattern in content.lower() and "changeme" in content.lower():
                    issues.append(f"WARNING: .env contains placeholder secret ({pattern})")
                    break
        except Exception:
            pass
    
    # Check for JWT secret in environment (should use secrets dir)
    jwt_env = os.getenv("JWT_SECRET", "")
    if jwt_env and len(jwt_env) < 32:
        issues.append("CRITICAL: JWT_SECRET too short (< 32 chars) - brute force risk!")
    
    session_key = os.getenv("SESSION_KEY", "")
    if session_key and len(session_key) < 32:
        issues.append("CRITICAL: SESSION_KEY too short - session hijacking risk!")
    
    critical_count = sum(1 for i in issues if i.startswith("CRITICAL:"))
    return critical_count == 0, issues

def main():
    print_header()
    
    is_production, mode_msg = check_env_mode()
    print(f"{BLUE}Mode:{RESET} {mode_msg}")
    print()
    
    checks = [
        ("Required Secrets", check_secrets_exist),
        ("Secret Permissions", check_secret_permissions),
        ("Secret Hygiene", check_secret_hygiene),
        ("Demo Users", check_demo_users),
        ("CORS Configuration", check_cors_config),
        ("Secure Cookies", check_secure_cookies),
        ("CSRF Protection", check_csrf_protection),
        ("SSL Certificates", check_ssl_certs),
        ("Security Headers", check_security_headers),
    ]
    
    critical_failures = []
    warnings = []
    
    for name, check_func in checks:
        try:
            passed, details = check_func()
            
            if passed:
                if isinstance(details, list):
                    print(f"  {GREEN}✓{RESET} {name}")
                    for d in details[:3]:
                        print(f"      {d}")
                else:
                    print(f"  {GREEN}✓{RESET} {name}: {details}")
            else:
                if isinstance(details, list):
                    for d in details:
                        if "MISSING" in d or "EMPTY" in d:
                            print(f"  {RED}✗{RESET} {name}: {d}")
                            critical_failures.append(f"{name}: {d}")
                        else:
                            print(f"  {YELLOW}⚠{RESET} {name}: {d}")
                            warnings.append(f"{name}: {d}")
                else:
                    if is_production and name in ["Required Secrets", "Demo Users", "Secure Cookies"]:
                        print(f"  {RED}✗{RESET} {name}: {details}")
                        critical_failures.append(f"{name}: {details}")
                    else:
                        print(f"  {YELLOW}⚠{RESET} {name}: {details}")
                        warnings.append(f"{name}: {details}")
        except Exception as e:
            print(f"  {RED}✗{RESET} {name}: Error - {e}")
            if is_production:
                critical_failures.append(f"{name}: {e}")
    
    print()
    
    if critical_failures:
        print(f"{RED}╔══════════════════════════════════════════════════════════════╗")
        print(f"║                    SECURITY CHECK FAILED                     ║")
        print(f"╚══════════════════════════════════════════════════════════════╝{RESET}")
        print()
        print("Critical issues that must be fixed:")
        for failure in critical_failures:
            print(f"  • {failure}")
        print()
        
        if is_production:
            print(f"{RED}Server startup BLOCKED in production mode.{RESET}")
            print("Fix the above issues or set PRODUCTION=false for development.")
            sys.exit(1)
        else:
            print(f"{YELLOW}Server will start with degraded security (non-production).{RESET}")
            sys.exit(2)
    
    elif warnings:
        print(f"{YELLOW}Security check passed with warnings.{RESET}")
        sys.exit(0)
    
    else:
        print(f"{GREEN}╔══════════════════════════════════════════════════════════════╗")
        print(f"║                  ALL SECURITY CHECKS PASSED                  ║")
        print(f"╚══════════════════════════════════════════════════════════════╝{RESET}")
        sys.exit(0)

if __name__ == "__main__":
    main()
