# Security Policy

## Vulnerability Reporting

### Preferred Method
We use GitHub Private Vulnerability Reporting.
1. Navigate to **Security** → **Advisories** → **Report a vulnerability**
2. Fill out the report form.
3. We will respond within **48 hours**.

### Alternative Contact
If GitHub PVR is not available, contact: **security@voxamelior.dev**

## Supported Versions

| Version | Status |
| ------- | ------------------ |
| 1.x.x   | Supported |
| < 1.0   | End of Life |

## Security Architecture

This project implements a Defense in Depth strategy:

### 1. Transport Security
- **TLS 1.2+** enforced (ECDHE, CHACHA20, AES-GCM).
- **HSTS** enabled with 1-year max-age.
- **mTLS** service mesh using Istio for all internal traffic.

### 2. Authentication & Authorization
- **JWT** with key rotation and replay protection (Redis-backed JTI).
- **SPIFFE/SPIRE** workload identities.
- **RBAC** for service-to-service communication.

### 3. Data Protection
- **Secrets Management**: Docker secrets (mounted at `/run/secrets/` with 600 permissions).
- **Encryption**: AES-256-GCM for backups and sensitive data at rest.
- **FIPS 140-2**: Compliant algorithms for cryptographic operations.

### 4. Application Security
- **Container Hardening**: Read-only root filesystems, capabilities dropped (`cap_drop: [ALL]`), non-root execution.
- **WAF**: OWASP CRS-based detection rules.
- **Input Validation**: Pydantic models and strict serialization.

## Compliance

### SOC 2 & Audit
- **Audit Logging**: CEF format, HMAC-sealed logs for immutability.
- **Policy Engine**: Automated enforcement of 12 security policies.

## Verification

To verify the security posture:

```bash
# Run security hardening check
python3 scripts/security_hardening.py

# Run comprehensive security verification
python3 scripts/verify_security.py
```

## Safe Harbor

We support safe harbor for security researchers who:
- Make good faith efforts to avoid privacy violations.
- Avoid disruption to production systems.
- Report findings privately.

## Scope

**In Scope**:
- Authentication/Authorization flaws
- Data exposure
- Injection vulnerabilities
- Cryptographic weaknesses

**Out of Scope**:
- Denial of Service (DoS)
- Social Engineering
