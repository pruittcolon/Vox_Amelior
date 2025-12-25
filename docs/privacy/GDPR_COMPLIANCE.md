# GDPR Compliance Documentation

## Overview

This document outlines Nemo Server's compliance with the General Data Protection Regulation (GDPR) for processing personal data of EU residents.

---

## Lawful Basis for Processing

| Data Type | Lawful Basis | Justification |
|-----------|--------------|---------------|
| User credentials | Contractual necessity | Required for service access |
| Usage analytics | Legitimate interest | Service improvement |
| Transcriptions | Consent | User-initiated |
| AI chat logs | Consent | User-initiated conversations |

---

## Data Subject Rights

### Right of Access (Article 15)
- **Endpoint**: `GET /api/v1/user/data-export`
- **Response Time**: 30 days maximum
- **Format**: JSON export

### Right to Rectification (Article 16)
- **Endpoint**: `PUT /api/v1/user/profile`
- Users can update personal information

### Right to Erasure (Article 17)
- **Endpoint**: `DELETE /api/v1/user/account`
- Cascading deletion of all user data
- Tenant admin can delete tenant data

### Right to Data Portability (Article 20)
- **Endpoint**: `GET /api/v1/user/data-export?format=portable`
- Machine-readable JSON format

---

## Data Processing Inventory

| Category | Data Elements | Retention | Processor |
|----------|---------------|-----------|-----------|
| Authentication | Username, email, password hash | Account lifetime | Internal |
| Transcription | Audio files, transcripts | 30 days | Internal |
| AI Chat | Messages, responses | 90 days | Internal |
| Analytics | Usage metrics | 1 year | Internal |
| Audit Logs | Request metadata, security events | 1 year | Internal |

---

## Technical Measures (Article 32)

### Encryption
| Layer | Implementation | Evidence |
|-------|---------------|----------|
| In Transit | TLS 1.3 + HSTS preload | `docker/nginx/` |
| At Rest | AES-256 for sensitive data | Docker secrets |
| Session Tokens | AES-GCM (AEAD) | `auth_manager.py` |

### Access Control
- Role-Based Access Control (RBAC): `shared/auth/rbac.py`
- Tenant isolation: `shared/models/tenant.py`
- Session management with CSRF protection: `core/middleware.py`
- JWT with JTI replay protection: `service_auth.py`

### PII Protection
- Pre-ingestion PII detection: `shared/security/pii_detector.py`
- Multiple redaction strategies: MASK, HASH, REMOVE
- Configurable sensitivity levels per data type

### Audit Logging (Accountability)
- **Location**: `shared/audit/__init__.py`
- **Tamper Detection**: HMAC chain sealing for log integrity
- **Verification**: `verify_log_integrity()` detects modifications
- **Fields Logged**: Timestamp, request_id, user_id, IP, endpoint, status

---

## Data Processing Agreements (DPA)

Required for all subprocessors:
- Cloud hosting provider
- Database services
- Third-party integrations (Salesforce, Fiserv)

---

## Breach Notification Procedure

1. **Detection**: Automated monitoring + incident detection
2. **Assessment**: Within 24 hours
3. **Authority Notification**: Within 72 hours if applicable
4. **User Notification**: Without undue delay if high risk
5. **Documentation**: Audit log preserved as evidence

Runbook: `docs/runbooks/incident_response.md`

---

## Privacy by Design

Implemented throughout:
- Data minimization in API responses
- Purpose limitation in data collection
- Default privacy-preserving settings
- AI guardrails for content safety: `shared/security/guardrails.py`
- Security headers (CSP, Permissions-Policy): `main.py`

---

*Last Updated: 2024-12-24*

