# PII Handling Policy

## Overview

This document defines procedures for handling Personally Identifiable Information (PII) within the Nemo Server platform in compliance with GDPR, SOC2, and enterprise security requirements.

---

## PII Classification

### Definition

PII includes any data that can identify an individual directly or indirectly:

| Category | Examples | Classification |
|----------|----------|----------------|
| Direct Identifiers | SSN, email, phone, name | RESTRICTED |
| Financial | Credit card, account number | RESTRICTED |
| Biometric | Voice recordings, embeddings | RESTRICTED |
| Technical | IP address, device ID | RESTRICTED |
| Behavioral | Usage patterns, transcripts | CONFIDENTIAL |

---

## Technical Controls

### Detection (`shared/security/pii_detector.py`)

The PII detector scans all data before ingestion:

```python
from shared.security.pii_detector import PIIDetector

detector = PIIDetector()
result = detector.scan(text)
# Returns: {"has_pii": True, "types": ["email", "phone"], "locations": [...]}
```

### Redaction Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `MASK` | Replace with `***` | Display to users |
| `HASH` | SHA-256 hash | Analytics (preserves uniqueness) |
| `REMOVE` | Delete entirely | Storage/export |
| `TOKENIZE` | Replace with reversible token | Processing with recoverable data |

### Classification (`shared/security/data_classification.py`)

All fields are classified automatically:

```python
from shared.security.data_classification import classify_field, ClassificationLevel

level = classify_field("email")  # Returns ClassificationLevel.RESTRICTED
```

---

## Handling Procedures

### Ingestion

1. **Scan**: All incoming data passes through `PIIDetector`
2. **Classify**: Fields assigned classification level
3. **Encrypt**: RESTRICTED/CONFIDENTIAL data encrypted at rest
4. **Log**: Audit event recorded (without PII content)

### Processing

1. **Access Control**: RBAC enforced per classification level
2. **Tenant Isolation**: Data scoped to tenant context
3. **Minimization**: Only necessary PII accessed

### Export (Data Subject Rights)

1. **Verify Identity**: Authentication required
2. **Scope to User**: Only requester's data exported
3. **Format**: JSON with optional redaction
4. **Audit**: Export event logged

### Deletion (Right to Erasure)

1. **Request Validation**: Verify data subject identity
2. **Cascading Delete**: All related records removed
3. **Backup Purge**: Scheduled removal from backups
4. **Confirmation**: Audit log entry

---

## Retention

| Data Type | Retention Period | Legal Basis |
|-----------|-----------------|-------------|
| User credentials | Account lifetime | Contractual |
| Transcriptions | 30 days | Consent |
| AI chat logs | 90 days | Consent |
| Audit logs | 1 year | Legitimate interest |
| Anonymized analytics | Indefinite | Legitimate interest |

---

## Breach Response

If PII breach detected:

1. **Contain**: Isolate affected systems
2. **Assess**: Determine scope and impact
3. **Notify**: Authority within 72 hours, users if high risk
4. **Document**: Record in audit log

See: [incident_response.md](../runbooks/incident_response.md)

---

## Testing

PII handling verified by:
- Integration tests: `tests/integration/test_pii_detection.py`
- Playwright E2E: `tests/playwright/month1_foundation.spec.ts`

---

*Last Updated: 2024-12-24*
