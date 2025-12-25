# Service Level Objectives (SLO)

## Overview

This document defines the Service Level Objectives for the Nemo Server platform. These objectives guide reliability engineering practices and inform capacity planning.

## Definitions

| Term | Definition |
|------|------------|
| **SLO** | Service Level Objective - target reliability level |
| **SLI** | Service Level Indicator - metric measuring performance |
| **SLA** | Service Level Agreement - contractual commitment |
| **Error Budget** | Allowable failures before SLO breach |

## Platform SLOs

### 1. API Availability

| Tier | Target | Error Budget (30 days) |
|------|--------|------------------------|
| Enterprise | 99.95% | ~22 minutes |
| Standard | 99.9% | ~43 minutes |
| Development | 99.5% | ~216 minutes |

**SLI**: Percentage of non-5xx responses at API Gateway
```promql
sum(rate(http_requests_total{status!~"5.."}[5m])) 
/ sum(rate(http_requests_total[5m]))
```

### 2. API Latency (P95)

| Endpoint Category | Target | Measurement |
|-------------------|--------|-------------|
| Health/Ready | <50ms | `histogram_quantile(0.95, http_request_duration_seconds{endpoint="/health"})` |
| Standard API | <500ms | All non-streaming endpoints |
| File Upload | <30s | `/api/upload`, `/api/ingest` |
| GenAI Inference | <2s TTFT | Time to first token |

**Exclusions**: WebSocket connections, SSE streaming endpoints

### 3. Error Rate

**Target**: <1% of requests return 5xx errors

```promql
sum(rate(http_requests_total{status=~"5.."}[1h])) 
/ sum(rate(http_requests_total[1h])) < 0.01
```

### 4. Throughput

**Target**: Sustain 1000 requests/second per service instance

**SLI**: `sum(rate(http_requests_total[1m]))`

## Per-Tenant SLOs (Enterprise)

Enterprise customers may have custom SLOs tracked separately:

| Tenant Type | Availability | Latency P95 | Support Response |
|-------------|--------------|-------------|------------------|
| Enterprise Plus | 99.99% | 200ms | 15 min |
| Enterprise | 99.95% | 500ms | 1 hour |
| Business | 99.9% | 1s | 4 hours |

### Tenant SLO Tracking

```python
from shared.telemetry.slo_tracker import SLOTracker, SLODefinition

# Create tenant-specific tracker
tenant_slos = [
    SLODefinition(
        name="tenant_availability",
        slo_type=SLOType.AVAILABILITY,
        target=0.9999,  # 99.99%
        window_days=30,
    )
]
tracker = SLOTracker(slos=tenant_slos)
```

## Error Budget Policy

### Calculation

```
Error Budget = (1 - SLO Target) × Total Window
Budget Consumed = (1 - Current SLI) × Total Window
Budget Remaining = Error Budget - Budget Consumed
```

### Burn Rate Alerts

| Alert Level | Condition | Action |
|-------------|-----------|--------|
| **Warning** | Burn rate >2x normal | Notify on-call |
| **Critical** | Burn rate >10x normal | Page on-call, freeze deploys |
| **Breach** | Budget exhausted | Incident declaration |

## Monitoring Integration

### OpenTelemetry Setup

```python
from shared.telemetry import setup_telemetry, get_tracer, get_meter

# Initialize in service startup
setup_telemetry("api-gateway")

# Create custom metrics
meter = get_meter()
request_counter = meter.create_counter("requests_total")
latency_histogram = meter.create_histogram("request_latency_ms")
```

### Prometheus Queries

**Availability (30-day)**:
```promql
1 - (sum(increase(http_requests_total{status=~"5.."}[30d])) 
   / sum(increase(http_requests_total[30d])))
```

**Error Budget Remaining (minutes)**:
```promql
(0.001 * 30 * 24 * 60) - (
  (1 - (sum(increase(http_requests_total{status!~"5.."}[30d])) 
      / sum(increase(http_requests_total[30d])))) * 30 * 24 * 60
)
```

## Incident Response

### SLO Breach Severity

| Severity | Criteria | Response Time |
|----------|----------|---------------|
| P1 | >1% users affected | 15 min |
| P2 | Single SLO breached | 1 hour |
| P3 | Warning threshold hit | 4 hours |

### Remediation Steps

1. **Acknowledge** incident within response time
2. **Assess** impact scope and blast radius
3. **Mitigate** with rollback or feature flag
4. **Communicate** status to stakeholders
5. **Resolve** root cause
6. **Post-mortem** within 48 hours

## Dashboard Reference

- **Grafana**: Platform Health (`/grafana/d/platform-health`)
- **SLO Status**: `/api/slo/status` endpoint
- **Error Budget**: `/api/slo/budget` endpoint

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-24 | 2.0 | Added per-tenant SLOs, error budget policy |
| 2024-01-01 | 1.0 | Initial SLO definitions |

