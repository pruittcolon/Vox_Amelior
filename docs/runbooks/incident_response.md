# Incident Response Runbook

## Purpose

This runbook provides step-by-step procedures for responding to production incidents affecting the Nemo Server platform.

---

## Severity Levels

| Level | Definition | Response Time | Examples |
|-------|------------|---------------|----------|
| **SEV1** | Complete outage | 15 minutes | All services down |
| **SEV2** | Major degradation | 30 minutes | Core feature unavailable |
| **SEV3** | Minor issue | 4 hours | Single endpoint slow |
| **SEV4** | Low impact | 24 hours | Minor bug reported |

---

## Incident Response Process

### 1. Detection
- Automated alerts from monitoring
- User reports via support channels
- Health check failures

### 2. Triage (First 5 minutes)
```bash
# Check overall health
curl http://localhost:8000/health

# Check container status
docker ps --format "table {{.Names}}\t{{.Status}}"

# Check recent logs
docker logs refactored_gateway --tail 100
```

### 3. Assess Severity
- How many users affected?
- Which services impacted?
- Data integrity at risk?

### 4. Communicate
- Update status page
- Notify stakeholders (SEV1/SEV2)
- Create incident channel

### 5. Investigate
```bash
# Check API gateway logs
docker logs refactored_gateway --since 10m

# Check database connectivity
docker exec refactored_postgres pg_isready

# Check Redis
docker exec refactored_redis redis-cli ping

# Check GPU services
docker logs refactored_gemma --tail 50
```

### 6. Mitigate
- Rollback if deployment-related
- Scale up if capacity issue
- Restart affected containers

### 7. Resolve
- Verify fix in production
- Update status page
- Monitor for recurrence

### 8. Post-Incident
- Document timeline
- Conduct blameless postmortem
- Create follow-up action items

---

## Common Issues and Fixes

### API Gateway Not Responding
```bash
docker restart refactored_gateway
```

### Database Connection Issues
```bash
# Check PostgreSQL
docker logs refactored_postgres
docker exec refactored_postgres pg_isready -U postgres

# Restart if needed
docker restart refactored_postgres
```

### GPU Service Failures
```bash
# Check VRAM
nvidia-smi

# Restart Gemma service
docker restart refactored_gemma
```

### Memory Issues
```bash
# Check system memory
free -h

# Identify memory-heavy containers
docker stats --no-stream
```

---

## Escalation Contacts

| Role | Responsibility |
|------|----------------|
| On-call Engineer | First responder |
| Team Lead | SEV1/SEV2 escalation |
| Security | Data breach concerns |

---

## Post-Incident Template

```markdown
## Incident Summary
- **Date**: YYYY-MM-DD
- **Duration**: X hours
- **Severity**: SEVX
- **Impact**: Description

## Timeline
- HH:MM - Event occurred
- HH:MM - Detected
- HH:MM - Mitigated
- HH:MM - Resolved

## Root Cause
Description of what caused the issue

## Resolution
Steps taken to fix

## Action Items
- [ ] Item 1
- [ ] Item 2
```

---

*Last Updated: 2024-12-23*
