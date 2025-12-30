# Incident Response Runbook

## Purpose

This runbook provides step-by-step procedures for responding to production incidents affecting the Nemo Server platform.

---

## Severity Levels

## Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| **P0** | Complete outage | 15 minutes | All services down, data loss risk |
| **P1** | Major degradation | 1 hour | GPU failure, LLM unavailable |
| **P2** | Minor degradation | 4 hours | Slow responses, single engine down |
| **P3** | Minimal impact | 24 hours | UI bugs, non-critical errors |

---

## P0: Complete Outage

### Immediate Actions (0-15 minutes)

1. **Verify the outage**
   ```bash
   curl -s http://localhost:8000/health | jq
   # Expected: {"status": "ok"}
   ```

2. **Check Docker status**
   ```bash
   cd docker && docker compose ps -a
   ```

3. **View recent logs**
   ```bash
   docker compose logs --tail=100 --timestamps
   ```

4. **Attempt restart**
   ```bash
   ./nemo --no-browser
   ```

5. **Resource Check**
   ```bash
   df -h      # Disk
   free -h    # Memory
   nvidia-smi # GPU
   ```

### Escalation
- Create incident channel
- Page on-call if not resolved in 15 minutes

---

## P1: GPU/LLM Failure

### Symptoms
- 503 errors from Gemma service
- Slow/no LLM responses

### Diagnosis & Resolution

1. **Check Service Health**
   ```bash
   curl -s http://localhost:8000/api/gemma/stats | jq
   # Verify "model_on_gpu": true
   ```

2. **Force GPU Warmup**
   ```bash
   curl -X POST http://localhost:8000/api/gemma/warmup
   ```

3. **Restart Stack (if stuck)**
   ```bash
   docker compose restart gpu-coordinator
   sleep 5
   docker compose restart gemma-service transcription-service
   ```

---

## P2: Database/Auth Issues

### Diagnosis
```bash
# Check Postgres
docker logs refactored_postgres --tail=50
docker exec refactored_postgres pg_isready -U postgres

# Check Redis
docker logs refactored_redis --tail=50
docker exec refactored_redis redis-cli ping
```

### Resolution
```bash
docker compose restart postgres redis
```

---

## Common Issues and Fixes

### API Gateway Not Responding
```bash
docker restart refactored_gateway
```

### Memory Issues
```bash
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
