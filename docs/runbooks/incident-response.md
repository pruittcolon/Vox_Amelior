# Incident Response Runbook

## Overview

This runbook provides step-by-step guidance for responding to incidents affecting the Nemo Server platform.

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

2. **Check Docker container status**
   ```bash
   cd /home/pruittcolon/Desktop/Nemo_Server/docker
   docker compose ps -a
   ```

3. **View recent logs**
   ```bash
   docker compose logs --tail=100 --timestamps
   ```

4. **Attempt restart**
   ```bash
   cd /home/pruittcolon/Desktop/Nemo_Server
   ./start.sh --no-browser
   ```

5. **If restart fails, check resources**
   ```bash
   # Check disk space
   df -h

   # Check memory
   free -h

   # Check GPU
   nvidia-smi
   ```

### Escalation

- Create incident channel
- Page on-call if not resolved in 15 minutes
- Document timeline in incident log

---

## P1: GPU/LLM Failure

### Symptoms
- Gemma service returns 503
- GPU not detected
- Slow or no LLM responses

### Diagnosis

1. **Check Gemma service health**
   ```bash
   curl -s http://localhost:8000/api/gemma/stats | jq
   # Look for: "model_on_gpu": true
   ```

2. **Check GPU status**
   ```bash
   nvidia-smi
   # Look for: GPU processes, memory usage
   ```

3. **Check GPU coordinator**
   ```bash
   docker logs nemo-gpu-coordinator --tail=50
   ```

### Resolution

1. **Restart Gemma service**
   ```bash
   docker compose restart gemma-service
   ```

2. **Force GPU warmup**
   ```bash
   curl -X POST http://localhost:8000/api/gemma/warmup
   ```

3. **If GPU is stuck, restart coordinator first**
   ```bash
   docker compose restart gpu-coordinator
   sleep 10
   docker compose restart gemma-service transcription-service
   ```

---

## P2: Database Issues

### Symptoms
- Auth failures
- Session errors
- Search not returning results

### Diagnosis

1. **Check PostgreSQL**
   ```bash
   docker logs nemo-postgres --tail=50
   docker exec nemo-postgres pg_isready
   ```

2. **Check Redis**
   ```bash
   docker logs nemo-redis --tail=50
   docker exec nemo-redis redis-cli ping
   ```

### Resolution

1. **Restart database services**
   ```bash
   docker compose restart postgres redis
   ```

2. **Check disk space for data volumes**
   ```bash
   docker system df -v
   ```

---

## Post-Incident

### Required Actions

1. **Document the incident** - Timeline, actions taken, resolution
2. **Identify root cause** - What failed and why
3. **Create follow-up tickets** - Improvements to prevent recurrence
4. **Update runbooks** - Add any new procedures learned

### Incident Report Template

```markdown
# Incident Report: [TITLE]

**Date:** YYYY-MM-DD
**Duration:** X hours Y minutes
**Severity:** P0/P1/P2/P3

## Summary
Brief description of what happened.

## Timeline
- HH:MM - First alert
- HH:MM - Actions taken
- HH:MM - Resolution

## Root Cause
What caused the incident.

## Resolution
How it was fixed.

## Follow-up Actions
- [ ] Action item 1
- [ ] Action item 2
```

---

## Useful Commands Reference

```bash
# Full system restart
./start.sh --no-browser

# View all logs
cd docker && docker compose logs -f

# Run health check
make test

# Run full test suite
make test-all

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```
