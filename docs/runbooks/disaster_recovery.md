# Disaster Recovery Runbook

## Overview

This runbook provides step-by-step procedures for disaster recovery scenarios.
All procedures should be tested quarterly during DR drills.

## RTO/RPO Targets

| Tier | RTO (Recovery Time) | RPO (Recovery Point) |
|------|---------------------|----------------------|
| Critical | 1 hour | 5 minutes |
| High | 4 hours | 1 hour |
| Medium | 24 hours | 4 hours |
| Low | 72 hours | 24 hours |

## Contacts

| Role | Primary | Backup |
|------|---------|--------|
| Incident Commander | On-Call Lead | Platform Manager |
| Database Admin | DBA Team | Backend Lead |
| Infrastructure | SRE Team | DevOps Lead |

---

## Scenario 1: Database Corruption

### Symptoms
- Application errors: "database is malformed"
- Query failures with SQLITE_CORRUPT
- Inconsistent data returned

### Immediate Actions

1. **Confirm the issue**
   ```bash
   sqlite3 /data/models.db "PRAGMA integrity_check;"
   ```

2. **Stop affected services**
   ```bash
   docker-compose stop api-gateway ml-service
   ```

3. **Identify last good backup**
   ```python
   from shared.ops.backup import get_backup_manager
   manager = get_backup_manager()
   backups = manager.list_backups(status=BackupStatus.VERIFIED, limit=5)
   print([b.to_dict() for b in backups])
   ```

4. **Restore from backup**
   ```python
   result = await manager.restore_backup(
       backup_id="<backup-id>",
       components=["database"]
   )
   print(f"Restored: {result.components_restored}")
   ```

5. **Verify restoration**
   ```bash
   sqlite3 /data/models.db "PRAGMA integrity_check;"
   # Should return: ok
   ```

6. **Restart services**
   ```bash
   docker-compose up -d api-gateway ml-service
   ```

### Post-Incident
- Document data loss window (if any)
- Update backup frequency if RPO exceeded
- Schedule post-mortem within 48 hours

---

## Scenario 2: Service Outage (API Gateway)

### Symptoms
- Health checks failing
- 502/504 errors from load balancer
- No logs being written

### Immediate Actions

1. **Check service status**
   ```bash
   docker-compose ps
   docker logs api-gateway --tail 100
   ```

2. **Check resource constraints**
   ```bash
   docker stats api-gateway --no-stream
   free -h
   df -h
   ```

3. **Restart service**
   ```bash
   docker-compose restart api-gateway
   ```

4. **If restart fails, recreate**
   ```bash
   docker-compose up -d --force-recreate api-gateway
   ```

5. **Verify recovery**
   ```bash
   curl -f http://localhost:8000/health
   ```

### Escalation
- If issue persists > 15 min, page Infrastructure team
- If affecting > 1% users, declare P1 incident

---

## Scenario 3: Model Service Failure

### Symptoms
- Gemma/Transcription endpoints return 500
- GPU lock waiting for extended periods
- OOM errors in logs

### Immediate Actions

1. **Check GPU status**
   ```bash
   nvidia-smi
   ```

2. **Check coordinator status**
   ```bash
   curl http://localhost:9000/health
   ```

3. **Clear GPU memory**
   ```bash
   # Restart coordinator to release locks
   docker-compose restart coordinator
   ```

4. **Reduce context size temporarily**
   ```bash
   curl -X POST http://localhost:9001/config \
     -H "Content-Type: application/json" \
     -d '{"context_size": 4096}'
   ```

5. **Monitor recovery**
   ```bash
   watch -n 5 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
   ```

---

## Scenario 4: Data Center Failover

### Prerequisites
- Secondary region configured
- Database replication active
- DNS failover configured

### Procedure

1. **Confirm primary failure**
   - Multiple sources confirm outage
   - Estimated recovery > RTO

2. **Initiate failover**
   ```bash
   # Update DNS to secondary
   ./scripts/failover.sh --region secondary --confirm
   ```

3. **Verify secondary health**
   ```bash
   curl https://secondary.api.example.com/health
   ```

4. **Update configurations**
   ```bash
   # Point services to secondary databases
   kubectl set env deployment/api-gateway \
     DATABASE_URL=postgres://secondary-db:5432/nemo
   ```

5. **Notify stakeholders**
   - Post to status page
   - Email to enterprise customers
   - Slack to internal channels

### Failback Procedure
1. Wait for primary region recovery confirmation
2. Sync data from secondary to primary
3. Gradual traffic shift (10% -> 50% -> 100%)
4. Monitor for 24 hours before full cutover

---

## Scenario 5: Security Incident

### Symptoms
- Anomalous login patterns detected
- Unauthorized API calls
- Data exfiltration alerts

### Immediate Actions

1. **Isolate affected systems**
   ```bash
   # Revoke all active sessions
   ./scripts/security/revoke-sessions.sh --all
   ```

2. **Rotate credentials**
   ```bash
   # Generate new API keys
   ./scripts/security/rotate-keys.sh
   ```

3. **Enable enhanced logging**
   ```bash
   export LOG_LEVEL=DEBUG
   export AUDIT_LEVEL=VERBOSE
   docker-compose up -d
   ```

4. **Notify security team**
   - Page security on-call immediately
   - Do not communicate details via non-secure channels

5. **Preserve evidence**
   ```bash
   # Snapshot logs before any changes
   cp -r /var/log/nemo /data/incident-$(date +%Y%m%d_%H%M%S)
   ```

---

## Backup Procedures

### Daily Automated Backup

```python
from shared.ops.backup import get_backup_manager, BackupType

async def daily_backup():
    manager = get_backup_manager()
    
    backup = await manager.create_backup(
        components=["database", "configs", "prompts"],
        backup_type=BackupType.FULL,
        encrypt=True,
        encryption_key=os.getenv("BACKUP_ENCRYPTION_KEY"),
        notes="Daily automated backup",
    )
    
    # Verify immediately
    result = await manager.verify_backup(backup.id)
    
    if not result["valid"]:
        # Alert on-call
        send_alert("Backup verification failed", backup.id)
```

### Manual Backup Before Changes

```bash
# Before any risky operation
python3 -c "
import asyncio
from shared.ops.backup import get_backup_manager

async def backup():
    m = get_backup_manager()
    b = await m.create_backup(notes='Pre-deployment backup')
    print(f'Backup ID: {b.id}')

asyncio.run(backup())
"
```

---

## Health Check Commands

```bash
# Full system health check
./scripts/health-check.sh

# Individual service checks
curl http://localhost:8000/health    # API Gateway
curl http://localhost:9000/health    # Coordinator
curl http://localhost:9001/health    # Gemma Service
curl http://localhost:8501/health    # ML Service

# Database health
sqlite3 /data/models.db "SELECT COUNT(*) FROM models;"
sqlite3 /data/prompts.db "SELECT COUNT(*) FROM prompts;"
```

---

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-12-24 | 1.0 | System | Initial runbook |
