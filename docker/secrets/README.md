# Nemo Server - Docker Secrets

> **SECURITY CRITICAL**: This directory contains encrypted secrets. Never commit actual secret files to Git.

## Required Secrets

The following secret files are required for the Nemo Server to function:

| Secret File | Description | Required |
|-------------|-------------|----------|
| `jwt_secret_primary` | Primary JWT signing key (32+ bytes) | Yes |
| `jwt_secret_previous` | Previous JWT key for rotation (32+ bytes) | Yes |
| `jwt_secret` | Service-to-service JWT key | Yes |
| `service_api_key` | **Legacy** - Replaced by JWT authentication | No (deprecated) |
| `session_key` | Session encryption key (32+ bytes) | Yes |
| `postgres_user` | PostgreSQL username | Yes |
| `postgres_password` | PostgreSQL password (24+ chars) | Yes |
| `users_db_key` | User database encryption key | Yes |
| `rag_db_key` | RAG database encryption key | Yes |
| `redis_password` | Redis authentication password | Yes |
| `huggingface_token` | HuggingFace API token | Optional |
| `email_db_key` | Email database encryption key | Optional |

## Setup Instructions

### Automatic Setup (Recommended)

Run the secret generation script:

```bash
cd /path/to/Nemo_Server
./scripts/setup_secrets.sh
```

This script will:
1. Generate cryptographically secure random secrets
2. Preserve any existing secrets (non-destructive)
3. Set proper file permissions (600)
4. Validate secret strength

### Manual Setup

Create each file manually with secure values:

```bash
# Generate a 32-byte random secret
openssl rand -base64 32 > docker/secrets/jwt_secret_primary

# Generate password
openssl rand -base64 24 > docker/secrets/postgres_password

# Set permissions (CRITICAL!)
chmod 600 docker/secrets/*
```

## Security Requirements

1. **File Permissions**: All secret files MUST have mode `600` (read/write owner only)
2. **Never Commit**: All files in this directory are gitignored. Never force-add them.
3. **Rotation**: Rotate secrets regularly. Use `jwt_secret_previous` for graceful JWT rotation.
4. **Backup**: Keep encrypted backups of secrets offline.

## Pre-flight Checklist

Before starting the server, verify:

```bash
# Check all required secrets exist
ls -la docker/secrets/

# Verify permissions (should be -rw-------)
stat -c "%a %n" docker/secrets/*

# Run security verification
python3 scripts/verify_security.py
```

## Troubleshooting

### "Secret file not found" on startup
- Run `./scripts/setup_secrets.sh` to generate missing secrets

### "Permission denied" errors
- Run `chmod 600 docker/secrets/*` to fix permissions

### "Invalid secret length" errors
- Regenerate the specific secret with the required minimum length

## Migration from Old Secrets

If upgrading from a previous version:

1. Backup existing secrets to `archive/secrets_backup/`
2. Run `./scripts/setup_secrets.sh --migrate`
3. Verify services start correctly
4. Securely delete old backups after verification
