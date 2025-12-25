#!/bin/bash
# Backup RAG Database
# Usage: ./backup_rag_db.sh

BACKUP_DIR="/home/pruittcolon/Desktop/Nemo_Server/backups/rag"
DB_PATH="/home/pruittcolon/Desktop/Nemo_Server/docker/rag_instance/rag.db"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
KEEP_DAYS=7

mkdir -p "$BACKUP_DIR"

if [ -f "$DB_PATH" ]; then
    cp "$DB_PATH" "$BACKUP_DIR/rag.db.$TIMESTAMP"
    echo "[$(date)] Backed up $DB_PATH to $BACKUP_DIR/rag.db.$TIMESTAMP"
    
    # Rotation: Delete backups older than 7 days
    find "$BACKUP_DIR" -name "rag.db.*" -type f -mtime +$KEEP_DAYS -delete
    echo "[$(date)] Cleaned up backups older than $KEEP_DAYS days"
else
    echo "[$(date)] Error: Database file not found at $DB_PATH"
    exit 1
fi
