#!/bin/bash
# Initialize PostgreSQL databases for Nemo Server
# This script runs on first PostgreSQL startup

set -e

# Create nemo_queue database (already created via POSTGRES_DB env)
# Create call_center database for Call Intelligence Platform

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    -- Create call_center database if not exists
    SELECT 'CREATE DATABASE call_center'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'call_center')\gexec
    
    GRANT ALL PRIVILEGES ON DATABASE call_center TO $POSTGRES_USER;
EOSQL

# Run call center schema on call_center database
if [ -f /docker-entrypoint-initdb.d/call_center_schema.sql ]; then
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "call_center" -f /docker-entrypoint-initdb.d/call_center_schema.sql
    echo "Call Center schema initialized"
fi
