#!/bin/bash
#
# Emergency API Gateway Container Start Script
# Use this when docker-compose fails with KeyError: 'ContainerConfig' bug
#
# This script manually starts the api-gateway container with all required
# secrets and volume mounts that docker-compose would normally provide.
#
# Usage:
#   ./scripts/start-container-manual.sh
#
# Prerequisites:
#   - Run this from the Nemo_Server directory
#   - Docker network 'docker_nemo_remote_network' must exist
#   - Secrets must exist in docker/secrets/
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$REPO_DIR/docker"

echo -e "${YELLOW}🚨 Emergency API Gateway Start Script${NC}"
echo "Using repo directory: $REPO_DIR"

# Check if container already running
if docker ps | grep -q "refactored_gateway\|nemo_remote_gateway"; then
    echo -e "${YELLOW}WARNING: Gateway container already running. Stopping it first...${NC}"
    docker stop refactored_gateway nemo_remote_gateway 2>/dev/null || true
    docker rm refactored_gateway nemo_remote_gateway 2>/dev/null || true
fi

# Check for required secrets
REQUIRED_SECRETS="jwt_secret jwt_secret_primary jwt_secret_previous session_key users_db_key"
for secret in $REQUIRED_SECRETS; do
    if [ ! -f "$DOCKER_DIR/secrets/$secret" ]; then
        echo -e "${RED}ERROR: Missing secret: $DOCKER_DIR/secrets/$secret${NC}"
        echo "Run: python scripts/security_hardening.py to generate secrets"
        exit 1
    fi
done
echo -e "${GREEN}✓ All required secrets found${NC}"

# Check/create network
if ! docker network ls | grep -q "nemo_remote_network"; then
    echo "Creating Docker network..."
    docker network create docker_nemo_remote_network
fi

# Build image if not exists
if ! docker images | grep -q "refactored-gateway"; then
    echo "Building API Gateway image..."
    cd "$DOCKER_DIR"
    docker build -f Dockerfile.gateway -t refactored-gateway:latest ..
fi

echo -e "${GREEN}Starting API Gateway container...${NC}"

docker run -d \
    --name nemo_remote_gateway \
    --restart unless-stopped \
    --network docker_nemo_remote_network \
    -p 0.0.0.0:8000:8000 \
    -e RAG_URL=http://rag-service:8004 \
    -e EMOTION_URL=http://emotion-service:8005 \
    -e INSIGHTS_URL=http://insights-service:8010 \
    -e ML_SERVICE_URL=http://ml-service:8006 \
    -e FISERV_SERVICE_URL=http://fiserv-service:8015 \
    -e RATE_LIMIT_ENABLED=true \
    -e ENABLE_DEMO_USERS=true \
    -e SECURE_MODE=false \
    -e SESSION_COOKIE_SECURE=false \
    -v "$DOCKER_DIR/gateway_instance:/app/instance" \
    -v "$REPO_DIR/frontend:/app/frontend:ro" \
    -v "$REPO_DIR/services/api-gateway/src:/app/src:ro" \
    -v "$REPO_DIR/shared:/app/shared:ro" \
    -v "$DOCKER_DIR/secrets/session_key:/run/secrets/session_key:ro" \
    -v "$DOCKER_DIR/secrets/users_db_key:/run/secrets/users_db_key:ro" \
    -v "$DOCKER_DIR/secrets/jwt_secret_primary:/run/secrets/jwt_secret_primary:ro" \
    -v "$DOCKER_DIR/secrets/jwt_secret_previous:/run/secrets/jwt_secret_previous:ro" \
    -v "$DOCKER_DIR/secrets/jwt_secret:/run/secrets/jwt_secret:ro" \
    refactored-gateway:latest

echo ""
echo -e "${GREEN}✅ Container started!${NC}"
echo ""
echo "Waiting for health check..."
sleep 5

# Health check
if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API Gateway is healthy!${NC}"
    echo "Access at: http://localhost:8000"
else
    echo -e "${YELLOW}Container started but health check failed. Check logs:${NC}"
    echo "  docker logs nemo_remote_gateway --tail 50"
fi
