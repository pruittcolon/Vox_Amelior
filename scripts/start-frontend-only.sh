#!/bin/bash
#
# Nemo Server - Frontend Only Mode
# Minimal services for HTML/CSS/JS development
#
# Usage:
#   ./start-frontend-only.sh           # Start minimal services
#   ./start-frontend-only.sh --stop    # Stop all services
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC}  $1"; }
print_info() { echo -e "${BLUE}ℹ${NC}  $1"; }

compose_cmd() {
    if command -v docker-compose &> /dev/null; then
        docker-compose -f docker/docker-compose.light.yml "$@"
    else
        docker compose -f docker/docker-compose.light.yml "$@"
    fi
}

ACTION="start"
for arg in "$@"; do
    case "$arg" in
        --stop) ACTION="stop" ;;
    esac
done

echo -e "${BLUE}"
cat << "EOF"
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║           🎨  NEMO SERVER - FRONTEND ONLY  🎨                 ║
    ║                                                               ║
    ║              Minimal Mode for UI Development                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

if [ "$ACTION" = "stop" ]; then
    print_header "Stopping Services"
    compose_cmd down
    print_success "All services stopped!"
    exit 0
fi

# Stop all variants
print_header "Stopping Existing Services"
docker compose -f docker/docker-compose.yml down 2>/dev/null || true
docker compose -f docker/docker-compose.light.yml down 2>/dev/null || true
print_success "Cleanup complete"

# Security
if [ -f "scripts/security_hardening.py" ]; then
    python3 scripts/security_hardening.py 2>/dev/null || true
fi

# Start ONLY minimal services needed for frontend
print_header "Starting Minimal Services"

print_info "Starting infrastructure..."
compose_cmd up -d redis postgres
sleep 3

print_info "Starting API Gateway only..."
# Use light compose but only bring up gateway and dependencies
compose_cmd up -d gpu-coordinator rag-service emotion-service api-gateway
sleep 2

print_success "Minimal services started!"

# Quick health check
print_header "Checking Health"
for i in {1..30}; do
    if curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
        print_success "API Gateway is healthy!"
        break
    fi
    sleep 2
done

print_header "Frontend Development Mode"

echo -e "${GREEN}Services Running:${NC}"
echo ""
echo "  📡 API Gateway:    http://localhost:8000"
echo "  💾 PostgreSQL:     localhost:5432"
echo "  🔴 Redis:          localhost:6379"
echo ""

echo -e "${YELLOW}NOT Running (minimal mode):${NC}"
echo "  ⏸️  Gemma LLM"
echo "  ⏸️  Transcription"  
echo "  ⏸️  ML Service"
echo "  ⏸️  Fiserv Service"
echo ""

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  OPEN: http://localhost:8000/ui/login.html${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Best for:${NC}"
echo "  ✅ HTML/CSS editing"
echo "  ✅ JavaScript development"
echo "  ✅ Styling and layout work"
echo "  ✅ Fastest startup time"
echo ""

if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:8000/ui/login.html" 2>/dev/null &
fi

print_success "Frontend mode ready! Minimal resources used."
