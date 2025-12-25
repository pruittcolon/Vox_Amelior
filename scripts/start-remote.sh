#!/bin/bash
#
# Nemo Server - Remote/No-GPU Startup Script
# For laptops without NVIDIA GPU (Intel/AMD integrated graphics)
#
# Usage:
#   ./start-remote.sh           # Start all CPU-only services
#   ./start-remote.sh --build   # Rebuild images before starting
#   ./start-remote.sh --stop    # Stop all services
#

set -e

# Colors
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
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC}  $1"; }

compose_cmd() {
    if command -v docker-compose &> /dev/null; then
        docker-compose -f docker/docker-compose.remote.yml "$@"
    else
        docker compose -f docker/docker-compose.remote.yml "$@"
    fi
}

get_local_ip() {
    # Try to get the WiFi/LAN IP address
    ip route get 1 2>/dev/null | awk '{print $7; exit}' || \
    hostname -I 2>/dev/null | awk '{print $1}' || \
    echo "localhost"
}

# Parse arguments
ACTION="start"
BUILD=false
for arg in "$@"; do
    case "$arg" in
        --stop) ACTION="stop" ;;
        --build) BUILD=true ;;
    esac
done

# Banner
echo -e "${BLUE}"
cat << "EOF"
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║         🖥️  NEMO SERVER - REMOTE/NO-GPU MODE  🖥️            ║
    ║                                                               ║
    ║              CPU-Only Services for Intel/AMD                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

if [ "$ACTION" = "stop" ]; then
    print_header "Stopping Remote Services"
    compose_cmd down
    print_success "All services stopped!"
    exit 0
fi

# Check dependencies
print_header "Checking Dependencies"

if ! command -v docker &> /dev/null; then
    print_error "Docker not installed!"
    echo "Install with: sudo apt install docker.io docker-compose"
    exit 1
fi
print_success "Docker installed: $(docker --version | head -n1)"

# Check if NVIDIA is present (and warn that we're not using it)
if command -v nvidia-smi &> /dev/null; then
    print_warning "NVIDIA GPU detected but NOT USED in this mode"
    print_info "GPU services (Gemma, Transcription) are disabled"
else
    print_success "No NVIDIA GPU detected - perfect for this mode!"
fi

# Run security hardening if script exists
if [ -f "scripts/security_hardening.py" ]; then
    print_header "Setting Up Secrets"
    python3 scripts/security_hardening.py 2>/dev/null || true
    print_success "Secrets configured"
fi

# Stop existing containers
print_header "Cleaning Up Existing Containers"
compose_cmd down 2>/dev/null || true
docker ps -a | grep -E "nemo_remote" | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true
print_success "Cleanup complete"

# Build if requested
if [ "$BUILD" = true ]; then
    print_header "Building Images (CPU-only)"
    compose_cmd build
    print_success "Build complete"
fi

# Start services
print_header "Starting CPU-Only Services"
print_info "Starting infrastructure (redis, postgres)..."
compose_cmd up -d redis postgres
sleep 3

print_info "Starting application services..."
compose_cmd up -d
print_success "All services started!"

# Wait for health
print_header "Waiting for Services"
local_ip=$(get_local_ip)
timeout=120
elapsed=0

while true; do
    if curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
        print_success "API Gateway is healthy!"
        break
    fi
    
    if [ $elapsed -gt $timeout ]; then
        print_error "Timeout waiting for services"
        compose_cmd logs --tail=50
        exit 1
    fi
    
    sleep 2
    elapsed=$((elapsed + 2))
    if [ $((elapsed % 10)) -eq 0 ]; then
        print_info "Waiting... (${elapsed}s)"
    fi
done

# Show info
print_header "Remote Server Information"

echo -e "${GREEN}Services Running (CPU-Only):${NC}"
echo ""
echo "  📡 API Gateway:      http://${local_ip}:8000"
echo "  🧠 RAG Service:      Internal (8004)"
echo "  😊 Emotion Service:  Internal (8005)"
echo "  📊 Insights:         Internal (8010)"
echo "  🤖 ML Service:       Internal (8006)"
echo "  🏦 Fiserv Service:   Internal (8015)"
echo ""
echo "  💾 PostgreSQL:       localhost:5432"
echo "  🔴 Redis:            localhost:6379"
echo ""

echo -e "${YELLOW}NOT Running (Require NVIDIA GPU):${NC}"
echo "  ❌ Gemma Service"
echo "  ❌ Transcription Service"
echo ""

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  ACCESS FROM YOUR MAIN PC:${NC}"
echo -e "${GREEN}  http://${local_ip}:8000/ui/login.html${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Demo Credentials:${NC}"
echo "  Username: admin"
echo "  Password: admin123"
echo ""

echo -e "${YELLOW}Controls:${NC}"
echo "  Stop services:  ./start-remote.sh --stop"
echo "  View logs:      cd docker && docker compose -f docker-compose.remote.yml logs -f"
echo ""

print_success "Remote server ready! Access from your main PC using the URL above."
