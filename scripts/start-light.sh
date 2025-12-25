#!/bin/bash
#
# Nemo Server - Lightweight Mode (No Gemma, No Transcription)
# Saves ~4GB RAM and GPU VRAM by excluding heavy AI services
#
# Use this when:
#   - Working with Banking/Fiserv features
#   - Doing ML analysis
#   - Need your GPU for other applications
#
# Usage:
#   ./start-light.sh           # Start lightweight services
#   ./start-light.sh --build   # Rebuild before starting
#   ./start-light.sh --stop    # Stop all services
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
        docker-compose -f docker-compose.light.yml "$@"
    else
        docker compose -f docker-compose.light.yml "$@"
    fi
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
    ║          ⚡  NEMO SERVER - LIGHTWEIGHT MODE  ⚡               ║
    ║                                                               ║
    ║         No Gemma | No Transcription | Low Resources           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

cd docker

if [ "$ACTION" = "stop" ]; then
    print_header "Stopping Services"
    compose_cmd down
    print_success "All services stopped!"
    exit 0
fi

# Stop everything first
print_header "Stopping Existing Services"
compose_cmd down 2>/dev/null || true
print_success "Cleanup complete"

# Run security hardening
if [ -f "../scripts/security_hardening.py" ]; then
    print_info "Configuring secrets..."
    python3 ../scripts/security_hardening.py 2>/dev/null || true
fi

# Build if requested
if [ "$BUILD" = true ]; then
    print_header "Building Images"
    # Only build what we need
    compose_cmd build api-gateway rag-service emotion-service ml-service fiserv-service insights-service
    print_success "Build complete"
fi

# Start services in order (excluding gemma-service and transcription-service)
print_header "Starting Lightweight Services"

print_info "Starting all lightweight services..."
compose_cmd up -d

print_success "Services starting!"

# Health check
print_header "Checking Health"
timeout=60
elapsed=0

while true; do
    if curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
        print_success "API Gateway is healthy!"
        break
    fi
    
    if [ $elapsed -gt $timeout ]; then
        print_error "Timeout waiting for services"
        compose_cmd logs api-gateway --tail=20
        exit 1
    fi
    
    sleep 2
    elapsed=$((elapsed + 2))
done

# Show info
print_header "Service Information"

echo -e "${GREEN}Services Running:${NC}"
echo ""
echo "  📡 API Gateway:      http://localhost:8000"
echo "  🧠 RAG Service:      http://localhost:8004 (internal)"
echo "  😊 Emotion Service:  http://localhost:8005 (internal)"
echo "  📊 Insights:         http://localhost:8010 (internal)"
echo "  🤖 ML Service:       http://localhost:8006 (internal)"
echo "  🏦 Fiserv Service:   http://localhost:8015 (internal)"
echo ""

echo -e "${YELLOW}NOT Running (to save resources):${NC}"
echo "  ⏸️  Gemma Service    (~2GB VRAM saved)"
echo "  ⏸️  Transcription    (~2GB VRAM + CPU saved)"
echo ""

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  OPEN: http://localhost:8000/ui/login.html${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Features Available:${NC}"
echo "  ✅ Banking Dashboard (banking.html)"
echo "  ✅ Fiserv API Integration"
echo "  ✅ ML Engines (Cross-Sell, Churn, Loan Pricing)"
echo "  ✅ Emotion/Sentiment Analysis"
echo "  ✅ RAG Memory & Search"
echo "  ✅ Analytics & Insights"
echo ""
echo "  ❌ Gemma Chat (use ./start.sh for full mode)"
echo "  ❌ Audio Transcription (use ./start.sh for full mode)"
echo ""

echo -e "${YELLOW}Commands:${NC}"
echo "  Stop:      ./start-light.sh --stop"
echo "  Full mode: ./start.sh"
echo "  Logs:      cd docker && docker compose -f docker-compose.light.yml logs -f"
echo ""

# Open browser
if command -v xdg-open &> /dev/null; then
    sleep 2
    xdg-open "http://localhost:8000/ui/login.html" 2>/dev/null &
fi

print_success "Lightweight mode ready! GPU is free for other tasks."
