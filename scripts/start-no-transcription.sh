#!/bin/bash
#
# Nemo Server - No Transcription Mode (Chat + Banking)
# Gemma LLM enabled, Transcription disabled (~2GB VRAM saved)
#
# Usage:
#   ./start-no-transcription.sh           # Start services
#   ./start-no-transcription.sh --build   # Rebuild before starting
#   ./start-no-transcription.sh --stop    # Stop all services
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
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC}  $1"; }

compose_cmd() {
    if command -v docker-compose &> /dev/null; then
        docker-compose -f docker/docker-compose.yml "$@"
    else
        docker compose -f docker/docker-compose.yml "$@"
    fi
}

ACTION="start"
BUILD=false
for arg in "$@"; do
    case "$arg" in
        --stop) ACTION="stop" ;;
        --build) BUILD=true ;;
    esac
done

echo -e "${BLUE}"
cat << "EOF"
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║         💬  NEMO SERVER - CHAT MODE (NO TRANSCRIPTION)  💬   ║
    ║                                                               ║
    ║          Gemma LLM Enabled | No Audio Processing              ║
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

# Stop all variants first
print_header "Stopping Existing Services"
compose_cmd down 2>/dev/null || true
print_success "Cleanup complete"

# Security hardening
if [ -f "scripts/security_hardening.py" ]; then
    print_info "Configuring secrets..."
    python3 scripts/security_hardening.py 2>/dev/null || true
fi

if [ "$BUILD" = true ]; then
    print_header "Building Images"
    compose_cmd build api-gateway gemma-service rag-service emotion-service ml-service fiserv-service
    print_success "Build complete"
fi

# Start services WITHOUT transcription
print_header "Starting Services (No Transcription)"

print_info "Starting infrastructure..."
compose_cmd up -d redis postgres
sleep 3

print_info "Starting GPU coordinator..."
compose_cmd up -d gpu-coordinator
sleep 2

print_info "Starting Gemma (LLM)..."
compose_cmd up -d gemma-service

# Wait for Gemma to load
print_info "Waiting for Gemma to load on GPU..."
for i in {1..60}; do
    if docker logs refactored_gemma 2>&1 | grep -q "Model loaded on GPU successfully\|Loaded on GPU successfully"; then
        print_success "Gemma loaded on GPU!"
        break
    fi
    sleep 2
    if [ $((i % 10)) -eq 0 ]; then
        print_info "Still waiting... (${i}s)"
    fi
done

print_info "Starting remaining services (skipping transcription)..."
# Start everything EXCEPT transcription-service
compose_cmd up -d rag-service emotion-service insights-service ml-service fiserv-service n8n-service api-gateway nginx

print_success "Services started!"

# Health check
print_header "Checking Health"
timeout=90
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
echo "  🤖 Gemma LLM:        Active (GPU)"
echo "  🧠 RAG Service:      Active"
echo "  😊 Emotion Service:  Active"
echo "  📊 ML Engines:       Active"
echo "  🏦 Fiserv Banking:   Active"
echo ""

echo -e "${YELLOW}NOT Running (saving resources):${NC}"
echo "  ⏸️  Transcription    (~2GB VRAM + CPU saved)"
echo ""

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  OPEN: http://localhost:8000/ui/login.html${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Features Available:${NC}"
echo "  ✅ Gemma Chat (gemma.html)"
echo "  ✅ Banking Dashboard (banking.html)"
echo "  ✅ ML Engines (Cross-Sell, Churn, Loan)"
echo "  ✅ Emotion/Sentiment Analysis"
echo "  ✅ RAG Memory & Search"
echo ""
echo "  ❌ Audio Transcription (use ./start.sh for full mode)"
echo ""

if command -v xdg-open &> /dev/null; then
    sleep 2
    xdg-open "http://localhost:8000/ui/login.html" 2>/dev/null &
fi

print_success "Chat mode ready! Transcription disabled to save VRAM."
