#!/bin/bash
#
# Nemo Server - Stop All Services
# Stops ALL docker compose variants
#

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}"
cat << "EOF"
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              🛑  NEMO SERVER - STOPPING ALL  🛑               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

echo -e "${BLUE}Stopping all Nemo services...${NC}"

# Stop all compose variants
cd docker
docker compose down 2>/dev/null || true
docker compose -f docker-compose.light.yml down 2>/dev/null || true
docker compose -f docker-compose.remote.yml down 2>/dev/null || true
cd ..

# Force kill any remaining containers
docker ps -a | grep -E "refactored|nemo" | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true

echo -e "\n${GREEN}✓ All Nemo services stopped!${NC}"
echo ""
echo -e "${BLUE}To restart:${NC}"
echo "  ./start.sh                   # Full mode (all services)"
echo "  ./start-no-transcription.sh  # Chat mode (no audio)"
echo "  ./start-light.sh             # Light mode (no GPU)"
echo "  ./start-frontend-only.sh     # Frontend dev only"
echo ""
