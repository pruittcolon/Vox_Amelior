#!/bin/bash
#
# n8n Integration Service - Test Runner
#
# Usage:
#   ./test_n8n.sh           # Run all tests locally
#   ./test_n8n.sh http      # Run against running service
#   ./test_n8n.sh voice     # Test voice commands only
#   ./test_n8n.sh emotion   # Test emotion alerts only
#   ./test_n8n.sh api       # Test command API only
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/tests/test_n8n_cli.py"
SERVICE_URL="${N8N_SERVICE_URL:-http://localhost:8011}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}n8n Integration Service - Test Runner${NC}"
echo "========================================"

case "${1:-local}" in
    "local"|"l")
        echo -e "${YELLOW}Running local tests (no HTTP)...${NC}"
        python3 "$TEST_SCRIPT" --local -v
        ;;
    "http"|"h"|"service")
        echo -e "${YELLOW}Running against service at $SERVICE_URL...${NC}"
        python3 "$TEST_SCRIPT" --url "$SERVICE_URL" -v
        ;;
    "voice"|"v")
        echo -e "${YELLOW}Running voice command tests...${NC}"
        python3 "$TEST_SCRIPT" --local --test voice_command -v
        ;;
    "emotion"|"e")
        echo -e "${YELLOW}Running emotion alert tests...${NC}"
        python3 "$TEST_SCRIPT" --local --test emotion_alert -v
        ;;
    "api"|"a")
        echo -e "${YELLOW}Running command API tests...${NC}"
        python3 "$TEST_SCRIPT" --local --test command_api -v
        ;;
    "e2e")
        echo -e "${YELLOW}Running end-to-end tests (requires running service)...${NC}"
        python3 "$TEST_SCRIPT" --url "$SERVICE_URL" --test e2e -v
        ;;
    "help"|"-h"|"--help")
        echo "
Usage: ./test_n8n.sh [MODE]

Modes:
  local     Run all tests using local modules (default)
  http      Run all tests against running n8n-service
  voice     Test voice command pattern matching
  emotion   Test emotion alert threshold and tracking
  api       Test command registry REST API
  e2e       Run end-to-end flow tests (requires service)
  help      Show this help message

Environment:
  N8N_SERVICE_URL   Service URL for HTTP tests (default: http://localhost:8011)

Examples:
  ./test_n8n.sh                           # Local tests
  ./test_n8n.sh http                      # Test running service
  N8N_SERVICE_URL=http://n8n:8011 ./test_n8n.sh http
"
        ;;
    *)
        echo "Unknown mode: $1"
        echo "Run './test_n8n.sh help' for usage"
        exit 1
        ;;
esac
