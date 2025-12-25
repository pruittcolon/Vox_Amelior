#!/bin/bash
# ==============================================================================
# NeMo Server Comprehensive Verification Script
# Tests ALL endpoints via curl and reports pass/fail
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BASE_URL="${BASE_URL:-http://localhost:8000}"
PASSED=0
FAILED=0
TESTS=()

# ==============================================================================
# Helper Functions
# ==============================================================================

log_header() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
}

log_test() {
    echo -ne "  Testing: $1... "
}

log_pass() {
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
    TESTS+=("PASS: $1")
}

log_fail() {
    echo -e "${RED}✗ FAILED${NC} - $1"
    ((FAILED++))
    TESTS+=("FAIL: $1 - $2")
}

log_skip() {
    echo -e "${YELLOW}○ SKIPPED${NC} - $1"
    TESTS+=("SKIP: $1 - $2")
}

# ==============================================================================
# 1. Health Checks (No Auth Required)
# ==============================================================================
test_health() {
    log_header "PHASE 1: Health Checks (No Auth)"
    
    log_test "Gateway health endpoint"
    response=$(curl -s -w "%{http_code}" -o /tmp/health.json "$BASE_URL/health" 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_pass
    else
        log_fail "HTTP $response"
    fi
    
    log_test "Static file serving (index.html)"
    response=$(curl -s -w "%{http_code}" -o /dev/null "$BASE_URL/ui/index.html" 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_pass
    else
        log_fail "HTTP $response"
    fi
}

# ==============================================================================
# 2. Authentication Flow
# ==============================================================================
test_auth() {
    log_header "PHASE 2: Authentication Flow"
    
    log_test "Login with valid credentials"
    response=$(curl -s -X POST "$BASE_URL/api/auth/login" \
        -H "Content-Type: application/json" \
        -d '{"username":"admin","password":"admin123"}' \
        -c /tmp/cookies.txt 2>/dev/null)
    
    if echo "$response" | grep -q '"success"\s*:\s*true'; then
        log_pass
        # Extract session token for later tests
        SESSION_TOKEN=$(echo "$response" | grep -o '"session_token"\s*:\s*"[^"]*"' | cut -d'"' -f4)
        CSRF_TOKEN=$(echo "$response" | grep -o '"csrf_token"\s*:\s*"[^"]*"' | cut -d'"' -f4)
        export SESSION_TOKEN CSRF_TOKEN
    else
        log_fail "Login failed: $response"
        # Try alternate credentials
        log_test "Login with alternate credentials (PruittColon)"
        response=$(curl -s -X POST "$BASE_URL/api/auth/login" \
            -H "Content-Type: application/json" \
            -d '{"username":"PruittColon","password":"Pruitt12!"}' \
            -c /tmp/cookies.txt 2>/dev/null)
        
        if echo "$response" | grep -q '"success"\s*:\s*true'; then
            log_pass
            SESSION_TOKEN=$(echo "$response" | grep -o '"session_token"\s*:\s*"[^"]*"' | cut -d'"' -f4)
            CSRF_TOKEN=$(echo "$response" | grep -o '"csrf_token"\s*:\s*"[^"]*"' | cut -d'"' -f4)
            export SESSION_TOKEN CSRF_TOKEN
        else
            log_fail "Both login attempts failed"
        fi
    fi
    
    log_test "Session validation"
    response=$(curl -s -w "%{http_code}" -o /tmp/check.json "$BASE_URL/api/auth/check" \
        -b "ws_session=$SESSION_TOKEN" 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_pass
    else
        log_fail "HTTP $response"
    fi
    
    log_test "Protected endpoint without auth (should fail)"
    response=$(curl -s -w "%{http_code}" -o /dev/null "$BASE_URL/api/transcripts/recent" 2>/dev/null)
    if [ "$response" = "401" ] || [ "$response" = "403" ]; then
        log_pass
    else
        log_fail "Expected 401/403, got HTTP $response"
    fi
}

# ==============================================================================
# 3. Gemma API Endpoints
# ==============================================================================
test_gemma() {
    log_header "PHASE 3: Gemma AI Endpoints"
    
    if [ -z "$SESSION_TOKEN" ]; then
        log_skip "Gemma tests" "No valid session token"
        return
    fi
    
    log_test "Gemma stats endpoint"
    response=$(curl -s -w "%{http_code}" -o /tmp/gemma_stats.json "$BASE_URL/api/gemma/stats" \
        -b "ws_session=$SESSION_TOKEN" \
        -H "X-CSRF-Token: $CSRF_TOKEN" 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_pass
    else
        log_fail "HTTP $response"
    fi
    
    log_test "Gemma warmup endpoint"
    response=$(curl -s -w "%{http_code}" -o /tmp/gemma_warmup.json -X POST "$BASE_URL/api/gemma/warmup" \
        -b "ws_session=$SESSION_TOKEN" \
        -H "X-CSRF-Token: $CSRF_TOKEN" \
        -H "Content-Type: application/json" 2>/dev/null)
    # 200 or 202 are both valid
    if [ "$response" = "200" ] || [ "$response" = "202" ]; then
        log_pass
    else
        log_fail "HTTP $response"
    fi
    
    log_test "Gemma chat endpoint (simple test)"
    response=$(curl -s -w "%{http_code}" -o /tmp/gemma_chat.json -X POST "$BASE_URL/api/gemma/chat" \
        -b "ws_session=$SESSION_TOKEN" \
        -H "X-CSRF-Token: $CSRF_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"message":"Hello","context":"test"}' 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_pass
    else
        # May timeout or have GPU issues, warn but don't fail hard
        log_fail "HTTP $response (GPU may not be available)"
    fi
}

# ==============================================================================
# 4. Enterprise Endpoints
# ==============================================================================
test_enterprise() {
    log_header "PHASE 4: Enterprise Endpoints"
    
    if [ -z "$SESSION_TOKEN" ]; then
        log_skip "Enterprise tests" "No valid session token"
        return
    fi
    
    # Test each enterprise module's stats endpoint
    for module in qa automation knowledge analytics meetings; do
        log_test "Enterprise $module stats"
        response=$(curl -s -w "%{http_code}" -o /tmp/enterprise_${module}.json \
            "$BASE_URL/api/enterprise/$module/stats" \
            -b "ws_session=$SESSION_TOKEN" \
            -H "X-CSRF-Token: $CSRF_TOKEN" 2>/dev/null)
        if [ "$response" = "200" ]; then
            log_pass
        else
            log_fail "HTTP $response"
        fi
    done
}

# ==============================================================================
# 5. Frontend Pages
# ==============================================================================
test_frontend() {
    log_header "PHASE 5: Frontend Pages"
    
    pages=(
        "index.html"
        "gemma.html"
        "login.html"
        "admin_qa.html"
        "automation.html"
        "knowledge.html"
        "analytics.html"
        "meetings.html"
    )
    
    for page in "${pages[@]}"; do
        log_test "Frontend page: $page"
        response=$(curl -s -w "%{http_code}" -o /dev/null "$BASE_URL/ui/$page" 2>/dev/null)
        if [ "$response" = "200" ]; then
            log_pass
        else
            log_fail "HTTP $response"
        fi
    done
}

# ==============================================================================
# 6. RAG/Search Endpoints
# ==============================================================================
test_rag() {
    log_header "PHASE 6: RAG & Search Endpoints"
    
    if [ -z "$SESSION_TOKEN" ]; then
        log_skip "RAG tests" "No valid session token"
        return
    fi
    
    log_test "Semantic search endpoint"
    response=$(curl -s -w "%{http_code}" -o /tmp/search.json -X POST "$BASE_URL/api/search/semantic" \
        -b "ws_session=$SESSION_TOKEN" \
        -H "X-CSRF-Token: $CSRF_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"query":"test","top_k":5}' 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_pass
    else
        log_fail "HTTP $response"
    fi
    
    log_test "Analytics signals endpoint"
    response=$(curl -s -w "%{http_code}" -o /tmp/signals.json "$BASE_URL/api/analytics/signals" \
        -b "ws_session=$SESSION_TOKEN" \
        -H "X-CSRF-Token: $CSRF_TOKEN" 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_pass
    else
        log_fail "HTTP $response"
    fi
}

# ==============================================================================
# 7. GPU Coordinator Status
# ==============================================================================
test_gpu() {
    log_header "PHASE 7: GPU Coordinator"
    
    log_test "GPU coordinator status"
    response=$(curl -s -w "%{http_code}" -o /tmp/gpu_status.json "http://localhost:8002/gpu/status" 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_pass
        echo "    GPU State: $(cat /tmp/gpu_status.json | grep -o '"state"\s*:\s*"[^"]*"' | cut -d'"' -f4)"
    else
        log_fail "HTTP $response (GPU coordinator may not be running)"
    fi
}

# ==============================================================================
# Summary
# ==============================================================================
print_summary() {
    log_header "TEST SUMMARY"
    
    echo -e "\n  ${GREEN}Passed:${NC} $PASSED"
    echo -e "  ${RED}Failed:${NC} $FAILED"
    echo -e "  Total:  $((PASSED + FAILED))"
    
    if [ $FAILED -gt 0 ]; then
        echo -e "\n  ${RED}Failed Tests:${NC}"
        for test in "${TESTS[@]}"; do
            if [[ $test == FAIL* ]]; then
                echo "    - ${test#FAIL: }"
            fi
        done
    fi
    
    echo ""
    if [ $FAILED -eq 0 ]; then
        echo -e "  ${GREEN}✓ ALL TESTS PASSED!${NC}"
        exit 0
    else
        echo -e "  ${RED}✗ SOME TESTS FAILED${NC}"
        exit 1
    fi
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    echo -e "${BLUE}"
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║       NeMo Server Comprehensive Verification Suite           ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo "  Base URL: $BASE_URL"
    echo "  Date: $(date)"
    
    test_health
    test_auth
    test_gemma
    test_enterprise
    test_frontend
    test_rag
    test_gpu
    
    print_summary
}

main "$@"
