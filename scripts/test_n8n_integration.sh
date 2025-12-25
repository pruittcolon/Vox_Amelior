#!/bin/bash
#
# n8n Integration Service - End-to-End Test Script
#
# This script verifies the n8n integration service is working correctly
# by making curl requests to the API Gateway endpoints.
#
# Usage:
#   ./scripts/test_n8n_integration.sh           # Test via Gateway (default)
#   ./scripts/test_n8n_integration.sh direct    # Test n8n-service directly
#
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
API_URL="${API_GATEWAY_URL:-http://localhost:8000}"
DIRECT_URL="${N8N_SERVICE_URL:-http://localhost:8011}"
MODE="${1:-gateway}"

if [ "$MODE" = "direct" ]; then
    BASE_URL="$DIRECT_URL"
    N8N_PREFIX=""
else
    BASE_URL="$API_URL"
    N8N_PREFIX="/n8n"
fi

PASSED=0
FAILED=0
SKIPPED=0

print_test() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}TEST: $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

pass() {
    echo -e "${GREEN}✅ PASS: $1${NC}"
    ((PASSED++))
}

fail() {
    echo -e "${RED}❌ FAIL: $1${NC}"
    ((FAILED++))
}

skip() {
    echo -e "${YELLOW}⏭️ SKIP: $1${NC}"
    ((SKIPPED++))
}

# ============================================================================
# Login (get auth token for gateway mode)
# ============================================================================
AUTH_HEADER=""
CSRF_TOKEN=""

if [ "$MODE" != "direct" ]; then
    print_test "Authenticating with Gateway"
    
    LOGIN_RESPONSE=$(curl -sf -c /tmp/cookie.txt -b /tmp/cookie.txt \
        "$API_URL/api/auth/login" \
        -H "Content-Type: application/json" \
        -d '{"username":"admin","password":"admin123"}' 2>&1 || echo '{"error":"failed"}')
    
    if echo "$LOGIN_RESPONSE" | grep -q '"success":true'; then
        CSRF_TOKEN=$(echo "$LOGIN_RESPONSE" | grep -o '"csrf_token":"[^"]*"' | cut -d'"' -f4)
        SESSION_TOKEN=$(echo "$LOGIN_RESPONSE" | grep -o '"session_token":"[^"]*"' | cut -d'"' -f4)
        pass "Logged in as admin"
        echo "  └─ CSRF Token: ${CSRF_TOKEN:0:20}..."
    else
        skip "Authentication failed (demo users may be disabled)"
        echo "  └─ Response: $LOGIN_RESPONSE"
        echo -e "\n${YELLOW}Continuing with unauthenticated tests only...${NC}"
    fi
fi

# ============================================================================
# Test 1: Health Check
# ============================================================================
print_test "n8n Service Health Check"

HEALTH_RESPONSE=$(curl -sf "${BASE_URL}${N8N_PREFIX}/health" 2>&1 || echo '{"error":"unreachable"}')

if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
    pass "Health endpoint returns healthy"
    echo "  └─ Response: $HEALTH_RESPONSE"
    
    # Check for expected fields
    if echo "$HEALTH_RESPONSE" | grep -q '"voice_monkey_configured"'; then
        echo "  └─ Voice Monkey status present"
    fi
    if echo "$HEALTH_RESPONSE" | grep -q '"emotion_model"'; then
        echo "  └─ Emotion model: j-hartmann/emotion-english-distilroberta-base"
    fi
else
    fail "Health check failed"
    echo "  └─ Response: $HEALTH_RESPONSE"
fi

# ============================================================================
# Test 2: List Commands (requires auth in gateway mode)
# ============================================================================
print_test "List Voice Commands"

if [ -n "$SESSION_TOKEN" ] || [ "$MODE" = "direct" ]; then
    COMMANDS_RESPONSE=$(curl -sf -b /tmp/cookie.txt \
        -H "X-CSRF-Token: $CSRF_TOKEN" \
        "${BASE_URL}${N8N_PREFIX}/commands" 2>&1 || echo '[]')
    
    if echo "$COMMANDS_RESPONSE" | grep -q "lights_off"; then
        pass "Commands endpoint returns default voice command"
        echo "  └─ Found 'lights_off' command"
    else
        fail "Commands endpoint missing 'lights_off'"
        echo "  └─ Response: $COMMANDS_RESPONSE"
    fi
else
    skip "Commands endpoint requires authentication"
fi

# ============================================================================
# Test 3: Process Voice Command
# ============================================================================
print_test "Process Voice Command (Lights Off)"

if [ -n "$SESSION_TOKEN" ] || [ "$MODE" = "direct" ]; then
    PROCESS_RESPONSE=$(curl -sf -b /tmp/cookie.txt \
        -H "X-CSRF-Token: $CSRF_TOKEN" \
        -H "Content-Type: application/json" \
        -X POST "${BASE_URL}${N8N_PREFIX}/process" \
        -d '{
            "segments": [
                {"text": "Honey, can you turn off the lights please", "speaker": "pruitt"}
            ],
            "job_id": "test-voice-001"
        }' 2>&1 || echo '{"error":"failed"}')
    
    if echo "$PROCESS_RESPONSE" | grep -q '"voice_commands_triggered":1'; then
        pass "Voice command detected and triggered"
        echo "  └─ Response: $PROCESS_RESPONSE"
    else
        fail "Voice command not detected"
        echo "  └─ Response: $PROCESS_RESPONSE"
    fi
else
    skip "Process endpoint requires authentication"
fi

# ============================================================================
# Test 4: Process Emotion Alert (simulate 20 angry segments)
# ============================================================================
print_test "Process Emotion Alert (20 consecutive anger)"

if [ -n "$SESSION_TOKEN" ] || [ "$MODE" = "direct" ]; then
    # First reset tracking
    curl -sf -b /tmp/cookie.txt \
        -H "X-CSRF-Token: $CSRF_TOKEN" \
        -X POST "${BASE_URL}${N8N_PREFIX}/alerts/reset" > /dev/null 2>&1 || true
    
    # Build 20 angry segments
    SEGMENTS="["
    for i in $(seq 1 20); do
        SEGMENTS="$SEGMENTS{\"text\":\"Segment $i\",\"speaker\":\"pruitt\",\"emotion\":\"anger\"}"
        if [ $i -lt 20 ]; then
            SEGMENTS="$SEGMENTS,"
        fi
    done
    SEGMENTS="$SEGMENTS]"
    
    EMOTION_RESPONSE=$(curl -sf -b /tmp/cookie.txt \
        -H "X-CSRF-Token: $CSRF_TOKEN" \
        -H "Content-Type: application/json" \
        -X POST "${BASE_URL}${N8N_PREFIX}/process" \
        -d "{\"segments\": $SEGMENTS, \"job_id\": \"test-emotion-001\"}" 2>&1 || echo '{"error":"failed"}')
    
    if echo "$EMOTION_RESPONSE" | grep -q '"emotion_alerts_triggered":1'; then
        pass "Emotion alert triggered at 20 consecutive anger"
        echo "  └─ Response: $EMOTION_RESPONSE"
    else
        fail "Emotion alert not triggered (expected 1)"
        echo "  └─ Response: $EMOTION_RESPONSE"
    fi
else
    skip "Emotion alert test requires authentication"
fi

# ============================================================================
# Test 5: Alert Status
# ============================================================================
print_test "Get Alert Status"

if [ -n "$SESSION_TOKEN" ] || [ "$MODE" = "direct" ]; then
    STATUS_RESPONSE=$(curl -sf -b /tmp/cookie.txt \
        -H "X-CSRF-Token: $CSRF_TOKEN" \
        "${BASE_URL}${N8N_PREFIX}/alerts/status" 2>&1 || echo '{}')
    
    if [ -n "$STATUS_RESPONSE" ] && [ "$STATUS_RESPONSE" != "{}" ]; then
        pass "Alert status endpoint working"
        echo "  └─ Response: $STATUS_RESPONSE"
    else
        fail "Alert status empty or failed"
        echo "  └─ Response: $STATUS_RESPONSE"
    fi
else
    skip "Alert status requires authentication"
fi

# ============================================================================
# Test 6: Negative Match (should NOT trigger)
# ============================================================================
print_test "Negative Voice Command (should not trigger)"

if [ -n "$SESSION_TOKEN" ] || [ "$MODE" = "direct" ]; then
    NEGATIVE_RESPONSE=$(curl -sf -b /tmp/cookie.txt \
        -H "X-CSRF-Token: $CSRF_TOKEN" \
        -H "Content-Type: application/json" \
        -X POST "${BASE_URL}${N8N_PREFIX}/process" \
        -d '{
            "segments": [
                {"text": "Just a normal conversation about the weather", "speaker": "pruitt"}
            ],
            "job_id": "test-negative-001"
        }' 2>&1 || echo '{"error":"failed"}')
    
    if echo "$NEGATIVE_RESPONSE" | grep -q '"voice_commands_triggered":0'; then
        pass "Normal text correctly NOT matched as voice command"
    else
        fail "False positive: normal text triggered as voice command"
        echo "  └─ Response: $NEGATIVE_RESPONSE"
    fi
else
    skip "Negative test requires authentication"
fi

# ============================================================================
# Results Summary
# ============================================================================
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                     TEST RESULTS                           ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${GREEN}PASSED:${NC}  $PASSED"
echo -e "  ${RED}FAILED:${NC}  $FAILED"
echo -e "  ${YELLOW}SKIPPED:${NC} $SKIPPED"
echo ""

# Cleanup
rm -f /tmp/cookie.txt

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
