#!/bin/bash
#
# VRAM High Context Stress Test
# Tests 24K, 32K, and 64K context sizes
#
# Usage: ./test_high_context.sh
#
# Requirements:
# - Docker running
# - Gemma service available
# - nvidia-smi installed
#

set -e

echo "============================================================"
echo "🧪 HIGH CONTEXT VRAM STRESS TEST"
echo "    Testing: 24K, 32K, 64K context with Q8_0 cache"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to get VRAM usage
get_vram() {
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1
}

# Function to test a context size
test_context() {
    local ctx=$1
    local cache=$2
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Testing: Context=$ctx, Cache=$cache"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Get baseline VRAM
    VRAM_BEFORE=$(get_vram)
    echo "   VRAM before: $VRAM_BEFORE"
    
    # Stop Gemma
    echo "   Stopping Gemma..."
    docker stop refactored_gemma 2>/dev/null || true
    sleep 3
    
    # Start with new context
    echo "   Starting Gemma with context=$ctx, cache=$cache..."
    cd /home/pruittcolon/Desktop/Nemo_Server/docker
    GEMMA_CONTEXT_SIZE=$ctx GEMMA_CACHE_TYPE=$cache docker compose up -d gemma-service 2>&1 | tail -3
    
    # Wait for healthy
    echo "   Waiting for Gemma to load model..."
    local attempts=0
    local max_attempts=120
    while [ $attempts -lt $max_attempts ]; do
        if docker exec refactored_gemma curl -sf http://localhost:8001/health >/dev/null 2>&1; then
            HEALTH=$(docker exec refactored_gemma curl -s http://localhost:8001/health)
            if echo "$HEALTH" | grep -q '"model_loaded":true'; then
                echo -e "   ${GREEN}✓ Gemma healthy after $((attempts * 2))s${NC}"
                break
            fi
        fi
        sleep 2
        attempts=$((attempts + 1))
        if [ $((attempts % 15)) -eq 0 ]; then
            echo "   Still loading... ${attempts}s elapsed"
        fi
    done
    
    if [ $attempts -ge $max_attempts ]; then
        echo -e "   ${RED}✗ Gemma failed to start within ${max_attempts}s${NC}"
        return 1
    fi
    
    # Get VRAM after load
    sleep 5
    VRAM_AFTER=$(get_vram)
    echo "   VRAM after: $VRAM_AFTER"
    
    # Get health details
    HEALTH=$(docker exec refactored_gemma curl -s http://localhost:8001/health 2>/dev/null)
    VRAM_REPORTED=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('vram_used_mb', '?'))" 2>/dev/null || echo "?")
    CACHE_REPORTED=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('cache_type', '?'))" 2>/dev/null || echo "?")
    CTX_REPORTED=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('context_size', '?'))" 2>/dev/null || echo "?")
    
    echo "   Model reports: VRAM=${VRAM_REPORTED}MB, cache=${CACHE_REPORTED}, context=${CTX_REPORTED}"
    
    # Test generation with a longer prompt to use context
    echo "   Testing generation..."
    PROMPT="You are an AI assistant. Here is a long context test. Please summarize the Transformer architecture in 3 sentences: The Transformer uses self-attention to process all positions simultaneously. It consists of encoder and decoder stacks with multi-head attention and feed-forward layers. The key innovation is eliminating recurrence for parallelization."
    
    START=$(date +%s.%N)
    RESULT=$(docker exec refactored_gemma curl -sf -X POST http://localhost:8001/generate \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": 100, \"temperature\": 0.7}" 2>&1)
    END=$(date +%s.%N)
    GEN_TIME=$(echo "$END - $START" | bc)
    
    if echo "$RESULT" | grep -q '"text"\|"response"'; then
        RESPONSE=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print((d.get('text') or d.get('response',''))[:100])" 2>/dev/null || echo "?")
        echo -e "   ${GREEN}✓ Generation succeeded in ${GEN_TIME}s${NC}"
        echo "   Response: ${RESPONSE}..."
        return 0
    else
        echo -e "   ${RED}✗ Generation failed: $RESULT${NC}"
        return 1
    fi
}

# Run tests
RESULTS=()

for CTX in 24576 32768 65536; do
    if test_context $CTX "q8_0"; then
        RESULTS+=("$CTX|q8_0|PASS")
    else
        RESULTS+=("$CTX|q8_0|FAIL")
    fi
done

# Print summary
echo ""
echo "============================================================"
echo "📋 HIGH CONTEXT STRESS TEST RESULTS"
echo "============================================================"
printf "%-10s | %-8s | %-8s\n" "Context" "Cache" "Status"
echo "------------------------------------------------------------"
for R in "${RESULTS[@]}"; do
    IFS='|' read -r CTX CACHE STATUS <<< "$R"
    if [ "$STATUS" == "PASS" ]; then
        printf "%-10s | %-8s | ${GREEN}✅ PASS${NC}\n" "$CTX" "$CACHE"
    else
        printf "%-10s | %-8s | ${RED}❌ FAIL${NC}\n" "$CTX" "$CACHE"
    fi
done

echo ""
echo "============================================================"
echo "🏁 Test complete! Final VRAM: $(get_vram)"
echo "============================================================"

# Reset to safe default
echo ""
echo "Resetting to safe default (8192 context)..."
GEMMA_CONTEXT_SIZE=8192 GEMMA_CACHE_TYPE=q8_0 docker compose up -d gemma-service 2>&1 | tail -1
echo "Done!"
