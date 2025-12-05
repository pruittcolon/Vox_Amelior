#!/bin/bash
set -e

echo "🚀 Starting NeMo Server Test Suite..."

# 1. Run ML Service Tests
echo "🧪 Running ML Service Tests..."
pytest services/ml-service/tests -v

# 2. Run Gateway Tests (if they exist)
if [ -d "services/api-gateway/tests" ]; then
    echo "🧪 Running API Gateway Tests..."
    pytest services/api-gateway/tests -v
fi

# 3. Run Gemma Service Tests (if they exist)
if [ -d "services/gemma-service/tests" ]; then
    echo "🧪 Running Gemma Service Tests..."
    pytest services/gemma-service/tests -v || {
        if [ $? -eq 5 ]; then
            echo "⚠️  No tests found for Gemma Service (skipping)"
        else
            exit 1
        fi
    }
fi

echo "✅ All tests passed!"
