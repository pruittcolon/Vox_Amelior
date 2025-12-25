#!/usr/bin/env python3
"""
Cache Comparison Test Script

Tests Gemma model output quality with different cache configurations.
Uses the planet question as a benchmark.

Usage:
    python3 scripts/test_cache_quality.py
"""

import json
import subprocess
import sys
import time
from datetime import datetime


PLANET_PROMPT = """Name every single planet in the solar system. For each planet, provide:
1. Number of known moons
2. One interesting fact

Format as a numbered list."""


def get_gemma_health():
    """Get current Gemma service health status."""
    try:
        result = subprocess.run(
            ["docker", "exec", "refactored_gemma", "curl", "-s", "http://localhost:8001/health"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return json.loads(result.stdout) if result.returncode == 0 else None
    except Exception as e:
        print(f"Error getting health: {e}")
        return None


def get_service_token():
    """Get a valid service token for API calls."""
    try:
        result = subprocess.run(
            ["docker", "exec", "refactored_gateway", "curl", "-s", 
             "-X", "POST", "http://localhost:8000/api/v1/auth/service-token",
             "-H", "Content-Type: application/json",
             "-d", '{"service_id": "test-client"}'],
            capture_output=True,
            text=True,
            timeout=10
        )
        data = json.loads(result.stdout)
        return data.get("token")
    except Exception:
        return None


def call_gemma_chat(prompt: str, max_tokens: int = 512) -> dict:
    """Call Gemma via the gateway chat endpoint."""
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    })
    
    try:
        # Call internal Gemma endpoint directly via docker
        result = subprocess.run(
            ["docker", "exec", "refactored_gemma", "curl", "-s",
             "-X", "POST", "http://localhost:8001/chat",
             "-H", "Content-Type: application/json",
             "-H", "X-Service-Token: internal-test",
             "-d", payload],
            capture_output=True,
            text=True,
            timeout=120  # 2 min timeout for generation
        )
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {"error": f"curl failed: {result.stderr}"}
            
    except subprocess.TimeoutExpired:
        return {"error": "Request timed out (120s)"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}", "raw": result.stdout[:500]}
    except Exception as e:
        return {"error": str(e)}


def run_test():
    """Run the planet test and report results."""
    print("=" * 70)
    print("GEMMA CACHE QUALITY TEST")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Get health info
    health = get_gemma_health()
    if not health:
        print("❌ Could not reach Gemma service")
        return False
    
    print(f"\n📊 Current Configuration:")
    print(f"   Model: {health.get('model_path', 'unknown')}")
    print(f"   GPU: {health.get('model_on_gpu', False)}")
    print(f"   Context: {health.get('context_size', 'unknown')}")
    print(f"   Cache Type: {health.get('cache_type', 'not set (FP16 default)')}")
    print(f"   VRAM: {health.get('vram_used_mb', '?')}/{health.get('vram_total_mb', '?')} MB")
    
    print(f"\n🔄 Sending test prompt...")
    print(f"   Prompt: {PLANET_PROMPT[:60]}...")
    
    start_time = time.time()
    response = call_gemma_chat(PLANET_PROMPT, max_tokens=800)
    elapsed = time.time() - start_time
    
    if "error" in response:
        print(f"\n❌ Error: {response['error']}")
        if "raw" in response:
            print(f"   Raw response: {response['raw']}")
        return False
    
    # Extract response text
    text = response.get("text", response.get("response", ""))
    if not text and "choices" in response:
        text = response["choices"][0].get("text", "")
    
    print(f"\n✅ Response received in {elapsed:.2f}s")
    print(f"\n{'=' * 70}")
    print("RESPONSE:")
    print("=" * 70)
    print(text if text else json.dumps(response, indent=2))
    print("=" * 70)
    
    # Count planets mentioned
    planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    found = [p for p in planets if p.lower() in text.lower()]
    print(f"\n📊 Planets mentioned: {len(found)}/8 ({', '.join(found)})")
    
    # Save result
    result_file = f"/tmp/gemma_test_{health.get('cache_type', 'default')}_{int(time.time())}.json"
    with open(result_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "health": health,
            "prompt": PLANET_PROMPT,
            "response": text,
            "elapsed_seconds": elapsed,
            "planets_found": len(found)
        }, f, indent=2)
    print(f"\n💾 Results saved to: {result_file}")
    
    return True


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
