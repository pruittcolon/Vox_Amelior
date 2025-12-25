#!/usr/bin/env python3
"""
Minimal Gemma V2 Test - Single Chunk
=====================================

Tests V2 scoring with real Gemma using a small, optimized prompt
designed for 6GB VRAM.

Run: python3 test_gemma_mini.py
"""

import asyncio
import json
import sys

# Gemma service URL - Internal Docker Network
GEMMA_URL = "http://gemma-service:8001"


# MINIMAL prompt - optimized for 6GB VRAM
# ~800 tokens total (prompt + response)
MINI_PROMPT = """Analyze this earnings call segment. Return JSON only.

TEXT:
"{text}"

Return ONLY this JSON (no other text):
{{"stress":1-10,"confidence":1-10,"tone":"positive|negative|neutral","summary":"one sentence"}}"""


def get_service_token():
    """Generate a service-to-service token using internal library."""
    try:
        import sys
        import os
        
        # Add app to path to find shared modules
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
            
        try:
            from shared.security.service_auth import get_service_auth, load_service_jwt_keys
        except ImportError:
            # Fallback if running outside container structure (shouldn't happen in this test)
            print("⚠️ Could not import shared.security.service_auth")
            return ""

        # Load keys for ml-service
        # We act as ml-service talking to gemma-service
        try:
            keys = load_service_jwt_keys("ml-service")
            auth = get_service_auth("ml-service", keys)
            
            # Create token with correct audience
            token = auth.create_token(aud="internal")
            return token
            
        except Exception as e:
            print(f"⚠️ Error creating token with internal lib: {e}")
            import traceback
            traceback.print_exc()
            return ""

    except Exception as e:
        print(f"⚠️ Could not generate token: {e}")
        return ""


async def test_gemma_health():
    """Check if Gemma is reachable."""
    import httpx
    
    print("🔍 Checking Gemma service...")
    token = get_service_token()
    headers = {"X-Service-Token": token}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Health might not need auth, but chat does
            r = await client.get(f"{GEMMA_URL}/health")
            if r.status_code == 200:
                print(f"✅ Gemma is healthy: {r.json()}")
                print(f"   Using Auth Token: {token[:10]}...")
                return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    return False


async def score_one_chunk(text: str) -> dict:
    """Score a single text chunk with Gemma."""
    import httpx
    
    # Keep text very short for 6GB VRAM
    text = text[:500]
    
    prompt = MINI_PROMPT.format(text=text)
    token = get_service_token()
    headers = {"X-Service-Token": token}
    
    print(f"\n📝 Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
    
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{GEMMA_URL}/chat",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,  # Small response
                    "temperature": 0.1,
                },
                headers=headers
            )
            
            if response.status_code != 200:
                print(f"❌ Gemma error: {response.status_code}")
                print(f"   Response: {response.text[:300]}")
                return None
            
            data = response.json()
            response_text = data.get("message", "") or data.get("response", "")
            
            print(f"\n📤 Gemma response:")
            print(f"   {response_text[:500]}")
            
            # Try to parse JSON
            try:
                # Find JSON in response
                import re
                json_match = re.search(r'\{[^}]+\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parse error: {e}")
            
            return {"raw_response": response_text}
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def main():
    print("\n" + "=" * 60)
    print("  🧪 MINIMAL GEMMA V2 TEST")
    print("  Testing single chunk with 6GB VRAM-safe prompt")
    print("=" * 60)
    
    # Check health first
    healthy = await test_gemma_health()
    
    if not healthy:
        print("\n⚠️ Gemma service not reachable!")
        print("   Start Gemma and retry, or check the URL:")
        print(f"   Current: {GEMMA_URL}")
        print("\n   To start: docker compose up gemma-service")
        return
    
    # Test sample from NVIDIA earnings call
    sample = """Thank you. Good afternoon, everyone, and welcome to NVIDIA's conference call for the second quarter of fiscal 2021. With me on the call today from NVIDIA are Jensen Huang, President and Chief Executive Officer; and Colette Kress, Executive Vice President and Chief Financial Officer. I'd like to remind you that our call is being webcast live."""
    
    print("\n" + "=" * 60)
    print("  📄 Testing with NVIDIA earnings call sample")
    print("=" * 60)
    
    result = await score_one_chunk(sample)
    
    if result:
        print("\n" + "=" * 60)
        print("  ✅ RESULT")
        print("=" * 60)
        print(f"\n{json.dumps(result, indent=2)}")
        
        if "stress" in result:
            print(f"\n📊 Parsed Scores:")
            print(f"   Stress:     {result.get('stress', '?')}/10")
            print(f"   Confidence: {result.get('confidence', '?')}/10")
            print(f"   Tone:       {result.get('tone', '?')}")
            print(f"   Summary:    {result.get('summary', '?')}")
    else:
        print("\n❌ Failed to get result from Gemma")


if __name__ == "__main__":
    try:
        import httpx
    except ImportError:
        print("Installing httpx...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "httpx", "--quiet", "--break-system-packages"])
        import httpx
    
    asyncio.run(main())
