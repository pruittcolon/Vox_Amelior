#!/usr/bin/env python3
"""
QA Pipeline Test Script
Tests the full QA flow: Auth -> Chunking -> Gemma Analysis -> Vectorization
"""
import asyncio
import json
import time
import httpx

BASE_URL = "http://localhost:8000"

SAMPLE_TRANSCRIPT = """
AGENT: Thank you for calling Service Credit Union, my name is Sarah. May I have your name please?
MEMBER: Hi Sarah, this is John Smith. I'm calling about my checking account.
AGENT: Hello John, thank you for being a valued member. Before I can access your account information, I'll need to verify your identity. Can you please provide your date of birth?
MEMBER: Sure, it's January 15th, 1985.
AGENT: Thank you. And can you confirm the last four digits of your Social Security Number?
MEMBER: Yes, it's 1234.
AGENT: Perfect, I've verified your identity. How can I assist you with your checking account today?
MEMBER: I noticed there's a charge on my account for $150 that I don't recognize.
AGENT: I understand how concerning that can be. Let me pull up your recent transactions. I can see the charge from December 14th. Would you like me to open a dispute for this transaction?
MEMBER: Yes please, I didn't make that purchase.
AGENT: I've initiated a dispute investigation. You'll receive a provisional credit within 3-5 business days. Is there anything else I can help you with?
MEMBER: No, that's all. Thank you for your help.
AGENT: You're welcome, John. Thank you for being a member. Have a great day!
"""

async def test_qa_pipeline():
    """Test the full QA pipeline"""
    print("=" * 60)
    print("🧪 QA Pipeline Test - Starting")
    print("=" * 60)
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0, cookies=httpx.Cookies()) as client:
        # Step 1: Login
        print("\n📋 Step 1: Authentication")
        login_resp = await client.post("/api/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        if login_resp.status_code != 200:
            print(f"❌ Login failed: {login_resp.status_code}")
            print(login_resp.text)
            return
        
        login_data = login_resp.json()
        token = login_data.get("session_token")
        csrf = login_data.get("csrf_token")
        
        # Update cookies from response
        for name, value in login_resp.cookies.items():
            client.cookies.set(name, value)
        
        # Also set ws_csrf cookie if present
        if csrf:
            client.cookies.set("ws_csrf", csrf)
        
        print(f"✅ Logged in (token: {len(token)} chars, csrf: {len(csrf) if csrf else 0} chars)")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "X-CSRF-Token": csrf,
            "Content-Type": "application/json"
        }
        
        # Step 2: Test Chunking
        print("\n📋 Step 2: Testing Chunking")
        chunk_resp = await client.post(
            "/api/v1/calls/qa/test-chunking",
            headers=headers,
            json={"transcript": SAMPLE_TRANSCRIPT}
        )
        
        if chunk_resp.status_code != 200:
            print(f"❌ Chunking failed: {chunk_resp.status_code}")
            print(chunk_resp.text[:500])
            return
        
        chunk_data = chunk_resp.json()
        print(f"✅ Chunking successful: {chunk_data.get('total_chunks', 0)} chunks")
        print(f"   Average tokens per chunk: {chunk_data.get('avg_tokens', 0):.1f}")
        
        for c in chunk_data.get("chunks", [])[:2]:
            print(f"   - Chunk {c['index']}: {c['token_count']} tokens, speaker: {c['primary_speaker']}")
        
        # Step 3: Full QA Processing
        print("\n📋 Step 3: Full QA Processing (this may take 30-60 seconds)")
        print("   ⏳ Calling /api/v1/calls/qa/process...")
        
        start_time = time.time()
        qa_resp = await client.post(
            "/api/v1/calls/qa/process",
            headers=headers,
            json={
                "call_id": f"test-{int(time.time())}",
                "agent_id": "agent-sarah",
                "transcript": SAMPLE_TRANSCRIPT
            }
        )
        elapsed = time.time() - start_time
        
        if qa_resp.status_code != 200:
            print(f"❌ QA Processing failed: {qa_resp.status_code}")
            print(qa_resp.text[:1000])
            return
        
        qa_data = qa_resp.json()
        
        print(f"\n✅ QA Processing Complete in {elapsed:.1f}s")
        print(f"   Call ID: {qa_data.get('call_id')}")
        print(f"   Chunks analyzed: {qa_data.get('chunk_count', 0)}")
        print(f"   Total processing time: {qa_data.get('total_processing_time_sec', 0):.1f}s")
        
        # Print scores
        avg_scores = qa_data.get("avg_scores", {})
        print("\n📊 Average Scores:")
        for key, val in avg_scores.items():
            icon = "🟢" if val >= 7 else "🟡" if val >= 5 else "🔴"
            print(f"   {icon} {key}: {val}")
        
        # Check for default scores (all 5.0 = problem)
        scores = list(avg_scores.values())
        if scores and all(s == 5.0 for s in scores):
            print("\n⚠️  WARNING: All scores are 5.0 - this indicates Gemma analysis failed!")
        else:
            print("\n✅ Scores look legitimate (not all defaults)")
        
        # Check vectorization
        chunks = qa_data.get("chunks", [])
        vectorized = sum(1 for c in chunks if c.get("vector_id"))
        print(f"\n📦 Vectorization: {vectorized}/{len(chunks)} chunks stored in RAG")
        
        # Check compliance flags
        flags_count = qa_data.get("compliance_flags_count", 0)
        review_count = qa_data.get("requires_review_count", 0)
        print(f"🚩 Compliance flags: {flags_count}")
        print(f"📝 Requires review: {review_count}")
        
        print("\n" + "=" * 60)
        print("🎉 QA Pipeline Test Complete!")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_qa_pipeline())
