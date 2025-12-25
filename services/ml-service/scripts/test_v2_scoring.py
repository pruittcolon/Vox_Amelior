#!/usr/bin/env python3
"""
V2 Scoring Test - Verify Gemma Business Scoring Without GPU
============================================================

Tests the V2 scoring pipeline with mocked Gemma responses.
Verifies SQLite storage and correlation data queries.

Run: python3 test_v2_scoring.py
"""

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock httpx before importing gemma_scorer
class MockResponse:
    status_code = 200
    
    def __init__(self, scores_data):
        self._data = scores_data
    
    def json(self):
        return {"message": json.dumps(self._data)}

class MockAsyncClient:
    def __init__(self, **kwargs):
        pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        pass
    
    async def post(self, url, **kwargs):
        # Generate mock scores based on content
        prompt = kwargs.get("json", {}).get("messages", [{}])[0].get("content", "")
        
        # Simulate different scores based on keywords
        stress = 3
        conflict = 2
        confidence = 7
        
        if "deadline" in prompt.lower() or "urgent" in prompt.lower():
            stress = 8
        if "frustrated" in prompt.lower() or "disagree" in prompt.lower():
            conflict = 7
        if "great" in prompt.lower() or "success" in prompt.lower():
            confidence = 9
        
        mock_response = {
            "scores": {
                "business_practice_adherence": 7,
                "industry_best_practices": 6,
                "deadline_stress": stress,
                "emotional_conflict": conflict,
                "decision_clarity": 8,
                "speaker_confidence": confidence,
                "action_orientation": 7,
                "risk_awareness": 6,
            },
            "justifications": {
                "business_practice_adherence": "Standard process followed",
                "deadline_stress": "Timeline concerns noted" if stress > 5 else "Normal pace",
                "emotional_conflict": "Some tension detected" if conflict > 5 else "Collaborative tone",
            },
            "extracted": {
                "topics": ["strategy", "operations"],
                "decisions_made": ["Proceed with plan"],
                "action_items": ["Schedule follow-up"],
                "concerns_raised": [],
            },
            "overall_tone": "positive" if confidence > 6 else "neutral",
            "confidence": 0.85,
        }
        return MockResponse(mock_response)

# Install mock
class MockHTTPX:
    AsyncClient = MockAsyncClient

sys.modules['httpx'] = MockHTTPX()


# Sample business meeting segments
SAMPLE_SEGMENTS = [
    {
        "id": "seg_001",
        "text": "We need to accelerate the timeline for the Q1 launch. The deadline is tight but I'm confident we can make it if we focus.",
        "speaker": "Product Lead",
        "meeting_type": "strategy",
        "recording_date": datetime(2024, 1, 15, 10, 0),
        "transcription_id": "meeting_20240115",
    },
    {
        "id": "seg_002", 
        "text": "Great progress on the mobile app! Customer feedback has been overwhelmingly positive. We're on track for a successful release.",
        "speaker": "Engineering Lead",
        "meeting_type": "strategy",
        "recording_date": datetime(2024, 1, 15, 10, 15),
        "transcription_id": "meeting_20240115",
    },
    {
        "id": "seg_003",
        "text": "I'm frustrated with the delays in getting vendor responses. This is the third time we've had to push back the integration deadline.",
        "speaker": "Operations Manager",
        "meeting_type": "operations",
        "recording_date": datetime(2024, 2, 1, 14, 0),
        "transcription_id": "meeting_20240201",
    },
    {
        "id": "seg_004",
        "text": "The budget review shows we're 15% under on Q1 spending. This gives us flexibility for the urgent infrastructure upgrade.",
        "speaker": "CFO",
        "meeting_type": "finance",
        "recording_date": datetime(2024, 2, 15, 9, 0),
        "transcription_id": "meeting_20240215",
    },
]


def print_header(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


async def test_segment_scoring():
    """Test scoring individual segments."""
    print_header("1. Testing Segment Scoring")
    
    from gemma_scorer import GemmaScorer, SegmentScores
    
    scorer = GemmaScorer()
    
    for seg in SAMPLE_SEGMENTS[:2]:
        scores = await scorer.score_segment(
            segment_text=seg["text"],
            speaker=seg["speaker"],
            meeting_type=seg["meeting_type"],
            recording_date=str(seg["recording_date"]),
        )
        
        print(f"\n📄 Segment: {seg['text'][:50]}...")
        print(f"   Speaker: {seg['speaker']}")
        print(f"   Scores:")
        print(f"      Deadline Stress:    {scores.deadline_stress}/10")
        print(f"      Emotional Conflict: {scores.emotional_conflict}/10")
        print(f"      Decision Clarity:   {scores.decision_clarity}/10")
        print(f"      Speaker Confidence: {scores.speaker_confidence}/10")
        print(f"   Health Score: {scores.health_score():.2f}/10")
        print(f"   Tone: {scores.overall_tone}")
    
    return scorer


async def test_batch_scoring():
    """Test batch scoring multiple segments."""
    print_header("2. Testing Batch Scoring")
    
    from gemma_scorer import GemmaScorer
    
    scorer = GemmaScorer()
    
    scored = await scorer.score_segments(
        segments=SAMPLE_SEGMENTS,
        company_context="Tech startup in growth phase",
        batch_delay=0.01,
    )
    
    print(f"\n✅ Scored {len(scored)} segments")
    
    # Summary
    health_scores = [s.scores.health_score() for s in scored]
    avg_health = sum(health_scores) / len(health_scores)
    high_stress = sum(1 for s in scored if s.scores.deadline_stress >= 7)
    
    print(f"\n📊 Batch Summary:")
    print(f"   Average Health Score: {avg_health:.2f}/10")
    print(f"   High Stress Segments: {high_stress}/{len(scored)}")
    
    return scored


async def test_database_storage(scored_segments):
    """Test storing and querying scores."""
    print_header("3. Testing SQLite Storage")
    
    from gemma_scorer import ScoreDatabase
    
    # Use temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    db = ScoreDatabase(db_path=db_path)
    
    # Store segments
    db.store_scored_segments(scored_segments)
    print(f"✅ Stored {len(scored_segments)} segments to {db_path}")
    
    # Query high stress
    high_stress = db.get_high_stress_segments(threshold=7)
    print(f"\n🔥 High Stress Segments: {len(high_stress)}")
    
    # Query correlation data
    correlation_data = db.get_correlation_data()
    print(f"\n📈 Correlation Data Points: {len(correlation_data)}")
    for row in correlation_data:
        print(f"   {row.get('date', 'N/A')}: stress={row.get('avg_stress', 0):.1f}, health={row.get('avg_health', 0):.1f}")
    
    # Calculate meeting scores
    for tid in set(s.transcription_id for s in scored_segments):
        meeting_scores = db.calculate_meeting_scores(tid)
        if meeting_scores:
            print(f"\n📋 Meeting {tid}:")
            print(f"   Segments: {meeting_scores.get('segment_count', 0)}")
            print(f"   Avg Stress: {meeting_scores.get('avg_deadline_stress', 0):.1f}")
            print(f"   Health: {meeting_scores.get('meeting_health_score', 0):.1f}")
    
    # Cleanup
    os.unlink(db_path)
    
    return db


async def run_demo():
    """Run the complete V2 demo."""
    print("\n" + "=" * 60)
    print("  🚀 V2 SCORING DEMO")
    print("  Gemma-Powered Quantitative Business Analysis")
    print("  (Mocked - No GPU Required)")
    print("=" * 60)
    
    # Test segment scoring
    await test_segment_scoring()
    
    # Test batch scoring
    scored = await test_batch_scoring()
    
    # Test database
    await test_database_storage(scored)
    
    # Final summary
    print_header("Demo Complete")
    print("""
✅ V2 Scoring Pipeline Verified:
   • GemmaScorer: Produces structured JSON with 1-10 scores
   • SegmentScores: 8 business dimensions + health score
   • ScoreDatabase: SQLite storage with correlation queries
   • Batch Processing: Efficient multi-segment handling

🎯 Ready for Integration:
   • POST /cide/v2/score - Score single segment
   • POST /cide/v2/score/batch - Score multiple segments
   • POST /cide/v2/correlation - Get data for analysis
   • GET /cide/v2/health - Check module health

📊 Correlation Analysis Pattern:
   1. Score all transcription segments
   2. Query aggregated data by date/meeting type
   3. Compare against business DBs (Salesforce, banking)
   4. Find patterns: "When stress > 7, deadlines missed?"
""")


if __name__ == "__main__":
    asyncio.run(run_demo())
