import asyncio
import glob
import logging
import os
import re
import sys
from datetime import datetime

# Add project root to path to allow importing from src and shared
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # ml-service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))) # Nemo_Server (for shared)

try:
    from src.gemma_scorer import get_gemma_scorer, get_score_database, ScoredSegment
except ImportError:
    # Container environment: src contents are in /app root
    sys.path.append("/app")
    from gemma_scorer import get_gemma_scorer, get_score_database, ScoredSegment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRANSCRIPT_DIR = os.path.join(os.path.dirname(__file__), "../data/earnings_transcripts")

def chunk_transcript(text: str, chunk_size: int = 1500) -> list[dict]:
    """
    Split transcript into analyzable chunks.
    Preserves speaker context and timestamps.
    """
    # Split by double newline or timestamp patterns to identify sections
    # This is a simple heuristic; can be improved
    lines = text.split('\n')
    chunks = []
    
    current_chunk = []
    current_length = 0
    current_speaker = "Unknown"
    current_timestamp = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try to detect speaker (e.g., "John Smith:")
        speaker_match = re.match(r'^\[?([\d:]+)?\]?\s*([A-Za-z\s]+)(?:\s\((.+?)\))?:\s', line)
        if speaker_match:
            # If we have a current chunk, see if we should save it before changing speaker
            if current_length > chunk_size:
                chunks.append({
                    "text": "\n".join(current_chunk),
                    "speaker": current_speaker,
                    "recording_date": current_timestamp or datetime.now().isoformat()
                })
                current_chunk = []
                current_length = 0
            
            # Update context
            if speaker_match.group(1):
                current_timestamp = speaker_match.group(1)
            current_speaker = speaker_match.group(2).strip()
        
        current_chunk.append(line)
        current_length += len(line)
        
        # If chunk is too big, split it
        if current_length >= chunk_size:
            chunks.append({
                "text": "\n".join(current_chunk),
                "speaker": current_speaker,
                "recording_date": current_timestamp or datetime.now().isoformat()
            })
            current_chunk = []
            current_length = 0
            
    # Add final chunk
    if current_chunk:
        chunks.append({
            "text": "\n".join(current_chunk),
            "speaker": current_speaker,
            "recording_date": current_timestamp or datetime.now().isoformat()
        })
        
    return chunks

async def analyze_file(file_path: str, scorer, db):
    """Analyze a single transcript file."""
    filename = os.path.basename(file_path)
    logger.info(f"Processing {filename}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # extract metadata from filename (e.g. ABC_Q4_2024.txt)
    company = "Unknown"
    meeting_type = "Earnings Call"
    if "_" in filename:
        parts = filename.split("_")
        company = parts[0]
        
    segments = chunk_transcript(text)
    logger.info(f"-> Generated {len(segments)} segments for {filename}")
    
    # Process segments
    # Enrich segments with file metadata
    transcription_id = filename # unique enough for now
    
    formatted_segments = []
    for i, seg in enumerate(segments):
        formatted_segments.append({
            "id": f"{transcription_id}_seg_{i}",
            "transcription_id": transcription_id,
            "text": seg["text"],
            "speaker": seg["speaker"],
            "meeting_type": meeting_type,
            "recording_date": seg.get("recording_date") or datetime.now().isoformat()
        })
        
    # Score
    logger.info(f"-> Sending to Gemma Scorer ({len(formatted_segments)} items)...")
    scored_results = await scorer.score_segments(
        formatted_segments,
        company_context=f"{company} Earnings Call",
        batch_delay=1.0 # Be nice to the API
    )
    
    # Store
    logger.info(f"-> Storing {len(scored_results)} scored segments to DB...")
    db.store_scored_segments(scored_results)
    logger.info(f"-> Done with {filename}")

async def main():
    logger.info("Initializing V2 Business Scoring Pipeline...")
    
    # Parse args
    use_mock = "--mock" in sys.argv
    
    try:
        # Initialize Services
        # Use local DB path to avoid /app permission issues on host
        db_path = os.path.join(os.path.dirname(__file__), "../data/transcript_scores.db")
        scorer = get_gemma_scorer()
        
        if use_mock:
            logger.warning("⚠️ RUNNING IN MOCK MODE - No real API calls will be made")
            async def mock_score(*args, **kwargs):
                from src.gemma_scorer import SegmentScores
                return SegmentScores(
                    business_practice_adherence=7,
                    industry_best_practices=5,
                    deadline_stress=5,
                    emotional_conflict=3,
                    decision_clarity=8,
                    speaker_confidence=9,
                    action_orientation=6,
                    risk_awareness=4,
                    justifications={"reasoning": "[MOCK] Simulated score for testing pipeline."}
                )
            scorer.score_segment = mock_score

        db = get_score_database(db_path=db_path)
        
        # Find Transcripts
        search_path = os.path.join(TRANSCRIPT_DIR, "*.txt")
        files = glob.glob(search_path)
        
        if not files:
            logger.warning(f"No transcript files found in {TRANSCRIPT_DIR}")
            return
            
        logger.info(f"Found {len(files)} transcripts to analyze.")
        
        for file in files:
            await analyze_file(file, scorer, db)
            
        logger.info("Batch analysis complete!")
        
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
