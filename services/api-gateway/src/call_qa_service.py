"""
Call QA Service - Quality Assurance Analysis for Call Transcriptions
=====================================================================

Implements automatic QA analysis using Gemma AI with GPU coordination.
Follows existing patterns from gemma-service/src/main.py for GPU management.

Features:
- Transcript chunking (200-250 tokens with overlap)
- Gemma-powered QA scoring (professionalism, compliance, customer service, protocol)
- Vectorization via RAG service
- Agent metrics aggregation

Author: Service Credit Union AI Platform
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

# Configure logging
logger = logging.getLogger("call_qa_service")

# Configuration from environment
GPU_COORDINATOR_URL = os.getenv("GPU_COORDINATOR_URL", "http://gpu-coordinator:8002")
GEMMA_SERVICE_URL = os.getenv("GEMMA_SERVICE_URL", "http://gemma-service:8001")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8004")

# Chunking configuration
TARGET_TOKENS = 200
OVERLAP_TOKENS = 25
MAX_TOKENS_PER_CHUNK = 250


@dataclass
class TranscriptChunk:
    """Represents a chunk of transcript for QA analysis."""

    index: int
    text: str
    text_redacted: str | None = None
    token_count: int = 0
    start_time_sec: float | None = None
    end_time_sec: float | None = None
    primary_speaker: str = "unknown"


@dataclass
class QAResult:
    """Result from Gemma QA analysis."""

    scores: dict[str, int] = field(default_factory=dict)
    rationales: dict[str, str] = field(default_factory=dict)
    compliance_flags: list[str] = field(default_factory=list)
    requires_review: bool = False
    review_reason: str | None = None
    raw_response: dict | None = None
    task_id: str | None = None
    processing_time_ms: int = 0


@dataclass
class CallContext:
    """Context information for QA analysis."""

    call_id: str
    agent_id: str | None
    member_id: str | None
    total_chunks: int
    call_started_at: datetime | None = None


class CallQAService:
    """
    Automatic QA analysis and vectorization of call transcriptions.

    Uses GPU coordinator pattern from existing Gemma integration:
    1. Request GPU slot via POST /gemma/request
    2. Run analysis
    3. Release GPU slot via POST /gemma/release/{task_id} (ALWAYS in finally)
    """

    # QA System Prompt for Gemma
    SYSTEM_PROMPT = """You are a quality assurance supervisor at Service Credit Union's call center.
You evaluate ONLY the SERVICE EMPLOYEE's (agent's) performance, not the member.

RESPOND WITH ONLY VALID JSON. Never add explanations outside the JSON structure.

COMPLIANCE STANDARDS (BANK-SPECIFIC):
- PCI-DSS: Never request full card numbers verbally
- Identity verification BEFORE accessing account info
- Required disclosures for products/rates
- GLBA privacy policy adherence
- TCPA compliance for callbacks
- Fair lending practices

SCORING SCALE (1-10):
- 1-3: Poor - Significant issues, retraining needed
- 4-5: Below Average - Room for improvement
- 6-7: Satisfactory - Meets basic expectations
- 8-9: Good - Exceeds expectations
- 10: Excellent - Model behavior

IMPORTANT: This is an INCOMPLETE transcription chunk. Judge only what you see."""

    def __init__(
        self,
        gpu_coordinator_url: str = GPU_COORDINATOR_URL,
        gemma_url: str = GEMMA_SERVICE_URL,
        rag_url: str = RAG_SERVICE_URL,
        service_auth_getter=None,
    ):
        self.coordinator_url = gpu_coordinator_url
        self.gemma_url = gemma_url
        self.rag_url = rag_url
        self._get_service_headers = service_auth_getter or self._default_headers

    def _default_headers(self, expires_in: int = 60) -> dict[str, str]:
        """Default headers when no service auth is configured."""
        return {}

    # =========================================================================
    # CHUNKING
    # =========================================================================

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count. Uses simple word-based approximation.
        For accuracy, use tiktoken in production.
        ~1.3 tokens per word is a reasonable estimate for English.
        """
        words = len(text.split())
        return int(words * 1.3)

    def chunk_transcript(
        self,
        transcript: str,
        segments: list[dict] | None = None,
        target_tokens: int = TARGET_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
    ) -> list[TranscriptChunk]:
        """
        Split transcript into chunks of approximately target_tokens size.

        Args:
            transcript: Full transcript text
            segments: Optional speaker segments with timing
            target_tokens: Target tokens per chunk (default 200)
            overlap_tokens: Tokens of overlap between chunks (default 25)

        Returns:
            List of TranscriptChunk objects
        """
        if not transcript or not transcript.strip():
            return []

        # Split by sentences for better semantic boundaries
        sentences = self._split_into_sentences(transcript)

        chunks: list[TranscriptChunk] = []
        current_chunk_sentences: list[str] = []
        current_token_count = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)

            # If adding this sentence exceeds max, finalize current chunk
            if current_token_count + sentence_tokens > MAX_TOKENS_PER_CHUNK and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(
                    TranscriptChunk(
                        index=chunk_index,
                        text=chunk_text,
                        token_count=current_token_count,
                        primary_speaker=self._detect_primary_speaker(chunk_text),
                    )
                )
                chunk_index += 1

                # Start new chunk with overlap (last few sentences)
                overlap_sentences = self._get_overlap_sentences(current_chunk_sentences, overlap_tokens)
                current_chunk_sentences = overlap_sentences
                current_token_count = sum(self.estimate_tokens(s) for s in overlap_sentences)

            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens

        # Add final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(
                TranscriptChunk(
                    index=chunk_index,
                    text=chunk_text,
                    token_count=current_token_count,
                    primary_speaker=self._detect_primary_speaker(chunk_text),
                )
            )

        # Add timing info from segments if available
        if segments:
            self._add_timing_from_segments(chunks, segments)

        logger.info(f"Chunked transcript into {len(chunks)} chunks")
        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences, preserving speaker labels."""
        # Handle common speaker patterns like "AGENT:" or "[Member]"
        # Split on sentence-ending punctuation
        pattern = r"(?<=[.!?])\s+(?=[A-Z\[\(])"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(self, sentences: list[str], target_tokens: int) -> list[str]:
        """Get last N sentences totaling approximately target_tokens."""
        result = []
        token_count = 0

        for sentence in reversed(sentences):
            sentence_tokens = self.estimate_tokens(sentence)
            if token_count + sentence_tokens > target_tokens:
                break
            result.insert(0, sentence)
            token_count += sentence_tokens

        return result

    def _detect_primary_speaker(self, text: str) -> str:
        """Detect primary speaker in chunk (agent, member, mixed)."""
        text_lower = text.lower()

        agent_markers = ["agent:", "representative:", "rep:", "csr:", "employee:"]
        member_markers = ["member:", "customer:", "caller:", "client:"]

        agent_count = sum(text_lower.count(m) for m in agent_markers)
        member_count = sum(text_lower.count(m) for m in member_markers)

        if agent_count > member_count * 2:
            return "agent"
        elif member_count > agent_count * 2:
            return "member"
        elif agent_count > 0 or member_count > 0:
            return "mixed"
        else:
            return "unknown"

    def _add_timing_from_segments(self, chunks: list[TranscriptChunk], segments: list[dict]):
        """Add timing information from segments to chunks."""
        if not segments:
            return

        # Simple approach: divide total time across chunks
        if segments:
            total_start = segments[0].get("start_time_sec", 0)
            total_end = segments[-1].get("end_time_sec", 0)

            if total_end > total_start and len(chunks) > 0:
                duration = total_end - total_start
                chunk_duration = duration / len(chunks)

                for i, chunk in enumerate(chunks):
                    chunk.start_time_sec = total_start + (i * chunk_duration)
                    chunk.end_time_sec = total_start + ((i + 1) * chunk_duration)

    # =========================================================================
    # GEMMA QA ANALYSIS (with GPU Coordination)
    # =========================================================================

    async def analyze_chunk_with_gemma(
        self, chunk: TranscriptChunk, context: CallContext, timeout: float = 120.0
    ) -> QAResult:
        """
        Analyze a transcript chunk using Gemma.

        NOTE: The /chat endpoint handles GPU coordination internally.
        We do NOT call GPU coordinator here - that would cause double coordination deadlock.

        Args:
            chunk: Transcript chunk to analyze
            context: Call context information
            timeout: Request timeout in seconds

        Returns:
            QAResult with scores, rationales, and flags
        """
        task_id = f"qa-{context.call_id[:8]}-{chunk.index}-{uuid.uuid4().hex[:6]}"
        start_time = datetime.now()

        # Build the analysis prompt
        prompt = self._build_qa_prompt(chunk, context)

        try:
            # Call Gemma /chat endpoint - it handles GPU coordination internally
            logger.info(f"🧠 [QA {task_id}] Calling Gemma for QA analysis...")
            async with httpx.AsyncClient(timeout=timeout) as client:
                gemma_response = await client.post(
                    f"{self.gemma_url}/chat",
                    json={
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 256,  # Reduced from 512 for faster inference
                        "temperature": 0.2,  # Low for consistent scoring
                        "top_p": 0.9,
                    },
                    headers=self._get_service_headers(120),
                )

                if gemma_response.status_code != 200:
                    logger.error(f"❌ [QA {task_id}] Gemma call failed: {gemma_response.status_code}")
                    return QAResult(
                        scores={"professionalism": 5, "compliance": 5, "customer_service": 5, "protocol_adherence": 5},
                        rationales={"error": f"Gemma call failed: {gemma_response.status_code}"},
                        task_id=task_id,
                    )

                gemma_data = gemma_response.json()

            # Parse the response
            result = self._parse_gemma_response(gemma_data, task_id)

            elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            result.processing_time_ms = elapsed_ms

            logger.info(f"✅ [QA {task_id}] Analysis complete in {elapsed_ms}ms")
            return result

        except Exception as e:
            logger.error(f"❌ [QA {task_id}] Error: {e}")
            return QAResult(
                scores={"professionalism": 5, "compliance": 5, "customer_service": 5, "protocol_adherence": 5},
                rationales={"error": str(e)},
                task_id=task_id,
            )

    def _build_qa_prompt(self, chunk: TranscriptChunk, context: CallContext) -> str:
        """Build the QA analysis prompt for Gemma."""
        time_range = ""
        if chunk.start_time_sec is not None and chunk.end_time_sec is not None:
            time_range = f"{chunk.start_time_sec:.1f}s - {chunk.end_time_sec:.1f}s"
        else:
            time_range = "N/A"

        return f"""Analyze this call center transcription chunk.

CONTEXT:
- Call ID: {context.call_id}
- Agent ID: {context.agent_id or "Unknown"}
- Chunk {chunk.index + 1} of {context.total_chunks}
- Timestamp: {time_range}

=== TRANSCRIPTION CHUNK ===
{chunk.text}
=== END CHUNK ===

EVALUATE AGENT ONLY. Return JSON:
{{
  "scores": {{
    "professionalism": <1-10>,
    "compliance": <1-10>,
    "customer_service": <1-10>,
    "protocol_adherence": <1-10>
  }},
  "rationales": {{
    "professionalism": "<brief rationale with quotes>",
    "compliance": "<note any violations>",
    "customer_service": "<brief rationale>",
    "protocol_adherence": "<brief rationale>"
  }},
  "compliance_flags": ["<list concerns if any>"],
  "requires_human_review": <true/false>,
  "review_reason": "<if true, explain>"
}}"""

    def _parse_gemma_response(self, gemma_data: dict, task_id: str) -> QAResult:
        """Parse Gemma response into QAResult."""
        try:
            # Get the response text from various possible formats
            # NOTE: /chat endpoint returns 'message', /generate returns 'text'
            text = None
            if "message" in gemma_data:
                # /chat endpoint response format
                text = gemma_data["message"]
            elif "response" in gemma_data:
                text = gemma_data["response"]
            elif "text" in gemma_data:
                text = gemma_data["text"]
            elif "choices" in gemma_data and gemma_data["choices"]:
                choice = gemma_data["choices"][0]
                text = choice.get("text") or choice.get("message", {}).get("content", "")

            if not text:
                logger.warning(f"[QA {task_id}] No text in response: {list(gemma_data.keys())}")
                text = str(gemma_data)

            logger.debug(f"[QA {task_id}] Gemma response text: {text[:500]}...")

            # Try to extract JSON from response - find the outermost braces
            # Look for the main JSON object
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
            if not json_match:
                # Try simpler approach - find { to last }
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_str = text[start : end + 1]
                else:
                    logger.warning(f"[QA {task_id}] No JSON braces found in response")
                    return QAResult(
                        scores={"professionalism": 5, "compliance": 5, "customer_service": 5, "protocol_adherence": 5},
                        rationales={"parse_error": "No JSON in response"},
                        raw_response={"text": text[:500]},
                        task_id=task_id,
                    )
            else:
                json_str = json_match.group()

            # Parse the JSON
            parsed = json.loads(json_str)
            logger.info(f"[QA {task_id}] Parsed scores: {parsed.get('scores', {})}")

            # Extract and validate scores
            scores = parsed.get("scores", {})
            for key in ["professionalism", "compliance", "customer_service", "protocol_adherence"]:
                if key not in scores or not isinstance(scores[key], (int, float)):
                    scores[key] = 5
                else:
                    scores[key] = max(1, min(10, int(scores[key])))

            return QAResult(
                scores=scores,
                rationales=parsed.get("rationales", {}),
                compliance_flags=parsed.get("compliance_flags", []),
                requires_review=parsed.get("requires_human_review", False),
                review_reason=parsed.get("review_reason"),
                raw_response=parsed,
                task_id=task_id,
            )

        except json.JSONDecodeError as e:
            logger.error(f"[QA {task_id}] JSON parse error: {e}")
            logger.error(f"[QA {task_id}] Failed to parse: {json_str[:200] if 'json_str' in dir() else 'N/A'}...")
            return QAResult(
                scores={"professionalism": 5, "compliance": 5, "customer_service": 5, "protocol_adherence": 5},
                rationales={"parse_error": str(e)},
                raw_response=gemma_data,
                task_id=task_id,
            )

    # =========================================================================
    # VECTORIZATION
    # =========================================================================

    async def vectorize_chunk(self, chunk: TranscriptChunk, call_id: str, metadata: dict | None = None) -> str | None:
        """
        Store chunk in RAG service for vector search.

        Args:
            chunk: Transcript chunk
            call_id: Associated call ID
            metadata: Additional metadata

        Returns:
            vector_id if successful, None otherwise
        """
        try:
            vector_metadata = {
                "call_id": call_id,
                "chunk_index": chunk.index,
                "primary_speaker": chunk.primary_speaker,
                "start_time_sec": chunk.start_time_sec,
                "end_time_sec": chunk.end_time_sec,
                "content_type": "call_transcript_chunk",
                **(metadata or {}),
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.rag_url}/memories",
                    json={
                        "content": chunk.text,
                        "metadata": vector_metadata,
                        "source": f"call:{call_id}:chunk:{chunk.index}",
                    },
                    headers=self._get_service_headers(60),
                )

                if response.status_code == 200:
                    data = response.json()
                    vector_id = data.get("memory_id") or data.get("id")
                    logger.info(f"📦 Vectorized chunk {chunk.index} -> {vector_id}")
                    return vector_id
                else:
                    logger.warning(f"Failed to vectorize chunk {chunk.index}: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"Vectorization error for chunk {chunk.index}: {e}")
            return None

    # =========================================================================
    # GEMMA WARMUP (reduces first-chunk latency)
    # =========================================================================

    async def _warmup_gemma(self) -> bool:
        """
        Pre-warm Gemma model to reduce first-chunk latency.

        The first Gemma inference after startup takes ~60s due to:
        1. GPU lock acquisition
        2. Model loading to GPU VRAM
        3. First inference warm-up

        Subsequent calls are fast (~5-15s). This warmup ensures
        all chunks process quickly.

        Returns:
            True if warmup succeeded, False otherwise (still safe to proceed)
        """
        try:
            logger.info("🔥 [QA] Warming up Gemma model for QA analysis...")
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(f"{self.gemma_url}/warmup", headers=self._get_service_headers(120))
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ [QA] Gemma warmup complete: {data.get('status', 'ready')}")
                    return True
                logger.warning(f"⚠️ [QA] Warmup returned {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ [QA] Warmup failed: {e} - proceeding with analysis anyway")
            return False

    # =========================================================================
    # FULL PIPELINE
    # =========================================================================

    async def process_completed_call(
        self,
        call_id: str,
        agent_id: str | None,
        transcript: str,
        segments: list[dict] | None = None,
        member_id: str | None = None,
        call_started_at: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Main entry point: Process a completed call for QA analysis.

        This method:
        1. Chunks the transcript
        2. Analyzes each chunk with Gemma (GPU coordinated)
        3. Vectorizes chunks
        4. Returns results for database storage

        Args:
            call_id: Unique call identifier
            agent_id: Agent who handled the call
            transcript: Full call transcript
            segments: Optional speaker segments
            member_id: Optional member ID
            call_started_at: Call start time

        Returns:
            Dict with processed chunks and metrics
        """
        logger.info(f"🔄 Starting QA processing for call {call_id}")
        start_time = datetime.now()

        # Pre-warm Gemma to avoid slow first chunk
        await self._warmup_gemma()

        # Step 1: Chunk the transcript
        chunks = self.chunk_transcript(transcript, segments)

        if not chunks:
            logger.warning(f"No chunks generated for call {call_id}")
            return {"call_id": call_id, "chunks": [], "error": "No chunks generated"}

        context = CallContext(
            call_id=call_id,
            agent_id=agent_id,
            member_id=member_id,
            total_chunks=len(chunks),
            call_started_at=call_started_at,
        )

        results = []

        # Step 2 & 3: Analyze and vectorize each chunk
        for chunk in chunks:
            # Analyze with Gemma
            qa_result = await self.analyze_chunk_with_gemma(chunk, context)

            # Vectorize
            vector_id = await self.vectorize_chunk(chunk, call_id)

            # Combine results
            chunk_result = {
                "chunk_index": chunk.index,
                "chunk_text": chunk.text,
                "token_count": chunk.token_count,
                "start_time_sec": chunk.start_time_sec,
                "end_time_sec": chunk.end_time_sec,
                "primary_speaker": chunk.primary_speaker,
                "scores": qa_result.scores,
                "rationales": qa_result.rationales,
                "compliance_flags": qa_result.compliance_flags,
                "requires_review": qa_result.requires_review,
                "review_reason": qa_result.review_reason,
                "vector_id": vector_id,
                "gemma_task_id": qa_result.task_id,
                "gemma_raw_response": qa_result.raw_response,
                "processing_time_ms": qa_result.processing_time_ms,
            }
            results.append(chunk_result)

            # Small delay between chunks to avoid overwhelming GPU coordinator
            await asyncio.sleep(0.1)

        # Calculate summary metrics
        total_time = (datetime.now() - start_time).total_seconds()
        avg_scores = self._calculate_average_scores(results)

        # Release Gemma's GPU slot so transcription can use GPU again
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(f"{self.gemma_url}/release-session", headers=self._get_service_headers(30))
            logger.info("[QA] Released Gemma GPU slot - transcription can use GPU")
        except Exception as e:
            logger.warning(f"[QA] Failed to release Gemma session: {e}")

        logger.info(
            f"✅ QA processing complete for call {call_id}: "
            f"{len(results)} chunks in {total_time:.1f}s, "
            f"avg_overall={avg_scores.get('overall', 0):.1f}"
        )

        return {
            "call_id": call_id,
            "agent_id": agent_id,
            "chunks": results,
            "chunk_count": len(results),
            "avg_scores": avg_scores,
            "total_processing_time_sec": total_time,
            "requires_review_count": sum(1 for r in results if r.get("requires_review")),
            "compliance_flags_count": sum(len(r.get("compliance_flags", [])) for r in results),
        }

    def _calculate_average_scores(self, chunk_results: list[dict]) -> dict[str, float]:
        """Calculate average scores across all chunks."""
        if not chunk_results:
            return {}

        score_keys = ["professionalism", "compliance", "customer_service", "protocol_adherence"]
        averages = {}

        for key in score_keys:
            values = [r["scores"].get(key, 5) for r in chunk_results if "scores" in r]
            if values:
                averages[key] = round(sum(values) / len(values), 1)

        # Calculate weighted overall
        if all(k in averages for k in score_keys):
            averages["overall"] = round(
                averages["professionalism"] * 0.25
                + averages["compliance"] * 0.30
                + averages["customer_service"] * 0.25
                + averages["protocol_adherence"] * 0.20,
                1,
            )

        return averages


# Singleton instance
_qa_service: CallQAService | None = None
_service_auth = None


def init_qa_service(service_auth=None):
    """Initialize the QA service with service authentication.

    Should be called from main.py after ServiceAuth is initialized.
    """
    global _qa_service, _service_auth
    _service_auth = service_auth

    def get_headers(expires_in: int = 60) -> dict[str, str]:
        if _service_auth:
            return _service_auth.get_auth_header()
        return {}

    _qa_service = CallQAService(service_auth_getter=get_headers)
    logger.info("✅ CallQAService initialized with service auth")
    return _qa_service


def get_qa_service() -> CallQAService:
    """Get or create the QA service singleton."""
    global _qa_service
    if _qa_service is None:
        # Fallback: create without auth (will fail on authenticated endpoints)
        logger.warning("CallQAService created without service auth - GPU coordinator will reject requests")
        _qa_service = CallQAService()
    return _qa_service
