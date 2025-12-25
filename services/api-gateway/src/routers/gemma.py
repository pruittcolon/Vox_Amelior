"""
Gemma AI Router - Local LLM endpoints.

Provides text generation, chat, RAG-enhanced chat, streaming analysis,
personality analysis, and emotion-focused summaries.
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

# Import auth
try:
    from src.auth.permissions import Session, require_auth
except ImportError:

    def require_auth():
        return None

    Session = None

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["gemma"])

# Service URLs from environment
GEMMA_URL = os.getenv("GEMMA_URL", "http://gemma-service:8004")
RAG_URL = os.getenv("RAG_URL", "http://rag-service:8001")

# In-memory job storage (imported from main or initialized here)
personality_jobs: dict[str, dict[str, Any]] = {}
analysis_jobs: dict[str, dict[str, Any]] = {}


def _get_proxy_request():
    """Lazy import proxy_request to avoid circular imports."""
    from src.main import proxy_request

    return proxy_request


def _get_service_jwt_headers():
    """Lazy import service JWT headers."""
    from src.main import _service_jwt_headers

    return _service_jwt_headers


def _get_gemma_generate_with_fallback():
    """Lazy import gemma generate with fallback."""
    from src.main import _gemma_generate_with_fallback

    return _gemma_generate_with_fallback


def format_sse(event: str, payload: dict[str, Any]) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


# =============================================================================
# Basic Gemma Endpoints
# =============================================================================


@router.post("/api/gemma/warmup")
async def gemma_warmup(session: Session = Depends(require_auth)):
    """Warmup Gemma - moves model to GPU and waits until ready."""
    proxy_request = _get_proxy_request()
    result = await proxy_request(f"{GEMMA_URL}/warmup", "POST", json={})
    return result


@router.post("/api/gemma/release-session")
async def gemma_release_session(request: Request, body: dict[str, Any] = None):
    """
    Release GPU session when Gemma page closes.

    Supports two authentication modes:
    1. Standard: Authorization header + X-CSRF-Token header (normal fetch)
    2. Beacon: Session cookie + csrf_token in body (sendBeacon on page close)

    sendBeacon cannot send custom headers, but DOES send cookies.
    CSRF token must be included in the JSON body for beacon requests.
    """
    proxy_request = _get_proxy_request()

    # Get auth manager for session validation
    from src.auth.auth_manager import get_auth_manager
    from src.config import SecurityConfig as SecConf

    auth_manager = get_auth_manager()
    session = None

    # Check if session was already validated by middleware (request.state.session)
    if hasattr(request.state, "session") and request.state.session:
        session = request.state.session
        logger.info("[GEMMA] release-session: Using middleware-validated session")
    else:
        # Fallback: Try to validate session from cookie directly (for sendBeacon)
        session_cookie = request.cookies.get(SecConf.SESSION_COOKIE_NAME)
        if session_cookie:
            session = auth_manager.validate_session(session_cookie)
            if session:
                logger.info("[GEMMA] release-session: Validated session from cookie")

    if not session:
        logger.warning("[GEMMA] release-session: No valid session found")
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Validate CSRF - check header first, then body (for sendBeacon)
    csrf_header = request.headers.get("X-CSRF-Token") or request.headers.get("x-csrf-token")
    csrf_body = (body or {}).get("csrf_token") if body else None
    csrf_cookie = request.cookies.get("ws_csrf")

    # Accept CSRF from either header OR body (body is for sendBeacon)
    csrf_provided = csrf_header or csrf_body

    if not csrf_provided:
        logger.warning("[GEMMA] release-session: No CSRF token provided")
        raise HTTPException(status_code=401, detail="CSRF token required")

    if csrf_provided != csrf_cookie:
        logger.warning("[GEMMA] release-session: CSRF token mismatch")
        raise HTTPException(status_code=401, detail="Invalid CSRF token")

    logger.info("[GEMMA] release-session: Auth validated, releasing GPU")

    # Release GPU
    try:
        result = await proxy_request(f"{GEMMA_URL}/release-session", "POST", json={})
        return {"status": "released", "result": result}
    except Exception as e:
        logger.error(f"[GEMMA] release-session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/gemma/generate")
async def gemma_generate(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Generate text using Gemma LLM."""
    proxy_request = _get_proxy_request()
    result = await proxy_request(f"{GEMMA_URL}/generate", "POST", json=request)
    return result


@router.post("/api/gemma/chat")
async def gemma_chat(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Multi-turn chat with Gemma LLM."""
    proxy_request = _get_proxy_request()
    result = await proxy_request(f"{GEMMA_URL}/chat", "POST", json=request)
    return result


@router.post("/api/public/chat")
async def public_chat(request: dict[str, Any], http_request: Request):
    """Public chat endpoint for unauthenticated chatbot access (IP rate-limited)."""
    proxy_request = _get_proxy_request()
    result = await proxy_request(f"{GEMMA_URL}/chat", "POST", json=request)
    return result


@router.get("/api/gemma/stats")
async def gemma_stats(session: Session = Depends(require_auth)):
    """Get Gemma service statistics (GPU status, VRAM, etc.)."""
    proxy_request = _get_proxy_request()
    result = await proxy_request(f"{GEMMA_URL}/stats", "GET")
    return result


@router.post("/api/gemma/chat-rag")
async def gemma_chat_rag(request: dict[str, Any], session: Session = Depends(require_auth)):
    """RAG-enhanced chat via Gemma service."""
    proxy_request = _get_proxy_request()
    result = await proxy_request(f"{GEMMA_URL}/chat/rag", "POST", json=request)
    return result


@router.post("/api/gemma/analyze")
async def gemma_analyze(request: dict[str, Any], http_request: Request, session: Session = Depends(require_auth)):
    """Proxy legacy Gemma analyzer call (batch)."""
    proxy_request = _get_proxy_request()
    analysis_id = http_request.headers.get("X-Analysis-Id") or f"analysis_{uuid.uuid4().hex[:10]}"
    headers = {"X-Analysis-Id": analysis_id} if analysis_id else None
    result = await proxy_request(
        f"{GEMMA_URL}/analyze",
        "POST",
        json=request,
        extra_headers=headers,
    )
    if isinstance(result, dict):
        result.setdefault("analysis_id", analysis_id)
    return result


# =============================================================================
# Streaming Analysis Endpoints
# =============================================================================


@router.post("/api/gemma/analyze/stream")
async def gemma_analyze_stream_create(
    payload: dict[str, Any], http_request: Request, session: Session = Depends(require_auth)
):
    """Create a streaming Gemma analysis job."""
    job_id = f"gemma-stream-{uuid.uuid4().hex[:10]}"
    analysis_id = payload.get("analysis_id") or http_request.headers.get("X-Analysis-Id")
    if not analysis_id:
        analysis_id = f"analysis_{uuid.uuid4().hex[:10]}"
    analysis_jobs[job_id] = {
        "payload": payload,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "analysis_id": analysis_id,
        "user_id": getattr(session, "user_id", None),
    }
    logger.info(
        "[ANALYZE-STREAM] Created job %s analysis_id=%s user=%s",
        job_id,
        analysis_id,
        getattr(session, "user_id", None),
    )
    return {"success": True, "job_id": job_id}


@router.get("/api/gemma/analyze/stream/{job_id}")
async def gemma_analyze_stream(job_id: str, http_request: Request, session: Session = Depends(require_auth)):
    """Stream analyzer progress via Server-Sent Events."""
    job = analysis_jobs.pop(job_id, None)
    if not job:
        logger.warning("[ANALYZE-STREAM] Unknown job_id=%s", job_id)
        raise HTTPException(status_code=404, detail="Analysis job not found or already started")

    # Import dependencies
    proxy_request = _get_proxy_request()
    _gemma_generate_with_fallback = _get_gemma_generate_with_fallback()

    payload = job.get("payload") or {}
    filters = payload.get("filters") or {}
    max_tokens = int(payload.get("max_tokens", 256) or 256)
    temperature = float(payload.get("temperature", 0.4) or 0.4)
    analysis_id = http_request.query_params.get("analysis_id") or job.get("analysis_id")
    analysis_headers = {"X-Analysis-Id": analysis_id} if analysis_id else None

    def _normalize_filter_list(value: Any | None) -> list[str]:
        if not value:
            return []
        if isinstance(value, (str, bytes)):
            return [str(value)]
        normalized: list[str] = []
        iterable = value if isinstance(value, (list, tuple, set)) else [value]
        for item in iterable:
            if isinstance(item, dict):
                for key in ("value", "name", "id", "speaker", "emotion"):
                    if item.get(key):
                        normalized.append(str(item[key]))
                        break
                else:
                    normalized.append(str(item))
            else:
                normalized.append(str(item))
        return normalized

    raw_max = payload.get("max_statements")
    try:
        analysis_limit = int(raw_max)
    except (TypeError, ValueError):
        analysis_limit = None
    if not analysis_limit or analysis_limit <= 0:
        try:
            analysis_limit = int(filters.get("limit", 20) or 20)
        except (TypeError, ValueError):
            analysis_limit = 20
    analysis_limit = max(1, min(analysis_limit, 200))

    async def _fetch_fallback_items(target_limit: int) -> list[dict[str, Any]]:
        limit = max(1, min(int(target_limit or 20), 200))
        context_lines = max(0, min(int(filters.get("context_lines", 3) or 0), 10))
        speakers_filter = {s.lower() for s in _normalize_filter_list(filters.get("speakers"))}
        emotions_filter = {e.lower() for e in _normalize_filter_list(filters.get("emotions"))}
        raw_keywords = str(filters.get("keywords", "") or "")
        keywords = [k.strip().lower() for k in raw_keywords.split(",") if k.strip()]
        match_all = filters.get("match", "any") == "all"
        start_date_str = filters.get("start_date")
        end_date_str = filters.get("end_date")

        def parse_date(value: str | None, end_of_day: bool = False) -> datetime | None:
            if not value:
                return None
            for fmt in (
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f%z",
            ):
                try:
                    dt = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
            if end_of_day and dt.tzinfo is None:
                return dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            if end_of_day:
                return dt + timedelta(hours=23, minutes=59, seconds=59, microseconds=999999)
            return dt

        start_dt = parse_date(start_date_str) if start_date_str else None
        end_dt = parse_date(end_date_str, end_of_day=True) if end_date_str else None

        recent_limit = max(limit * 5, 200)
        recent_response = await proxy_request(
            f"{RAG_URL}/transcripts/recent?limit={recent_limit}",
            "GET",
            params=None,
            extra_headers=analysis_headers,
        )
        transcripts = recent_response.get("transcripts") or []

        items: list[dict[str, Any]] = []
        for transcript in transcripts:
            created_value = transcript.get("created_at") or transcript.get("timestamp")
            transcript_dt = parse_date(created_value)
            if start_dt and transcript_dt and transcript_dt < start_dt:
                continue
            if end_dt and transcript_dt and transcript_dt > end_dt:
                continue

            segments = transcript.get("segments") or []
            for idx, segment in enumerate(segments):
                speaker_value = (segment.get("speaker") or "").strip()
                if speakers_filter and speaker_value.lower() not in speakers_filter:
                    continue

                emotion_value = (segment.get("emotion") or segment.get("dominant_emotion") or "").strip().lower()
                if emotions_filter and emotion_value not in emotions_filter:
                    continue

                text_value = (segment.get("text") or "").strip()
                if not text_value:
                    continue
                if keywords:
                    text_lower = text_value.lower()
                    matches = [kw for kw in keywords if kw in text_lower]
                    if match_all and len(matches) != len(keywords):
                        continue
                    if not match_all and not matches:
                        continue

                context: list[dict[str, Any]] = []
                if context_lines:
                    start_idx = max(0, idx - context_lines)
                    for ctx_segment in segments[start_idx:idx]:
                        context.append(
                            {
                                "speaker": ctx_segment.get("speaker"),
                                "text": ctx_segment.get("text"),
                                "emotion": ctx_segment.get("emotion") or ctx_segment.get("dominant_emotion"),
                            }
                        )

                items.append(
                    {
                        "segment_id": segment.get("id"),
                        "transcript_id": transcript.get("id"),
                        "job_id": transcript.get("job_id"),
                        "speaker": segment.get("speaker"),
                        "emotion": segment.get("emotion") or segment.get("dominant_emotion"),
                        "text": text_value,
                        "created_at": created_value,
                        "start_time": segment.get("start_time"),
                        "end_time": segment.get("end_time"),
                        "context_before": context,
                    }
                )

                if len(items) >= limit:
                    return items

        return items

    logger.info(
        "[ANALYZE-STREAM] job=%s analysis_id=%s max_statements=%s filters=%s user=%s",
        job_id,
        analysis_id,
        analysis_limit,
        {k: filters.get(k) for k in ("limit", "speakers", "emotions", "start_date", "end_date", "search_type")},
        getattr(session, "user_id", None),
    )

    ANALYZE_FALLBACK_ENABLED = os.getenv("ANALYZE_FALLBACK_ENABLED", "true").lower() in {"1", "true", "yes"}

    async def event_generator():
        last_model = None
        started_at = datetime.utcnow().isoformat() + "Z"
        MAX_PROMPT_CHARS = 6000
        combined_sections: list[str] = []
        artifact_id: str | None = None
        transcripts_sentinel = "<END_OF_TRANSCRIPTS>"
        analysis_stop_sequences: list[str] = []
        prompt_template = (payload.get("custom_prompt") or "").strip()
        has_transcript_placeholder = "{transcripts}" in prompt_template
        apply_short_instruction = bool(prompt_template) and not has_transcript_placeholder
        guardrail_instruction = (
            "You are an analyst. Please answer the user's question about the given transcript section."
            if apply_short_instruction
            else ""
        )

        try:
            # Warmup GPU (best effort)
            try:
                await proxy_request(
                    f"{GEMMA_URL}/warmup",
                    "POST",
                    json={},
                    extra_headers=analysis_headers,
                )
                logger.info("[ANALYZE-STREAM] Warmup complete job=%s analysis_id=%s", job_id, analysis_id)
                yield format_sse("meta", {"job_id": job_id, "message": "GPU warmup complete"})
            except Exception as warmup_error:
                logger.warning(
                    "[ANALYZE-STREAM] Warmup failed job=%s analysis_id=%s error=%s", job_id, analysis_id, warmup_error
                )
                yield format_sse("meta", {"job_id": job_id, "message": "GPU warmup failed; continuing"})

            rag_payload = dict(filters)
            rag_payload["limit"] = analysis_limit
            rag_payload.setdefault("context_lines", max(0, min(int(filters.get("context_lines", 3) or 0), 10)))

            fallback_used = False
            rag_query_status: int | None = None
            raw_items: list[dict[str, Any]] = []
            dataset_total = None
            try:
                rag_result = await proxy_request(
                    f"{RAG_URL}/transcripts/query",
                    "POST",
                    json=rag_payload,
                    extra_headers=analysis_headers,
                )
                rag_query_status = 200
                raw_items = rag_result.get("items") or []
                try:
                    if isinstance(rag_result, dict):
                        dataset_total = rag_result.get("total") or rag_result.get("count")
                except Exception:
                    dataset_total = None
            except HTTPException as exc:
                if exc.status_code == 404:
                    logger.error(
                        "[ANALYZE-STREAM] RAG query endpoint 404 job=%s analysis_id=%s (fallback=%s)",
                        job_id,
                        analysis_id,
                        ANALYZE_FALLBACK_ENABLED,
                    )
                    rag_query_status = 404
                    if ANALYZE_FALLBACK_ENABLED:
                        fallback_used = True
                        raw_items = await _fetch_fallback_items(analysis_limit)
                    else:
                        yield format_sse("server_error", {"job_id": job_id, "detail": "RAG query unavailable"})
                        return
                else:
                    rag_query_status = exc.status_code
                    raise

            items = [item for item in raw_items if isinstance(item, dict) and (item.get("text") or "").strip()]
            if len(items) > analysis_limit:
                items = items[:analysis_limit]

            if isinstance(dataset_total, int):
                dataset_total_int = dataset_total
            else:
                dataset_total_int = len(raw_items) if raw_items else len(items)
            total = len(items)

            if fallback_used:
                logger.info(
                    "[ANALYZE-STREAM] job=%s analysis_id=%s fallback produced %s candidate statements",
                    job_id,
                    analysis_id,
                    total,
                )
            else:
                logger.info(
                    "[ANALYZE-STREAM] job=%s analysis_id=%s fetched %s candidate statements", job_id, analysis_id, total
                )

            def _alpha_label(position: int) -> str:
                if position <= 0:
                    return str(position)
                label = ""
                while position > 0:
                    position, rem = divmod(position - 1, 26)
                    label = chr(65 + rem) + label
                return label

            for idx, item in enumerate(items, start=1):
                if isinstance(item, dict):
                    item.setdefault("label", _alpha_label(idx))

            meta_payload = {
                "job_id": job_id,
                "total": total,
                "started_at": started_at,
                "max_statements": analysis_limit,
            }
            if isinstance(dataset_total_int, int):
                meta_payload["dataset_total"] = dataset_total_int
            if fallback_used:
                meta_payload["fallback"] = "transcripts/recent"
                meta_payload["fallback_reason"] = "rag_query_404"
            if rag_query_status is not None:
                meta_payload["rag_query_status"] = rag_query_status
            yield format_sse("meta", meta_payload)
            if fallback_used:
                yield format_sse(
                    "meta", {"job_id": job_id, "message": "Using fallback query (transcripts/recent filter)"}
                )

            if total == 0:
                yield format_sse(
                    "done", {"job_id": job_id, "completed_at": datetime.utcnow().isoformat() + "Z", "model": None}
                )
                logger.info("[ANALYZE-STREAM] job=%s analysis_id=%s completed with no matches", job_id, analysis_id)
                return

            for index, item in enumerate(items, start=1):
                if await http_request.is_disconnected():
                    logger.warning("[ANALYZE-STREAM] Client disconnected job=%s analysis_id=%s", job_id, analysis_id)
                    break

                label = item.get("label") or _alpha_label(index) if isinstance(item, dict) else _alpha_label(index)
                if isinstance(item, dict):
                    item["label"] = label

                context_before = item.get("context_before") or []
                context_plain_lines = [
                    f"{ctx.get('speaker') or 'Speaker'}: {ctx.get('text')}" for ctx in context_before if ctx.get("text")
                ]
                context_plain = "\n".join(context_plain_lines).strip()
                statement_plain = f"{item.get('speaker') or 'Speaker'}: {item.get('text')}".strip()

                log_block_parts: list[str] = []
                if context_plain:
                    log_block_parts.append("Context:\n" + context_plain)
                log_block_parts.append("Statement:\n" + statement_plain)
                transcript_block = "\n".join(log_block_parts)

                statement_header = f"Statement {label or index}"
                formatted_sections: list[str] = [statement_header]
                if context_plain:
                    formatted_sections.extend(["Context:", "```", context_plain, "```"])
                formatted_sections.extend(["Statement:", "```", statement_plain, "```"])
                formatted_sections.append(transcripts_sentinel)
                formatted_transcripts = "\n".join(formatted_sections)

                if has_transcript_placeholder:
                    user_instruction = prompt_template.replace("{transcripts}", formatted_transcripts)
                else:
                    base_prompt = prompt_template or "Analyze the following transcript section."
                    user_instruction = base_prompt.strip() + "\n\nTranscript Section:\n" + formatted_transcripts

                if guardrail_instruction:
                    final_prompt = guardrail_instruction.strip() + "\n\n" + user_instruction.strip()
                else:
                    final_prompt = user_instruction.strip()

                prompt_trimmed = False
                if len(final_prompt) > MAX_PROMPT_CHARS:
                    head = final_prompt[:2000]
                    tail = final_prompt[-(MAX_PROMPT_CHARS - 2000) :]
                    final_prompt = head + "\n[TRUNCATED FOR LENGTH]\n" + tail
                    prompt_trimmed = True

                prompt_len = len(final_prompt)
                approx_prompt_tokens = max(1, prompt_len // 4)
                prompt_hash = hashlib.sha256(final_prompt.encode("utf-8")).hexdigest()[:12]
                logger.info(
                    "[ANALYZE-STREAM] job=%s analysis_id=%s statement=%s/%s prompt_hash=%s chars=%s approx_tokens=%s trimmed=%s",
                    job_id,
                    analysis_id,
                    index,
                    total,
                    prompt_hash,
                    prompt_len,
                    approx_prompt_tokens,
                    prompt_trimmed,
                )

                step_payload = {
                    "job_id": job_id,
                    "i": index,
                    "total": total,
                    "status": "sending",
                    "prompt_fragment": final_prompt[:160],
                }
                if label:
                    step_payload["label"] = label
                yield format_sse("step", step_payload)

                gemma_request = {
                    "prompt": final_prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stop": analysis_stop_sequences,
                }

                waiting_payload = {"job_id": job_id, "i": index, "total": total, "status": "waiting"}
                if label:
                    waiting_payload["label"] = label
                yield format_sse("step", waiting_payload)

                try:
                    response_text, gen_resp = await _gemma_generate_with_fallback(gemma_request, analysis_headers)
                    last_model = (gen_resp or {}).get("model") or last_model
                except HTTPException as exc:
                    logger.error("[ANALYZE-STREAM] Gemma error job=%s: %s", job_id, exc.detail)
                    yield format_sse("error", {"job_id": job_id, "i": index, "detail": exc.detail})
                    break

                if not isinstance(response_text, str) or not response_text.strip():
                    response_text = (
                        "Gemma returned no text for this statement. Try refining the prompt or reducing stop sequences."
                    )
                combined_sections.append(
                    "\n".join(
                        [
                            f"{statement_header} ({index}/{total})",
                            "Context:",
                            transcript_block,
                            "",
                            "Gemma Response:",
                            response_text,
                            "",
                        ]
                    )
                )

                result_payload = {"job_id": job_id, "i": index, "total": total, "response": response_text, "item": item}
                if label:
                    result_payload["label"] = label
                yield format_sse("result", result_payload)

            # Archive combined artifact (best effort)
            if analysis_id and combined_sections:
                archive_payload = {
                    "analysis_id": analysis_id,
                    "title": payload.get("title")
                    or f"Streaming Analysis - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
                    "body": "\n".join(combined_sections),
                    "metadata": {
                        "filters": filters,
                        "total_statements": total,
                        "fallback_used": fallback_used,
                        "job_id": job_id,
                        "model": last_model,
                        "started_at": started_at,
                        "completed_at": datetime.utcnow().isoformat() + "Z",
                        "user_id": job.get("user_id"),
                    },
                }
                try:
                    archive_result = await proxy_request(
                        f"{RAG_URL}/analysis/archive", "POST", json=archive_payload, extra_headers=analysis_headers
                    )
                    artifact_id = archive_result.get("artifact_id") if isinstance(archive_result, dict) else None
                    if artifact_id:
                        logger.info(
                            "[ANALYZE-STREAM] job=%s analysis_id=%s archived artifact_id=%s",
                            job_id,
                            analysis_id,
                            artifact_id,
                        )
                except Exception as exc:
                    logger.error("[ANALYZE-STREAM] job=%s analysis_id=%s archive failed: %s", job_id, analysis_id, exc)

            # Produce executive summary
            summary_text = ""
            try:
                if combined_sections:
                    summary_context = "\n\n".join(combined_sections)[-12000:]
                    summary_prompt = (
                        "You are an expert conversation analyst. Based on the following analysis sections (each contains Context and Gemma Response), "
                        "write a concise executive summary with 5-8 bullet points and a 2-3 sentence conclusion. Be precise and avoid repetition.\n\n"
                        f"{summary_context}\n\nNow provide the executive summary:"
                    )
                    gen_req = {"prompt": summary_prompt, "max_tokens": 384, "temperature": 0.3}
                    summary_text, _ = await _gemma_generate_with_fallback(gen_req, analysis_headers)
            except Exception as exc:
                logger.warning(
                    "[ANALYZE-STREAM] summary generation failed job=%s analysis_id=%s: %s", job_id, analysis_id, exc
                )

            yield format_sse(
                "done",
                {
                    "job_id": job_id,
                    "completed_at": datetime.utcnow().isoformat() + "Z",
                    "model": last_model,
                    "analysis_id": analysis_id,
                    "artifact_id": artifact_id,
                    **(({"summary": summary_text}) if summary_text else {}),
                },
            )
            logger.info("[ANALYZE-STREAM] job=%s analysis_id=%s finished model=%s", job_id, analysis_id, last_model)
        except asyncio.CancelledError:
            logger.warning("[ANALYZE-STREAM] Stream cancelled job=%s", job_id)
            raise
        except Exception as exc:
            logger.error("[ANALYZE-STREAM] Unexpected error job=%s analysis_id=%s: %s", job_id, analysis_id, exc)
            yield format_sse("error", {"job_id": job_id, "detail": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _prepare_stream_payload(raw: str) -> dict[str, Any]:
    """Decode a base64-encoded JSON payload from query string."""
    if not raw:
        return {}
    try:
        # Try standard base64
        decoded = base64.b64decode(raw)
    except Exception:
        try:
            # Try URL-safe base64
            decoded = base64.urlsafe_b64decode(raw + "==")
        except Exception:
            return {}
    try:
        return json.loads(decoded)
    except Exception:
        return {}


@router.get("/api/gemma/analyze/stream/inline/start")
async def gemma_analyze_stream_inline(
    payload: str = Query(..., description="Base64 payload"),
    http_request: Request = None,
    session: Session = Depends(require_auth),
):
    """Stream analyzer progress via Server-Sent Events without a separate job-creation step.

    Mirrors gemma_analyze_stream but accepts a base64-encoded JSON payload directly in the query string.
    """
    # Import dependencies
    proxy_request = _get_proxy_request()
    _gemma_generate_with_fallback = _get_gemma_generate_with_fallback()

    decoded_payload = _prepare_stream_payload(payload)
    analysis_id = decoded_payload.get("analysis_id") or (
        http_request.query_params.get("analysis_id") if http_request else None
    )
    analysis_headers = {"X-Analysis-Id": analysis_id} if analysis_id else None

    filters = decoded_payload.get("filters") or {}
    try:
        raw_max = decoded_payload.get("max_statements")
        analysis_limit = int(raw_max) if raw_max is not None else int(filters.get("limit", 20) or 20)
    except Exception:
        analysis_limit = 20
    analysis_limit = max(1, min(analysis_limit, 200))

    async def event_generator():
        last_model = None
        started_at = datetime.utcnow().isoformat() + "Z"
        MAX_PROMPT_CHARS = 6000
        combined_sections: list[str] = []
        transcripts_sentinel = "<END_OF_TRANSCRIPTS>"
        analysis_stop_sequences: list[str] = []
        prompt_template = (decoded_payload.get("custom_prompt") or "").strip()
        has_transcript_placeholder = "{transcripts}" in prompt_template
        apply_short_instruction = bool(prompt_template) and not has_transcript_placeholder
        guardrail_instruction = (
            "You are an analyst. Please answer the user's question about the given transcript section."
            if apply_short_instruction
            else ""
        )

        ANALYZE_FALLBACK_ENABLED = os.getenv("ANALYZE_FALLBACK_ENABLED", "true").lower() in {"1", "true", "yes"}

        try:
            # Warmup GPU (best effort)
            try:
                await proxy_request(f"{GEMMA_URL}/warmup", "POST", json={}, extra_headers=analysis_headers)
                yield format_sse("meta", {"message": "GPU warmup complete"})
            except Exception as warmup_error:
                logger.warning("[ANALYZE-STREAM] Warmup failed (inline): %s", warmup_error)
                yield format_sse("meta", {"message": "GPU warmup failed; continuing"})

            rag_payload = dict(filters)
            rag_payload["limit"] = analysis_limit
            rag_payload.setdefault("context_lines", max(0, min(int(filters.get("context_lines", 3) or 0), 10)))

            fallback_used = False
            raw_items: list[dict[str, Any]] = []
            dataset_total = None

            try:
                rag_result = await proxy_request(
                    f"{RAG_URL}/transcripts/query", "POST", json=rag_payload, extra_headers=analysis_headers
                )
                raw_items = rag_result.get("items") or []
                if isinstance(rag_result, dict):
                    dataset_total = rag_result.get("total") or rag_result.get("count")
            except HTTPException as exc:
                if exc.status_code == 404 and ANALYZE_FALLBACK_ENABLED:
                    fallback_used = True
                    # Fallback: fetch from recent transcripts
                    recent_response = await proxy_request(
                        f"{RAG_URL}/transcripts/recent?limit={analysis_limit * 5}",
                        "GET",
                        extra_headers=analysis_headers,
                    )
                    transcripts = recent_response.get("transcripts") or []
                    for tr in transcripts:
                        for seg in tr.get("segments") or []:
                            if (seg.get("text") or "").strip():
                                raw_items.append(
                                    {
                                        "speaker": seg.get("speaker"),
                                        "text": seg.get("text"),
                                        "emotion": seg.get("emotion"),
                                        "context_before": [],
                                    }
                                )
                            if len(raw_items) >= analysis_limit:
                                break
                        if len(raw_items) >= analysis_limit:
                            break
                else:
                    raise

            items = [item for item in raw_items if isinstance(item, dict) and (item.get("text") or "").strip()][
                :analysis_limit
            ]
            total = len(items)

            meta_payload = {"total": total, "max_statements": analysis_limit}
            if fallback_used:
                meta_payload["fallback"] = "transcripts/recent"
            if isinstance(dataset_total, int):
                meta_payload["dataset_total"] = dataset_total
            yield format_sse("meta", meta_payload)

            if total == 0:
                yield format_sse("done", {"completed_at": datetime.utcnow().isoformat() + "Z", "model": None})
                return

            def _alpha_label(position: int) -> str:
                if position <= 0:
                    return str(position)
                label = ""
                while position > 0:
                    position, rem = divmod(position - 1, 26)
                    label = chr(65 + rem) + label
                return label

            for index, item in enumerate(items, start=1):
                if http_request and await http_request.is_disconnected():
                    break

                label = _alpha_label(index)
                context_before = item.get("context_before") or []
                context_plain = "\n".join(
                    [
                        f"{ctx.get('speaker') or 'Speaker'}: {ctx.get('text')}"
                        for ctx in context_before
                        if ctx.get("text")
                    ]
                ).strip()
                statement_plain = f"{item.get('speaker') or 'Speaker'}: {item.get('text')}".strip()

                formatted_sections: list[str] = [f"Statement {label}"]
                if context_plain:
                    formatted_sections.extend(["Context:", "```", context_plain, "```"])
                formatted_sections.extend(["Statement:", "```", statement_plain, "```"])
                formatted_sections.append(transcripts_sentinel)
                formatted_transcripts = "\n".join(formatted_sections)

                if has_transcript_placeholder:
                    user_instruction = prompt_template.replace("{transcripts}", formatted_transcripts)
                else:
                    base_prompt = prompt_template or "Analyze the following transcript section."
                    user_instruction = base_prompt.strip() + "\n\nTranscript Section:\n" + formatted_transcripts
                final_prompt = (guardrail_instruction.strip() + "\n\n" + user_instruction.strip()).strip()

                if len(final_prompt) > MAX_PROMPT_CHARS:
                    final_prompt = final_prompt[-MAX_PROMPT_CHARS:]

                yield format_sse("step", {"i": index, "total": total, "status": "prompting"})

                try:
                    gemma_request = {
                        "prompt": final_prompt,
                        "max_tokens": int(decoded_payload.get("max_tokens", 512) or 512),
                        "temperature": float(decoded_payload.get("temperature", 0.4) or 0.4),
                        "stop": analysis_stop_sequences,
                    }
                    answer_text, gen_resp = await _gemma_generate_with_fallback(gemma_request, analysis_headers)
                    last_model = (gen_resp or {}).get("model") or last_model
                    if not isinstance(answer_text, str) or not answer_text.strip():
                        answer_text = "Gemma returned no text for this statement."
                except HTTPException as exc:
                    yield format_sse("server_error", {"detail": f"Gemma error: {getattr(exc, 'detail', exc)}"})
                    return

                combined_sections.append(
                    "\n".join(
                        [
                            f"Statement {label}",
                            "Statement:",
                            statement_plain,
                            "Gemma Response:",
                            answer_text or "(empty)",
                        ]
                    )
                )
                yield format_sse("result", {"i": index, "total": total, "response": answer_text, "item": item})

            # Summary generation
            summary_text = ""
            try:
                if combined_sections:
                    summary_context = "\n\n".join(combined_sections)[-12000:]
                    summary_prompt = (
                        "You are an expert conversation analyst. Based on the following analysis sections, "
                        "write a concise executive summary with 5-8 bullet points and a 2-3 sentence conclusion.\n\n"
                        f"{summary_context}\n\nNow provide the executive summary:"
                    )
                    gen_req = {"prompt": summary_prompt, "max_tokens": 384, "temperature": 0.3}
                    summary_text, _ = await _gemma_generate_with_fallback(gen_req, analysis_headers)
            except Exception as exc:
                logger.warning("[ANALYZE-STREAM] summary generation failed (inline): %s", exc)

            yield format_sse(
                "done",
                {
                    "completed_at": datetime.utcnow().isoformat() + "Z",
                    "model": last_model,
                    **({"summary": summary_text} if summary_text else {}),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[ANALYZE-STREAM] Unexpected error (inline): %s", exc)
            yield format_sse("error", {"detail": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# =============================================================================
# Summary & Personality Endpoints
# =============================================================================


@router.post("/api/analyze/gemma_summary")
async def analyze_gemma_summary(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Generate a conversational summary using Gemma based on provided context."""
    proxy_request = _get_proxy_request()

    context_raw = str(request.get("context", "") or "").strip()
    if not context_raw:
        raise HTTPException(status_code=400, detail="context is required")

    context = context_raw[:8000]
    emotion_focus = (request.get("emotion") or "neutral").strip().lower()
    max_tokens = int(request.get("max_tokens") or 320)
    temperature = float(request.get("temperature") or 0.4)

    prompt = (
        "You are an expert conversation analyst. Review the transcript excerpts "
        "and produce a concise summary for an executive audience. Highlight the most "
        "important events, decisions, risks, and opportunities. If an emotion focus is provided, "
        "weave in how that emotion manifests.\n\n"
        f"Emotion focus: {emotion_focus or 'neutral'}\n\n"
        "Transcript context:\n"
        f"{context}\n\n"
        "Write the summary as bullet points followed by a short paragraph of key insights."
    )

    generation_payload = {
        "prompt": prompt,
        "max_tokens": max(120, min(max_tokens, 512)),
        "temperature": max(0.1, min(temperature, 1.0)),
    }

    gemma_response = await proxy_request(f"{GEMMA_URL}/generate", "POST", json=generation_payload)

    summary_text = ""
    if isinstance(gemma_response, dict):
        summary_text = gemma_response.get("text") or gemma_response.get("response") or ""

    return {
        "success": True,
        "summary": summary_text.strip(),
        "emotion": emotion_focus,
        "model": gemma_response.get("model") if isinstance(gemma_response, dict) else None,
        "raw": gemma_response,
    }


@router.post("/api/analyze/prepare_emotion_analysis")
async def analyze_prepare_emotion_analysis(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Fetch emotion statistics from RAG service and tailor them for the UI."""
    proxy_request = _get_proxy_request()

    start_date = request.get("start_date")
    end_date = request.get("end_date")

    params: dict[str, Any] = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    rag_stats = await proxy_request(f"{RAG_URL}/memory/emotions/stats", "GET", params=params or None)

    emotions_requested = request.get("emotions")
    emotion_counts: dict[str, Any] = {}

    if isinstance(emotions_requested, list) and rag_stats:
        for emotion_name in emotions_requested:
            if isinstance(emotion_name, str):
                emotion_counts[emotion_name] = rag_stats.get(emotion_name, 0)
    else:
        for key, value in (rag_stats or {}).items():
            if isinstance(value, (int, float)):
                emotion_counts[key] = value

    return {
        "success": True,
        "start_date": start_date,
        "end_date": end_date,
        "emotions": emotion_counts,
        "raw": rag_stats,
    }


@router.post("/api/analyze/personality")
async def analyze_personality(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Run personality analysis by collecting recent transcripts and prompting Gemma."""
    proxy_request = _get_proxy_request()

    last_n = request.get("last_n_transcripts") or request.get("limit") or 15
    try:
        last_n_int = max(5, min(int(last_n), 40))
    except (TypeError, ValueError):
        last_n_int = 15

    transcripts_payload = await proxy_request(f"{RAG_URL}/transcripts/recent", "GET", params={"limit": last_n_int})

    transcript_items = []
    if isinstance(transcripts_payload, dict):
        transcript_items = transcripts_payload.get("transcripts") or transcripts_payload.get("items") or []
    elif isinstance(transcripts_payload, list):
        transcript_items = transcripts_payload

    snippets = []
    max_snippets = 200
    for item in transcript_items[:last_n_int]:
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker") or item.get("primary_speaker") or "Speaker"
        text = item.get("snippet") or item.get("text") or item.get("full_text") or ""
        if text:
            snippets.append(f"{speaker}: {text}")
        if len(snippets) >= max_snippets:
            break
        segments = item.get("segments") or []
        if isinstance(segments, list):
            for seg in segments[:5]:
                if isinstance(seg, dict):
                    seg_speaker = seg.get("speaker") or speaker
                    seg_text = seg.get("text") or ""
                    if seg_text:
                        snippets.append(f"{seg_speaker}: {seg_text}")
                    if len(snippets) >= max_snippets:
                        break
            if len(snippets) >= max_snippets:
                break

    conversation_sample = "\n".join(snippets)[:6000] or "No transcript data available."

    prompt = (
        "You are a professional psychologist analyzing the conversation below. "
        "Provide a personality profile using Big Five traits, communication style, strengths, "
        "risks, and actionable coaching suggestions. Be balanced and evidence-driven.\n\n"
        "Conversation sample:\n"
        f"{conversation_sample}\n\n"
        "Respond with structured sections titled: 'Overview', 'Trait Breakdown', "
        "'Communication Style', 'Strengths', 'Risks', and 'Coaching Tips'."
    )

    gemma_response = await proxy_request(
        f"{GEMMA_URL}/generate", "POST", json={"prompt": prompt, "max_tokens": 512, "temperature": 0.5}
    )

    analysis_text = ""
    if isinstance(gemma_response, dict):
        analysis_text = gemma_response.get("text") or gemma_response.get("response") or ""

    job_id = f"persona_{uuid.uuid4().hex[:12]}"
    result_payload = {
        "job_id": job_id,
        "status": "complete",
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "result": {
            "analysis": analysis_text.strip(),
            "model": gemma_response.get("model") if isinstance(gemma_response, dict) else None,
            "raw": gemma_response,
        },
    }

    personality_jobs[job_id] = result_payload
    return result_payload


@router.get("/api/analyze/personality/{job_id}")
async def get_personality_result(job_id: str, session: Session = Depends(require_auth)):
    """Return previously computed personality analysis."""
    job = personality_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Personality analysis not found")
    return job


# =============================================================================
# Artifact Chat Endpoints
# =============================================================================


@router.post("/api/gemma/chat-on-artifact")
async def chat_on_artifact_api(
    payload: dict[str, Any],
    session: Session = Depends(require_auth),
):
    """Chat about a specific analysis artifact."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{GEMMA_URL}/chat/on-artifact", "POST", json=payload)


@router.post("/api/gemma/chat-on-artifact/v2")
async def chat_on_artifact_v2_api(
    payload: dict[str, Any],
    session: Session = Depends(require_auth),
):
    """Chat about a specific analysis artifact (v2 API)."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{GEMMA_URL}/chat/on-artifact/v2", "POST", json=payload)


logger.info("✅ Gemma Router initialized with AI endpoints")
