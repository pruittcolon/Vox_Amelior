"""
Email Processor - Gemma-powered email analysis.

Handles batch processing of emails through the Gemma LLM service
with streaming progress support and GPU coordination.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncGenerator

import httpx

from .models import (
    AnalysisPreset,
    AnalysisStreamEvent,
    EmailAnalysisResult,
    EmailAnalyzeRequest,
    EmailMessage,
    get_analysis_prompt,
)

logger = logging.getLogger(__name__)

# Configuration
GEMMA_SERVICE_URL = os.getenv("GEMMA_SERVICE_URL", "http://gemma-service:8001")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8004")
GPU_COORDINATOR_URL = os.getenv("GPU_COORDINATOR_URL", "http://gpu-coordinator:8002")

# Processing limits
MAX_BODY_LENGTH = 4000  # Characters to include per email
MAX_CONCURRENT_REQUESTS = 3  # Parallel Gemma requests
REQUEST_TIMEOUT = 120.0  # Seconds per analysis request


class EmailProcessor:
    """Processes emails through Gemma LLM for analysis.

    Handles batch processing with progress tracking, GPU coordination,
    and optional RAG storage of results.
    """

    def __init__(self, service_headers: dict[str, str] | None = None) -> None:
        """Initialize email processor.

        Args:
            service_headers: JWT service auth headers for internal communication.
        """
        self.service_headers = service_headers or {}
        self._http_client: httpx.AsyncClient | None = None
        self._service_auth = None
        
        # Initialize service auth for generating tokens
        try:
            from shared.security.service_auth import get_service_auth, load_service_jwt_keys
            jwt_keys = load_service_jwt_keys("gmail-service")
            self._service_auth = get_service_auth(
                service_id="gmail-service",
                service_secret=jwt_keys,
            )
            logger.debug("[PROCESSOR] Service auth initialized")
        except Exception as e:
            logger.warning(f"[PROCESSOR] Service auth init failed: {e}")

    def _get_auth_headers(self) -> dict[str, str]:
        """Get auth headers for internal service calls.
        
        Returns:
            Headers dict with X-Service-Token if available.
        """
        # Use passed headers if available
        if self.service_headers.get("X-Service-Token"):
            return self.service_headers
        
        # Generate our own token using the correct API
        if self._service_auth:
            try:
                return self._service_auth.get_auth_header()
            except Exception as e:
                logger.warning(f"[PROCESSOR] Failed to generate token: {e}")
        
        return {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def _build_analysis_prompt(
        self,
        email: EmailMessage,
        analysis_prompt: str,
    ) -> str:
        """Build the full prompt for email analysis.

        Args:
            email: Email to analyze.
            analysis_prompt: The analysis instruction.

        Returns:
            Complete prompt string.
        """
        # Truncate body if too long
        body = email.body or email.snippet or ""
        if len(body) > MAX_BODY_LENGTH:
            body = body[:MAX_BODY_LENGTH] + "... [truncated]"

        return f"""You are an email analysis assistant. Analyze the following email and respond to the task.

EMAIL DETAILS:
From: {email.sender_name} <{email.sender}>
Subject: {email.subject}
Date: {email.date.strftime('%Y-%m-%d %H:%M')}

EMAIL CONTENT:
{body}

ANALYSIS TASK:
{analysis_prompt}

Provide a clear, concise response:"""

    async def _request_gpu_access(self, session_id: str) -> bool:
        """Request GPU access from coordinator.

        Args:
            session_id: Session ID for GPU retention.

        Returns:
            True if GPU access granted.
        """
        try:
            client = await self._get_client()
            response = await client.post(
                f"{GPU_COORDINATOR_URL}/request",
                json={
                    "service": "gmail-analysis",
                    "priority": "normal",
                    "session_id": session_id,
                },
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"[PROCESSOR] GPU coordinator request failed: {e}")
            # Continue anyway - Gemma service handles its own GPU
            return False

    async def _release_gpu_access(self, session_id: str) -> None:
        """Release GPU access to coordinator.

        Args:
            session_id: Session ID to release.
        """
        try:
            client = await self._get_client()
            await client.post(
                f"{GPU_COORDINATOR_URL}/release",
                json={"session_id": session_id},
                headers=self._get_auth_headers(),
            )
        except Exception as e:
            logger.debug(f"[PROCESSOR] GPU release notification failed: {e}")

    async def analyze_single(
        self,
        email: EmailMessage,
        request: EmailAnalyzeRequest,
    ) -> EmailAnalysisResult:
        """Analyze a single email.

        Args:
            email: Email to analyze.
            request: Analysis request parameters.

        Returns:
            Analysis result.

        Raises:
            httpx.HTTPStatusError: If Gemma service returns error.
        """
        start_time = time.monotonic()

        # Get the analysis prompt
        analysis_prompt = get_analysis_prompt(request.preset, request.custom_prompt)

        # Build full prompt
        full_prompt = self._build_analysis_prompt(email, analysis_prompt)

        # Call Gemma service
        client = await self._get_client()
        response = await client.post(
            f"{GEMMA_SERVICE_URL}/generate",
            json={
                "prompt": full_prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stop": ["EMAIL DETAILS:", "From:"],  # Prevent hallucinating next email
            },
            headers=self._get_auth_headers(),
        )
        response.raise_for_status()
        result_data = response.json()

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        return EmailAnalysisResult(
            email_id=email.id,
            subject=email.subject,
            sender=email.sender,
            analysis=result_data.get("text", result_data.get("response", "")),
            tokens_used=result_data.get("tokens_generated", 0),
            processing_time_ms=elapsed_ms,
        )

    async def analyze_batch(
        self,
        emails: list[EmailMessage],
        request: EmailAnalyzeRequest,
    ) -> list[EmailAnalysisResult]:
        """Analyze multiple emails in batch.

        Uses controlled concurrency to avoid overwhelming the GPU.

        Args:
            emails: List of emails to analyze.
            request: Analysis request parameters.

        Returns:
            List of analysis results.
        """
        session_id = f"gmail-batch-{int(time.time())}"
        results: list[EmailAnalysisResult] = []

        # Request GPU access for batch
        await self._request_gpu_access(session_id)

        try:
            # Process in batches with controlled concurrency
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

            async def process_with_semaphore(email: EmailMessage) -> EmailAnalysisResult | None:
                async with semaphore:
                    try:
                        return await self.analyze_single(email, request)
                    except Exception as e:
                        logger.error(f"[PROCESSOR] Analysis failed for {email.id}: {e}")
                        return EmailAnalysisResult(
                            email_id=email.id,
                            subject=email.subject,
                            sender=email.sender,
                            analysis=f"Analysis failed: {str(e)}",
                            tokens_used=0,
                            processing_time_ms=0,
                        )

            tasks = [process_with_semaphore(email) for email in emails]
            completed = await asyncio.gather(*tasks)

            results = [r for r in completed if r is not None]

        finally:
            await self._release_gpu_access(session_id)

        return results

    async def analyze_stream(
        self,
        emails: list[EmailMessage],
        request: EmailAnalyzeRequest,
    ) -> AsyncGenerator[AnalysisStreamEvent, None]:
        """Stream analysis results as they complete.

        Yields SSE-formatted events for real-time progress.

        Args:
            emails: List of emails to analyze.
            request: Analysis request parameters.

        Yields:
            AnalysisStreamEvent objects.
        """
        session_id = f"gmail-stream-{int(time.time())}"
        total = len(emails)
        completed = 0

        # Initial progress event
        yield AnalysisStreamEvent(
            event_type="progress",
            progress=0.0,
            message=f"Starting analysis of {total} emails...",
        )

        # Request GPU access
        await self._request_gpu_access(session_id)

        try:
            for email in emails:
                try:
                    # Progress update
                    yield AnalysisStreamEvent(
                        event_type="progress",
                        email_id=email.id,
                        progress=completed / total,
                        message=f"Analyzing: {email.subject[:50]}...",
                    )

                    # Analyze
                    result = await self.analyze_single(email, request)
                    completed += 1

                    # Result event
                    yield AnalysisStreamEvent(
                        event_type="result",
                        email_id=email.id,
                        progress=completed / total,
                        result=result,
                    )

                except Exception as e:
                    logger.error(f"[PROCESSOR] Stream analysis failed for {email.id}: {e}")
                    completed += 1
                    yield AnalysisStreamEvent(
                        event_type="error",
                        email_id=email.id,
                        progress=completed / total,
                        error=str(e),
                    )

        finally:
            await self._release_gpu_access(session_id)

        # Done event
        yield AnalysisStreamEvent(
            event_type="done",
            progress=1.0,
            message=f"Completed analysis of {completed} emails",
        )

    async def store_results_in_rag(
        self,
        results: list[EmailAnalysisResult],
        user_id: str,
    ) -> bool:
        """Store analysis results in RAG service.

        Args:
            results: Analysis results to store.
            user_id: User ID for ownership.

        Returns:
            True if storage was successful.
        """
        try:
            client = await self._get_client()

            # Store each result as a document
            for result in results:
                await client.post(
                    f"{RAG_SERVICE_URL}/index/document",
                    json={
                        "content": result.analysis,
                        "metadata": {
                            "source": "gmail_analysis",
                            "email_id": result.email_id,
                            "subject": result.subject,
                            "sender": result.sender,
                            "user_id": user_id,
                        },
                    },
                    headers=self.service_headers,
                )

            logger.info(f"[PROCESSOR] Stored {len(results)} analysis results in RAG")
            return True

        except Exception as e:
            logger.error(f"[PROCESSOR] Failed to store results in RAG: {e}")
            return False


def format_sse_event(event: AnalysisStreamEvent) -> str:
    """Format an event for SSE transmission.

    Args:
        event: Event to format.

    Returns:
        SSE-formatted string.
    """
    data = event.model_dump(exclude_none=True)
    return f"event: {event.event_type}\ndata: {json.dumps(data)}\n\n"
