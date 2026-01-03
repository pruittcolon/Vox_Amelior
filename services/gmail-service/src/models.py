"""
Pydantic models for Gmail Automation Service.

Provides request/response schemas for OAuth, email fetching, and analysis endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TimeframePreset(str, Enum):
    """Predefined email fetch timeframes."""

    LAST_24H = "24h"
    LAST_7D = "7d"
    LAST_30D = "30d"
    LAST_90D = "90d"
    CUSTOM = "custom"


class AnalysisPreset(str, Enum):
    """Predefined analysis prompt templates."""

    SUMMARIZE = "summarize"
    ACTION_ITEMS = "action_items"
    SENDER_INTENT = "sender_intent"
    SENTIMENT = "sentiment"
    CUSTOM = "custom"


# =============================================================================
# OAuth Models
# =============================================================================


class OAuthURLResponse(BaseModel):
    """Response containing OAuth authorization URL."""

    authorization_url: str = Field(..., description="URL to redirect user to Google OAuth")
    state: str = Field(..., description="OAuth state parameter for CSRF protection")


class OAuthCallbackRequest(BaseModel):
    """Request containing OAuth callback data."""

    code: str = Field(..., description="Authorization code from Google")
    state: str = Field(..., description="State parameter for verification")


class OAuthStatusResponse(BaseModel):
    """Response containing OAuth connection status."""

    connected: bool = Field(..., description="Whether Gmail is connected")
    email: str | None = Field(None, description="Connected Gmail address")
    expires_at: datetime | None = Field(None, description="Token expiration time")
    scopes: list[str] = Field(default_factory=list, description="Granted OAuth scopes")


class OAuthDisconnectResponse(BaseModel):
    """Response after disconnecting OAuth."""

    success: bool
    message: str


# =============================================================================
# Email Models
# =============================================================================


class EmailFetchRequest(BaseModel):
    """Request to fetch emails from Gmail."""

    timeframe: TimeframePreset = Field(
        TimeframePreset.LAST_7D,
        description="Predefined timeframe for email fetch",
    )
    start_date: datetime | None = Field(
        None,
        description="Custom start date (required when timeframe is 'custom')",
    )
    end_date: datetime | None = Field(
        None,
        description="Custom end date (defaults to now)",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Gmail labels to filter (e.g., ['INBOX', 'IMPORTANT'])",
    )
    max_results: int = Field(
        50,
        ge=1,
        le=500,
        description="Maximum number of emails to fetch",
    )
    include_body: bool = Field(
        True,
        description="Whether to include email body content",
    )
    query: str | None = Field(
        None,
        description="Gmail search query (e.g., 'from:boss@example.com')",
    )


class EmailMessage(BaseModel):
    """Represents a single email message."""

    id: str = Field(..., description="Gmail message ID")
    thread_id: str = Field(..., description="Gmail thread ID")
    subject: str = Field("", description="Email subject")
    sender: str = Field(..., description="Sender email address")
    sender_name: str = Field("", description="Sender display name")
    recipients: list[str] = Field(default_factory=list, description="Recipient addresses")
    date: datetime = Field(..., description="Email date")
    snippet: str = Field("", description="Email preview snippet")
    body: str | None = Field(None, description="Full email body (if requested)")
    labels: list[str] = Field(default_factory=list, description="Gmail labels")
    is_unread: bool = Field(False, description="Whether email is unread")
    has_attachments: bool = Field(False, description="Whether email has attachments")


class EmailFetchResponse(BaseModel):
    """Response containing fetched emails."""

    success: bool
    emails: list[EmailMessage] = Field(default_factory=list)
    total_count: int = Field(0, description="Total matching emails")
    fetched_count: int = Field(0, description="Number of emails fetched")
    timeframe_start: datetime | None = None
    timeframe_end: datetime | None = None


# =============================================================================
# Analysis Models
# =============================================================================


class EmailAnalyzeRequest(BaseModel):
    """Request to analyze emails with Gemma."""

    email_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Gmail message IDs to analyze",
    )
    preset: AnalysisPreset = Field(
        AnalysisPreset.SUMMARIZE,
        description="Analysis preset template",
    )
    custom_prompt: str | None = Field(
        None,
        description="Custom prompt (required when preset is 'custom')",
    )
    max_tokens: int = Field(
        512,
        ge=64,
        le=2048,
        description="Maximum tokens per analysis response",
    )
    temperature: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="LLM temperature for analysis",
    )
    store_results: bool = Field(
        False,
        description="Whether to store results in RAG service",
    )


class EmailAnalysisResult(BaseModel):
    """Analysis result for a single email."""

    email_id: str
    subject: str
    sender: str
    analysis: str
    tokens_used: int = 0
    processing_time_ms: int = 0


class EmailAnalyzeResponse(BaseModel):
    """Response containing analysis results."""

    success: bool
    results: list[EmailAnalysisResult] = Field(default_factory=list)
    total_emails: int = 0
    total_tokens: int = 0
    total_time_ms: int = 0
    model: str = "gemma-3-4b"


class AnalysisStreamEvent(BaseModel):
    """SSE event for streaming analysis progress."""

    event_type: str = Field(..., description="Event type: progress, result, error, done")
    email_id: str | None = None
    progress: float | None = Field(None, ge=0.0, le=1.0)
    message: str | None = None
    result: EmailAnalysisResult | None = None
    error: str | None = None


# =============================================================================
# Prompt Templates
# =============================================================================


ANALYSIS_PROMPTS: dict[AnalysisPreset, str] = {
    AnalysisPreset.SUMMARIZE: (
        "Provide a concise 2-3 sentence summary of this email. "
        "Focus on the main point and any important details."
    ),
    AnalysisPreset.ACTION_ITEMS: (
        "Extract all action items, tasks, or requests from this email. "
        "Format as a numbered list. If no action items exist, state 'No action items found.'"
    ),
    AnalysisPreset.SENDER_INTENT: (
        "Analyze the sender's intent and purpose in sending this email. "
        "Identify: (1) Primary goal, (2) Urgency level, (3) Expected response."
    ),
    AnalysisPreset.SENTIMENT: (
        "Analyze the overall sentiment and tone of this email. "
        "Identify: (1) Sentiment (positive/neutral/negative), (2) Tone (formal/casual), "
        "(3) Any emotional indicators."
    ),
}


def get_analysis_prompt(preset: AnalysisPreset, custom_prompt: str | None = None) -> str:
    """Get the analysis prompt for a given preset.

    Args:
        preset: The analysis preset to use.
        custom_prompt: Custom prompt text (required if preset is CUSTOM).

    Returns:
        The prompt string to use for analysis.

    Raises:
        ValueError: If preset is CUSTOM but no custom_prompt provided.
    """
    if preset == AnalysisPreset.CUSTOM:
        if not custom_prompt or not custom_prompt.strip():
            raise ValueError("custom_prompt is required when using CUSTOM preset")
        return custom_prompt.strip()

    return ANALYSIS_PROMPTS.get(preset, ANALYSIS_PROMPTS[AnalysisPreset.SUMMARIZE])
