"""Pydantic schemas for the Email Analyzer module."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EmailQueryFilters(BaseModel):
    users: list[str] | None = Field(default=None, description="Mailbox owners to filter")
    participants: list[str] | None = Field(default=None, description="Email participants")
    labels: list[str] | None = Field(default=None, description="Label filters")
    start_date: str | None = Field(default=None, description="ISO8601 start date")
    end_date: str | None = Field(default=None, description="ISO8601 end date")
    keywords: str | None = Field(default=None, description="Comma separated keywords")
    match: Literal["any", "all"] = Field(default="any")


class EmailQueryRequest(BaseModel):
    filters: EmailQueryFilters = Field(default_factory=EmailQueryFilters)
    limit: int = Field(default=25, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    sort_by: Literal["date", "sender", "thread"] = Field(default="date")
    order: Literal["asc", "desc"] = Field(default="desc")


class EmailQuickAnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    filters: EmailQueryFilters = Field(default_factory=EmailQueryFilters)


class EmailStreamRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=2000)
    filters: EmailQueryFilters = Field(default_factory=EmailQueryFilters)
    max_chunks: int = Field(default=20, ge=1, le=200)


class EmailCancelRequest(BaseModel):
    analysis_id: str | None = None


class EmailSummary(BaseModel):
    summary: str
    tokens_used: int = 128
    gpu_seconds: float = 0.0
    artifact_id: str | None = None


class EmailListResponse(BaseModel):
    success: bool = True
    items: list[dict[str, Any]]
    count: int
    offset: int
    has_more: bool


class EmailStatsResponse(BaseModel):
    success: bool = True
    totals: dict[str, Any]
    by_day: list[dict[str, Any]]
    top_senders: list[dict[str, Any]]
    top_threads: list[dict[str, Any]]


class EmailUsersResponse(BaseModel):
    success: bool = True
    items: list[dict[str, Any]]
    count: int


class EmailLabelsResponse(BaseModel):
    success: bool = True
    items: list[dict[str, Any]]
    count: int
