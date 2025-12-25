"""
Prompts Router - Prompt Registry API.

Provides endpoints for:
- Prompt CRUD with versioning
- Prompt version history
- Prompt templates
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/prompts", tags=["prompts"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class PromptCreate(BaseModel):
    """Prompt creation request."""

    name: str
    description: str = ""
    template: str
    variables: list[str] = Field(default_factory=list)
    category: str = "general"
    metadata: dict = Field(default_factory=dict)


class PromptUpdate(BaseModel):
    """Prompt update request (creates new version)."""

    template: str
    description: str = ""
    change_notes: str = ""


class PromptResponse(BaseModel):
    """Prompt response."""

    id: str
    name: str
    description: str
    template: str
    variables: list[str]
    category: str
    version: int
    created_at: str
    updated_at: str
    created_by: Optional[str] = None


class PromptVersionResponse(BaseModel):
    """Prompt version response."""

    version: int
    template: str
    change_notes: str
    created_at: str
    created_by: Optional[str] = None


class PromptListResponse(BaseModel):
    """Prompt list response."""

    prompts: list[PromptResponse]
    total: int


# =============================================================================
# IN-MEMORY STORAGE (Would use DB in production)
# =============================================================================

PROMPTS: dict[str, dict] = {}
PROMPT_VERSIONS: dict[str, list[dict]] = {}


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("", status_code=201, response_model=PromptResponse)
async def create_prompt(request: PromptCreate):
    """
    Create a new prompt.

    Args:
        request: Prompt creation request

    Returns:
        Created prompt
    """
    prompt_id = str(uuid4())
    now = datetime.utcnow().isoformat()

    prompt = {
        "id": prompt_id,
        "name": request.name,
        "description": request.description,
        "template": request.template,
        "variables": request.variables,
        "category": request.category,
        "metadata": request.metadata,
        "version": 1,
        "created_at": now,
        "updated_at": now,
        "created_by": None,
    }

    PROMPTS[prompt_id] = prompt
    PROMPT_VERSIONS[prompt_id] = [{
        "version": 1,
        "template": request.template,
        "change_notes": "Initial version",
        "created_at": now,
        "created_by": None,
    }]

    return PromptResponse(**prompt)


@router.get("", response_model=PromptListResponse)
async def list_prompts(
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
):
    """
    List all prompts.

    Args:
        category: Filter by category
        search: Search in name/description
        limit: Max results
        offset: Pagination offset

    Returns:
        List of prompts
    """
    prompts = list(PROMPTS.values())

    if category:
        prompts = [p for p in prompts if p["category"] == category]

    if search:
        search_lower = search.lower()
        prompts = [
            p for p in prompts
            if search_lower in p["name"].lower() or search_lower in p["description"].lower()
        ]

    total = len(prompts)
    prompts = prompts[offset : offset + limit]

    return PromptListResponse(
        prompts=[PromptResponse(**p) for p in prompts],
        total=total,
    )


@router.get("/{prompt_id}", response_model=PromptResponse)
async def get_prompt(prompt_id: str):
    """
    Get prompt by ID.

    Args:
        prompt_id: Prompt UUID

    Returns:
        Prompt details
    """
    prompt = PROMPTS.get(prompt_id)

    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    return PromptResponse(**prompt)


@router.put("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(prompt_id: str, request: PromptUpdate):
    """
    Update prompt (creates new version).

    Args:
        prompt_id: Prompt UUID
        request: Update request

    Returns:
        Updated prompt
    """
    prompt = PROMPTS.get(prompt_id)

    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    now = datetime.utcnow().isoformat()
    new_version = prompt["version"] + 1

    # Update prompt
    prompt["template"] = request.template
    prompt["description"] = request.description or prompt["description"]
    prompt["version"] = new_version
    prompt["updated_at"] = now

    # Add version entry
    PROMPT_VERSIONS[prompt_id].append({
        "version": new_version,
        "template": request.template,
        "change_notes": request.change_notes,
        "created_at": now,
        "created_by": None,
    })

    return PromptResponse(**prompt)


@router.delete("/{prompt_id}", status_code=204)
async def delete_prompt(prompt_id: str):
    """
    Delete prompt.

    Args:
        prompt_id: Prompt UUID
    """
    if prompt_id not in PROMPTS:
        raise HTTPException(status_code=404, detail="Prompt not found")

    del PROMPTS[prompt_id]
    if prompt_id in PROMPT_VERSIONS:
        del PROMPT_VERSIONS[prompt_id]


@router.get("/{prompt_id}/versions", response_model=list[PromptVersionResponse])
async def get_prompt_versions(prompt_id: str):
    """
    Get version history for a prompt.

    Args:
        prompt_id: Prompt UUID

    Returns:
        List of versions
    """
    if prompt_id not in PROMPTS:
        raise HTTPException(status_code=404, detail="Prompt not found")

    versions = PROMPT_VERSIONS.get(prompt_id, [])

    return [PromptVersionResponse(**v) for v in reversed(versions)]


@router.get("/{prompt_id}/versions/{version}", response_model=PromptVersionResponse)
async def get_prompt_version(prompt_id: str, version: int):
    """
    Get specific version of a prompt.

    Args:
        prompt_id: Prompt UUID
        version: Version number

    Returns:
        Version details
    """
    if prompt_id not in PROMPTS:
        raise HTTPException(status_code=404, detail="Prompt not found")

    versions = PROMPT_VERSIONS.get(prompt_id, [])
    for v in versions:
        if v["version"] == version:
            return PromptVersionResponse(**v)

    raise HTTPException(status_code=404, detail="Version not found")


logger.info("✅ Prompts Router initialized with versioning endpoints")
