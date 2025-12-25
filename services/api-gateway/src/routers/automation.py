"""
Automation Router - Workflow Orchestration API.

Provides endpoints for:
- Workflow CRUD
- Workflow execution
- Job status monitoring
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/automation", tags=["automation"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class StepConfig(BaseModel):
    """Workflow step configuration."""

    type: str = "action"
    name: str = ""
    config: dict = Field(default_factory=dict)


class WorkflowCreate(BaseModel):
    """Workflow creation request."""

    name: str
    description: str = ""
    steps: list[StepConfig]
    trigger: str = "manual"
    trigger_config: dict = Field(default_factory=dict)


class WorkflowResponse(BaseModel):
    """Workflow response."""

    id: str
    name: str
    description: str
    status: str
    trigger: str
    step_count: int
    created_at: str


class ExecutionResponse(BaseModel):
    """Workflow execution response."""

    id: str
    workflow_id: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ExecuteRequest(BaseModel):
    """Workflow execution request."""

    input_data: dict = Field(default_factory=dict)


# =============================================================================
# WORKFLOW ENGINE SINGLETON
# =============================================================================

_engine = None


def get_engine():
    """Get or create workflow engine singleton."""
    global _engine
    if _engine is None:
        from shared.automation.workflow_engine import WorkflowEngine
        _engine = WorkflowEngine()
    return _engine


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/workflows", status_code=201, response_model=WorkflowResponse)
async def create_workflow(request: WorkflowCreate):
    """
    Create a new workflow.

    Args:
        request: Workflow creation request

    Returns:
        Created workflow
    """
    try:
        from shared.automation.workflow_engine import TriggerType

        engine = get_engine()

        steps = [{"type": s.type, "name": s.name, "config": s.config} for s in request.steps]
        trigger = TriggerType(request.trigger) if request.trigger else TriggerType.MANUAL

        workflow = engine.create_workflow(
            name=request.name,
            steps=steps,
            trigger=trigger,
            description=request.description,
        )

        return WorkflowResponse(
            id=str(workflow.id),
            name=workflow.name,
            description=workflow.description,
            status=workflow.status.value,
            trigger=workflow.trigger.value,
            step_count=len(workflow.steps),
            created_at=workflow.created_at.isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows", response_model=list[WorkflowResponse])
async def list_workflows(
    tenant_id: Optional[str] = Query(None),
):
    """
    List all workflows.

    Args:
        tenant_id: Optional tenant filter

    Returns:
        List of workflows
    """
    engine = get_engine()

    tid = UUID(tenant_id) if tenant_id else None
    workflows = engine.list_workflows(tenant_id=tid)

    return [
        WorkflowResponse(
            id=str(w.id),
            name=w.name,
            description=w.description,
            status=w.status.value,
            trigger=w.trigger.value,
            step_count=len(w.steps),
            created_at=w.created_at.isoformat(),
        )
        for w in workflows
    ]


@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str):
    """
    Get workflow by ID.

    Args:
        workflow_id: Workflow UUID

    Returns:
        Workflow details
    """
    engine = get_engine()
    workflow = engine.get_workflow(UUID(workflow_id))

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return WorkflowResponse(
        id=str(workflow.id),
        name=workflow.name,
        description=workflow.description,
        status=workflow.status.value,
        trigger=workflow.trigger.value,
        step_count=len(workflow.steps),
        created_at=workflow.created_at.isoformat(),
    )


@router.post("/workflows/{workflow_id}/run", response_model=ExecutionResponse)
async def execute_workflow(workflow_id: str, request: ExecuteRequest = None):
    """
    Execute a workflow.

    Args:
        workflow_id: Workflow UUID
        request: Optional execution input

    Returns:
        Execution status
    """
    engine = get_engine()
    workflow = engine.get_workflow(UUID(workflow_id))

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    input_data = request.input_data if request else {}
    execution = await engine.execute(workflow, input_data)

    return ExecutionResponse(
        id=str(execution.id),
        workflow_id=str(execution.workflow_id),
        status=execution.status.value,
        started_at=execution.started_at.isoformat(),
        completed_at=execution.completed_at.isoformat() if execution.completed_at else None,
        error=execution.error,
    )


@router.get("/jobs/{job_id}", response_model=ExecutionResponse)
async def get_job_status(job_id: str):
    """
    Get job/execution status.

    Args:
        job_id: Execution UUID

    Returns:
        Execution status
    """
    engine = get_engine()
    execution = engine.get_execution(UUID(job_id))

    if not execution:
        raise HTTPException(status_code=404, detail="Job not found")

    return ExecutionResponse(
        id=str(execution.id),
        workflow_id=str(execution.workflow_id),
        status=execution.status.value,
        started_at=execution.started_at.isoformat(),
        completed_at=execution.completed_at.isoformat() if execution.completed_at else None,
        error=execution.error,
    )


@router.get("/jobs", response_model=list[ExecutionResponse])
async def list_jobs(
    workflow_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    """
    List all executions/jobs.

    Args:
        workflow_id: Filter by workflow
        status: Filter by status

    Returns:
        List of executions
    """
    from shared.automation.workflow_engine import WorkflowStatus

    engine = get_engine()

    wid = UUID(workflow_id) if workflow_id else None
    ws = WorkflowStatus(status) if status else None

    executions = engine.list_executions(workflow_id=wid, status=ws)

    return [
        ExecutionResponse(
            id=str(e.id),
            workflow_id=str(e.workflow_id),
            status=e.status.value,
            started_at=e.started_at.isoformat(),
            completed_at=e.completed_at.isoformat() if e.completed_at else None,
            error=e.error,
        )
        for e in executions
    ]


logger.info("✅ Automation Router initialized with workflow endpoints")
