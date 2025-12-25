"""
Workflow Execution Engine.

Provides workflow definition, execution, and monitoring.
Supports step types: trigger, action, condition, delay.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
from uuid import UUID, uuid4


logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Types of workflow steps."""

    TRIGGER = "trigger"
    ACTION = "action"
    CONDITION = "condition"
    DELAY = "delay"
    PARALLEL = "parallel"
    LOOP = "loop"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    DRAFT = "draft"
    ACTIVE = "active"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class TriggerType(str, Enum):
    """Types of workflow triggers."""

    MANUAL = "manual"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    EVENT = "event"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""

    id: UUID = field(default_factory=uuid4)
    type: StepType = StepType.ACTION
    name: str = ""
    config: dict = field(default_factory=dict)
    next_steps: list[UUID] = field(default_factory=list)
    on_error: Optional[UUID] = None
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "type": self.type.value,
            "name": self.name,
            "config": self.config,
            "next_steps": [str(s) for s in self.next_steps],
            "on_error": str(self.on_error) if self.on_error else None,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class Workflow:
    """A workflow definition."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    tenant_id: Optional[UUID] = None
    steps: list[WorkflowStep] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.DRAFT
    trigger: TriggerType = TriggerType.MANUAL
    trigger_config: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "tenant_id": str(self.tenant_id) if self.tenant_id else None,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "trigger": self.trigger.value,
            "trigger_config": self.trigger_config,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowExecution:
    """A workflow execution instance."""

    id: UUID = field(default_factory=uuid4)
    workflow_id: UUID = field(default_factory=uuid4)
    status: WorkflowStatus = WorkflowStatus.RUNNING
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    current_step: Optional[UUID] = None
    step_results: dict = field(default_factory=dict)
    error: Optional[str] = None
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "workflow_id": str(self.workflow_id),
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_step": str(self.current_step) if self.current_step else None,
            "step_results": self.step_results,
            "error": self.error,
        }


# =============================================================================
# BUILT-IN ACTIONS
# =============================================================================

BUILTIN_ACTIONS: dict[str, Callable] = {}


def register_action(name: str):
    """Decorator to register a built-in action."""
    def decorator(func: Callable):
        BUILTIN_ACTIONS[name] = func
        return func
    return decorator


@register_action("ping")
async def action_ping(config: dict, context: dict) -> dict:
    """Simple ping action for testing."""
    return {"status": "pong", "timestamp": datetime.utcnow().isoformat()}


@register_action("log")
async def action_log(config: dict, context: dict) -> dict:
    """Log a message."""
    message = config.get("message", "Workflow log")
    logger.info(f"Workflow log: {message}")
    return {"logged": True, "message": message}


@register_action("http_request")
async def action_http_request(config: dict, context: dict) -> dict:
    """Make an HTTP request."""
    try:
        import httpx

        url = config.get("url")
        method = config.get("method", "GET").upper()
        headers = config.get("headers", {})
        data = config.get("data")
        timeout = config.get("timeout", 30)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, url, headers=headers, json=data)
            return {
                "status_code": response.status_code,
                "body": response.text[:1000],
            }
    except Exception as e:
        return {"error": str(e)}


@register_action("delay")
async def action_delay(config: dict, context: dict) -> dict:
    """Wait for a specified duration."""
    seconds = config.get("seconds", 1)
    await asyncio.sleep(seconds)
    return {"delayed": seconds}


# =============================================================================
# WORKFLOW ENGINE
# =============================================================================


class WorkflowEngine:
    """
    Workflow execution engine.

    Usage:
        engine = WorkflowEngine()
        workflow = engine.create_workflow("My Workflow", steps=[...])
        execution = await engine.execute(workflow)
    """

    def __init__(self):
        """Initialize workflow engine."""
        self._workflows: dict[UUID, Workflow] = {}
        self._executions: dict[UUID, WorkflowExecution] = {}
        self._custom_actions: dict[str, Callable] = {}

    def register_action(self, name: str, func: Callable) -> None:
        """Register a custom action."""
        self._custom_actions[name] = func

    def create_workflow(
        self,
        name: str,
        steps: list[dict],
        trigger: TriggerType = TriggerType.MANUAL,
        tenant_id: Optional[UUID] = None,
        **kwargs,
    ) -> Workflow:
        """
        Create a new workflow.

        Args:
            name: Workflow name
            steps: List of step configurations
            trigger: Trigger type
            tenant_id: Optional tenant ID

        Returns:
            Created Workflow
        """
        workflow_steps = []
        for i, step_config in enumerate(steps):
            step = WorkflowStep(
                type=StepType(step_config.get("type", "action")),
                name=step_config.get("name", f"Step {i + 1}"),
                config=step_config.get("config", {}),
            )
            workflow_steps.append(step)

        # Link steps sequentially by default
        for i in range(len(workflow_steps) - 1):
            workflow_steps[i].next_steps = [workflow_steps[i + 1].id]

        workflow = Workflow(
            name=name,
            steps=workflow_steps,
            trigger=trigger,
            tenant_id=tenant_id,
            status=WorkflowStatus.ACTIVE,
            **kwargs,
        )

        self._workflows[workflow.id] = workflow
        return workflow

    async def execute(
        self,
        workflow: Workflow,
        input_data: Optional[dict] = None,
    ) -> WorkflowExecution:
        """
        Execute a workflow.

        Args:
            workflow: Workflow to execute
            input_data: Optional input data

        Returns:
            WorkflowExecution instance
        """
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            input_data=input_data or {},
        )
        self._executions[execution.id] = execution

        try:
            context = {
                "workflow_id": str(workflow.id),
                "execution_id": str(execution.id),
                "input": input_data or {},
                "step_results": {},
            }

            # Execute steps sequentially for now
            for step in workflow.steps:
                execution.current_step = step.id

                result = await self._execute_step(step, context)
                execution.step_results[str(step.id)] = result
                context["step_results"][str(step.id)] = result

                if result.get("error"):
                    execution.status = WorkflowStatus.FAILED
                    execution.error = result["error"]
                    break

            if execution.status != WorkflowStatus.FAILED:
                execution.status = WorkflowStatus.COMPLETED

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)

        finally:
            execution.completed_at = datetime.utcnow()
            execution.output_data = context.get("step_results", {})

        return execution

    async def _execute_step(self, step: WorkflowStep, context: dict) -> dict:
        """Execute a single workflow step."""
        try:
            if step.type == StepType.TRIGGER:
                # Triggers are entry points, just log
                return {"triggered": True, "config": step.config}

            elif step.type == StepType.DELAY:
                seconds = step.config.get("seconds", 1)
                await asyncio.sleep(seconds)
                return {"delayed": seconds}

            elif step.type == StepType.CONDITION:
                # Simple condition evaluation
                condition = step.config.get("condition", "true")
                # TODO: Implement proper condition parsing
                return {"evaluated": True, "result": True}

            elif step.type == StepType.ACTION:
                action_name = step.config.get("action", "ping")

                # Check custom actions first
                if action_name in self._custom_actions:
                    return await self._custom_actions[action_name](step.config, context)

                # Then built-in actions
                if action_name in BUILTIN_ACTIONS:
                    return await BUILTIN_ACTIONS[action_name](step.config, context)

                return {"error": f"Unknown action: {action_name}"}

            else:
                return {"error": f"Unknown step type: {step.type}"}

        except asyncio.TimeoutError:
            return {"error": "Step timed out"}
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return {"error": str(e)}

    def get_workflow(self, workflow_id: UUID) -> Optional[Workflow]:
        """Get workflow by ID."""
        return self._workflows.get(workflow_id)

    def get_execution(self, execution_id: UUID) -> Optional[WorkflowExecution]:
        """Get execution by ID."""
        return self._executions.get(execution_id)

    def list_workflows(self, tenant_id: Optional[UUID] = None) -> list[Workflow]:
        """List all workflows, optionally filtered by tenant."""
        workflows = list(self._workflows.values())
        if tenant_id:
            workflows = [w for w in workflows if w.tenant_id == tenant_id]
        return workflows

    def list_executions(
        self,
        workflow_id: Optional[UUID] = None,
        status: Optional[WorkflowStatus] = None,
    ) -> list[WorkflowExecution]:
        """List executions with optional filters."""
        executions = list(self._executions.values())

        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        if status:
            executions = [e for e in executions if e.status == status]

        return sorted(executions, key=lambda e: e.started_at, reverse=True)
