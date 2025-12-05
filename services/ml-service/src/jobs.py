import asyncio
import uuid
from typing import Dict, Any, Callable, Awaitable
from enum import Enum
import traceback

class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    def submit_job(self, task_func: Callable[..., Awaitable[Any]], *args, **kwargs) -> str:
        """
        Submits an async task to be run in the background.
        Returns the job_id.
        """
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.PENDING.value,
            "result": None,
            "error": None,
            "progress": "Queued"
        }
        
        # Create async task
        task = asyncio.create_task(self._run_job(job_id, task_func, *args, **kwargs))
        self._tasks[job_id] = task
        return job_id

    async def _run_job(self, job_id: str, task_func: Callable, *args, **kwargs):
        self.jobs[job_id]["status"] = JobStatus.RUNNING.value
        self.jobs[job_id]["progress"] = "Starting..."
        
        try:
            # Pass job_id to the task function so it can update progress if needed
            # We assume task_func accepts job_id as first argument or we wrap it?
            # For simplicity, let's assume task_func takes job_id as a keyword arg if it wants to update progress
            # But to be generic, let's just run it.
            # Ideally, we'd pass a progress_callback.
            
            def update_progress(msg: str):
                self.jobs[job_id]["progress"] = msg
            
            result = await task_func(update_progress, *args, **kwargs)
            
            self.jobs[job_id]["result"] = result
            self.jobs[job_id]["status"] = JobStatus.COMPLETED.value
            self.jobs[job_id]["progress"] = "Done"
            
        except asyncio.CancelledError:
            self.jobs[job_id]["status"] = JobStatus.CANCELLED.value
            self.jobs[job_id]["progress"] = "Cancelled"
            
        except Exception as e:
            traceback.print_exc()
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["status"] = JobStatus.FAILED.value
            self.jobs[job_id]["progress"] = "Failed"
            
        finally:
            # Cleanup task reference
            self._tasks.pop(job_id, None)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str):
        if job_id in self._tasks:
            self._tasks[job_id].cancel()
            return True
        return False
