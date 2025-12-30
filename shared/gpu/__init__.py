"""
GPU Coordination Module

Provides centralized GPU state management, protocol definitions, and clients
for coordinating GPU access between services (Gemma, Transcription, ML).

Architecture:
- state.py: GPU ownership states and transition validation
- protocol.py: Redis message protocol definitions (Pydantic models)
- client.py: Async HTTP client for GPU coordinator
- listener.py: Redis pub/sub listener for services

Usage:
    # In Gemma/ML service (GPU requesters):
    from shared.gpu.client import GPUCoordinatorClient
    
    client = GPUCoordinatorClient()
    async with client.gpu_session("session-123") as acquired:
        if acquired:
            # Run inference on GPU
            pass

    # In Transcription service (GPU owner):
    from shared.gpu.listener import GPUCommandListener
    
    listener = GPUCommandListener()
    listener.on_pause(my_pause_handler)
    listener.on_resume(my_resume_handler)
    await listener.start()
"""

from shared.gpu.state import GPUOwner, GPUState, StateTransitionError
from shared.gpu.protocol import (
    GPUAcquireRequest,
    GPUAcquireResponse,
    GPUReleaseRequest,
    GPUReleaseResponse,
    GPUCommand,
    GPUAck,
)
from shared.gpu.client import GPUCoordinatorClient, get_gpu_client
from shared.gpu.listener import GPUCommandListener

__all__ = [
    # State
    "GPUOwner",
    "GPUState",
    "StateTransitionError",
    # Protocol
    "GPUAcquireRequest",
    "GPUAcquireResponse",
    "GPUReleaseRequest",
    "GPUReleaseResponse",
    "GPUCommand",
    "GPUAck",
    # Client
    "GPUCoordinatorClient",
    "get_gpu_client",
    # Listener
    "GPUCommandListener",
]
