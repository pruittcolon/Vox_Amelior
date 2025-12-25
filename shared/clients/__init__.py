"""
Shared clients for inter-service communication.

Phase 2: Architecture decomposition - centralized HTTP clients with retry/circuit breaker.
"""

from .base import BaseServiceClient, CircuitBreaker

__all__ = [
    "BaseServiceClient",
    "CircuitBreaker",
]
