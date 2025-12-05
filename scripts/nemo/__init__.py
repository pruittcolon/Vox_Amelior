"""
Nemo CLI - Unified Command-Line Interface for Nemo Server

Provides comprehensive CLI access to all Nemo Server functionality:
- API interactions (transcription, chat, search, analysis)
- Service management (start, stop, health checks)
- Testing (service-specific and full suite)
- Documentation validation

Usage:
    python3 -m nemo <command> [options]
    OR (after installation): nemo <command> [options]
"""

__version__ = "1.0.0"
__all__ = ["api_client", "service_manager", "test_runner", "validators"]
