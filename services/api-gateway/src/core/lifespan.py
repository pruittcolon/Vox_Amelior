"""
Core Lifespan Module - Application Lifecycle Management.

This module manages FastAPI application startup and shutdown events,
including security checks, authentication initialization, and
service authentication setup.

Follows fail-closed security principles - blocks startup if critical
security components are unavailable.
"""

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup/shutdown.

    Performs the following during startup:
    1. Fail-closed security checks (blocks if SECURE_MODE=true and unsafe)
    2. Auth module availability verification
    3. Session key loading from Docker secrets or environment
    4. AuthManager initialization with users database
    5. ServiceAuth initialization for inter-service JWT communication
    6. QA service initialization (optional)

    During shutdown:
    - Logs completion message

    Args:
        app: FastAPI application instance.

    Yields:
        Control back to FastAPI after startup is complete.

    Raises:
        RuntimeError: If auth module failed to import (fail-closed).
        RuntimeError: If service auth fails to initialize.

    Example:
        >>> app = FastAPI(lifespan=lifespan)
    """
    logger.info("Starting API Gateway...")

    # PHASE 0: Fail-closed security checks
    from shared.security.startup_checks import assert_secure_mode

    assert_secure_mode()  # Blocks if SECURE_MODE=true and unsafe flags enabled

    # Check if auth modules are available
    try:
        from src.auth.auth_manager import AuthManager

        _auth_loaded = True
    except ImportError as e:
        raise RuntimeError(
            f"SECURITY: AuthManager failed to import: {e}. Cannot start API Gateway without authentication."
        ) from e

    # Initialize auth manager via dependency injection
    from src.core.dependencies import get_auth_manager, get_service_auth

    try:
        auth_manager = get_auth_manager()
        logger.info("Auth manager initialized via dependency injection")
    except Exception as e:
        raise RuntimeError(f"SECURITY: AuthManager initialization failed: {e}") from e

    # Initialize service auth for inter-service JWTs
    logger.info("🔍 DEBUG: About to initialize JWT service auth")
    try:
        service_auth = get_service_auth()
        logger.info("✅ JWT service auth initialized for gateway")

        # Initialize QA service with service auth for GPU coordinator access
        try:
            from src.call_qa_service import init_qa_service

            init_qa_service(service_auth)
        except Exception as qa_err:
            logger.warning("QA service init failed (non-critical): %s", qa_err)
    except RuntimeError:
        raise  # Re-raise security failures
    except Exception as e:
        raise RuntimeError(f"SECURITY: ServiceAuth initialization failed: {e}. Run: ./scripts/setup_secrets.sh") from e

    yield

    logger.info("API Gateway shutdown complete")
