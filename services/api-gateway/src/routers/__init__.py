"""
Routers package for API Gateway.

Phase 2: Architecture decomposition - split 155KB main.py into focused modules.

Routers:
- auth: Authentication and user management
- health: Health checks and status
- gemma: Gemma AI endpoints (chat, generate, analyze, streaming)
- rag: RAG queries and memory management
- ml: ML analytics, vectorization, AutoML
- email: Email analysis endpoints
- transcription: Audio transcription and emotion
- transcripts: Transcript query and analytics
- analysis: Analysis artifact management
- enrollment: Speaker voice enrollment
- websocket: Real-time streaming proxy
- enterprise: Enterprise features (QA, Automation, Knowledge, Analytics, Meetings)
"""

from .analysis import router as analysis_router
from .auth import router as auth_router
from .email_analyzer import router as email_router
from .enrollment import router as enrollment_router

# New modular routers (Phase 2)
from .gemma import router as gemma_router
from .health import router as health_router
from .ml import router as ml_router
from .rag import router as rag_router
from .transcription import router as transcription_router
from .transcripts import router as transcripts_router
from .websocket import router as websocket_router

# Enterprise router (already existed)
try:
    from .enterprise import router as enterprise_router
except ImportError:
    enterprise_router = None

__all__ = [
    "auth_router",
    "health_router",
    "gemma_router",
    "rag_router",
    "ml_router",
    "email_router",
    "transcription_router",
    "transcripts_router",
    "analysis_router",
    "enrollment_router",
    "websocket_router",
    "enterprise_router",
]
