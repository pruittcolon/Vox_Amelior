"""Analytics Routers Package.

Modular router architecture for ML Service analytics endpoints.
Imports all sub-routers for easy assembly.

Usage in main.py:
    from routers import (
        core_router,
        premium_router,
        financial_router,
        quick_router,
        history_router,
        cide_router,  # Contextual Insight Discovery Engine
        database_viewer_router,  # Database content viewer
    )

    app.include_router(core_router, prefix="/analytics")
    app.include_router(premium_router, prefix="/analytics")
    app.include_router(financial_router, prefix="/analytics")
    app.include_router(quick_router, prefix="/analytics")
    app.include_router(history_router, prefix="/analytics")
    app.include_router(cide_router)  # Mounted at /cide
    app.include_router(database_viewer_router)  # Mounted at /databases
"""

try:
    from .core_analytics import router as core_router
    from .premium_engines import router as premium_router
    from .financial_analytics import router as financial_router
    from .quick_analysis import router as quick_router
    from .analysis_history import router as history_router
    from .cide import router as cide_router
    from .database_viewer import router as database_viewer_router
except ImportError:
    from core_analytics import router as core_router
    from premium_engines import router as premium_router
    from financial_analytics import router as financial_router
    from quick_analysis import router as quick_router
    from analysis_history import router as history_router
    try:
        from cide import router as cide_router
    except ImportError:
        cide_router = None
    try:
        from database_viewer import router as database_viewer_router
    except ImportError:
        database_viewer_router = None

__all__ = [
    "core_router",
    "premium_router",
    "financial_router",
    "quick_router",
    "history_router",
    "cide_router",
    "database_viewer_router",
]

