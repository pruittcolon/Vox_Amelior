"""
Salesforce Routes Package

Modular FastAPI route handlers organized by resource type.
"""

from .accounts import router as accounts_router
from .bulk import router as bulk_router
from .cases import router as cases_router
from .contacts import router as contacts_router
from .leads import router as leads_router
from .opportunities import router as opportunities_router
from .query import router as query_router

__all__ = [
    "accounts_router",
    "contacts_router",
    "opportunities_router",
    "leads_router",
    "cases_router",
    "bulk_router",
    "query_router",
]
