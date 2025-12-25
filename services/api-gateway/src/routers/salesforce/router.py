"""
Salesforce Main Router

Assembles all modular route handlers into the main API router.
This is the entry point imported by the API Gateway.
"""

from fastapi import APIRouter

from . import streaming
from .routes.accounts import router as accounts_router
from .routes.analytics import router as analytics_router
from .routes.bulk import router as bulk_router
from .routes.cases import router as cases_router
from .routes.contacts import router as contacts_router
from .routes.leads import router as leads_router
from .routes.opportunities import router as opportunities_router
from .routes.query import router as query_router

# Main router with prefix
router = APIRouter(prefix="/api/v1/salesforce", tags=["salesforce"])

# Include all sub-routers
router.include_router(accounts_router)
router.include_router(contacts_router)
router.include_router(opportunities_router)
router.include_router(leads_router)
router.include_router(cases_router)
router.include_router(bulk_router)
router.include_router(query_router)
router.include_router(streaming.router)
router.include_router(analytics_router)
