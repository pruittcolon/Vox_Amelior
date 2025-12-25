"""
Salesforce Enterprise Integration Module

Modular, enterprise-grade Salesforce CRM integration.

Modules:
- config: Configuration management
- errors: Custom exception classes
- models: Pydantic request/response schemas
- client: Enterprise API client with retry logic
- routes: FastAPI endpoint routers
"""

from .client import SalesforceClient, get_client
from .config import SALESFORCE_ENABLED, SalesforceConfig, get_config
from .errors import SalesforceAuthError, SalesforceError, SalesforceRateLimitError
from .router import router

__all__ = [
    # Configuration
    "SalesforceConfig",
    "get_config",
    "SALESFORCE_ENABLED",
    # Errors
    "SalesforceError",
    "SalesforceAuthError",
    "SalesforceRateLimitError",
    # Client
    "SalesforceClient",
    "get_client",
    # Router
    "router",
]
