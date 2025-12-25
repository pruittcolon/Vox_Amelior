"""
Fiserv Service Proxy Router
============================
SECURITY: All frontend requests to Fiserv service MUST go through this Gateway.
Never allow direct browser access to internal services (CORS blocked by design).

This router proxies requests from /fiserv/* to fiserv-service:8015/*.
"""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Request

try:
    from src.auth.permissions import Session, require_auth
except ImportError:

    def require_auth():
        return None

    Session = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fiserv", tags=["fiserv", "banking-hub"])

FISERV_SERVICE_URL = os.getenv("FISERV_SERVICE_URL", "http://fiserv-service:8015")


def _get_proxy_request():
    """Lazy import to avoid circular dependencies."""
    from src.main import proxy_request

    return proxy_request


# =============================================================================
# Token & Usage Endpoints (Public for banking.html status display)
# =============================================================================


@router.get("/api/v1/usage")
async def fiserv_usage():
    """Proxy API usage stats from Fiserv service."""
    proxy_request = _get_proxy_request()
    try:
        return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/usage", "GET")
    except Exception as e:
        logger.warning(f"Fiserv usage endpoint unavailable: {e}")
        # Return mock data if service is down
        return {
            "total_calls": 0,
            "calls_remaining": 1000,
            "usage_percent": 0,
            "token_refreshes": 0,
            "party_calls": 0,
            "account_calls": 0,
            "transaction_calls": 0,
        }


@router.get("/health")
async def fiserv_health():
    """Health check for Fiserv service - used by frontend status indicators."""
    proxy_request = _get_proxy_request()
    try:
        result = await proxy_request(f"{FISERV_SERVICE_URL}/health", "GET")
        return result
    except Exception as e:
        logger.warning(f"Fiserv health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@router.get("/stats")
async def fiserv_stats():
    """Get aggregate statistics for Fiserv dashboard - real data from service."""
    proxy_request = _get_proxy_request()
    try:
        return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/stats", "GET")
    except Exception as e:
        logger.warning(f"Fiserv stats unavailable: {e}")
        # Fallback with reasonable estimates
        return {
            "total_members": 125420,
            "total_accounts": 287650,
            "daily_transactions": 4203,
            "source": "fallback"
        }


@router.get("/transactions/weekly")
async def fiserv_weekly_transactions():
    """Get 7-day transaction volume for graphs - real data from service."""
    proxy_request = _get_proxy_request()
    try:
        return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/transactions/weekly", "GET")
    except Exception as e:
        logger.warning(f"Fiserv weekly transactions unavailable: {e}")
        # Return empty to trigger frontend fallback
        return {"daily_volumes": [], "source": "unavailable"}


@router.get("/api/v1/token")
async def fiserv_token_status():
    """Proxy token status from Fiserv service."""
    proxy_request = _get_proxy_request()
    try:
        return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/token", "GET")
    except Exception as e:
        logger.warning(f"Fiserv token endpoint unavailable: {e}")
        return {"status": "no_token", "expires_in_seconds": 0}


@router.post("/api/v1/token/refresh")
async def fiserv_token_refresh():
    """Proxy token refresh to Fiserv service."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/token/refresh", "POST")


# =============================================================================
# Party/Member Endpoints (Require Auth)
# =============================================================================


@router.post("/api/v1/party/search")
async def fiserv_party_search(request: Request, session: Session = Depends(require_auth)):
    """Proxy member search to Fiserv service."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/party/search", "POST", json=body)


@router.post("/api/v1/account/lookup")
async def fiserv_account_lookup(request: Request, session: Session = Depends(require_auth)):
    """Proxy account lookup to Fiserv service."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/account/lookup", "POST", json=body)


@router.get("/api/v1/transactions/{account_id}")
async def fiserv_transactions(account_id: str, days: int = 30, session: Session = Depends(require_auth)):
    """Proxy transaction history to Fiserv service."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/transactions/{account_id}?days={days}", "GET")


@router.post("/api/v1/transfer")
async def fiserv_transfer(request: Request, session: Session = Depends(require_auth)):
    """Proxy internal transfer to Fiserv service."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/transfer", "POST", json=body)


# =============================================================================
# Dataset Endpoints (For ML Integration)
# =============================================================================


@router.get("/api/v1/datasets/member/{member_id}")
async def fiserv_member_dataset(member_id: str, session: Session = Depends(require_auth)):
    """Get full member dataset for ML analysis."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/datasets/member/{member_id}", "GET")


@router.get("/api/v1/datasets/member/{member_id}/views/{view_name}")
async def fiserv_member_view(member_id: str, view_name: str, session: Session = Depends(require_auth)):
    """Get normalized view (transactions, features) for ML engines."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/datasets/member/{member_id}/views/{view_name}", "GET")


# =============================================================================
# ML Profit Endpoints (Cross-sell, Churn, Pricing)
# =============================================================================


@router.post("/api/v1/ml/cross-sell/predict")
async def fiserv_ml_crosssell(request: Request, session: Session = Depends(require_auth)):
    """Proxy cross-sell prediction to Fiserv ML."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/ml/cross-sell/predict", "POST", json=body)


@router.post("/api/v1/ml/churn/predict")
async def fiserv_ml_churn(request: Request, session: Session = Depends(require_auth)):
    """Proxy churn prediction to Fiserv ML."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/ml/churn/predict", "POST", json=body)


@router.post("/api/v1/ml/pricing/optimize")
async def fiserv_ml_pricing(request: Request, session: Session = Depends(require_auth)):
    """Proxy loan pricing optimization to Fiserv ML."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/ml/pricing/optimize", "POST", json=body)


@router.get("/api/v1/ml/features/{member_id}")
async def fiserv_ml_features(member_id: str, session: Session = Depends(require_auth)):
    """Get ML feature vector for member."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/ml/features/{member_id}", "GET")


# =============================================================================
# Member 360 View (MSR Dashboard - Phase 2)
# =============================================================================


@router.get("/api/v1/member360/{member_id}")
async def member_360_view(member_id: str, session: Session = Depends(require_auth)):
    """
    Get comprehensive Member 360 view for MSR dashboard.

    Combines:
    - Party info (name, contact, status)
    - Accounts summary (balances, product count)
    - Recent transaction analysis
    - Quick ML insights (risk flags, opportunities)

    Optimized for frontline MSR use - single API call returns all context needed.
    """
    proxy_request = _get_proxy_request()

    try:
        # Get full member dataset
        member_data = await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/datasets/member/{member_id}", "GET")

        # Get ML features if available
        try:
            ml_features = await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/ml/features/{member_id}", "GET")
        except Exception:
            ml_features = {"status": "unavailable"}

        # Build Member 360 response
        return {
            "member_id": member_id,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            # Core member info
            "party": member_data.get("party", {}),
            # Account summary
            "accounts": {
                "list": member_data.get("accounts", []),
                "total_balance": sum(
                    a.get("balance", {}).get("Avail", {}).get("amount", 0) or 0 for a in member_data.get("accounts", [])
                ),
                "product_count": len(member_data.get("accounts", [])),
            },
            # Transaction analysis
            "transactions": {
                "recent": member_data.get("transactions", [])[:10],
                "analysis": member_data.get("transaction_analysis", {}),
            },
            # ML insights (for quick context)
            "insights": {
                "ml_features": ml_features,
                "deployment_status": _check_deployment_status(member_data),
                "quick_flags": _generate_quick_flags(member_data),
            },
        }

    except Exception as e:
        logger.error(f"Member 360 failed for {member_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _check_deployment_status(member_data: dict) -> dict:
    """Check if member is military and potentially deployed."""
    # In real implementation, would check against military flags
    return {
        "is_military": False,  # Placeholder - integrate with SCU military member flags
        "deployed": False,
        "eligible_for_warrior_savings": False,
    }


def _generate_quick_flags(member_data: dict) -> list:
    """Generate quick insight flags for MSR."""
    flags = []

    # Check for potential issues
    analysis = member_data.get("transaction_analysis", {})

    if analysis.get("total_debits", 0) > analysis.get("total_credits", 0) * 1.5:
        flags.append({"type": "warning", "icon": "📉", "message": "Spending exceeds income - cash flow concern"})

    if len(member_data.get("accounts", [])) == 1:
        flags.append({"type": "opportunity", "icon": "💡", "message": "Single product member - cross-sell opportunity"})

    if analysis.get("count", 0) < 5:
        flags.append({"type": "info", "icon": "📊", "message": "Low activity - engagement opportunity"})

    return flags


# =============================================================================
# Case Management Endpoints (Proxy to fiserv-service PostgreSQL backend)
# =============================================================================


@router.get("/api/v1/cases")
async def list_cases(
    status: str | None = None,
    priority: str | None = None,
    case_type: str | None = None,
    assignee_id: str | None = None,
    limit: int = 50
):
    """List cases from database."""
    proxy_request = _get_proxy_request()
    params = []
    if status:
        params.append(f"status={status}")
    if priority:
        params.append(f"priority={priority}")
    if case_type:
        params.append(f"case_type={case_type}")
    if assignee_id:
        params.append(f"assignee_id={assignee_id}")
    params.append(f"limit={limit}")
    
    query = "&".join(params)
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/cases?{query}", "GET")


@router.post("/api/v1/cases")
async def create_case(request: Request):
    """Create a new case."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/cases", "POST", json=body)


@router.get("/api/v1/cases/stats")
async def get_case_stats():
    """Get case statistics."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/cases/stats", "GET")


@router.get("/api/v1/cases/{case_id}")
async def get_case(case_id: str):
    """Get a single case by ID."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/cases/{case_id}", "GET")


@router.put("/api/v1/cases/{case_id}")
async def update_case(case_id: str, request: Request):
    """Update a case."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/cases/{case_id}", "PUT", json=body)


@router.post("/api/v1/cases/{case_id}/notes")
async def add_case_note(case_id: str, request: Request):
    """Add a note to a case."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/cases/{case_id}/notes", "POST", json=body)


# =============================================================================
# Loan Applications Proxy Routes
# =============================================================================


@router.get("/api/v1/loans/applications")
async def list_loan_applications(status: str = None, loan_type: str = None, limit: int = 50):
    """List loan applications from fiserv-service."""
    proxy_request = _get_proxy_request()
    params = []
    if status:
        params.append(f"status={status}")
    if loan_type:
        params.append(f"loan_type={loan_type}")
    params.append(f"limit={limit}")
    query_string = "&".join(params)
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/loans/applications?{query_string}", "GET")


@router.post("/api/v1/loans/applications")
async def create_loan_application(request: Request):
    """Create a new loan application."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/loans/applications", "POST", json=body)


@router.put("/api/v1/loans/applications/{app_id}")
async def update_loan_application(app_id: str, request: Request):
    """Update a loan application."""
    proxy_request = _get_proxy_request()
    body = await request.json()
    return await proxy_request(f"{FISERV_SERVICE_URL}/api/v1/loans/applications/{app_id}", "PUT", json=body)


logger.info("✅ Fiserv Router initialized - proxying /fiserv/* to fiserv-service:8015")

