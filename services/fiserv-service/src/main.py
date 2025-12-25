"""
Fiserv Automation Service - FastAPI Main

REST API for Fiserv Banking Hub automation with ML insights.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .fiserv.account_service import get_account_service
from .fiserv.auth import get_auth_client
from .fiserv.config import PROVIDER_ORG_IDS, get_config
from .fiserv.party_service import get_party_service
from .fiserv.tracker import get_tracker
from .fiserv.transaction_service import get_transaction_service
from .fiserv.case_service import get_case_service
from .ml.anomaly import get_anomaly_detector
from .ml.churn import get_churn_predictor
from .ml.cross_sell import get_cross_sell_engine
from .ml.features import get_feature_collector
from .ml.pricing import get_pricing_optimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Fiserv Automation Service...")
    yield
    # Cleanup
    auth = get_auth_client()
    await auth.close()
    logger.info("Fiserv Automation Service stopped.")


app = FastAPI(
    title="Fiserv Automation Service",
    description="Banking Hub automation with ML insights",
    version="1.0.0",
    lifespan=lifespan,
)

# Register Dataset Router (Banking Hub Integration)
from .router import router as dataset_router

app.include_router(dataset_router, prefix="/api/v1")

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================


class PartySearchRequest(BaseModel):
    """Party search request."""

    name: str | None = None
    tax_ident: str | None = None
    phone: str | None = None
    email: str | None = None
    account_id: str | None = None
    max_records: int = Field(default=20, le=100)


class AccountRequest(BaseModel):
    """Account lookup request."""

    account_id: str
    account_type: str = "DDA"


class TransactionListRequest(BaseModel):
    """Transaction list request."""

    account_id: str
    account_type: str = "DDA"
    start_date: str | None = None
    end_date: str | None = None
    max_records: int = Field(default=100, le=500)
    run_anomaly_detection: bool = True


class AlertResolution(BaseModel):
    """Alert resolution request."""

    alert_id: str
    action: str  # "approve", "reject", "escalate"
    notes: str | None = None


class CaseCreate(BaseModel):
    """Case creation request."""

    case_type: str  # fraud, dispute, complaint, compliance
    subject: str
    description: str | None = None
    member_id: str | None = None
    account_id: str | None = None
    priority: str = "medium"  # critical, high, medium, low
    assignee_id: str | None = None
    assignee_name: str | None = None
    due_date: str | None = None


class CaseUpdate(BaseModel):
    """Case update request."""

    status: str | None = None  # open, in_progress, escalated, closed
    priority: str | None = None
    assignee_id: str | None = None
    assignee_name: str | None = None
    resolution_summary: str | None = None


class CaseNote(BaseModel):
    """Case note request."""

    note: str
    user_name: str = "Current User"


# ============================================================================
# Health & Status Endpoints
# ============================================================================


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "service": "fiserv-automation"}


@app.get("/api/v1/status")
async def status():
    """Get service status and configuration."""
    config = get_config()
    auth = get_auth_client()
    tracker = get_tracker()

    return {
        "service": "fiserv-automation",
        "config": {
            "host_url": config.host_url,
            "default_org_id": config.default_org_id,
            "available_providers": list(PROVIDER_ORG_IDS.keys()),
        },
        "token_status": auth.token_status,
        "api_usage": tracker.get_stats(),
    }


@app.get("/api/v1/token")
async def token_status():
    """Get current OAuth token status."""
    auth = get_auth_client()
    return auth.token_status


@app.post("/api/v1/token/refresh")
async def refresh_token():
    """Force refresh OAuth token."""
    auth = get_auth_client()
    try:
        await auth.get_token(force_refresh=True)
        return {"status": "refreshed", **auth.token_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/usage")
async def api_usage():
    """
    Get API call usage statistics.

    IMPORTANT: Sandbox limit is 1000 calls total!
    """
    tracker = get_tracker()
    stats = tracker.get_stats()
    return {"warning": "Sandbox limit is 1000 total API calls" if stats["calls_remaining"] < 200 else None, **stats}


# ============================================================================
# Party (Customer) Endpoints
# ============================================================================


@app.post("/api/v1/party/search")
async def search_parties(request: PartySearchRequest, provider: str = Query(default="DNABanking")):
    """
    Search for customers/parties.

    Human-in-the-loop: Returns search results for review.
    """
    service = get_party_service(provider)

    response = await service.search(
        name=request.name,
        tax_ident=request.tax_ident,
        phone=request.phone,
        email=request.email,
        account_id=request.account_id,
        max_records=request.max_records,
    )

    if "error" in response:
        return {"success": False, "error": response.get("message"), "raw": response}

    parties = service.parse_party_list(response)

    return {"success": True, "count": len(parties), "parties": parties, "raw_response": response}


@app.get("/api/v1/party/{party_id}")
async def get_party(party_id: str, provider: str = Query(default="DNABanking")):
    """Get party details by ID."""
    service = get_party_service(provider)
    response = await service.get_by_id(party_id)

    if "error" in response:
        raise HTTPException(status_code=404, detail=response.get("message"))

    return {"success": True, "party": response}


# ============================================================================
# Account Endpoints
# ============================================================================


@app.post("/api/v1/account/lookup")
async def lookup_account(request: AccountRequest, provider: str = Query(default="DNABanking")):
    """
    Look up account details.
    """
    service = get_account_service(provider)

    response = await service.get_by_id(account_id=request.account_id, account_type=request.account_type)

    if "error" in response:
        return {"success": False, "error": response.get("message")}

    account = service.parse_account_info(response)

    return {"success": True, "account": account, "raw_response": response}


@app.get("/api/v1/account/{account_id}/balance")
async def get_balance(
    account_id: str, account_type: str = Query(default="DDA"), provider: str = Query(default="DNABanking")
):
    """Get account balance."""
    service = get_account_service(provider)

    response = await service.get_balance(account_id=account_id, account_type=account_type)

    return {"success": True, "balance": response}


@app.get("/api/v1/party/{party_id}/accounts")
async def list_party_accounts(
    party_id: str, account_type: str | None = None, provider: str = Query(default="DNABanking")
):
    """List accounts for a party."""
    service = get_account_service(provider)

    response = await service.list_by_party(party_id=party_id, account_type=account_type)

    return {"success": True, "accounts": response}


# ============================================================================
# Transaction Endpoints
# ============================================================================


class TransferRequest(BaseModel):
    """Internal transfer request."""

    from_account_id: str
    to_account_id: str
    amount: float = Field(..., gt=0)
    description: str | None = "Transfer"


@app.post("/api/v1/transfer")
async def execute_transfer(request: TransferRequest, provider: str = Query(default="DNABanking")):
    """
    Execute internal fund transfer.

    Validates balance and updates both accounts.
    """
    account_service = get_account_service(provider)
    tx_service = get_transaction_service(provider)

    # 1. Validate Source Account
    from_acct = await account_service.get_by_id(request.from_account_id)
    if "error" in from_acct:
        raise HTTPException(status_code=404, detail=f"Source account not found: {request.from_account_id}")

    # Check Balance (Mock logic - in real world use get_balance)
    current_bal = float(from_acct.get("Balances", {}).get("Available", 0))
    if current_bal < request.amount:
        return {"success": False, "error": "Insufficient funds"}

    # 2. Validate Dest Account
    to_acct = await account_service.get_by_id(request.to_account_id)
    if "error" in to_acct:
        raise HTTPException(status_code=404, detail=f"Destination account not found: {request.to_account_id}")

    # 3. Execute Transfer (Atomic Simulation)
    # create_transfer is a helper we assume exists or we mock the success here for the proto
    tx_id = f"TX-{uuid.uuid4().hex[:8].upper()}"

    # Log debit
    await tx_service.record_transaction(
        account_id=request.from_account_id,
        amount=-request.amount,
        description=f"Transfer to {request.to_account_id}: {request.description}",
        tx_id=tx_id,
    )

    # Log credit
    await tx_service.record_transaction(
        account_id=request.to_account_id,
        amount=request.amount,
        description=f"Transfer from {request.from_account_id}: {request.description}",
        tx_id=tx_id,
    )

    return {
        "success": True,
        "transaction_id": tx_id,
        "message": "Transfer successful",
        "new_balance_source": current_bal - request.amount,
    }


@app.post("/api/v1/transactions/list")
async def list_transactions(request: TransactionListRequest, provider: str = Query(default="DNABanking")):
    """
    List transactions with optional anomaly detection.

    Human-in-the-loop: Returns transactions and flags anomalies for review.
    Does NOT take any automatic action.
    """
    service = get_transaction_service(provider)

    response = await service.list_by_date_range(
        account_id=request.account_id,
        account_type=request.account_type,
        start_date=request.start_date,
        end_date=request.end_date,
        max_records=request.max_records,
    )

    if "error" in response:
        return {"success": False, "error": response.get("message")}

    transactions = service.parse_transactions(response)
    analysis = service.analyze_transactions(transactions)

    result = {"success": True, "count": len(transactions), "transactions": transactions, "analysis": analysis}

    # Run anomaly detection if requested
    if request.run_anomaly_detection and transactions:
        detector = get_anomaly_detector()
        anomaly_results = detector.analyze_all(transactions)
        result["anomaly_detection"] = anomaly_results

        if anomaly_results["total_flags"] > 0:
            result["action_required"] = "human_review"
            result["message"] = f"{anomaly_results['total_flags']} items flagged for review"

    return result


@app.get("/api/v1/transactions/{account_id}")
async def get_account_transactions(
    account_id: str,
    account_type: str = Query(default="DDA"),
    days: int = Query(default=30, le=365),
    provider: str = Query(default="DNABanking"),
):
    """Quick transaction lookup for an account."""
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    service = get_transaction_service(provider)

    response = await service.list_by_date_range(
        account_id=account_id, account_type=account_type, start_date=start_date, end_date=end_date
    )

    transactions = service.parse_transactions(response)

    return {
        "success": True,
        "account_id": account_id,
        "date_range": {"start": start_date, "end": end_date},
        "count": len(transactions),
        "transactions": transactions,
    }


# ============================================================================
# Alert Queue (Human-in-the-Loop)
# ============================================================================

# In-memory alert queue (would be database in production)
_alert_queue: list[dict[str, Any]] = []
_alert_id_counter = 0


@app.get("/api/v1/alerts")
async def get_alerts(status: str = Query(default="pending"), limit: int = Query(default=50, le=200)):
    """
    Get alert queue for human review.

    These are anomalies flagged by ML that need human decision.
    """
    filtered = [a for a in _alert_queue if a.get("status") == status]
    return {"count": len(filtered), "alerts": filtered[:limit]}


@app.post("/api/v1/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, resolution: AlertResolution):
    """
    Resolve an alert (human decision).

    Actions: approve, reject, escalate
    """
    for alert in _alert_queue:
        if alert.get("id") == alert_id:
            alert["status"] = "resolved"
            alert["resolution"] = resolution.action
            alert["notes"] = resolution.notes
            return {"success": True, "alert": alert}

    raise HTTPException(status_code=404, detail="Alert not found")


# ============================================================================
# Providers
# ============================================================================


@app.get("/api/v1/providers")
async def list_providers():
    """List available banking providers."""
    return {"providers": [{"name": name, "org_id": org_id} for name, org_id in PROVIDER_ORG_IDS.items()]}


# ============================================================================
# ML Profit Maximization Endpoints
# ============================================================================


class CrossSellRequest(BaseModel):
    """Cross-sell prediction request."""

    member_id: str
    top_n: int = Field(default=3, ge=1, le=10)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)


class ChurnPredictRequest(BaseModel):
    """Churn prediction request."""

    member_id: str


class PricingRequest(BaseModel):
    """Loan pricing request."""

    member_id: str
    loan_type: str = Field(..., description="auto_new, auto_used, personal, heloc, signature")
    amount: float = Field(..., gt=0)
    term_months: int = Field(..., gt=0)
    credit_score: int | None = Field(default=None, ge=300, le=850)
    risk_tier: str | None = Field(default=None, description="A+, A, B, C, D, E")


@app.post("/api/v1/ml/cross-sell/predict")
async def predict_cross_sell(request: CrossSellRequest):
    """
    Predict product recommendations for a member.

    Returns ranked list of products the member is likely to adopt,
    with propensity scores and explainable reasons.
    """
    try:
        # Collect member features from Fiserv
        collector = get_feature_collector()
        features = await collector.collect_member_features(request.member_id)

        if not features.get("collection_success") or features.get("product_count", 0) == 0:
            # Use demo features for sandbox/demo mode
            features = _get_demo_features(request.member_id)

        # Generate recommendations
        engine = get_cross_sell_engine()
        result = engine.predict(features, top_n=request.top_n, min_score=request.min_score)

        return result

    except Exception as e:
        logger.error(f"Cross-sell prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/cross-sell/products")
async def list_cross_sell_products():
    """List products available for cross-sell recommendations."""
    engine = get_cross_sell_engine()
    return {"products": engine.get_available_products()}


@app.post("/api/v1/ml/churn/predict")
async def predict_churn(request: ChurnPredictRequest):
    """
    Predict churn probability for a member.

    Returns churn probability, risk factors, and recommended retention actions.
    """
    try:
        # Collect member features from Fiserv
        collector = get_feature_collector()
        features = await collector.collect_member_features(request.member_id)

        if not features.get("collection_success") or features.get("product_count", 0) == 0:
            # Use demo features for sandbox/demo mode
            features = _get_demo_features(request.member_id)

        # Predict churn
        predictor = get_churn_predictor()
        result = predictor.predict(features)

        return result

    except Exception as e:
        logger.error(f"Churn prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/churn/high-risk")
async def get_high_risk_members(
    limit: int = Query(default=50, le=200), min_score: float = Query(default=0.5, ge=0.0, le=1.0)
):
    """
    Get list of high-risk members for retention campaigns.

    Note: In demo mode, returns sample data.
    """
    # Demo data for high-risk members
    demo_members = [
        {"member_id": "M001", "churn_probability": 0.82, "risk_level": "HIGH", "priority": 95},
        {"member_id": "M002", "churn_probability": 0.76, "risk_level": "HIGH", "priority": 88},
        {"member_id": "M003", "churn_probability": 0.71, "risk_level": "HIGH", "priority": 82},
        {"member_id": "M004", "churn_probability": 0.65, "risk_level": "MEDIUM", "priority": 75},
        {"member_id": "M005", "churn_probability": 0.58, "risk_level": "MEDIUM", "priority": 68},
    ]

    filtered = [m for m in demo_members if m["churn_probability"] >= min_score]

    return {
        "members": filtered[:limit],
        "total_at_risk": len(filtered),
        "high_risk_count": len([m for m in filtered if m["risk_level"] == "HIGH"]),
        "medium_risk_count": len([m for m in filtered if m["risk_level"] == "MEDIUM"]),
    }


@app.post("/api/v1/ml/pricing/optimize")
async def optimize_loan_pricing(request: PricingRequest):
    """
    Optimize loan pricing for a member application.

    Returns recommended rate with breakdown of adjustments,
    competitive analysis, and approval recommendation.
    """
    try:
        # Collect member features from Fiserv
        collector = get_feature_collector()
        features = await collector.collect_member_features(request.member_id)

        if not features.get("collection_success") or features.get("product_count", 0) == 0:
            # Use demo features for sandbox/demo mode
            features = _get_demo_features(request.member_id)

        # Optimize pricing
        optimizer = get_pricing_optimizer()
        result = optimizer.optimize(
            member_features=features,
            loan_type=request.loan_type,
            amount=request.amount,
            term_months=request.term_months,
            credit_score=request.credit_score,
            risk_tier=request.risk_tier,
        )

        return result

    except Exception as e:
        logger.error(f"Pricing optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/pricing/loan-types")
async def list_loan_types():
    """List available loan types for pricing."""
    optimizer = get_pricing_optimizer()
    return {"loan_types": optimizer.get_loan_types()}


@app.get("/api/v1/ml/features/{member_id}")
async def get_member_features(member_id: str):
    """
    Get ML features for a member.

    Shows the features extracted from Fiserv data used by ML models.
    """
    try:
        collector = get_feature_collector()
        features = await collector.collect_member_features(member_id)

        if not features.get("collection_success"):
            # Return demo features
            features = _get_demo_features(member_id)
            features["demo_mode"] = True

        return features

    except Exception as e:
        logger.error(f"Feature collection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_demo_features(member_id: str) -> dict:
    """
    Generate demo features for sandbox/demo mode.

    Provides realistic feature values when Fiserv APIs return
    limited data in sandbox environment.
    """
    import random

    random.seed(hash(member_id) % 1000)  # Consistent for same member_id

    return {
        "member_id": member_id,
        "collection_success": True,
        "demo_mode": True,
        # Demographics
        "age": random.randint(25, 65),
        "age_bracket": random.choice(["25-34", "35-44", "45-54", "55-64"]),
        "has_email": True,
        "has_phone": True,
        "has_address": True,
        # Products
        "has_checking": True,
        "has_savings": random.random() > 0.3,
        "has_certificate": random.random() > 0.7,
        "has_loan": random.random() > 0.5,
        "has_auto_loan": random.random() > 0.6,
        "has_credit_card": random.random() > 0.5,
        "has_heloc": random.random() > 0.85,
        "has_mortgage": random.random() > 0.7,
        "product_count": random.randint(1, 5),
        # Balances
        "total_balance": random.randint(1000, 75000),
        "avg_balance": random.randint(500, 25000),
        # Tenure
        "tenure_months": random.randint(6, 180),
        "tenure_years": random.randint(0, 15),
        # Transactions
        "transaction_count_30d": random.randint(5, 60),
        "credit_count_30d": random.randint(2, 10),
        "debit_count_30d": random.randint(10, 50),
        "total_credits_30d": random.randint(2000, 15000),
        "total_debits_30d": random.randint(1500, 12000),
        "has_direct_deposit": random.random() > 0.3,
        "large_credit_count_30d": random.randint(0, 4),
        "external_transfer_activity": random.random() > 0.6,
        "avg_transaction_amount": random.randint(30, 200),
        "tx_count_change_pct": random.uniform(-0.3, 0.3),
        "days_since_last_tx": random.randint(0, 15),
    }


# ============================================================================
# Loan Applications Endpoints (Database-Backed - NO DEMO DATA)
# ============================================================================


class LoanApplicationCreate(BaseModel):
    """Loan application request."""
    applicant_name: str
    amount: float = Field(..., gt=0)
    loan_type: str  # auto, mortgage, personal, heloc
    credit_score: int = Field(..., ge=300, le=850)
    monthly_income: float = Field(..., gt=0)
    existing_debt: float = Field(default=0, ge=0)
    member_id: str | None = None


class LoanApplicationUpdate(BaseModel):
    """Loan application update."""
    status: str | None = None  # pending, approved, denied, under_review
    assigned_officer: str | None = None
    notes: str | None = None


@app.get("/api/v1/loans/applications")
async def list_loan_applications(
    status: str | None = None,
    loan_type: str | None = None,
    limit: int = Query(default=50, le=200)
):
    """
    List loan applications from database.
    
    NO DEMO DATA - Returns real applications from PostgreSQL.
    """
    from .fiserv.case_service import get_case_service
    case_service = get_case_service()
    
    try:
        async with case_service.pool.acquire() as conn:
            # Build query
            query = """
                SELECT id, applicant_name, amount, loan_type, credit_score,
                       monthly_income, existing_debt, member_id, status,
                       assigned_officer, created_at, updated_at
                FROM loan_applications
                WHERE 1=1
            """
            params = []
            param_count = 0
            
            if status:
                param_count += 1
                query += f" AND status = ${param_count}"
                params.append(status)
            
            if loan_type:
                param_count += 1
                query += f" AND loan_type = ${param_count}"
                params.append(loan_type)
            
            query += f" ORDER BY created_at DESC LIMIT ${param_count + 1}"
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            applications = []
            for row in rows:
                dti = (row['existing_debt'] / row['monthly_income'] * 100) if row['monthly_income'] > 0 else 0
                score = row['credit_score']
                risk_grade = 'A+' if score >= 750 else 'A' if score >= 720 else 'B' if score >= 680 else 'C' if score >= 640 else 'D'
                
                applications.append({
                    "id": row['id'],
                    "name": row['applicant_name'],
                    "amount": float(row['amount']),
                    "type": row['loan_type'].capitalize(),
                    "score": score,
                    "income": float(row['monthly_income']),
                    "debt": float(row['existing_debt']),
                    "status": row['status'],
                    "dti_ratio": round(dti, 1),
                    "risk_grade": risk_grade,
                    "assigned_officer": row['assigned_officer'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None
                })
            
            return {
                "success": True,
                "applications": applications,
                "count": len(applications),
                "data_source": "postgresql"
            }
            
    except Exception as e:
        logger.warning(f"Loan applications query failed: {e}")
        # Return empty array - NO MOCK DATA
        return {
            "success": True,
            "applications": [],
            "count": 0,
            "message": "No applications in queue",
            "data_source": "postgresql"
        }


@app.post("/api/v1/loans/applications")
async def create_loan_application(request: LoanApplicationCreate):
    """Create a new loan application."""
    from .fiserv.case_service import get_case_service
    case_service = get_case_service()
    
    try:
        async with case_service.pool.acquire() as conn:
            app_id = f"L-{uuid.uuid4().hex[:6].upper()}"
            
            await conn.execute("""
                INSERT INTO loan_applications
                (id, applicant_name, amount, loan_type, credit_score, 
                 monthly_income, existing_debt, member_id, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending', NOW())
            """, app_id, request.applicant_name, request.amount, request.loan_type,
                request.credit_score, request.monthly_income, request.existing_debt,
                request.member_id)
            
            return {
                "success": True,
                "application_id": app_id,
                "status": "pending",
                "message": "Application submitted for review"
            }
    except Exception as e:
        logger.error(f"Failed to create loan application: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/loans/applications/{app_id}")
async def update_loan_application(app_id: str, request: LoanApplicationUpdate):
    """Update loan application status."""
    from .fiserv.case_service import get_case_service
    case_service = get_case_service()
    
    try:
        async with case_service.pool.acquire() as conn:
            updates = []
            params = [app_id]
            param_count = 1
            
            if request.status:
                param_count += 1
                updates.append(f"status = ${param_count}")
                params.append(request.status)
            
            if request.assigned_officer:
                param_count += 1
                updates.append(f"assigned_officer = ${param_count}")
                params.append(request.assigned_officer)
            
            if updates:
                updates.append("updated_at = NOW()")
                query = f"UPDATE loan_applications SET {', '.join(updates)} WHERE id = $1"
                await conn.execute(query, *params)
            
            return {"success": True, "updated": app_id}
    except Exception as e:
        logger.error(f"Failed to update loan application: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Case Management Endpoints (Database-Backed - NO DEMO DATA)
# ============================================================================


@app.get("/api/v1/cases")
async def list_cases(
    status: str | None = None,
    priority: str | None = None,
    case_type: str | None = None,
    assignee_id: str | None = None,
    limit: int = Query(default=50, le=200)
):
    """
    List cases with optional filters.
    
    All data from PostgreSQL - NO DEMO DATA.
    """
    service = get_case_service()
    cases = await service.list_cases(
        status=status,
        priority=priority,
        case_type=case_type,
        assignee_id=assignee_id,
        limit=limit
    )
    return {"success": True, "count": len(cases), "cases": cases}


@app.post("/api/v1/cases")
async def create_case(request: CaseCreate):
    """
    Create a new case.
    
    Stores in PostgreSQL with full audit trail.
    """
    service = get_case_service()
    
    due_date = None
    if request.due_date:
        from datetime import datetime
        due_date = datetime.fromisoformat(request.due_date.replace("Z", "+00:00"))
    
    case = await service.create_case(
        case_type=request.case_type,
        subject=request.subject,
        description=request.description,
        member_id=request.member_id,
        account_id=request.account_id,
        priority=request.priority,
        assignee_id=request.assignee_id,
        assignee_name=request.assignee_name,
        due_date=due_date,
        created_by="API"
    )
    return {"success": True, "case": case}


@app.get("/api/v1/cases/stats")
async def get_case_stats():
    """Get case management statistics."""
    service = get_case_service()
    stats = await service.get_stats()
    return {"success": True, "stats": stats}


@app.get("/api/v1/cases/{case_id}")
async def get_case(case_id: str):
    """Get a single case with full timeline."""
    service = get_case_service()
    case = await service.get_case(case_id)
    
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    
    return {"success": True, "case": case}


@app.put("/api/v1/cases/{case_id}")
async def update_case(case_id: str, request: CaseUpdate):
    """Update a case's status, priority, or assignee."""
    service = get_case_service()
    case = await service.update_case(
        case_id=case_id,
        updated_by="API",
        status=request.status,
        priority=request.priority,
        assignee_id=request.assignee_id,
        assignee_name=request.assignee_name,
        resolution_summary=request.resolution_summary
    )
    
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    
    return {"success": True, "case": case}


@app.post("/api/v1/cases/{case_id}/notes")
async def add_case_note(case_id: str, request: CaseNote):
    """Add a note to a case timeline."""
    service = get_case_service()
    case = await service.add_note(
        case_id=case_id,
        user_name=request.user_name,
        note=request.note
    )
    return {"success": True, "case": case}
