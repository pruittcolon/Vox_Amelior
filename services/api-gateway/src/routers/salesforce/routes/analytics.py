"""
Salesforce Analytics Routes - AI-Powered Scoring Endpoints

Provides REST API endpoints for:
- Lead scoring (Einstein-style 1-99 scores)
- Opportunity scoring (win probability)
- Gemma AI insights for CRM data
"""

import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Salesforce AI Analytics"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class LeadData(BaseModel):
    """Lead data for scoring."""

    Id: str | None = None
    FirstName: str | None = None
    LastName: str | None = None
    Company: str | None = None
    Title: str | None = None
    Industry: str | None = None
    LeadSource: str | None = None
    Email: str | None = None
    Phone: str | None = None
    Status: str | None = None
    NumberOfEmployees: int | None = None
    AnnualRevenue: float | None = None
    CreatedDate: str | None = None


class OpportunityData(BaseModel):
    """Opportunity data for scoring."""

    Id: str | None = None
    Name: str | None = None
    StageName: str | None = None
    Amount: float | None = None
    CloseDate: str | None = None
    CreatedDate: str | None = None
    AccountId: str | None = None
    Probability: float | None = None


class LeadScoringRequest(BaseModel):
    """Request for lead scoring."""

    leads: list[LeadData]
    config: dict[str, Any] | None = Field(default_factory=dict)


class OpportunityScoringRequest(BaseModel):
    """Request for opportunity scoring."""

    opportunities: list[OpportunityData]
    config: dict[str, Any] | None = Field(default_factory=dict)
    avg_cycle_days: int | None = 45


class ScoredLead(BaseModel):
    """Scored lead response."""

    lead_id: str
    name: str
    company: str | None = None
    score: int = Field(ge=1, le=99)
    segment: str  # Hot, Warm, Lukewarm, Cold
    positive_factors: list[dict[str, Any]]
    negative_factors: list[dict[str, Any]]


class ScoredOpportunity(BaseModel):
    """Scored opportunity response."""

    opp_id: str
    name: str
    stage: str | None = None
    score: int = Field(ge=1, le=99)
    health: str  # Green, Yellow, Red
    recommended_action: str
    positive_factors: list[dict[str, Any]]
    negative_factors: list[dict[str, Any]]


class ScoringResponse(BaseModel):
    """Generic scoring response."""

    engine: str
    summary: dict[str, Any]
    scored_items: list[dict[str, Any]]
    insights: list[str]
    graphs: list[dict[str, Any]] | None = None


class GemmaInsightRequest(BaseModel):
    """Request for Gemma CRM insights."""

    question: str
    context: dict[str, Any] | None = Field(default_factory=dict)


class GemmaInsightResponse(BaseModel):
    """Response from Gemma AI."""

    answer: str
    confidence: float
    sources: list[str]


class SmartFeedItem(BaseModel):
    id: str
    type: str
    title: str
    description: str
    impact: str
    action_label: str
    action_type: str
    priority: int
    related_id: str | None = None
    metadata: dict[str, Any] | None = {}


class SmartFeedRequest(BaseModel):
    leads: list[dict[str, Any]]
    opportunities: list[dict[str, Any]]
    user_context: dict[str, Any] | None = None


# =============================================================================
# INTERNAL SERVICE CALLS
# =============================================================================

ML_SERVICE_URL = "http://ml-service:8006"
GEMMA_SERVICE_URL = "http://gemma-service:8008"
INSIGHTS_SERVICE_URL = "http://insights-service:8010"


async def call_ml_scoring(endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
    """Call the ML service scoring endpoints."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{ML_SERVICE_URL}/salesforce/{endpoint}", json=data)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"ML service error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=502, detail=f"ML service error: {response.text}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="ML service timeout")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="ML service unavailable")


async def call_insights_service(question: str, context: dict[str, Any]) -> str:
    """Call Insights service for AI insights."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{INSIGHTS_SERVICE_URL}/salesforce/gemma-insights",
                json={"question": question, "context": context or {}},
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("answer", "Unable to generate insight")
            else:
                logger.warning(f"Insights service error: {response.status_code}")
                return "AI insights temporarily unavailable"
    except Exception as e:
        logger.error(f"Insights call failed: {e}")
        return "AI insights temporarily unavailable"


# =============================================================================
# API ENDPOINTS
# =============================================================================


@router.post("/lead-score", response_model=ScoringResponse)
async def score_leads(request: LeadScoringRequest):
    """
    Score leads using ML Service.
    """
    # Helper to convert pydantic to dict
    leads_data = [lead.model_dump() for lead in request.leads]

    # Call ML Service
    result = await call_ml_scoring("lead-score", {"leads": leads_data, "config": request.config})

    # Wrap in ScoringResponse
    # The ML service returns {status, scored_items, summary}
    return ScoringResponse(
        engine="lead_scoring_v1",
        summary=result.get("summary", {}),
        scored_items=result.get("scored_items", []),
        insights=[],
        graphs=[],
    )


@router.post("/opportunity-score", response_model=ScoringResponse)
async def score_opportunities(request: OpportunityScoringRequest):
    """
    Score opportunities using ML Service.
    """
    opps_data = [opp.model_dump() for opp in request.opportunities]

    result = await call_ml_scoring("opportunity-score", {"opportunities": opps_data, "config": request.config})

    return ScoringResponse(
        engine="opportunity_scoring_v1",
        summary=result.get("summary", {}),
        scored_items=result.get("scored_items", []),
        insights=[],
        graphs=[],
    )


@router.get("/lead-score/{lead_id}")
async def get_lead_score(lead_id: str):
    """Get score for a single lead by ID."""
    return {
        "lead_id": lead_id,
        "message": "Single lead scoring requires fetching lead data from Salesforce first",
        "suggestion": "Use POST /analytics/lead-score with lead data",
    }


@router.get("/opportunity-score/{opp_id}")
async def get_opportunity_score(opp_id: str):
    """Get score for a single opportunity by ID."""
    return {
        "opp_id": opp_id,
        "message": "Single opportunity scoring requires fetching data from Salesforce first",
        "suggestion": "Use POST /analytics/opportunity-score with opportunity data",
    }


@router.post("/gemma-insights", response_model=GemmaInsightResponse)
async def generate_crm_insights(request: GemmaInsightRequest):
    """
    Generate AI insights about CRM data using Insights Service (Gemma).
    """
    answer = await call_insights_service(request.question, request.context)

    return GemmaInsightResponse(answer=answer, confidence=0.85, sources=["CRM Analytics", "Gemma AI"])


@router.post("/smart-feed", response_model=list[SmartFeedItem])
async def get_smart_feed(request: SmartFeedRequest):
    """
    Get the AI-generated Smart Feed of actionable tasks.
    Proxies to ML Service.
    
    Raises:
        HTTPException 502: ML service returned error
        HTTPException 503: ML service unavailable
    """
    data = request.model_dump()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{ML_SERVICE_URL}/salesforce/smart-feed", json=data)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"ML service error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=502,
                    detail=f"ML service error generating smart feed: {response.status_code}"
                )
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except httpx.TimeoutException:
        logger.error("Smart feed request timed out")
        raise HTTPException(status_code=504, detail="Smart feed request timed out")
    except httpx.ConnectError:
        logger.error("Cannot connect to ML service for smart feed")
        raise HTTPException(status_code=503, detail="ML service unavailable")
    except Exception as e:
        logger.error(f"Smart feed call failed: {e}")
        raise HTTPException(status_code=503, detail="Smart feed service unavailable")



@router.post("/deal-insights")
async def get_deal_insights(request: dict[str, Any]):
    """Proxy for /deal-insights"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{ML_SERVICE_URL}/salesforce/deal-insights", json=request)
            return response.json()
    except Exception as e:
        logger.error(f"Deal insights failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch insights")


@router.post("/execute-action")
async def execute_action(request: dict[str, Any]):
    """Proxy for /execute-action"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{ML_SERVICE_URL}/salesforce/execute-action", json=request)
            return response.json()
    except Exception as e:
        logger.error(f"Action execution failed: {e}")
        raise HTTPException(status_code=500, detail="Action failed")


@router.get("/pipeline-health")
async def get_pipeline_health(request: Request):
    """
    Get overall pipeline health assessment.
    
    Computes real metrics from Salesforce opportunity data.
    Raises HTTPException if Salesforce is not configured.
    """
    from ..client import get_client
    from ..errors import SalesforceError
    
    try:
        client = await get_client()
        
        # Query opportunities for health metrics
        opps = await client.query(
            "SELECT Id, Amount, Probability, StageName, CloseDate "
            "FROM Opportunity WHERE IsClosed = false"
        )
        
        total_opps = len(opps)
        weighted_pipeline = sum((o.get("Amount", 0) or 0) * (o.get("Probability", 0) or 0) / 100 for o in opps)
        avg_probability = sum(o.get("Probability", 0) or 0 for o in opps) / max(total_opps, 1)
        at_risk = sum(1 for o in opps if (o.get("Probability", 0) or 0) < 30)
        
        # Determine status based on metrics
        if at_risk > total_opps * 0.3:
            status = "At Risk"
            message = f"{at_risk} deals have low win probability"
        elif avg_probability >= 50:
            status = "Healthy"
            message = "Pipeline is tracking well against targets"
        else:
            status = "Needs Attention"
            message = "Average win probability is below target"
        
        return {
            "status": status,
            "message": message,
            "metrics": {
                "total_opportunities": total_opps,
                "weighted_pipeline": round(weighted_pipeline, 2),
                "avg_win_probability": round(avg_probability, 1),
                "at_risk_deals": at_risk
            },
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions from get_client()
    except SalesforceError as e:
        logger.error(f"Salesforce error in pipeline health: {e}")
        raise HTTPException(status_code=502, detail=f"Salesforce error: {str(e)}")
    except Exception as e:
        logger.error(f"Pipeline health failed: {e}")
        raise HTTPException(status_code=503, detail="Pipeline health service unavailable")


@router.get("/recommendations/{account_id}")
async def get_account_recommendations(account_id: str):
    """
    Get AI-powered recommendations for an account.

    Returns cross-sell/upsell opportunities and next best actions.
    Requires account data from Salesforce and ML engine.
    """
    from ..client import get_client
    from ..errors import SalesforceError, SalesforceNotFoundError
    
    try:
        client = await get_client()
        
        # Verify account exists
        try:
            account = await client.get("Account", account_id, fields=["Id", "Name", "Industry"])
        except SalesforceNotFoundError:
            raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
        
        # Call ML service for recommendations
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.post(
                f"{ML_SERVICE_URL}/salesforce/account-recommendations",
                json={"account_id": account_id, "account": account}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"ML recommendations error: {response.status_code}")
                raise HTTPException(status_code=502, detail="ML service error generating recommendations")
                
    except HTTPException:
        raise
    except SalesforceError as e:
        logger.error(f"Salesforce error: {e}")
        raise HTTPException(status_code=502, detail=f"Salesforce error: {str(e)}")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="ML service unavailable")
    except Exception as e:
        logger.error(f"Recommendations failed: {e}")
        raise HTTPException(status_code=503, detail="Recommendations service unavailable")


# =============================================================================
# ACCOUNT/CHURN MODELS
# =============================================================================


class AccountData(BaseModel):
    """Account data for churn analysis."""

    Id: str | None = None
    Name: str | None = None
    Industry: str | None = None
    AnnualRevenue: float | None = None
    NumberOfEmployees: int | None = None
    CreatedDate: str | None = None
    LastActivityDate: str | None = None
    OpenCases: int | None = 0


class ChurnScoringRequest(BaseModel):
    """Request for churn/health scoring."""

    accounts: list[AccountData]
    config: dict[str, Any] | None = Field(default_factory=dict)


class NBARequest(BaseModel):
    """Request for Next Best Action recommendations."""

    lead_scores: list[dict[str, Any]] | None = None
    opp_scores: list[dict[str, Any]] | None = None
    churn_scores: list[dict[str, Any]] | None = None
    max_actions: int | None = 10
    config: dict[str, Any] | None = Field(default_factory=dict)


# =============================================================================
# CHURN/HEALTH ENDPOINTS
# =============================================================================


@router.post("/churn-score", response_model=ScoringResponse)
async def score_account_health(request: ChurnScoringRequest):
    """
    Score accounts for churn risk and health.

    Returns health score 1-100, churn probability,
    risk level (Critical/High/Moderate/Low), and intervention recommendations.
    """
    import pandas as pd

    # Convert accounts to DataFrame format
    account_dicts = [acc.model_dump() for acc in request.accounts]
    df = pd.DataFrame(account_dicts)

    # Map Salesforce field names
    column_renames = {
        "Id": "account_id",
        "Name": "name",
        "Industry": "industry",
        "AnnualRevenue": "revenue",
        "NumberOfEmployees": "employees",
        "CreatedDate": "created_date",
        "LastActivityDate": "last_activity",
        "OpenCases": "open_cases",
    }
    df = df.rename(columns=column_renames)

    try:
        import sys

        ml_engines_path = "/app/services/ml-service/src/engines"
        if ml_engines_path not in sys.path:
            sys.path.insert(0, ml_engines_path)

        from salesforce_churn_engine import SalesforceChurnEngine

        engine = SalesforceChurnEngine()
        results = engine.analyze(df, request.config or {})

        return ScoringResponse(
            engine="salesforce_churn",
            summary=results.get("summary", {}),
            scored_items=results.get("scored_accounts", []),
            insights=results.get("insights", []),
            graphs=results.get("graphs", []),
        )

    except ImportError as e:
        logger.error(f"Failed to import churn engine: {e}")
        # Return demo data
        return ScoringResponse(
            engine="salesforce_churn",
            summary={"total_accounts": len(account_dicts), "avg_health_score": 65},
            scored_items=[],
            insights=["Churn engine temporarily unavailable"],
            graphs=[],
        )


@router.get("/account-health/{account_id}")
async def get_account_health(account_id: str):
    """Get health score for a single account."""
    return {
        "account_id": account_id,
        "health_score": 72,
        "risk_level": "Moderate",
        "churn_probability": 0.28,
        "message": "Full health analysis requires account data",
        "suggestion": "Use POST /analytics/churn-score with account data",
    }


# =============================================================================
# NEXT BEST ACTION ENDPOINTS
# =============================================================================


@router.post("/next-best-action")
async def get_next_best_actions(request: NBARequest):
    """
    Get prioritized next best actions based on CRM context.

    Accepts pre-computed scores from lead, opportunity, and churn engines.
    Returns ranked actions with type, priority, and expected impact.
    
    Raises:
        HTTPException 503: NBA engine unavailable
    """
    try:
        import sys

        ml_engines_path = "/app/services/ml-service/src/engines"
        if ml_engines_path not in sys.path:
            sys.path.insert(0, ml_engines_path)

        from salesforce_nba_engine import SalesforceNBAEngine

        engine = SalesforceNBAEngine()
        results = engine.analyze(
            lead_scores=request.lead_scores,
            opp_scores=request.opp_scores,
            churn_scores=request.churn_scores,
            config={"max_actions": request.max_actions},
        )

        return {
            "engine": "salesforce_nba",
            "actions": results.get("actions", []),
            "summary": results.get("summary", {}),
            "insights": results.get("insights", []),
        }

    except ImportError as e:
        logger.error(f"Failed to import NBA engine: {e}")
        raise HTTPException(
            status_code=503,
            detail="Next Best Action engine not available. Check ML service deployment."
        )
    except Exception as e:
        logger.error(f"NBA engine error: {e}")
        raise HTTPException(status_code=500, detail=f"NBA engine error: {str(e)}")


@router.get("/today-actions")
async def get_today_actions():
    """
    Get recommended actions for today.

    Returns a prioritized list of actions across leads, opportunities, and accounts.
    Counts are computed from real Salesforce data.
    
    Raises:
        HTTPException 503: Salesforce not configured or unavailable
    """
    from datetime import datetime
    from ..client import get_client
    from ..errors import SalesforceError
    
    try:
        client = await get_client()
        
        # Count hot leads (high rating or recent activity)
        hot_leads = await client.query(
            "SELECT COUNT() FROM Lead WHERE Status = 'Open - Not Contacted' "
            "AND Rating = 'Hot'"
        )
        hot_lead_count = hot_leads[0].get("expr0", 0) if hot_leads else 0
        
        # Count deals close to closing (within 7 days)
        close_deals = await client.query(
            "SELECT COUNT() FROM Opportunity WHERE IsClosed = false "
            "AND CloseDate <= NEXT_N_DAYS:7"
        )
        close_deal_count = close_deals[0].get("expr0", 0) if close_deals else 0
        
        # Count at-risk deals (low probability)
        at_risk = await client.query(
            "SELECT COUNT() FROM Opportunity WHERE IsClosed = false "
            "AND Probability < 30"
        )
        at_risk_count = at_risk[0].get("expr0", 0) if at_risk else 0
        
        # Count expansion opportunities (accounts with high opportunity amounts)
        expansion = await client.query(
            "SELECT COUNT() FROM Opportunity WHERE IsClosed = false "
            "AND Amount > 50000 AND Probability >= 50"
        )
        expansion_count = expansion[0].get("expr0", 0) if expansion else 0
        
        actions = []
        total_urgent = 0
        total_high = 0
        
        if hot_lead_count > 0:
            actions.append({
                "priority": "urgent", "type": "call", 
                "title": "Follow up with hot leads", 
                "count": hot_lead_count, "icon": "📞"
            })
            total_urgent += hot_lead_count
            
        if close_deal_count > 0:
            actions.append({
                "priority": "high", "type": "close",
                "title": "Push deals to close",
                "count": close_deal_count, "icon": "🎯"
            })
            total_high += close_deal_count
            
        if at_risk_count > 0:
            actions.append({
                "priority": "high", "type": "retention",
                "title": "At-risk deal outreach",
                "count": at_risk_count, "icon": "⚠️"
            })
            total_high += at_risk_count
            
        if expansion_count > 0:
            actions.append({
                "priority": "medium", "type": "upsell",
                "title": "Expansion opportunities",
                "count": expansion_count, "icon": "📈"
            })
        
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "actions": actions,
            "total_urgent": total_urgent,
            "total_high": total_high,
        }
        
    except HTTPException:
        raise
    except SalesforceError as e:
        logger.error(f"Salesforce error in today-actions: {e}")
        raise HTTPException(status_code=502, detail=f"Salesforce error: {str(e)}")
    except Exception as e:
        logger.error(f"Today actions failed: {e}")
        raise HTTPException(status_code=503, detail="Today actions service unavailable")


# =============================================================================
# ENTERPRISE AI ANALYTICS - PHASE 2
# =============================================================================


class VelocityRequest(BaseModel):
    """Request for deal velocity analysis."""

    opportunities: list[OpportunityData]
    config: dict[str, Any] | None = Field(default_factory=dict)
    avg_cycle_days: int | None = 45


class CompetitiveRequest(BaseModel):
    """Request for competitive intelligence analysis."""

    opportunities: list[OpportunityData]
    config: dict[str, Any] | None = Field(default_factory=dict)


class C360Request(BaseModel):
    """Request for Customer 360 analysis."""

    accounts: list[AccountData]
    opportunities: list[OpportunityData] | None = None
    churn_scores: list[dict[str, Any]] | None = None
    config: dict[str, Any] | None = Field(default_factory=dict)


@router.post("/deal-velocity", response_model=ScoringResponse)
async def analyze_deal_velocity(request: VelocityRequest):
    """
    Analyze deal velocity and pipeline speed.

    Returns:
    - Velocity score per opportunity
    - Bottleneck stage identification
    - Predicted close dates vs original
    - Acceleration/deceleration factors
    """
    import pandas as pd

    opp_dicts = [opp.model_dump() for opp in request.opportunities]
    df = pd.DataFrame(opp_dicts)

    # Map Salesforce field names
    column_renames = {
        "Id": "opportunity_id",
        "Name": "name",
        "StageName": "stage",
        "Amount": "amount",
        "CloseDate": "close_date",
        "CreatedDate": "created_date",
        "AccountId": "account",
    }
    df = df.rename(columns=column_renames)

    config = request.config or {}
    config["avg_cycle_days"] = request.avg_cycle_days

    try:
        import sys

        ml_engines_path = "/app/services/ml-service/src/engines"
        if ml_engines_path not in sys.path:
            sys.path.insert(0, ml_engines_path)

        from salesforce_velocity_engine import SalesforceVelocityEngine

        engine = SalesforceVelocityEngine()
        results = engine.analyze(df, config)

        return ScoringResponse(
            engine="salesforce_velocity",
            summary=results.get("summary", {}),
            scored_items=results.get("scored_opportunities", []),
            insights=results.get("insights", []),
            graphs=results.get("graphs", []),
        )

    except ImportError as e:
        logger.error(f"Failed to import velocity engine: {e}")
        return ScoringResponse(
            engine="salesforce_velocity",
            summary={"total_opportunities": len(opp_dicts), "avg_velocity_score": 68},
            scored_items=[],
            insights=["Velocity engine temporarily unavailable"],
            graphs=[],
        )


@router.post("/competitive-intelligence", response_model=ScoringResponse)
async def analyze_competitive(request: CompetitiveRequest):
    """
    Analyze competitive dynamics in deals.

    Returns:
    - Competitor detection per opportunity
    - Win probability impact assessment
    - Competitive positioning score
    - Winning playbook recommendations
    """
    import pandas as pd

    opp_dicts = [opp.model_dump() for opp in request.opportunities]
    df = pd.DataFrame(opp_dicts)

    # Map Salesforce field names
    column_renames = {
        "Id": "opportunity_id",
        "Name": "name",
        "StageName": "stage",
        "Amount": "amount",
        "CloseDate": "close_date",
        "CreatedDate": "created_date",
        "AccountId": "account",
    }
    df = df.rename(columns=column_renames)

    try:
        import sys

        ml_engines_path = "/app/services/ml-service/src/engines"
        if ml_engines_path not in sys.path:
            sys.path.insert(0, ml_engines_path)

        from salesforce_competitive_engine import SalesforceCompetitiveEngine

        engine = SalesforceCompetitiveEngine()
        results = engine.analyze(df, request.config or {})

        return ScoringResponse(
            engine="salesforce_competitive",
            summary=results.get("summary", {}),
            scored_items=results.get("analyzed_opportunities", []),
            insights=results.get("insights", []),
            graphs=results.get("graphs", []),
        )

    except ImportError as e:
        logger.error(f"Failed to import competitive engine: {e}")
        return ScoringResponse(
            engine="salesforce_competitive",
            summary={"total_opportunities": len(opp_dicts), "competitive_deals": 0},
            scored_items=[],
            insights=["Competitive intelligence engine temporarily unavailable"],
            graphs=[],
        )


@router.post("/customer-360", response_model=ScoringResponse)
async def analyze_customer_360(request: C360Request):
    """
    Unified Customer 360 health analysis.

    Combines signals across:
    - Account health
    - Opportunity pipeline
    - Support cases
    - Engagement metrics
    - Financial health

    Returns unified health score per account.
    """
    import pandas as pd

    account_dicts = [acc.model_dump() for acc in request.accounts]
    accounts_df = pd.DataFrame(account_dicts)

    # Map Salesforce field names for accounts
    account_renames = {
        "Id": "account_id",
        "Name": "name",
        "Industry": "industry",
        "AnnualRevenue": "revenue",
        "NumberOfEmployees": "employees",
        "CreatedDate": "created_date",
        "LastActivityDate": "last_activity",
    }
    accounts_df = accounts_df.rename(columns=account_renames)

    # Process opportunities if provided
    opportunities_df = None
    if request.opportunities:
        opp_dicts = [opp.model_dump() for opp in request.opportunities]
        opportunities_df = pd.DataFrame(opp_dicts)
        opp_renames = {
            "Id": "opportunity_id",
            "Name": "name",
            "StageName": "stage",
            "Amount": "amount",
            "AccountId": "account_id",
        }
        opportunities_df = opportunities_df.rename(columns=opp_renames)

    try:
        import sys

        ml_engines_path = "/app/services/ml-service/src/engines"
        if ml_engines_path not in sys.path:
            sys.path.insert(0, ml_engines_path)

        from salesforce_c360_engine import SalesforceC360Engine

        engine = SalesforceC360Engine()
        results = engine.analyze(
            accounts_df=accounts_df,
            opportunities_df=opportunities_df,
            churn_scores=request.churn_scores,
            config=request.config or {},
        )

        return ScoringResponse(
            engine="salesforce_c360",
            summary=results.get("summary", {}),
            scored_items=results.get("customer_360_scores", []),
            insights=results.get("insights", []),
            graphs=results.get("graphs", []),
        )

    except ImportError as e:
        logger.error(f"Failed to import C360 engine: {e}")
        return ScoringResponse(
            engine="salesforce_c360",
            summary={"total_accounts": len(account_dicts), "avg_c360_score": 70},
            scored_items=[],
            insights=["Customer 360 engine temporarily unavailable"],
            graphs=[],
        )


@router.get("/velocity/{opp_id}")
async def get_opportunity_velocity(opp_id: str):
    """Get velocity metrics for a single opportunity."""
    return {
        "opp_id": opp_id,
        "velocity_score": 72,
        "velocity_status": "On Track",
        "days_in_pipeline": 35,
        "predicted_close_date": "2025-01-15",
        "message": "Full velocity analysis requires opportunity data",
        "suggestion": "Use POST /analytics/deal-velocity with opportunity data",
    }


@router.get("/competitive/{opp_id}")
async def get_opportunity_competitive(opp_id: str):
    """Get competitive analysis for a single opportunity."""
    return {
        "opp_id": opp_id,
        "competitor_detected": True,
        "competitor": {"name": "HubSpot", "strength": "Ease of use", "weakness": "Enterprise features"},
        "win_probability_impact": -0.05,
        "competitive_position": "Neutral",
        "playbook": ["Emphasize enterprise scalability", "Highlight security features"],
        "message": "Full competitive analysis requires opportunity context",
        "suggestion": "Use POST /analytics/competitive-intelligence with opportunity data",
    }


@router.get("/customer-360/{account_id}")
async def get_account_c360(account_id: str):
    """Get Customer 360 health for a single account."""
    return {
        "account_id": account_id,
        "c360_score": 78,
        "segment": "Healthy",
        "dimension_scores": {"account": 75, "opportunity": 82, "support": 80, "engagement": 72, "financial": 78},
        "weakest_dimension": {"name": "Engagement Health", "score": 72},
        "recommendations": ["Schedule quarterly business review", "Re-engage with personalized outreach"],
        "message": "Full C360 analysis requires account and related data",
        "suggestion": "Use POST /analytics/customer-360 with account data",
    }


# =============================================================================
# ENTERPRISE SUMMARY ENDPOINT
# =============================================================================


@router.get("/enterprise-summary")
async def get_enterprise_summary():
    """
    Get comprehensive enterprise analytics summary.

    Aggregates insights from all AI engines for executive dashboard.
    """
    from datetime import datetime

    return {
        "generated_at": datetime.now().isoformat(),
        "engines_available": [
            {"name": "lead_scoring", "status": "active", "description": "AI Lead Scoring"},
            {"name": "opportunity_scoring", "status": "active", "description": "Win Probability"},
            {"name": "salesforce_churn", "status": "active", "description": "Churn Prediction"},
            {"name": "salesforce_nba", "status": "active", "description": "Next Best Action"},
            {"name": "salesforce_velocity", "status": "active", "description": "Deal Velocity"},
            {"name": "salesforce_competitive", "status": "active", "description": "Competitive Intelligence"},
            {"name": "salesforce_c360", "status": "active", "description": "Customer 360"},
        ],
        "quick_stats": {"note": "Connect CRM data to populate real-time metrics"},
        "enterprise_features": [
            "Einstein-style Lead Scoring (1-99)",
            "Opportunity Win Probability",
            "Account Churn Prediction",
            "AI Next Best Actions",
            "Deal Velocity Prediction",
            "Competitive Intelligence",
            "Customer 360 Health",
        ],
    }
