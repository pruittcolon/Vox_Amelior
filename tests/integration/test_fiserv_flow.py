import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

# Import the router logic
from src.routers.banking import analyze_member, UserRole
from src.auth.permissions import Session

# Mock data
MOCK_TRANSACTIONS = [
    {"date": "2023-01-01", "amount": 100.0, "amount_signed": -100.0, "description": "Test Tx"}
]

@pytest.mark.asyncio
async def test_msr_role_flow():
    """Verify MSR role triggers 'statistical' and 'cash_flow' engines."""
    
    # Setup Mock User Session (MSR)
    session = MagicMock(spec=Session)
    session.role = UserRole.MSR
    session.user_id = "test_user"

    # Mock httpx.AsyncClient
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value.__aenter__.return_value
        
        # Define side_effect for .get and .post
        async def mock_request(method, url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            
            if "fiserv" in url and "transactions" in url:
                mock_resp.json.return_value = MOCK_TRANSACTIONS
                mock_resp.status_code = 200
                return mock_resp
            
            if "ml-service" in url and "engine/run" in url:
                payload = kwargs.get("json", {})
                engine_id = payload.get("engine_id")
                if engine_id == "statistical":
                    mock_resp.json.return_value = {"stats": "ok"}
                elif engine_id == "cash_flow":
                    mock_resp.json.return_value = {"cash_flow": "ok"}
                else:
                    mock_resp.json.return_value = {}
                mock_resp.status_code = 200
                return mock_resp
                
            return mock_resp

        mock_client.get = AsyncMock(side_effect=lambda url, **k: mock_request("GET", url, **k))
        mock_client.post = AsyncMock(side_effect=lambda url, **k: mock_request("POST", url, **k))

        # Execute
        result = await analyze_member("12345", session)
        
        # Verify Structure
        assert result["member_id"] == "12345"
        assert result["role_context"] == UserRole.MSR
        assert "stats" in result["insights"]
        assert "cash_flow" in result["insights"]
        # LTV should NOT be present for MSR
        assert "ltv" not in result["insights"] 

@pytest.mark.asyncio
async def test_loan_officer_role_flow():
    """Verify LOAN_OFFICER triggers 'spend_pattern' and 'ltv'."""
    
    session = MagicMock(spec=Session)
    session.role = UserRole.LOAN_OFFICER
    
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value.__aenter__.return_value
        
        async def mock_request(method, url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            if "fiserv" in url:
                 mock_resp.json.return_value = MOCK_TRANSACTIONS
            elif "ml-service" in url:
                 payload = kwargs.get("json", {})
                 eid = payload.get("engine_id")
                 if eid == "spend_pattern":
                     mock_resp.json.return_value = {"spend": "high"}
                 elif eid == "customer_ltv":
                     mock_resp.json.return_value = {"ltv": 5000}
            return mock_resp

        mock_client.get = AsyncMock(side_effect=lambda url, **k: mock_request("GET", url, **k))
        mock_client.post = AsyncMock(side_effect=lambda url, **k: mock_request("POST", url, **k))
            
        result = await analyze_member("12345", session)
        
        assert "spend_pattern" in result["insights"]
        assert "ltv" in result["insights"]
        assert "anomalies" not in result["insights"]
