"""
Call Lifecycle API Router - Enterprise Call Center Platform
============================================================

REST API endpoints for CTI webhooks, verification, and call lifecycle.
This version uses in-memory storage for demo/testing - no PostgreSQL required.

For production, set CALL_CENTER_POSTGRES_URL environment variable.

Author: Service Credit Union AI Platform
Version: 1.0.0
"""

import logging
import re
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger("call_lifecycle_router")

# Create router with prefix
router = APIRouter(prefix="/api/v1/cti", tags=["Call Lifecycle", "CTI Integration"])


# =============================================================================
# IN-MEMORY STORAGE (for demo/testing without PostgreSQL)
# =============================================================================


class InMemoryCallStore:
    """Simple in-memory store for demo/testing."""

    def __init__(self):
        self.calls: dict[str, dict] = {}
        self.verification_attempts: dict[str, list[dict]] = {}

        # Pre-populated phone registry with demo members
        self.phone_registry = {
            "+16035551234": {
                "member_id": "SCU-000123",
                "name": "John Smith",
                "member_since": "2015-03-15",
                "verified": True,
            },
            "+16035555678": {
                "member_id": "SCU-000456",
                "name": "Jane Doe",
                "member_since": "2018-07-22",
                "verified": True,
            },
            "+16035559999": {
                "member_id": "SCU-000789",
                "name": "Robert Johnson",
                "member_since": "2020-01-10",
                "verified": False,
            },
        }

        # Demo member data (simulates Fiserv)
        self.members = {
            "SCU-000123": {
                "member_id": "SCU-000123",
                "name": "John Smith",
                "date_of_birth": "01/15/1985",
                "ssn_last4": "1234",
                "member_since": "2015-03-15",
                "accounts": [
                    {"type": "Checking", "account_number": "****1234", "balance": "$1,234.56"},
                    {"type": "Savings", "account_number": "****5678", "balance": "$5,000.00"},
                ],
                "email": "john.smith@email.com",
                "phone": "+16035551234",
                "vip": False,
            },
            "SCU-000456": {
                "member_id": "SCU-000456",
                "name": "Jane Doe",
                "date_of_birth": "06/20/1990",
                "ssn_last4": "5678",
                "member_since": "2018-07-22",
                "accounts": [
                    {"type": "Checking", "account_number": "****2345", "balance": "$8,432.10"},
                    {"type": "Money Market", "account_number": "****6789", "balance": "$25,000.00"},
                ],
                "email": "jane.doe@email.com",
                "phone": "+16035555678",
                "vip": True,
            },
            "SCU-000789": {
                "member_id": "SCU-000789",
                "name": "Robert Johnson",
                "date_of_birth": "12/05/1978",
                "ssn_last4": "9012",
                "member_since": "2020-01-10",
                "accounts": [
                    {"type": "Savings", "account_number": "****3456", "balance": "$500.00"},
                ],
                "email": "robert.j@email.com",
                "phone": "+16035559999",
                "vip": False,
            },
        }

    def normalize_phone(self, phone: str) -> str:
        """Normalize phone number to E.164 format."""
        if not phone:
            return ""
        cleaned = re.sub(r"[^0-9]", "", phone)
        if len(cleaned) == 10:
            return f"+1{cleaned}"
        elif len(cleaned) == 11 and cleaned.startswith("1"):
            return f"+{cleaned}"
        return f"+{cleaned}"

    def lookup_by_phone(self, ani: str) -> dict | None:
        """Lookup member by phone number."""
        normalized = self.normalize_phone(ani)
        return self.phone_registry.get(normalized)

    def get_member(self, member_id: str) -> dict | None:
        """Get member details."""
        return self.members.get(member_id)


# Global store
call_store = InMemoryCallStore()


# =============================================================================
# RESTRICTED ACTIONS (require verification)
# =============================================================================

RESTRICTED_UNTIL_VERIFIED = {
    "account_balances",
    "transaction_history",
    "loan_details",
    "card_numbers",
    "personal_info",
    "transfer_funds",
    "address_change",
    "password_reset",
    "pin_reset",
}

ALWAYS_ALLOWED = {"branch_hours", "general_rates", "product_info", "routing_number"}

MAX_VERIFICATION_ATTEMPTS = 3


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class CTICallStartedEvent(BaseModel):
    event: str = "call_started"
    call_sid: str = Field(..., description="External call ID from PBX")
    timestamp: str | None = None
    ani: str = Field(..., description="Caller's phone number")
    dnis: str | None = Field(None, description="Dialed number")
    direction: str = Field("inbound", description="Call direction")
    queue_id: str | None = None
    source_system: str = "cti"


class CTICallAnsweredEvent(BaseModel):
    event: str = "call_answered"
    call_sid: str
    timestamp: str | None = None
    agent_id: str
    agent_extension: str | None = None
    source_system: str = "cti"


class CTICallEndedEvent(BaseModel):
    event: str = "call_ended"
    call_sid: str
    timestamp: str | None = None
    disposition: str | None = None
    wrap_up_code: str | None = None
    source_system: str = "cti"


class VerificationRequest(BaseModel):
    call_id: str
    method: str
    answer: str


class AccessCheckRequest(BaseModel):
    call_id: str
    action: str


# =============================================================================
# CTI WEBHOOK ENDPOINTS
# =============================================================================


@router.post("/webhook/call_started", summary="CTI Webhook: Call Started")
async def webhook_call_started(event: CTICallStartedEvent, request: Request):
    """
    Process call started event from CTI/PBX.
    Performs ANI lookup and creates call record.
    """
    normalized_ani = call_store.normalize_phone(event.ani)

    # Lookup member by phone
    registry_match = call_store.lookup_by_phone(normalized_ani)
    member_data = None
    member_id = None
    confidence = 0.0

    if registry_match:
        member_id = registry_match["member_id"]
        member_data = call_store.get_member(member_id)
        confidence = 0.95 if registry_match.get("verified") else 0.70

    # Create call record
    call_id = str(uuid.uuid4())
    call_record = {
        "id": call_id,
        "call_sid": event.call_sid,
        "ani": normalized_ani,
        "dnis": event.dnis,
        "direction": event.direction,
        "queue_id": event.queue_id,
        "member_id": member_id,
        "member_verified": False,
        "verification_method": None,
        "verification_attempts": 0,
        "status": "ringing",
        "agent_id": None,
        "call_started_at": datetime.utcnow().isoformat(),
        "call_answered_at": None,
        "call_ended_at": None,
        "fiserv_member_data": member_data,
    }
    call_store.calls[call_id] = call_record

    # Build screen pop
    screen_pop = {
        "call_sid": event.call_sid,
        "ani": normalized_ani,
        "dnis": event.dnis,
        "member_found": bool(member_data),
        "member_confidence": confidence,
        "member_id": member_id,
        "member_name": member_data.get("name") if member_data else None,
        "member_since": member_data.get("member_since") if member_data else None,
        "verification_required": True,
        "suggested_verification": "kba_dob" if confidence >= 0.9 else "kba_ssn4",
        "accounts_preview": member_data.get("accounts", []) if member_data else [],
        "last_call_date": None,
        "last_call_reason": None,
        "open_action_items": 0,
        "vip_status": member_data.get("vip", False) if member_data else False,
    }

    logger.info(f"Call started: {event.call_sid} -> {call_id}, member_found={bool(member_data)}")

    return {"status": "success", "call_id": call_id, "screen_pop": screen_pop}


@router.post("/webhook/call_answered", summary="CTI Webhook: Call Answered")
async def webhook_call_answered(event: CTICallAnsweredEvent):
    """Process call answered event."""
    # Find call by SID
    call = None
    for c in call_store.calls.values():
        if c["call_sid"] == event.call_sid:
            call = c
            break

    if not call:
        raise HTTPException(status_code=404, detail=f"Call not found: {event.call_sid}")

    call["status"] = "in_progress"
    call["agent_id"] = event.agent_id
    call["call_answered_at"] = datetime.utcnow().isoformat()

    logger.info(f"Call answered: {event.call_sid} by agent {event.agent_id}")

    return {"status": "success", "call_sid": event.call_sid, "agent_id": event.agent_id}


@router.post("/webhook/call_ended", summary="CTI Webhook: Call Ended")
async def webhook_call_ended(event: CTICallEndedEvent):
    """Process call ended event."""
    call = None
    call_id = None
    for cid, c in call_store.calls.items():
        if c["call_sid"] == event.call_sid:
            call = c
            call_id = cid
            break

    if not call:
        raise HTTPException(status_code=404, detail=f"Call not found: {event.call_sid}")

    # Determine final status
    if not call.get("call_answered_at"):
        call["status"] = "abandoned"
    elif event.disposition and "transfer" in event.disposition.lower():
        call["status"] = "transferred"
    else:
        call["status"] = "completed"

    call["call_ended_at"] = datetime.utcnow().isoformat()

    logger.info(f"Call ended: {event.call_sid}, status={call['status']}")

    return {"status": "success", "call_sid": event.call_sid, "disposition": event.disposition}


# =============================================================================
# VERIFICATION ENDPOINTS
# =============================================================================


@router.post("/verify", summary="Verify Member Identity")
async def verify_member_identity(req: VerificationRequest, request: Request):
    """
    Verify member identity using specified method.

    Methods: kba_dob, kba_ssn4, kba_account, mfa_sms, manual
    """
    call = call_store.calls.get(req.call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    # Check lockout
    if call["verification_attempts"] >= MAX_VERIFICATION_ATTEMPTS:
        return JSONResponse(
            status_code=423,
            content={
                "success": False,
                "locked": True,
                "attempts_remaining": 0,
                "message": "⚠️ Maximum verification attempts exceeded. Please escalate to supervisor.",
            },
        )

    member_data = call.get("fiserv_member_data")
    if not member_data:
        return JSONResponse(status_code=400, content={"success": False, "message": "No member linked to this call."})

    # Verify based on method
    success = False
    answer = req.answer.strip()

    if req.method == "kba_dob":
        expected = member_data.get("date_of_birth", "")
        # Fuzzy date match (remove slashes/dashes for comparison)
        answer_clean = re.sub(r"[^0-9]", "", answer)
        expected_clean = re.sub(r"[^0-9]", "", expected)
        success = answer_clean == expected_clean

    elif req.method == "kba_ssn4":
        expected = member_data.get("ssn_last4", "")
        success = answer == expected

    elif req.method == "kba_account":
        accounts = member_data.get("accounts", [])
        success = any(acc.get("account_number", "").endswith(answer[-4:]) for acc in accounts)

    elif req.method == "mfa_sms":
        # Accept any 6-digit code in demo mode
        success = len(answer) == 6 and answer.isdigit()

    elif req.method == "manual":
        success = answer.lower() in ["yes", "verified", "true", "1"]

    # Log attempt
    if req.call_id not in call_store.verification_attempts:
        call_store.verification_attempts[req.call_id] = []

    call_store.verification_attempts[req.call_id].append(
        {"method": req.method, "success": success, "timestamp": datetime.utcnow().isoformat()}
    )

    call["verification_attempts"] += 1

    if success:
        call["member_verified"] = True
        call["verification_method"] = req.method

        return {
            "success": True,
            "method": req.method,
            "message": "✅ Identity verified. Full access granted.",
            "verified_at": datetime.utcnow().isoformat(),
        }
    else:
        remaining = MAX_VERIFICATION_ATTEMPTS - call["verification_attempts"]

        if remaining <= 0:
            return JSONResponse(
                status_code=423,
                content={
                    "success": False,
                    "locked": True,
                    "attempts_remaining": 0,
                    "message": "❌ Verification failed. Maximum attempts reached. Escalating to supervisor.",
                },
            )

        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "locked": False,
                "attempts_remaining": remaining,
                "message": f"❌ Verification failed. {remaining} attempts remaining.",
            },
        )


@router.get("/verification/options", summary="Get Verification Options")
async def get_verification_options(call_id: str):
    """Get available verification methods for a call."""
    return {
        "call_id": call_id,
        "options": [
            {"code": "kba_dob", "name": "Date of Birth", "type": "knowledge", "available": True},
            {"code": "kba_ssn4", "name": "SSN Last 4", "type": "knowledge", "available": True},
            {"code": "kba_account", "name": "Account Number", "type": "knowledge", "available": True},
            {"code": "mfa_sms", "name": "SMS Code", "type": "possession", "available": True},
            {"code": "manual", "name": "Manual Verification", "type": "manual", "available": True},
        ],
    }


# =============================================================================
# ACCESS CONTROL
# =============================================================================


@router.post("/access/check", summary="Check Access Permission")
async def check_access_permission(req: AccessCheckRequest, request: Request):
    """Check if action is allowed for call based on verification status."""

    if req.action in ALWAYS_ALLOWED:
        return {"allowed": True, "reason": "public_info", "message": "This information is publicly accessible."}

    call = call_store.calls.get(req.call_id)
    if not call:
        return {"allowed": False, "reason": "call_not_found", "message": "Call not found."}

    if req.action in RESTRICTED_UNTIL_VERIFIED:
        if not call.get("member_verified"):
            if call["verification_attempts"] >= MAX_VERIFICATION_ATTEMPTS:
                return {
                    "allowed": False,
                    "reason": "verification_locked",
                    "message": "⚠️ Maximum verification attempts exceeded. Please escalate to supervisor.",
                    "prompt_verification": False,
                }

            return {
                "allowed": False,
                "reason": "verification_required",
                "message": "⚠️ Member identity must be verified before accessing this information.",
                "prompt_verification": True,
                "verification_options": [
                    {"code": "kba_dob", "name": "Date of Birth"},
                    {"code": "kba_ssn4", "name": "SSN Last 4"},
                    {"code": "mfa_sms", "name": "SMS Code"},
                ],
            }

    return {"allowed": True, "reason": "verified" if call.get("member_verified") else "unrestricted_action"}


# =============================================================================
# CALL DATA ENDPOINTS
# =============================================================================


@router.get("/call/{call_id}", summary="Get Call Details")
async def get_call_details(call_id: str, include_segments: bool = False):
    """Get call details by ID."""
    call = call_store.calls.get(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    result = dict(call)
    if include_segments:
        result["segments"] = []  # Would come from transcription service

    return result


@router.get("/screen-pop/{call_sid}", summary="Get Screen Pop Data")
async def get_screen_pop(call_sid: str):
    """Get screen pop data for a call."""
    for call in call_store.calls.values():
        if call["call_sid"] == call_sid:
            member_data = call.get("fiserv_member_data")
            return {
                "call_sid": call_sid,
                "call_id": call["id"],
                "ani": call["ani"],
                "dnis": call["dnis"],
                "member_found": bool(member_data),
                "member_id": call.get("member_id"),
                "member_name": member_data.get("name") if member_data else None,
                "member_confidence": 0.95 if member_data else 0.0,
                "verification_required": True,
                "verification_status": "verified" if call.get("member_verified") else "pending",
                "accounts_preview": member_data.get("accounts", []) if member_data else [],
                "vip_status": member_data.get("vip", False) if member_data else False,
            }

    raise HTTPException(status_code=404, detail="Active call not found")


@router.get("/active", summary="Get Active Calls")
async def get_active_calls(agent_id: str | None = None, queue_id: str | None = None):
    """Get all active calls, optionally filtered by agent or queue."""
    active = [c for c in call_store.calls.values() if c["status"] in ("ringing", "in_progress", "on_hold")]

    if agent_id:
        active = [c for c in active if c.get("agent_id") == agent_id]
    if queue_id:
        active = [c for c in active if c.get("queue_id") == queue_id]

    return {
        "count": len(active),
        "calls": [
            {
                "call_id": c["id"],
                "call_sid": c["call_sid"],
                "ani": c["ani"],
                "member_id": c.get("member_id"),
                "verified": c.get("member_verified", False),
                "agent_id": c.get("agent_id"),
                "queue_id": c.get("queue_id"),
                "status": c["status"],
                "started_at": c.get("call_started_at"),
            }
            for c in active
        ],
    }


@router.get("/health", summary="Health Check")
async def health_check():
    """CTI service health check."""
    return {
        "status": "healthy",
        "service": "call_lifecycle",
        "storage": "in_memory",
        "timestamp": datetime.utcnow().isoformat(),
    }
