"""
Call Lifecycle Manager - Enterprise Call Center Platform
=========================================================

Manages the complete call lifecycle for Service Credit Union:
- CTI webhook processing (call start/answer/end events)
- ANI (Automatic Number Identification) for caller recognition
- Member lookup via phone registry and Fiserv
- Verification gate enforcement
- Real-time call state tracking
- Screen pop data generation

Author: Service Credit Union AI Platform
Version: 1.0.0
"""

import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import asyncpg
from asyncpg import Pool

# Configure logging
logger = logging.getLogger("call_lifecycle_manager")
logger.setLevel(logging.INFO)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================


class CallStatus(str, Enum):
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    TRANSFERRED = "transferred"


class VerificationMethod(str, Enum):
    ANI_MATCH = "ani_match"
    KBA_DOB = "kba_dob"
    KBA_SSN4 = "kba_ssn4"
    KBA_ACCOUNT = "kba_account"
    KBA_RECENT_TX = "kba_recent_tx"
    MFA_SMS = "mfa_sms"
    MFA_EMAIL = "mfa_email"
    VOICE_BIO = "voice_bio"
    MANUAL = "manual"


class ChallengeType(str, Enum):
    KNOWLEDGE = "knowledge"
    POSSESSION = "possession"
    INHERENCE = "inherence"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemberLookupResult:
    """Result of ANI-based member lookup."""

    found: bool
    confidence: float = 0.0
    member_id: str | None = None
    member_data: dict[str, Any] | None = None
    match_type: str | None = None  # phone_registry, fiserv_search, multiple_matches, no_match
    phone_verified: bool = False
    possible_matches: list[dict] = field(default_factory=list)
    requires_verification: bool = True
    requires_manual_entry: bool = False
    requires_manual_selection: bool = False


@dataclass
class VerificationResult:
    """Result of a verification attempt."""

    success: bool
    method: str | None = None
    locked: bool = False
    attempts_remaining: int = 3
    message: str = ""
    confidence_score: float | None = None


@dataclass
class AccessResult:
    """Result of access check through verification gate."""

    allowed: bool
    reason: str = ""
    message: str = ""
    prompt_verification: bool = False
    verification_options: list[dict] = field(default_factory=list)


@dataclass
class ScreenPopData:
    """Data sent to agent desktop for screen pop."""

    call_sid: str
    ani: str
    dnis: str | None = None
    direction: str = "inbound"
    queue_id: str | None = None

    # Member lookup
    member_found: bool = False
    member_confidence: float = 0.0
    member_id: str | None = None
    member_name: str | None = None
    member_since: str | None = None

    # Verification
    verification_required: bool = True
    verification_status: str = "pending"
    suggested_verification: str | None = None

    # Quick context
    last_call_date: str | None = None
    last_call_reason: str | None = None
    open_action_items: int = 0
    account_alerts: list[str] = field(default_factory=list)
    vip_status: bool = False

    # Accounts preview (limited until verified)
    accounts_preview: list[dict] = field(default_factory=list)


# =============================================================================
# VERIFICATION GATE - Enforces Identity Verification
# =============================================================================


class VerificationGate:
    """
    Enforces verification before allowing access to member data.

    This is the core security component that ensures:
    1. Member identity is verified before sensitive actions
    2. Failed verifications are tracked and locked after max attempts
    3. All access attempts are audited
    """

    # Actions that require verified identity
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
        "statement_request",
        "card_replacement",
        "beneficiary_change",
        "wire_transfer",
        "ach_transfer",
        "stop_payment",
        "loan_application",
    }

    # Actions always allowed (public info)
    ALWAYS_ALLOWED = {
        "branch_hours",
        "general_rates",
        "product_info",
        "routing_number",
        "application_status",
        "general_inquiry",
    }

    # Maximum verification attempts before lockout
    MAX_VERIFICATION_ATTEMPTS = 3

    # Lockout duration after max attempts
    LOCKOUT_DURATION_MINUTES = 30

    def __init__(self, db_pool: Pool):
        self.db = db_pool

    async def check_access(self, call_id: str, requested_action: str, agent_id: str | None = None) -> AccessResult:
        """
        Check if agent can access requested action for this call.

        Args:
            call_id: UUID of the call
            requested_action: Action being requested
            agent_id: ID of the requesting agent

        Returns:
            AccessResult indicating if access is allowed
        """
        # Always allowed actions
        if requested_action in self.ALWAYS_ALLOWED:
            return AccessResult(allowed=True, reason="public_info", message="This information is publicly accessible.")

        # Get call verification status
        async with self.db.acquire() as conn:
            call = await conn.fetchrow(
                """
                SELECT id, member_id, member_verified, verification_method,
                       verification_attempts, verification_passed_at
                FROM calls WHERE id = $1
                """,
                uuid.UUID(call_id),
            )

        if not call:
            return AccessResult(allowed=False, reason="call_not_found", message="Call not found.")

        # Check if action requires verification
        if requested_action in self.RESTRICTED_UNTIL_VERIFIED:
            if not call["member_verified"]:
                # Get available verification methods
                verification_options = await self._get_verification_options(call_id)

                # Check if locked out
                if call["verification_attempts"] >= self.MAX_VERIFICATION_ATTEMPTS:
                    return AccessResult(
                        allowed=False,
                        reason="verification_locked",
                        message="⚠️ Maximum verification attempts exceeded. Please escalate to supervisor.",
                        prompt_verification=False,
                    )

                return AccessResult(
                    allowed=False,
                    reason="verification_required",
                    message="⚠️ Member identity must be verified before accessing this information.",
                    prompt_verification=True,
                    verification_options=verification_options,
                )

        # Access granted
        return AccessResult(allowed=True, reason="verified" if call["member_verified"] else "unrestricted_action")

    async def verify_challenge(
        self, call_id: str, method: str, answer: str, agent_id: str | None = None, ip_address: str | None = None
    ) -> VerificationResult:
        """
        Verify a challenge response.

        Args:
            call_id: UUID of the call
            method: Verification method (kba_dob, kba_ssn4, etc.)
            answer: Member's response to challenge
            agent_id: Agent submitting the verification
            ip_address: Client IP for audit

        Returns:
            VerificationResult with success/failure status
        """
        async with self.db.acquire() as conn:
            # Get call and member info
            call = await conn.fetchrow(
                """
                SELECT c.id, c.member_id, c.verification_attempts, c.fiserv_member_data
                FROM calls c WHERE c.id = $1
                """,
                uuid.UUID(call_id),
            )

            if not call:
                return VerificationResult(success=False, message="Call not found.")

            # Check for lockout
            if call["verification_attempts"] >= self.MAX_VERIFICATION_ATTEMPTS:
                return VerificationResult(
                    success=False,
                    locked=True,
                    attempts_remaining=0,
                    message="Maximum verification attempts exceeded. Escalating to supervisor.",
                )

            # Get member data (either cached or from Fiserv)
            member_data = call["fiserv_member_data"] or {}

            # Verify based on method
            success = False
            failure_reason = None
            confidence_score = None

            if method == VerificationMethod.KBA_DOB.value:
                expected = member_data.get("date_of_birth", "")
                success = self._fuzzy_date_match(answer, expected)
                failure_reason = "Date of birth does not match" if not success else None

            elif method == VerificationMethod.KBA_SSN4.value:
                expected = member_data.get("ssn_last4", "")
                success = answer.strip() == expected
                failure_reason = "SSN last 4 digits do not match" if not success else None

            elif method == VerificationMethod.KBA_ACCOUNT.value:
                # Check if answer matches any account number
                accounts = member_data.get("accounts", [])
                success = any(acc.get("account_number", "").endswith(answer.strip()[-4:]) for acc in accounts)
                failure_reason = "Account number not recognized" if not success else None

            elif method == VerificationMethod.MFA_SMS.value:
                # TODO: Integrate with OTP provider (Twilio)
                # For now, accept any 6-digit code in development
                success = len(answer.strip()) == 6 and answer.strip().isdigit()
                failure_reason = "Invalid OTP code" if not success else None
                confidence_score = 0.95 if success else None

            elif method == VerificationMethod.MANUAL.value:
                # Agent manually verified identity
                success = answer.lower() in ["yes", "verified", "true", "1"]
                confidence_score = 0.5 if success else None

            else:
                return VerificationResult(success=False, message=f"Unknown verification method: {method}")

            # Log the attempt
            await conn.execute(
                """
                INSERT INTO verification_attempts 
                (call_id, method, challenge_type, success, failure_reason, 
                 confidence_score, agent_id, ip_address)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                uuid.UUID(call_id),
                method,
                "knowledge" if method.startswith("kba") else "possession",
                success,
                failure_reason,
                confidence_score,
                agent_id,
                ip_address,
            )

            if success:
                # Update call as verified
                await conn.execute(
                    """
                    UPDATE calls SET 
                        member_verified = TRUE,
                        verification_method = $2,
                        verification_passed_at = NOW(),
                        verification_attempts = verification_attempts + 1,
                        updated_at = NOW()
                    WHERE id = $1
                    """,
                    uuid.UUID(call_id),
                    method,
                )

                # Audit log
                await self._audit_log(
                    conn,
                    event_type="verification_success",
                    resource_type="call",
                    resource_id=call_id,
                    actor_type="agent",
                    actor_id=agent_id,
                    action="verify_identity",
                    new_value={"method": method, "member_id": call["member_id"]},
                    verification_event=True,
                )

                return VerificationResult(
                    success=True,
                    method=method,
                    message="✅ Identity verified. Full access granted.",
                    confidence_score=confidence_score,
                )
            else:
                # Increment failed attempts
                await conn.execute(
                    """
                    UPDATE calls SET 
                        verification_attempts = verification_attempts + 1,
                        updated_at = NOW()
                    WHERE id = $1
                    """,
                    uuid.UUID(call_id),
                )

                attempts = call["verification_attempts"] + 1
                remaining = self.MAX_VERIFICATION_ATTEMPTS - attempts

                # Audit log
                await self._audit_log(
                    conn,
                    event_type="verification_failed",
                    resource_type="call",
                    resource_id=call_id,
                    actor_type="agent",
                    actor_id=agent_id,
                    action="verify_identity",
                    new_value={"method": method, "failure_reason": failure_reason, "attempts": attempts},
                    verification_event=True,
                )

                if remaining <= 0:
                    return VerificationResult(
                        success=False,
                        locked=True,
                        attempts_remaining=0,
                        message="❌ Verification failed. Maximum attempts reached. Escalating to supervisor.",
                    )

                return VerificationResult(
                    success=False,
                    attempts_remaining=remaining,
                    message=f"❌ Verification failed. {remaining} attempts remaining.",
                )

    async def _get_verification_options(self, call_id: str) -> list[dict]:
        """Get available verification methods for this call."""
        async with self.db.acquire() as conn:
            challenges = await conn.fetch(
                """
                SELECT challenge_code, display_name, challenge_type, security_level,
                       prompt_template, requires_otp_provider
                FROM verification_challenges
                WHERE active = TRUE
                ORDER BY sort_order
                """
            )

            return [
                {
                    "code": c["challenge_code"],
                    "name": c["display_name"],
                    "type": c["challenge_type"],
                    "security_level": c["security_level"],
                    "prompt": c["prompt_template"],
                    "available": not c["requires_otp_provider"],  # TODO: Check OTP provider status
                }
                for c in challenges
            ]

    def _fuzzy_date_match(self, answer: str, expected: str) -> bool:
        """Fuzzy match for date of birth (handles various formats)."""
        if not answer or not expected:
            return False

        # Remove non-alphanumeric characters
        answer_clean = re.sub(r"[^0-9]", "", answer)
        expected_clean = re.sub(r"[^0-9]", "", expected)

        # Try direct match
        if answer_clean == expected_clean:
            return True

        # Try various date formats
        date_formats = ["%m%d%Y", "%Y%m%d", "%d%m%Y", "%m/%d/%Y", "%Y-%m-%d"]

        for fmt in date_formats:
            try:
                if len(answer_clean) >= 8:
                    parsed = datetime.strptime(answer_clean[:8], fmt.replace("/", "").replace("-", ""))
                    expected_date = datetime.strptime(expected_clean[:8], "%m%d%Y")
                    if parsed == expected_date:
                        return True
            except ValueError:
                continue

        return False

    async def _audit_log(
        self,
        conn,
        event_type: str,
        resource_type: str,
        resource_id: str,
        actor_type: str,
        actor_id: str,
        action: str,
        old_value: dict = None,
        new_value: dict = None,
        pii_accessed: bool = False,
        financial_data: bool = False,
        verification_event: bool = False,
    ):
        """Write to audit log."""
        await conn.execute(
            """
            INSERT INTO audit_log 
            (event_type, resource_type, resource_id, actor_type, actor_id, action,
             old_value, new_value, pii_accessed, financial_data, verification_event)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            event_type,
            resource_type,
            resource_id,
            actor_type,
            actor_id,
            action,
            old_value,
            new_value,
            pii_accessed,
            financial_data,
            verification_event,
        )


# =============================================================================
# CALL LIFECYCLE MANAGER - Main Service
# =============================================================================


class CallLifecycleManager:
    """
    Manages the complete call lifecycle from ring to wrap-up.

    Responsibilities:
    - Process CTI webhooks for call events
    - Perform ANI-based member lookup
    - Generate screen pop data
    - Track call state transitions
    - Enforce verification via VerificationGate
    - Store transcription segments
    """

    def __init__(self, db_pool: Pool, fiserv_client=None):
        self.db = db_pool
        self.fiserv = fiserv_client
        self.verification_gate = VerificationGate(db_pool)

        # In-memory cache for active calls (for real-time updates)
        self._active_calls: dict[str, dict] = {}

        logger.info("CallLifecycleManager initialized")

    # =========================================================================
    # CTI WEBHOOK HANDLERS
    # =========================================================================

    async def handle_call_started(
        self,
        call_sid: str,
        ani: str,
        dnis: str,
        direction: str = "inbound",
        queue_id: str | None = None,
        source_system: str = "cti",
        raw_event: dict | None = None,
    ) -> tuple[str, ScreenPopData]:
        """
        Handle call started event from CTI.

        This is triggered when an inbound call arrives at the PBX.

        Returns:
            Tuple of (call_id, ScreenPopData)
        """
        logger.info(f"Call started: {call_sid} from {ani} to {dnis}")

        # Normalize phone number
        normalized_ani = self._normalize_phone(ani)

        # Perform ANI lookup
        lookup_result = await self._lookup_member_by_ani(normalized_ani)

        # Create call record
        call_id = str(uuid.uuid4())

        async with self.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO calls 
                (id, call_sid, ani, dnis, direction, queue_id, member_id,
                 call_started_at, status, fiserv_member_data, fiserv_accounts)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), 'ringing', $8, $9)
                """,
                uuid.UUID(call_id),
                call_sid,
                normalized_ani,
                dnis,
                direction,
                queue_id,
                lookup_result.member_id,
                lookup_result.member_data,
                lookup_result.member_data.get("accounts") if lookup_result.member_data else None,
            )

            # Log call event
            await conn.execute(
                """
                INSERT INTO call_events 
                (call_id, event_type, event_timestamp, queue_id, source_system, raw_event)
                VALUES ($1, 'started', NOW(), $2, $3, $4)
                """,
                uuid.UUID(call_id),
                queue_id,
                source_system,
                raw_event,
            )

        # Generate screen pop
        screen_pop = await self._generate_screen_pop(
            call_id=call_id,
            call_sid=call_sid,
            ani=normalized_ani,
            dnis=dnis,
            direction=direction,
            queue_id=queue_id,
            lookup_result=lookup_result,
        )

        # Cache for real-time updates
        self._active_calls[call_sid] = {
            "call_id": call_id,
            "status": CallStatus.RINGING,
            "member_id": lookup_result.member_id,
            "verified": False,
        }

        logger.info(f"Call {call_sid} created with ID {call_id}, member_found={lookup_result.found}")

        return call_id, screen_pop

    async def handle_call_answered(
        self,
        call_sid: str,
        agent_id: str,
        agent_extension: str | None = None,
        source_system: str = "cti",
        raw_event: dict | None = None,
    ) -> bool:
        """
        Handle call answered event.

        This is triggered when an agent picks up the call.
        """
        logger.info(f"Call answered: {call_sid} by agent {agent_id}")

        async with self.db.acquire() as conn:
            # Get call ID
            call = await conn.fetchrow("SELECT id FROM calls WHERE call_sid = $1", call_sid)

            if not call:
                logger.warning(f"Call not found for SID: {call_sid}")
                return False

            call_id = call["id"]

            # Update call status
            await conn.execute(
                """
                UPDATE calls SET 
                    status = 'in_progress',
                    agent_id = $2,
                    agent_extension = $3,
                    call_answered_at = NOW(),
                    updated_at = NOW()
                WHERE id = $1
                """,
                call_id,
                agent_id,
                agent_extension,
            )

            # Log event
            await conn.execute(
                """
                INSERT INTO call_events 
                (call_id, event_type, event_timestamp, previous_state, new_state,
                 agent_id, source_system, raw_event)
                VALUES ($1, 'answered', NOW(), 'ringing', 'in_progress', $2, $3, $4)
                """,
                call_id,
                agent_id,
                source_system,
                raw_event,
            )

            # Update agent session
            await conn.execute(
                """
                UPDATE agent_sessions SET 
                    status = 'on_call',
                    current_call_id = $2,
                    calls_handled = calls_handled + 1
                WHERE agent_id = $1 AND session_ended IS NULL
                """,
                agent_id,
                call_id,
            )

        # Update cache
        if call_sid in self._active_calls:
            self._active_calls[call_sid]["status"] = CallStatus.IN_PROGRESS
            self._active_calls[call_sid]["agent_id"] = agent_id

        return True

    async def handle_call_ended(
        self,
        call_sid: str,
        disposition: str | None = None,
        wrap_up_code: str | None = None,
        source_system: str = "cti",
        raw_event: dict | None = None,
    ) -> bool:
        """
        Handle call ended event.

        This is triggered when the call terminates (hang up, transfer, abandon).
        """
        logger.info(f"Call ended: {call_sid}, disposition={disposition}")

        async with self.db.acquire() as conn:
            call = await conn.fetchrow(
                "SELECT id, agent_id, call_started_at, call_answered_at FROM calls WHERE call_sid = $1", call_sid
            )

            if not call:
                logger.warning(f"Call not found for SID: {call_sid}")
                return False

            call_id = call["id"]
            agent_id = call["agent_id"]

            # Determine final status
            final_status = "completed"
            if not call["call_answered_at"]:
                final_status = "abandoned"
            elif disposition and "transfer" in disposition.lower():
                final_status = "transferred"

            # Update call
            await conn.execute(
                """
                UPDATE calls SET 
                    status = $2,
                    call_ended_at = NOW(),
                    disposition = $3,
                    updated_at = NOW()
                WHERE id = $1
                """,
                call_id,
                final_status,
                disposition or wrap_up_code,
            )

            # Log event
            await conn.execute(
                """
                INSERT INTO call_events 
                (call_id, event_type, event_timestamp, previous_state, new_state,
                 source_system, raw_event)
                VALUES ($1, 'ended', NOW(), 'in_progress', $2, $3, $4)
                """,
                call_id,
                final_status,
                source_system,
                raw_event,
            )

            # Update agent session if agent was assigned
            if agent_id:
                await conn.execute(
                    """
                    UPDATE agent_sessions SET 
                        status = 'wrap_up',
                        current_call_id = NULL
                    WHERE agent_id = $1 AND session_ended IS NULL
                    """,
                    agent_id,
                )

        # Remove from cache
        if call_sid in self._active_calls:
            del self._active_calls[call_sid]

        return True

    async def handle_call_hold(self, call_sid: str, hold_started: bool = True, source_system: str = "cti") -> bool:
        """Handle hold start/end events."""
        event_type = "hold_started" if hold_started else "hold_ended"
        new_status = "on_hold" if hold_started else "in_progress"

        async with self.db.acquire() as conn:
            call = await conn.fetchrow("SELECT id FROM calls WHERE call_sid = $1", call_sid)

            if not call:
                return False

            await conn.execute("UPDATE calls SET status = $2, updated_at = NOW() WHERE id = $1", call["id"], new_status)

            await conn.execute(
                """
                INSERT INTO call_events (call_id, event_type, event_timestamp, new_state, source_system)
                VALUES ($1, $2, NOW(), $3, $4)
                """,
                call["id"],
                event_type,
                new_status,
                source_system,
            )

        return True

    # =========================================================================
    # ANI LOOKUP AND MEMBER IDENTIFICATION
    # =========================================================================

    async def _lookup_member_by_ani(self, ani: str) -> MemberLookupResult:
        """
        Attempt to identify member from inbound phone number (ANI).

        Lookup order:
        1. Phone number registry (pre-synced from Fiserv)
        2. Direct Fiserv search (if enabled)
        3. Return not found
        """
        if not ani:
            return MemberLookupResult(found=False, match_type="no_ani", requires_manual_entry=True)

        async with self.db.acquire() as conn:
            # Step 1: Check phone registry
            registry_match = await conn.fetchrow(
                """
                SELECT member_id, phone_number, number_type, is_primary, verified,
                       fraud_flags, last_successful_auth
                FROM phone_number_registry
                WHERE phone_number = $1
                ORDER BY is_primary DESC, verified DESC
                LIMIT 1
                """,
                ani,
            )

            if registry_match:
                # Found in registry - get member data
                member_data = await self._get_member_from_fiserv(registry_match["member_id"])

                confidence = 0.95 if registry_match["verified"] else 0.70
                if registry_match["fraud_flags"] > 0:
                    confidence *= 0.5  # Reduce confidence for flagged numbers

                return MemberLookupResult(
                    found=True,
                    confidence=confidence,
                    member_id=registry_match["member_id"],
                    member_data=member_data,
                    match_type="phone_registry",
                    phone_verified=registry_match["verified"],
                    requires_verification=True,  # ALWAYS require verification
                )

            # Step 2: Search Fiserv directly (if client available)
            if self.fiserv:
                try:
                    fiserv_matches = await self.fiserv.search_by_phone(ani)

                    if len(fiserv_matches) == 1:
                        member = fiserv_matches[0]

                        # Add to registry for future lookups
                        await self._add_to_phone_registry(conn, ani, member["member_id"], "fiserv_search")

                        return MemberLookupResult(
                            found=True,
                            confidence=0.60,
                            member_id=member["member_id"],
                            member_data=member,
                            match_type="fiserv_search",
                            requires_verification=True,
                        )

                    elif len(fiserv_matches) > 1:
                        return MemberLookupResult(
                            found=False,
                            confidence=0.0,
                            possible_matches=fiserv_matches,
                            match_type="multiple_matches",
                            requires_verification=True,
                            requires_manual_selection=True,
                        )
                except Exception as e:
                    logger.warning(f"Fiserv search failed: {e}")

        # Step 3: No match found
        return MemberLookupResult(
            found=False, confidence=0.0, match_type="no_match", requires_verification=True, requires_manual_entry=True
        )

    async def _get_member_from_fiserv(self, member_id: str) -> dict | None:
        """Get member data from Fiserv or cache."""
        if self.fiserv:
            try:
                return await self.fiserv.get_member(member_id)
            except Exception as e:
                logger.warning(f"Failed to get member from Fiserv: {e}")

        # Return minimal mock data if Fiserv unavailable
        return {"member_id": member_id, "name": "Unknown", "member_since": None, "accounts": []}

    async def _add_to_phone_registry(self, conn, phone: str, member_id: str, source: str):
        """Add discovered phone number to registry."""
        await conn.execute(
            """
            INSERT INTO phone_number_registry 
            (phone_number, member_id, number_type, source)
            VALUES ($1, $2, 'unknown', $3)
            ON CONFLICT (phone_number, member_id) DO NOTHING
            """,
            phone,
            member_id,
            source,
        )

    # =========================================================================
    # SCREEN POP GENERATION
    # =========================================================================

    async def _generate_screen_pop(
        self,
        call_id: str,
        call_sid: str,
        ani: str,
        dnis: str,
        direction: str,
        queue_id: str | None,
        lookup_result: MemberLookupResult,
    ) -> ScreenPopData:
        """Generate screen pop data for agent desktop."""

        screen_pop = ScreenPopData(
            call_sid=call_sid, ani=ani, dnis=dnis, direction=direction, queue_id=queue_id, verification_required=True
        )

        if lookup_result.found and lookup_result.member_data:
            member = lookup_result.member_data

            screen_pop.member_found = True
            screen_pop.member_confidence = lookup_result.confidence
            screen_pop.member_id = lookup_result.member_id
            screen_pop.member_name = member.get("name", "Unknown")
            screen_pop.member_since = member.get("member_since")
            screen_pop.vip_status = member.get("vip", False)

            # Suggest verification method based on confidence
            if lookup_result.confidence >= 0.9 and lookup_result.phone_verified:
                screen_pop.suggested_verification = "kba_dob"
            elif lookup_result.confidence >= 0.7:
                screen_pop.suggested_verification = "kba_ssn4"
            else:
                screen_pop.suggested_verification = "mfa_sms"

            # Get limited account preview (last 4 only until verified)
            accounts = member.get("accounts", [])
            screen_pop.accounts_preview = [
                {
                    "type": acc.get("type", "Unknown"),
                    "last4": acc.get("account_number", "")[-4:],
                    "balance": "****.**",  # Hidden until verified
                }
                for acc in accounts[:3]
            ]

            # Get call history
            async with self.db.acquire() as conn:
                last_call = await conn.fetchrow(
                    """
                    SELECT call_started_at, intent_detected
                    FROM calls
                    WHERE member_id = $1 AND id != $2
                    ORDER BY call_started_at DESC
                    LIMIT 1
                    """,
                    lookup_result.member_id,
                    uuid.UUID(call_id),
                )

                if last_call:
                    screen_pop.last_call_date = (
                        last_call["call_started_at"].isoformat() if last_call["call_started_at"] else None
                    )
                    screen_pop.last_call_reason = last_call["intent_detected"]

                # Count open action items
                action_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM action_items
                    WHERE call_id IN (SELECT id FROM calls WHERE member_id = $1)
                    AND status = 'open'
                    """,
                    lookup_result.member_id,
                )
                screen_pop.open_action_items = action_count or 0

        elif lookup_result.requires_manual_selection:
            screen_pop.verification_status = "multiple_matches"
        else:
            screen_pop.verification_status = "unknown_caller"

        return screen_pop

    # =========================================================================
    # TRANSCRIPTION INTEGRATION
    # =========================================================================

    async def add_transcript_segment(
        self,
        call_id: str,
        speaker: str,
        text: str,
        start_time_sec: float,
        end_time_sec: float,
        speaker_id: str | None = None,
        sentiment_score: float | None = None,
        emotion: str | None = None,
    ) -> str:
        """Add a transcription segment to the call."""
        segment_id = str(uuid.uuid4())

        async with self.db.acquire() as conn:
            # Get segment index
            count = await conn.fetchval("SELECT COUNT(*) FROM call_segments WHERE call_id = $1", uuid.UUID(call_id))

            # Redact PII
            text_redacted = self._redact_pii(text)

            await conn.execute(
                """
                INSERT INTO call_segments
                (id, call_id, segment_index, speaker, speaker_id, start_time_sec, end_time_sec,
                 text, text_redacted, sentiment_score, emotion)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                uuid.UUID(segment_id),
                uuid.UUID(call_id),
                count + 1,
                speaker,
                speaker_id,
                start_time_sec,
                end_time_sec,
                text,
                text_redacted,
                sentiment_score,
                emotion,
            )

        return segment_id

    async def finalize_transcript(self, call_id: str) -> bool:
        """Compile all segments into full transcript."""
        async with self.db.acquire() as conn:
            segments = await conn.fetch(
                """
                SELECT speaker, text, text_redacted
                FROM call_segments
                WHERE call_id = $1
                ORDER BY segment_index
                """,
                uuid.UUID(call_id),
            )

            if not segments:
                return False

            # Build full transcript
            full_text = "\n".join(f"{s['speaker'].upper()}: {s['text']}" for s in segments)

            redacted_text = "\n".join(f"{s['speaker'].upper()}: {s['text_redacted']}" for s in segments)

            await conn.execute(
                """
                UPDATE calls SET
                    transcript_text = $2,
                    transcript_redacted = $3,
                    transcript_status = 'completed',
                    updated_at = NOW()
                WHERE id = $1
                """,
                uuid.UUID(call_id),
                full_text,
                redacted_text,
            )

        return True

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to E.164 format."""
        if not phone:
            return ""

        # Remove all non-digits
        cleaned = re.sub(r"[^0-9]", "", phone)

        # Handle US numbers
        if len(cleaned) == 10:
            return f"+1{cleaned}"
        elif len(cleaned) == 11 and cleaned.startswith("1"):
            return f"+{cleaned}"
        else:
            return f"+{cleaned}"

    def _redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
            (r"\b\d{9}\b", "[SSN_REDACTED]"),
            (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD_REDACTED]"),
            (r"\b\d{10,12}\b", "[ACCOUNT_REDACTED]"),
            (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]"),
            (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL_REDACTED]"),
        ]

        redacted = text
        for pattern, replacement in patterns:
            redacted = re.sub(pattern, replacement, redacted)

        return redacted

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    async def get_call(self, call_id: str) -> dict | None:
        """Get call details by ID."""
        async with self.db.acquire() as conn:
            call = await conn.fetchrow("SELECT * FROM calls WHERE id = $1", uuid.UUID(call_id))
            return dict(call) if call else None

    async def get_call_by_sid(self, call_sid: str) -> dict | None:
        """Get call details by external SID."""
        async with self.db.acquire() as conn:
            call = await conn.fetchrow("SELECT * FROM calls WHERE call_sid = $1", call_sid)
            return dict(call) if call else None

    async def check_access(self, call_id: str, action: str, agent_id: str = None) -> AccessResult:
        """Check if action is allowed for this call (wrapper for VerificationGate)."""
        return await self.verification_gate.check_access(call_id, action, agent_id)

    async def verify_identity(
        self, call_id: str, method: str, answer: str, agent_id: str = None, ip_address: str = None
    ) -> VerificationResult:
        """Verify member identity (wrapper for VerificationGate)."""
        return await self.verification_gate.verify_challenge(call_id, method, answer, agent_id, ip_address)


# =============================================================================
# DATABASE CONNECTION POOL
# =============================================================================

_db_pool: Pool | None = None
_call_manager: CallLifecycleManager | None = None


async def init_database(
    host: str = None, port: int = 5432, database: str = "call_center", user: str = "postgres", password: str = None
) -> Pool:
    """Initialize PostgreSQL connection pool."""
    global _db_pool

    host = host or os.environ.get("POSTGRES_HOST", "localhost")
    password = password or os.environ.get("POSTGRES_PASSWORD", "postgres")
    database = os.environ.get("POSTGRES_DB", database)
    user = os.environ.get("POSTGRES_USER", user)

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    _db_pool = await asyncpg.create_pool(dsn, min_size=5, max_size=20, command_timeout=60)

    logger.info(f"PostgreSQL pool initialized: {host}:{port}/{database}")

    return _db_pool


async def get_db_pool() -> Pool:
    """Get database connection pool."""
    global _db_pool
    if _db_pool is None:
        _db_pool = await init_database()
    return _db_pool


async def get_call_lifecycle_manager() -> CallLifecycleManager:
    """Get CallLifecycleManager singleton."""
    global _call_manager
    if _call_manager is None:
        pool = await get_db_pool()
        _call_manager = CallLifecycleManager(pool)
    return _call_manager


async def close_database():
    """Close database pool."""
    global _db_pool, _call_manager
    if _db_pool:
        await _db_pool.close()
        _db_pool = None
        _call_manager = None
        logger.info("PostgreSQL pool closed")
