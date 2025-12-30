# GPU Coordinator: Why It Was Failing and How It Was Fixed

## Executive Summary

The GPU coordinator was experiencing intermittent failures during GPU ownership switching between services (Gemma and Transcription). The root cause was a **complex 3-phase Redis pub/sub handshake** that was prone to race conditions, stale messages, and timeout cascades. The fix involved creating a **centralized shared module** with proper state management and maintaining backwards compatibility.

---

## The Problem: What Was Failing

### Symptom
GPU switching between services was unreliable - sometimes it worked, sometimes it didn't. Users experienced:
- Gemma failing to acquire GPU
- Transcription not resuming after Gemma released
- OOM errors when both services tried to use GPU simultaneously
- Long delays (up to 70 seconds) during GPU handoffs

### Root Causes

#### 1. Complex 3-Phase Handshake

The original protocol required three distinct signals to complete a GPU handoff:

```
Phase 1: Coordinator -> Transcription: "transcription_pause"
Phase 2: Transcription -> Coordinator: "transcription_paused" (ACK)
Phase 3: Transcription -> Coordinator: "vram_freed" (after model offload)
```

**Problem**: Any phase could fail silently, and the coordinator had no way to know which phase failed.

#### 2. Race Conditions in Redis Pub/Sub

The `gpu_lock_manager.py` had this critical code (lines 247-265):

```python
# CRITICAL FIX: Flush any stale messages from pubsub buffer before waiting
# This prevents race conditions where old vram_freed signals are processed
flushed_count = 0
while True:
    try:
        stale_msg = await asyncio.wait_for(
            self.pubsub.get_message(ignore_subscribe_messages=True), 
            timeout=0.05
        )
        # ...flush loop...
```

**The "CRITICAL FIX" comment indicates this was a known issue.** Old messages from previous sessions could be processed instead of the current session's messages, causing:
- Premature GPU grants (before VRAM was actually freed)
- Missed acknowledgments
- State desynchronization

#### 3. Timestamp Validation Was Fragile

Messages were validated using a 30-second timestamp window:

```python
msg_timestamp = datetime.fromisoformat(data.get("timestamp", ""))
if (datetime.now() - msg_timestamp).total_seconds() > 30:
    logger.warning("[GPU] Ignoring stale vram_freed message")
    continue
```

**Problem**: Clock skew between containers, message delays, and the arbitrary 30-second window caused valid messages to be rejected.

#### 4. Timeout Cascades

The combined timeouts created worst-case delays of 70+ seconds:

| Timeout | Value | Purpose |
|---------|-------|---------|
| `pause_ack_timeout` | 10s | Wait for transcription to pause |
| `_wait_for_vram_available` | 60s | Wait for VRAM freed signal |
| **Total worst case** | **70s** | User sees frozen UI |

During this time, additional requests could queue up, creating a cascade of timeouts.

#### 5. No State Machine Enforcement

The `GPUState` enum existed but transitions were not enforced:

```python
class GPUState(Enum):
    TRANSCRIPTION_ACTIVE = "transcription"
    PAUSING = "pausing"
    GEMMA_EXCLUSIVE = "gemma"
```

**Problem**: Code could set any state at any time without validation, leading to impossible states like going directly from `TRANSCRIPTION_ACTIVE` to `GEMMA_EXCLUSIVE` without the pausing phase.

---

## The Fix: What Changed

### 1. Centralized State Machine (`shared/gpu/state.py`)

Created a proper state machine with enforced transitions:

```python
VALID_TRANSITIONS = {
    GPUOwner.TRANSCRIPTION: {GPUOwner.ACQUIRING},
    GPUOwner.ACQUIRING: {GPUOwner.GEMMA, GPUOwner.ML_SERVICE, GPUOwner.TRANSCRIPTION},
    GPUOwner.GEMMA: {GPUOwner.RELEASING},
    GPUOwner.RELEASING: {GPUOwner.TRANSCRIPTION},
}

def transition_to(self, target: GPUOwner, ...) -> "GPUState":
    if not self.can_transition_to(target):
        raise StateTransitionError(...)  # Prevents invalid transitions
```

**Impact**: Impossible to get into invalid states - catches bugs at development time.

### 2. Unified Protocol (`shared/gpu/protocol.py`)

Replaced fragile Redis message parsing with Pydantic models:

```python
class GPUAcquireRequest(BaseModel):
    session_id: str
    requester: str
    priority: GPUPriority
    timeout_ms: int = Field(ge=1000, le=120000)  # Validated range
```

**Impact**: Type-safe serialization, automatic validation, clear API contracts.

### 3. Shared Client with Retry Logic (`shared/gpu/client.py`)

Created a unified client with proper error handling:

```python
async with client.gpu_session("session-123", "gemma") as acquired:
    if acquired:
        # GPU guaranteed available
    else:
        # Graceful fallback to CPU
```

**Impact**: Automatic retry, timeout handling, and fallback logic in one place.

### 4. Backwards Compatible Listener (`shared/gpu/listener.py`)

The new listener supports both old and new channels:

```python
channels = [
    REDIS_CHANNEL_COMMAND,      # New: "gpu:command"
    REDIS_CHANNEL_PAUSE,        # Legacy: "transcription_pause"
    REDIS_CHANNEL_RESUME,       # Legacy: "transcription_resume"
]
```

**Impact**: Existing services continue to work without modification.

### 5. New Simplified Endpoints

Added new endpoints that bypass the complex handshake:

| Endpoint | Purpose |
|----------|---------|
| `POST /gpu/acquire` | Single call to acquire GPU |
| `POST /gpu/release` | Single call to release GPU |
| `GET /gpu/state` | Query current state |

**Impact**: Future services can use the simplified protocol.

---

## Why It Works Now

### Session-Based Ownership

The fundamental insight was that rapid GPU switching is inherently fragile on constrained hardware (6GB GTX 1660 Ti). The fix embraces **session-based ownership**:

1. **Gemma acquires GPU when user opens chat page** (once per session)
2. **Gemma keeps GPU during entire session** (no per-request switching)
3. **Gemma releases GPU when user closes page** (clean release)

This eliminates the "back and forth" that caused most failures.

### State Machine Prevents Impossible States

```
TRANSCRIPTION -> ACQUIRING -> GEMMA -> RELEASING -> TRANSCRIPTION
       ^                                                  |
       +--------------------------------------------------+
```

Every state has defined valid transitions. The coordinator cannot skip steps.

### Unit Tests Catch Regressions

18 unit tests verify:
- All valid transitions work
- Invalid transitions raise `StateTransitionError`
- Serialization roundtrips correctly
- Force reset always works

---

## Files Changed

| File | Change |
|------|--------|
| `shared/gpu/__init__.py` | New module initialization |
| `shared/gpu/state.py` | New state machine |
| `shared/gpu/protocol.py` | New Pydantic models |
| `shared/gpu/client.py` | New unified client |
| `shared/gpu/listener.py` | New Redis listener |
| `services/queue-service/src/main.py` | Added new endpoints |
| `tests/unit/test_gpu_state.py` | 18 new unit tests |
| `tests/playwright/test_gpu_session.spec.ts` | E2E tests |

---

## Summary

| Before | After |
|--------|-------|
| 3-phase handshake | Single acquire/release |
| Stale message race conditions | Proper message flushing with protocol |
| No state validation | Enforced state machine |
| 70s worst-case timeout | Configurable timeout with fallback |
| No unit tests | 18 passing tests |
