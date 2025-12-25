"""
AI Guardrails - Safety mechanisms for LLM inputs and outputs.

Provides defense-in-depth protection:
- Prompt injection detection
- Content moderation
- PII filtering (integrates with pii_detector)
- Schema enforcement
- Rate limiting awareness

Critical for responsible AI deployment.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


logger = logging.getLogger(__name__)


class GuardrailType(str, Enum):
    """Types of guardrail checks."""

    PROMPT_INJECTION = "prompt_injection"
    HARMFUL_CONTENT = "harmful_content"
    PII_EXPOSURE = "pii_exposure"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    OUTPUT_TOXICITY = "output_toxicity"
    HALLUCINATION_RISK = "hallucination_risk"
    OFF_TOPIC = "off_topic"


class GuardrailAction(str, Enum):
    """Actions to take when guardrail triggers."""

    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    MODIFY = "modify"
    LOG_ONLY = "log_only"


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    guardrail_type: Optional[GuardrailType] = None
    action: GuardrailAction = GuardrailAction.ALLOW
    confidence: float = 1.0
    message: Optional[str] = None
    modified_text: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "guardrail_type": self.guardrail_type.value if self.guardrail_type else None,
            "action": self.action.value,
            "confidence": self.confidence,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# PATTERNS FOR DETECTION
# =============================================================================

# Prompt injection patterns
INJECTION_PATTERNS = [
    # Direct instruction overrides
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|prompts?)",
    r"disregard\s+(all\s+)?(previous|prior)\s+",
    r"forget\s+(everything|all|what)\s+",
    r"new\s+instructions?:\s*",
    r"system\s*:\s*you\s+are\s+now",
    # Role play attacks
    r"pretend\s+(you\s+are|to\s+be)\s+a",
    r"act\s+as\s+(if\s+you|a|an)\s+",
    r"roleplay\s+as\s+",
    r"you\s+are\s+now\s+(a|an)\s+",
    # Jailbreak patterns
    r"DAN\s*mode",
    r"developer\s*mode",
    r"evil\s*mode",
    r"god\s*mode",
    # Encoding attempts
    r"base64\s*:",
    r"decode\s+this\s*:",
    r"ROT13\s*:",
]

# Harmful content patterns
HARMFUL_PATTERNS = [
    # Violence
    r"\b(kill|murder|harm|hurt|attack)\s+(people|someone|them)\b",
    r"\bhow\s+to\s+(make|build)\s+(a\s+)?(bomb|weapon|explosive)\b",
    # Self-harm
    r"\b(suicide|self.?harm)\s+(method|way|how)\b",
    # Illegal activities
    r"\bhow\s+to\s+(hack|break\s+into|steal)\b",
    r"\billegal\s+(drugs?|substances?)\s+recipe\b",
]

# Jailbreak attempt patterns
JAILBREAK_PATTERNS = [
    r"bypass\s+(safety|content|moderation)",
    r"circumvent\s+(filters?|restrictions?)",
    r"override\s+(your|the)\s+(ethics|morals|guidelines)",
    r"ignore\s+(safety|ethical)\s+(guidelines?|constraints?)",
    r"without\s+(any\s+)?(restrictions?|limitations?|constraints?)",
]


# =============================================================================
# GUARDRAIL ENGINE
# =============================================================================


class GuardrailEngine:
    """
    AI safety guardrail engine.

    Usage:
        engine = GuardrailEngine()
        result = engine.check_input("user message")
        if not result.passed:
            # Block or modify the input
    """

    def __init__(
        self,
        enable_pii_check: bool = True,
        enable_injection_check: bool = True,
        enable_content_check: bool = True,
        strict_mode: bool = False,
        allowed_topics: Optional[list[str]] = None,
    ):
        """
        Initialize guardrail engine.

        Args:
            enable_pii_check: Check for PII exposure
            enable_injection_check: Check for prompt injection
            enable_content_check: Check for harmful content
            strict_mode: Use stricter confidence thresholds
            allowed_topics: Optional list of allowed topics (enables off-topic detection)
        """
        self.enable_pii_check = enable_pii_check
        self.enable_injection_check = enable_injection_check
        self.enable_content_check = enable_content_check
        self.strict_mode = strict_mode
        self.allowed_topics = allowed_topics

        # Compile patterns
        self._injection_patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
        self._harmful_patterns = [re.compile(p, re.IGNORECASE) for p in HARMFUL_PATTERNS]
        self._jailbreak_patterns = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]

    def check_input(self, text: str) -> GuardrailResult:
        """
        Check input text before sending to LLM.

        Args:
            text: User input text

        Returns:
            GuardrailResult with pass/fail and details
        """
        if not text or not text.strip():
            return GuardrailResult(passed=True)

        # Check for prompt injection
        if self.enable_injection_check:
            injection_result = self._check_prompt_injection(text)
            if not injection_result.passed:
                return injection_result

        # Check for jailbreak attempts
        jailbreak_result = self._check_jailbreak(text)
        if not jailbreak_result.passed:
            return jailbreak_result

        # Check for harmful content
        if self.enable_content_check:
            harmful_result = self._check_harmful_content(text)
            if not harmful_result.passed:
                return harmful_result

        # Check for PII
        if self.enable_pii_check:
            pii_result = self._check_pii(text)
            if not pii_result.passed:
                return pii_result

        return GuardrailResult(passed=True)

    def check_output(self, text: str) -> GuardrailResult:
        """
        Check LLM output before returning to user.

        Args:
            text: LLM generated text

        Returns:
            GuardrailResult with pass/fail and details
        """
        if not text or not text.strip():
            return GuardrailResult(passed=True)

        # Check for harmful content in output
        if self.enable_content_check:
            harmful_result = self._check_harmful_content(text)
            if not harmful_result.passed:
                harmful_result.guardrail_type = GuardrailType.OUTPUT_TOXICITY
                return harmful_result

        # Check for PII leakage in output
        if self.enable_pii_check:
            pii_result = self._check_pii(text)
            if not pii_result.passed:
                pii_result.guardrail_type = GuardrailType.PII_EXPOSURE
                return pii_result

        return GuardrailResult(passed=True)

    def _check_prompt_injection(self, text: str) -> GuardrailResult:
        """Check for prompt injection attempts."""
        for pattern in self._injection_patterns:
            if pattern.search(text):
                logger.warning(f"Prompt injection detected: {pattern.pattern[:50]}...")
                return GuardrailResult(
                    passed=False,
                    guardrail_type=GuardrailType.PROMPT_INJECTION,
                    action=GuardrailAction.BLOCK,
                    confidence=0.9,
                    message="Potential prompt injection detected",
                    metadata={"pattern_matched": pattern.pattern[:50]},
                )

        return GuardrailResult(passed=True)

    def _check_jailbreak(self, text: str) -> GuardrailResult:
        """Check for jailbreak attempts."""
        for pattern in self._jailbreak_patterns:
            if pattern.search(text):
                logger.warning(f"Jailbreak attempt detected: {pattern.pattern[:50]}...")
                return GuardrailResult(
                    passed=False,
                    guardrail_type=GuardrailType.JAILBREAK_ATTEMPT,
                    action=GuardrailAction.BLOCK,
                    confidence=0.85,
                    message="Jailbreak attempt detected",
                    metadata={"pattern_matched": pattern.pattern[:50]},
                )

        return GuardrailResult(passed=True)

    def _check_harmful_content(self, text: str) -> GuardrailResult:
        """Check for harmful content."""
        for pattern in self._harmful_patterns:
            if pattern.search(text):
                logger.warning(f"Harmful content detected: {pattern.pattern[:50]}...")
                return GuardrailResult(
                    passed=False,
                    guardrail_type=GuardrailType.HARMFUL_CONTENT,
                    action=GuardrailAction.BLOCK,
                    confidence=0.8,
                    message="Potentially harmful content detected",
                    metadata={"pattern_matched": pattern.pattern[:50]},
                )

        return GuardrailResult(passed=True)

    def _check_pii(self, text: str) -> GuardrailResult:
        """Check for PII using pii_detector integration."""
        try:
            from shared.security.pii_detector import PIIDetector

            detector = PIIDetector()
            matches = detector.scan(text)

            if matches:
                # In strict mode, block any PII
                if self.strict_mode:
                    return GuardrailResult(
                        passed=False,
                        guardrail_type=GuardrailType.PII_EXPOSURE,
                        action=GuardrailAction.BLOCK,
                        confidence=0.9,
                        message=f"PII detected: {len(matches)} items",
                        metadata={"pii_types": [m.pii_type.value for m in matches]},
                    )
                else:
                    # Non-strict: redact and allow
                    redacted = detector.redact(text)
                    return GuardrailResult(
                        passed=True,
                        guardrail_type=GuardrailType.PII_EXPOSURE,
                        action=GuardrailAction.MODIFY,
                        confidence=0.9,
                        message=f"PII redacted: {len(matches)} items",
                        modified_text=redacted,
                        metadata={"pii_types": [m.pii_type.value for m in matches]},
                    )

        except ImportError:
            logger.debug("PII detector not available")

        return GuardrailResult(passed=True)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def check_input(text: str) -> GuardrailResult:
    """Quick input check with default settings."""
    engine = GuardrailEngine()
    return engine.check_input(text)


def check_output(text: str) -> GuardrailResult:
    """Quick output check with default settings."""
    engine = GuardrailEngine()
    return engine.check_output(text)


def is_safe(text: str) -> bool:
    """Check if text is safe (both input and output)."""
    engine = GuardrailEngine()
    input_result = engine.check_input(text)
    if not input_result.passed:
        return False
    output_result = engine.check_output(text)
    return output_result.passed
