"""
Content Sanitizer Module

Detects and neutralizes LLM prompt injection patterns in uploaded data.
Prevents RAG poisoning attacks by scanning content before embedding.

Reference: OWASP LLM Top 10 - LLM01: Prompt Injection
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result of content scan for prompt injection."""
    is_suspicious: bool
    risk_level: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    should_quarantine: bool = False


RISK_LEVELS = ["none", "low", "medium", "high", "critical"]


INJECTION_PATTERNS: list[tuple[str, str, str]] = [
    (r"ignore\s+(all\s+)?previous\s+instructions?", "critical", "Instruction override"),
    (r"disregard\s+(all\s+)?previous", "critical", "Instruction override"),
    (r"forget\s+(all\s+)?previous", "high", "Memory manipulation"),
    (r"you\s+are\s+now\s+", "high", "Role manipulation"),
    (r"pretend\s+(you\s+are|to\s+be)", "high", "Role manipulation"),
    (r"(show|reveal|display)\s+(your\s+)?(system\s+)?prompt", "critical", "Prompt extraction"),
    (r"DAN\s+mode", "critical", "Known jailbreak"),
    (r"bypass\s+(safety|filter|restriction)", "critical", "Filter bypass"),
    (r"(send|transmit|email)\s+.*(data|secret)", "critical", "Data exfiltration"),
]


CONTROL_TOKENS = [
    "<|im_start|>", "<|im_end|>", "<<SYS>>", "<</SYS>>",
    "[INST]", "[/INST]", "<s>", "</s>"
]


def scan_text(text: str, max_length: int = 100000) -> ScanResult:
    """
    Scan text for prompt injection patterns.

    Args:
        text: Text content to scan
        max_length: Maximum text length to scan

    Returns:
        ScanResult with findings
    """
    if not text:
        return ScanResult(is_suspicious=False, risk_level="none")

    # Truncate for performance
    scan_text = text[:max_length].lower()
    findings = []
    max_risk = "none"

    # Check for control tokens
    for token in CONTROL_TOKENS:
        if token.lower() in scan_text:
            findings.append({
                "type": "control_token",
                "token": token,
                "risk": "critical",
            })
            max_risk = "critical"

    # Check injection patterns
    for pattern, risk, description in INJECTION_PATTERNS:
        try:
            if re.search(pattern, scan_text, re.IGNORECASE):
                findings.append({
                    "type": "injection_pattern",
                    "pattern": pattern,
                    "description": description,
                    "risk": risk,
                })
                if RISK_LEVELS.index(risk) > RISK_LEVELS.index(max_risk):
                    max_risk = risk
        except re.error:
            pass

    should_quarantine = max_risk in ("high", "critical")

    return ScanResult(
        is_suspicious=len(findings) > 0,
        risk_level=max_risk,
        findings=findings,
        should_quarantine=should_quarantine,
    )


def scan_dataframe_for_injection(df, columns: list[str] | None = None) -> ScanResult:
    """
    Scan DataFrame columns for prompt injection.

    Args:
        df: Pandas DataFrame
        columns: Specific columns to scan (None = all string columns)

    Returns:
        Aggregated ScanResult
    """
    import pandas as pd

    all_findings = []
    max_risk = "none"

    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        # Sample for large datasets
        sample_size = min(1000, len(df))
        sample = df[col].dropna().head(sample_size)

        for value in sample:
            if isinstance(value, str) and len(value) > 20:
                result = scan_text(value)
                if result.is_suspicious:
                    for finding in result.findings:
                        finding["column"] = col
                        all_findings.append(finding)
                    if RISK_LEVELS.index(result.risk_level) > RISK_LEVELS.index(max_risk):
                        max_risk = result.risk_level

    should_quarantine = max_risk in ("high", "critical")

    return ScanResult(
        is_suspicious=len(all_findings) > 0,
        risk_level=max_risk,
        findings=all_findings,
        should_quarantine=should_quarantine,
    )