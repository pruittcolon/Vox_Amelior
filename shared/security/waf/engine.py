"""
WAF Engine - Core request inspection and threat detection.

Evaluates incoming requests against WAF rules to detect attacks.
Provides threat scoring, rule matching, and action determination.

Usage:
    engine = WAFEngine()
    result = engine.check_request(request)
    if result.should_block:
        return JSONResponse({"error": "Blocked"}, status_code=403)
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from shared.security.waf.rules import (
    RuleAction,
    RuleCategory,
    WAFRule,
    get_default_rules,
)

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Overall threat level assessment."""
    
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RuleMatch:
    """A single rule match result."""
    
    rule: WAFRule
    matched_content: str
    target: str  # Where the match was found
    match_position: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "rule_id": self.rule.rule_id,
            "rule_name": self.rule.name,
            "category": self.rule.category.value,
            "severity": self.rule.severity,
            "target": self.target,
            "matched": self.matched_content[:100],  # Truncate for logging
        }


@dataclass
class WAFResult:
    """Result of WAF evaluation.
    
    Attributes:
        threat_level: Overall threat assessment
        threat_score: Numeric score (0-100)
        matches: List of matched rules
        should_block: Whether request should be blocked
        action: Recommended action
        request_id: Unique request identifier
    """
    
    threat_level: ThreatLevel = ThreatLevel.NONE
    threat_score: float = 0.0
    matches: list[RuleMatch] = field(default_factory=list)
    should_block: bool = False
    action: RuleAction = RuleAction.LOG
    request_id: str = ""
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/API response."""
        return {
            "threat_level": self.threat_level.name,
            "threat_score": self.threat_score,
            "match_count": len(self.matches),
            "should_block": self.should_block,
            "action": self.action.value,
            "request_id": self.request_id,
            "processing_time_ms": self.processing_time_ms,
            "matches": [m.to_dict() for m in self.matches],
        }


@dataclass
class WAFConfig:
    """WAF configuration."""
    
    # Enable/disable WAF
    enabled: bool = True
    
    # Action mode
    blocking_mode: bool = True  # False = detection only (log but don't block)
    
    # Threshold for blocking (0-100)
    block_threshold: float = 50.0
    
    # Maximum request body size to inspect (bytes)
    max_body_size: int = 1024 * 1024  # 1MB
    
    # Categories to enable (None = all)
    enabled_categories: Optional[list[RuleCategory]] = None
    
    # Rule IDs to disable
    disabled_rules: list[str] = field(default_factory=list)
    
    # Paths to exclude from inspection
    exclude_paths: list[str] = field(default_factory=lambda: [
        "/health",
        "/metrics",
        "/favicon.ico",
    ])
    
    # Log all requests (including clean ones)
    verbose_logging: bool = False


class WAFEngine:
    """Web Application Firewall engine.
    
    Inspects HTTP requests for malicious patterns
    and provides threat assessment.
    """
    
    def __init__(self, config: Optional[WAFConfig] = None, rules: Optional[list[WAFRule]] = None):
        """Initialize the WAF engine.
        
        Args:
            config: WAF configuration
            rules: Custom rules (uses default if None)
        """
        self.config = config or WAFConfig()
        self._rules = rules or get_default_rules()
        
        # Filter rules based on config
        self._active_rules = self._filter_rules()
        
        # Stats
        self._requests_checked = 0
        self._requests_blocked = 0
        self._rule_hits: dict[str, int] = {}
        
        logger.info(
            "WAF engine initialized: %d rules active, blocking=%s",
            len(self._active_rules),
            self.config.blocking_mode,
        )
    
    def _filter_rules(self) -> list[WAFRule]:
        """Filter rules based on configuration."""
        active = []
        for rule in self._rules:
            # Skip disabled rules
            if rule.rule_id in self.config.disabled_rules:
                continue
            
            # Skip categories not enabled
            if self.config.enabled_categories:
                if rule.category not in self.config.enabled_categories:
                    continue
            
            # Skip if rule is disabled
            if not rule.enabled:
                continue
            
            active.append(rule)
        
        return active
    
    def check_request(
        self,
        url: str = "",
        method: str = "GET",
        headers: Optional[dict[str, str]] = None,
        body: Optional[str] = None,
        cookies: Optional[dict[str, str]] = None,
        client_ip: str = "",
    ) -> WAFResult:
        """Check a request against WAF rules.
        
        Args:
            url: Request URL
            method: HTTP method
            headers: Request headers
            body: Request body
            cookies: Request cookies
            client_ip: Client IP address
            
        Returns:
            WAFResult with threat assessment
        """
        start_time = time.time()
        self._requests_checked += 1
        
        # Generate request ID
        request_id = hashlib.md5(
            f"{client_ip}{url}{time.time()}".encode()
        ).hexdigest()[:16]
        
        result = WAFResult(request_id=request_id)
        
        # Check if WAF is enabled
        if not self.config.enabled:
            return result
        
        # Check excluded paths
        parsed = urlparse(url)
        if any(parsed.path.startswith(p) for p in self.config.exclude_paths):
            return result
        
        # Prepare content to inspect
        headers = headers or {}
        cookies = cookies or {}
        
        # Decode URL for inspection
        decoded_url = unquote(url)
        
        # Build targets map
        targets = {
            "url": decoded_url,
            "body": body or "",
            "headers": json.dumps(headers),
            "cookies": json.dumps(cookies),
            "method": method,
        }
        
        # Truncate body if too large
        if len(targets["body"]) > self.config.max_body_size:
            targets["body"] = targets["body"][:self.config.max_body_size]
        
        # Check each rule
        matches = []
        total_severity = 0
        
        for rule in self._active_rules:
            for target_name in rule.targets:
                content = targets.get(target_name, "")
                if not content:
                    continue
                
                match = rule.matches(content)
                if match:
                    matches.append(RuleMatch(
                        rule=rule,
                        matched_content=match.group(0),
                        target=target_name,
                        match_position=match.start(),
                    ))
                    total_severity += rule.severity
                    
                    # Track rule hits
                    self._rule_hits[rule.rule_id] = self._rule_hits.get(rule.rule_id, 0) + 1
        
        # Calculate threat score
        if matches:
            # Score based on severity and number of matches
            result.threat_score = min(100, total_severity * 10 + len(matches) * 5)
            result.matches = matches
            
            # Determine threat level
            if result.threat_score >= 80:
                result.threat_level = ThreatLevel.CRITICAL
            elif result.threat_score >= 60:
                result.threat_level = ThreatLevel.HIGH
            elif result.threat_score >= 40:
                result.threat_level = ThreatLevel.MEDIUM
            elif result.threat_score > 0:
                result.threat_level = ThreatLevel.LOW
            
            # Determine action
            highest_severity_match = max(matches, key=lambda m: m.rule.severity)
            result.action = highest_severity_match.rule.action
            
            # Should we block?
            if self.config.blocking_mode:
                if result.threat_score >= self.config.block_threshold:
                    if result.action == RuleAction.BLOCK:
                        result.should_block = True
                        self._requests_blocked += 1
        
        # Calculate processing time
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        # Log if matches found or verbose mode
        if matches or self.config.verbose_logging:
            self._log_result(result, url, client_ip)
        
        return result
    
    def _log_result(self, result: WAFResult, url: str, client_ip: str) -> None:
        """Log WAF result."""
        if result.should_block:
            logger.warning(
                "WAF BLOCKED: ip=%s url=%s threat=%s score=%.1f matches=%d",
                client_ip, url[:100], result.threat_level.name,
                result.threat_score, len(result.matches),
            )
            for match in result.matches:
                logger.warning(
                    "  Rule %s (%s): %s in %s",
                    match.rule.rule_id, match.rule.category.value,
                    match.rule.name, match.target,
                )
        elif result.matches:
            logger.info(
                "WAF detected: ip=%s url=%s threat=%s score=%.1f matches=%d",
                client_ip, url[:100], result.threat_level.name,
                result.threat_score, len(result.matches),
            )
    
    def get_stats(self) -> dict[str, Any]:
        """Get WAF statistics."""
        return {
            "requests_checked": self._requests_checked,
            "requests_blocked": self._requests_blocked,
            "block_rate": (
                self._requests_blocked / self._requests_checked
                if self._requests_checked > 0 else 0
            ),
            "active_rules": len(self._active_rules),
            "top_rules": sorted(
                self._rule_hits.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }
    
    def add_rule(self, rule: WAFRule) -> None:
        """Add a custom rule."""
        self._rules.append(rule)
        if rule.enabled and rule.rule_id not in self.config.disabled_rules:
            self._active_rules.append(rule)
    
    def disable_rule(self, rule_id: str) -> None:
        """Disable a rule by ID."""
        self.config.disabled_rules.append(rule_id)
        self._active_rules = [r for r in self._active_rules if r.rule_id != rule_id]
    
    def enable_rule(self, rule_id: str) -> None:
        """Enable a previously disabled rule."""
        if rule_id in self.config.disabled_rules:
            self.config.disabled_rules.remove(rule_id)
            self._active_rules = self._filter_rules()


# Singleton WAF engine
_waf_engine: Optional[WAFEngine] = None


def get_waf_engine(config: Optional[WAFConfig] = None) -> WAFEngine:
    """Get or create the global WAF engine."""
    global _waf_engine
    
    if _waf_engine is None:
        _waf_engine = WAFEngine(config=config)
    
    return _waf_engine
