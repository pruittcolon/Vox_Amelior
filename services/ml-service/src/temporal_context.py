"""
Temporal Context Enrichment Module
===================================

Adds date awareness and world events context to transcription sections.
This enables the CIDE (Contextual Insight Discovery Engine) to create
temporally-aware embeddings that capture the context of when recordings
were made.

Key Features:
- Cyclical date encoding (day, month, quarter)
- World events fetching (Wikipedia Current Events / GDELT)
- Market context integration (optional)
- Fiscal period awareness

Author: NeMo Server Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)

# Configuration
EVENTS_CACHE_DIR = os.getenv("EVENTS_CACHE_DIR", "/tmp/temporal_events_cache")
EVENTS_LOOKBACK_DAYS = int(os.getenv("EVENTS_LOOKBACK_DAYS", "7"))
EVENTS_LOOKAHEAD_DAYS = int(os.getenv("EVENTS_LOOKAHEAD_DAYS", "2"))


@dataclass
class TemporalContext:
    """
    Temporal metadata for a transcription or section.
    
    Contains both raw date information and derived features
    useful for embedding generation and insight discovery.
    """
    
    recording_date: datetime
    day_of_week: str = ""
    day_of_week_num: int = 0
    month: int = 0
    quarter: int = 0
    fiscal_quarter: str = ""
    year: int = 0
    week_of_year: int = 0
    is_weekend: bool = False
    is_month_end: bool = False
    is_quarter_end: bool = False
    
    # External context
    notable_events: list[str] = field(default_factory=list)
    event_categories: list[str] = field(default_factory=list)
    market_context: dict[str, Any] = field(default_factory=dict)
    
    # Business context (optional, user-provided)
    business_events: list[str] = field(default_factory=list)
    meeting_type: str = ""
    
    def __post_init__(self):
        """Derive date features from recording_date."""
        if isinstance(self.recording_date, str):
            self.recording_date = datetime.fromisoformat(self.recording_date)
        
        dt = self.recording_date
        
        # Basic date features
        self.day_of_week = dt.strftime("%A")
        self.day_of_week_num = dt.weekday()
        self.month = dt.month
        self.quarter = (dt.month - 1) // 3 + 1
        self.year = dt.year
        self.week_of_year = dt.isocalendar()[1]
        self.is_weekend = dt.weekday() >= 5
        
        # Month/quarter end detection
        next_day = dt + timedelta(days=1)
        self.is_month_end = next_day.month != dt.month
        self.is_quarter_end = self.is_month_end and dt.month in [3, 6, 9, 12]
        
        # Fiscal quarter (assuming Oct fiscal year start, common in finance)
        fiscal_month = (dt.month - 10) % 12 + 1
        self.fiscal_quarter = f"FQ{(fiscal_month - 1) // 3 + 1}"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "recording_date": self.recording_date.isoformat(),
            "day_of_week": self.day_of_week,
            "day_of_week_num": self.day_of_week_num,
            "month": self.month,
            "quarter": self.quarter,
            "fiscal_quarter": self.fiscal_quarter,
            "year": self.year,
            "week_of_year": self.week_of_year,
            "is_weekend": self.is_weekend,
            "is_month_end": self.is_month_end,
            "is_quarter_end": self.is_quarter_end,
            "notable_events": self.notable_events,
            "event_categories": self.event_categories,
            "market_context": self.market_context,
            "business_events": self.business_events,
            "meeting_type": self.meeting_type,
        }
    
    def get_context_summary(self) -> str:
        """
        Generate a human-readable context summary for Gemma prompts.
        """
        parts = [
            f"Date: {self.recording_date.strftime('%B %d, %Y')} ({self.day_of_week})",
            f"Period: Q{self.quarter} {self.year} ({self.fiscal_quarter})",
        ]
        
        if self.is_quarter_end:
            parts.append("Note: Quarter-end period")
        elif self.is_month_end:
            parts.append("Note: Month-end period")
        
        if self.notable_events:
            events_str = "; ".join(self.notable_events[:3])
            parts.append(f"World Events: {events_str}")
        
        if self.market_context:
            market_str = ", ".join(
                f"{k}: {v}" for k, v in list(self.market_context.items())[:3]
            )
            parts.append(f"Market: {market_str}")
        
        if self.business_events:
            biz_str = "; ".join(self.business_events[:2])
            parts.append(f"Business Context: {biz_str}")
        
        return "\n".join(parts)


class TemporalEncoder:
    """
    Encode temporal context as dense feature vectors.
    
    Uses cyclical encoding for periodic features (day, month, etc.)
    to preserve the circular nature of time.
    """
    
    def __init__(self, feature_dim: int = 32):
        """
        Initialize encoder with target feature dimension.
        
        Args:
            feature_dim: Output dimension for temporal features.
                        Must be >= 16 for all features.
        """
        self.feature_dim = feature_dim
    
    def encode(self, ctx: TemporalContext) -> np.ndarray:
        """
        Encode temporal context as a dense vector.
        
        Features (total 32 dimensions):
        - Day of week: sin/cos (2d)
        - Month: sin/cos (2d)
        - Quarter: sin/cos (2d)
        - Week of year: sin/cos (2d)
        - Hour of day: sin/cos (2d) - if available
        - Year normalized: (1d)
        - Weekend flag: (1d)
        - Month-end flag: (1d)
        - Quarter-end flag: (1d)
        - Event density: (1d)
        - Event category flags: (8d)
        - Market features: (8d)
        - Padding: remaining to feature_dim
        
        Args:
            ctx: TemporalContext object
            
        Returns:
            numpy array of shape (feature_dim,)
        """
        features = []
        
        # Cyclical day of week (7-day cycle)
        features.extend([
            np.sin(2 * np.pi * ctx.day_of_week_num / 7),
            np.cos(2 * np.pi * ctx.day_of_week_num / 7),
        ])
        
        # Cyclical month (12-month cycle)
        features.extend([
            np.sin(2 * np.pi * (ctx.month - 1) / 12),
            np.cos(2 * np.pi * (ctx.month - 1) / 12),
        ])
        
        # Cyclical quarter (4-quarter cycle)
        features.extend([
            np.sin(2 * np.pi * (ctx.quarter - 1) / 4),
            np.cos(2 * np.pi * (ctx.quarter - 1) / 4),
        ])
        
        # Cyclical week of year (52-week cycle)
        features.extend([
            np.sin(2 * np.pi * ctx.week_of_year / 52),
            np.cos(2 * np.pi * ctx.week_of_year / 52),
        ])
        
        # Hour of day (if time is meaningful)
        hour = ctx.recording_date.hour
        features.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
        ])
        
        # Year normalized (assuming range 2020-2030)
        year_normalized = (ctx.year - 2020) / 10
        features.append(np.clip(year_normalized, 0, 1))
        
        # Binary flags
        features.append(1.0 if ctx.is_weekend else 0.0)
        features.append(1.0 if ctx.is_month_end else 0.0)
        features.append(1.0 if ctx.is_quarter_end else 0.0)
        
        # Event density (normalized count)
        event_density = min(len(ctx.notable_events) / 10.0, 1.0)
        features.append(event_density)
        
        # Event category flags (8 common categories)
        category_flags = self._encode_event_categories(ctx.event_categories)
        features.extend(category_flags)
        
        # Market features (if available)
        market_features = self._encode_market_context(ctx.market_context)
        features.extend(market_features)
        
        # Pad or truncate to target dimension
        features = np.array(features, dtype=np.float32)
        
        if len(features) < self.feature_dim:
            padding = np.zeros(self.feature_dim - len(features), dtype=np.float32)
            features = np.concatenate([features, padding])
        elif len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        
        return features
    
    def _encode_event_categories(self, categories: list[str]) -> list[float]:
        """
        Encode event categories as binary flags.
        
        Categories:
        - politics, economics, technology, business,
        - health, environment, sports, entertainment
        """
        known_categories = [
            "politics", "economics", "technology", "business",
            "health", "environment", "sports", "entertainment"
        ]
        
        flags = []
        categories_lower = [c.lower() for c in categories]
        
        for cat in known_categories:
            flags.append(1.0 if cat in categories_lower else 0.0)
        
        return flags
    
    def _encode_market_context(self, market: dict[str, Any]) -> list[float]:
        """
        Encode market context as normalized features.
        
        Expected keys: vix, sp500_change, fed_rate, etc.
        """
        features = [
            # VIX (fear index, normalize 10-80 range)
            np.clip((market.get("vix", 20) - 10) / 70, 0, 1),
            # S&P 500 daily change (-5% to +5%)
            np.clip((market.get("sp500_change", 0) + 5) / 10, 0, 1),
            # Fed rate (0-10%)
            np.clip(market.get("fed_rate", 5) / 10, 0, 1),
            # Unemployment (0-15%)
            np.clip(market.get("unemployment", 5) / 15, 0, 1),
            # Consumer sentiment (0-100)
            np.clip(market.get("consumer_sentiment", 70) / 100, 0, 1),
            # GDP growth (-5% to +10%)
            np.clip((market.get("gdp_growth", 2) + 5) / 15, 0, 1),
            # Inflation (0-15%)
            np.clip(market.get("inflation", 3) / 15, 0, 1),
            # Market volatility flag
            1.0 if market.get("high_volatility", False) else 0.0,
        ]
        
        return features


class WorldEventsProvider:
    """
    Fetches world events from public sources.
    
    Primary source: Wikipedia Current Events Portal
    Fallback: GDELT Project (more comprehensive but less curated)
    """
    
    def __init__(self, cache_dir: str = EVENTS_CACHE_DIR):
        """Initialize provider with caching."""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    async def fetch_events(
        self,
        target_date: datetime,
        lookback_days: int = EVENTS_LOOKBACK_DAYS,
        lookahead_days: int = EVENTS_LOOKAHEAD_DAYS,
    ) -> list[dict[str, Any]]:
        """
        Fetch world events around the target date.
        
        Args:
            target_date: Center date for event lookup
            lookback_days: Days before target to include
            lookahead_days: Days after target to include
            
        Returns:
            List of event dictionaries with title, date, category
        """
        all_events = []
        
        start_date = target_date - timedelta(days=lookback_days)
        end_date = target_date + timedelta(days=lookahead_days)
        
        current = start_date
        while current <= end_date:
            # Check cache first
            cached = self._load_from_cache(current)
            if cached:
                all_events.extend(cached)
            else:
                # Fetch from Wikipedia
                try:
                    events = await self._fetch_wikipedia_events(current)
                    if events:
                        self._save_to_cache(current, events)
                        all_events.extend(events)
                except Exception as e:
                    logger.warning(f"Failed to fetch events for {current}: {e}")
            
            current += timedelta(days=1)
        
        return all_events
    
    async def _fetch_wikipedia_events(self, date: datetime) -> list[dict[str, Any]]:
        """
        Fetch events from Wikipedia Current Events portal.
        
        Uses the Wikipedia API to fetch the "Portal:Current_events" page
        for the specified date.
        """
        # Format date for Wikipedia URL
        date_str = date.strftime("%Y_%B_%-d").replace(" ", "_")
        # Wikipedia page title pattern
        page_title = f"Portal:Current_events/{date_str}"
        
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "parse",
            "page": page_title,
            "format": "json",
            "prop": "text",
            "formatversion": "2",
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(api_url, params=params)
                
                if response.status_code != 200:
                    return []
                
                data = response.json()
                
                if "error" in data:
                    # Page might not exist for this date
                    return []
                
                html_content = data.get("parse", {}).get("text", "")
                
                # Parse events from HTML (simplified extraction)
                events = self._parse_wikipedia_html(html_content, date)
                return events
                
        except Exception as e:
            logger.debug(f"Wikipedia fetch error for {date}: {e}")
            return []
    
    def _parse_wikipedia_html(
        self, html: str, date: datetime
    ) -> list[dict[str, Any]]:
        """
        Extract events from Wikipedia Current Events HTML.
        
        This is a simplified parser - production would use BeautifulSoup.
        """
        events = []
        
        # Simple regex to find list items (events are typically in <li> tags)
        # This is basic - production should use proper HTML parsing
        li_pattern = r"<li[^>]*>([^<]+(?:<[^>]+>[^<]*</[^>]+>)*[^<]*)</li>"
        matches = re.findall(li_pattern, html, re.IGNORECASE)
        
        for match in matches[:20]:  # Limit to 20 events per day
            # Clean HTML tags
            text = re.sub(r"<[^>]+>", "", match).strip()
            
            if len(text) > 20 and len(text) < 500:
                # Detect category from keywords
                category = self._detect_category(text)
                
                events.append({
                    "title": text[:200],  # Truncate long events
                    "date": date.isoformat(),
                    "category": category,
                    "source": "wikipedia",
                })
        
        return events
    
    def _detect_category(self, text: str) -> str:
        """Detect event category from content keywords."""
        text_lower = text.lower()
        
        category_keywords = {
            "politics": ["president", "congress", "parliament", "election", "vote", "minister", "government"],
            "economics": ["economy", "inflation", "gdp", "unemployment", "trade", "tariff", "recession"],
            "technology": ["tech", "ai", "software", "internet", "cyber", "startup", "innovation"],
            "business": ["company", "merger", "acquisition", "ceo", "stock", "earnings", "billion"],
            "health": ["health", "covid", "vaccine", "hospital", "disease", "medical", "who"],
            "environment": ["climate", "environment", "carbon", "pollution", "wildfire", "hurricane"],
            "sports": ["game", "championship", "olympic", "world cup", "tournament", "athlete"],
            "entertainment": ["film", "movie", "music", "award", "celebrity", "streaming"],
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return category
        
        return "general"
    
    def _get_cache_path(self, date: datetime) -> str:
        """Get cache file path for a date."""
        date_str = date.strftime("%Y-%m-%d")
        return os.path.join(self.cache_dir, f"events_{date_str}.json")
    
    def _load_from_cache(self, date: datetime) -> list[dict[str, Any]] | None:
        """Load events from cache if available and fresh."""
        cache_path = self._get_cache_path(date)
        
        if not os.path.exists(cache_path):
            return None
        
        # Check cache age (24 hours for recent dates, indefinite for old)
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
        days_ago = (datetime.now() - date).days
        
        # For dates more than 7 days ago, cache is permanent
        # For recent dates, refresh daily
        if days_ago <= 7 and cache_age.days >= 1:
            return None
        
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    
    def _save_to_cache(self, date: datetime, events: list[dict[str, Any]]) -> None:
        """Save events to cache."""
        cache_path = self._get_cache_path(date)
        try:
            with open(cache_path, "w") as f:
                json.dump(events, f)
        except Exception as e:
            logger.warning(f"Failed to cache events: {e}")


class TemporalContextEnricher:
    """
    Main enricher class that combines date features with world events.
    
    Usage:
        enricher = TemporalContextEnricher()
        context = await enricher.enrich(
            recording_date=datetime(2024, 11, 15),
            meeting_type="strategy",
            business_events=["Q4 planning kickoff"]
        )
    """
    
    def __init__(self):
        """Initialize enricher with dependencies."""
        self.events_provider = WorldEventsProvider()
        self.encoder = TemporalEncoder()
    
    async def enrich(
        self,
        recording_date: datetime | str,
        meeting_type: str = "",
        business_events: list[str] | None = None,
        fetch_world_events: bool = True,
        market_context: dict[str, Any] | None = None,
    ) -> TemporalContext:
        """
        Create an enriched temporal context.
        
        Args:
            recording_date: When the recording was made
            meeting_type: Type of meeting (strategy, operations, etc.)
            business_events: Internal business events to include
            fetch_world_events: Whether to fetch external events
            market_context: Optional market data to include
            
        Returns:
            TemporalContext with all enrichments
        """
        if isinstance(recording_date, str):
            recording_date = datetime.fromisoformat(recording_date)
        
        # Create base context (derives date features)
        context = TemporalContext(
            recording_date=recording_date,
            meeting_type=meeting_type,
            business_events=business_events or [],
            market_context=market_context or {},
        )
        
        # Fetch world events if requested
        if fetch_world_events:
            try:
                events = await self.events_provider.fetch_events(recording_date)
                
                # Extract event titles and categories
                context.notable_events = [e["title"] for e in events[:10]]
                context.event_categories = list(set(e["category"] for e in events))
                
            except Exception as e:
                logger.warning(f"Failed to fetch world events: {e}")
        
        return context
    
    def encode_context(self, context: TemporalContext) -> np.ndarray:
        """
        Encode temporal context as a dense feature vector.
        
        Args:
            context: TemporalContext object
            
        Returns:
            numpy array suitable for embedding concatenation
        """
        return self.encoder.encode(context)


# Singleton instance for service use
_enricher: TemporalContextEnricher | None = None


def get_temporal_enricher() -> TemporalContextEnricher:
    """Get or create the temporal context enricher singleton."""
    global _enricher
    if _enricher is None:
        _enricher = TemporalContextEnricher()
    return _enricher
