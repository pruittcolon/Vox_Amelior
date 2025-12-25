"""
Configuration for Browser Tests
================================
Centralized configuration for URLs, timeouts, and settings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Test configuration settings"""
    
    # Base URL for the application
    base_url: str = "http://localhost:8000"
    
    # UI path prefix
    ui_prefix: str = "/ui"
    
    # Timeouts (milliseconds)
    # Increased for heavy pages like predictions.html, financial-dashboard.html
    page_load_timeout: int = 45000
    network_idle_timeout: int = 30000
    element_timeout: int = 5000
    
    # Browser settings
    headless: bool = True
    slow_mo: int = 0  # Slow down operations (ms)
    
    # Retry settings
    max_retries: int = 2
    
    def page_url(self, page_name: str) -> str:
        """Build full URL for a page"""
        return f"{self.base_url}{self.ui_prefix}/{page_name}"


# Default configuration
DEFAULT_CONFIG = Config()
