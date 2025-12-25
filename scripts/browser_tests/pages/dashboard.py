"""
Dashboard Page Tests
====================
Tests for index.html - main dashboard.
"""

from typing import List
from ..base import BasePageTest, BrowserTest


class DashboardTests(BasePageTest):
    """Tests for index.html (main dashboard)"""

    @property
    def page_name(self) -> str:
        return "index.html"

    async def run_tests(self) -> List[BrowserTest]:
        """Run all dashboard tests"""
        tests = []

        # Page load test
        tests.append(await self.test_page_loads(
            name="Dashboard - Page Load"
        ))

        # Note: Refresh button test removed as selector uses
        # unsupported :has-text() pseudo-class

        return tests
