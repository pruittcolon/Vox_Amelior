"""
Admin QA Page Tests
===================
Tests for admin_qa.html - QA administration dashboard.
"""

from typing import List
from ..base import BasePageTest, BrowserTest


class AdminQATests(BasePageTest):
    """Tests for admin_qa.html"""

    @property
    def page_name(self) -> str:
        return "admin_qa.html"

    async def run_tests(self) -> List[BrowserTest]:
        """Run all admin QA tests"""
        tests = []

        # Page load test
        tests.append(await self.test_page_loads(
            name="Admin QA - Page Load"
        ))

        # Note: Refresh button test removed as selector uses
        # unsupported :has-text() pseudo-class

        return tests
