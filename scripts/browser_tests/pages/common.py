"""
Common Page Tests
=================
Load tests for all other pages that don't need feature-specific testing.
"""

import asyncio
import time
from typing import List
from ..base import BasePageTest, BrowserTest, TestResult


class CommonPageTests(BasePageTest):
    """Tests for all other pages - just page load verification"""

    # List of all pages to test with display names
    # Format: (filename, display_name, timeout_override)
    # Comprehensive list - all pages should be tested
    PAGES = [
        ("analytics.html", "Analytics", None),
        ("meetings.html", "Meetings", None),
        ("transcripts.html", "Transcripts", None),
        ("databases.html", "Databases", None),
        ("automation.html", "Automation", None),
        ("knowledge.html", "Knowledge", None),
        ("salesforce.html", "Salesforce", None),
        ("emotions.html", "Emotions", None),
        ("memories.html", "Memories", None),
        ("settings.html", "Settings", None),
        ("login.html", "Login", None),
        ("chat.html", "Chat", None),
        ("chatbot.html", "Chatbot", None),
        ("speakers.html", "Speakers", None),
        ("ml_dashboard.html", "ML Dashboard", None),
        ("gemma_data.html", "Gemma Data", None),
        ("analysis.html", "Analysis", None),
        ("database_analysis.html", "Database Analysis", None),
        ("financial-dashboard.html", "Financial Dashboard", 30000),  # Reduced timeout
        ("about.html", "About", None),
        ("banking.html", "Banking", None),  # Ensure banking is tested
    ]

    @property
    def page_name(self) -> str:
        """Not used directly - we test multiple pages"""
        return "common"

    async def run_tests(self) -> List[BrowserTest]:
        """Run page load tests for all common pages"""
        tests = []

        for page_tuple in self.PAGES:
            filename, display_name, timeout = page_tuple
            tests.append(await self._test_single_page(filename, display_name, timeout))

        return tests

    async def _test_single_page(
        self, 
        page_file: str, 
        display_name: str,
        timeout_override: int = None
    ) -> BrowserTest:
        """Test a single page load"""
        test = BrowserTest(name=f"{display_name} - Load", page=page_file)
        start = time.time()

        url = self.config.page_url(page_file)
        timeout = timeout_override or self.config.page_load_timeout
        network_timeout = (timeout_override or self.config.network_idle_timeout) + 15000

        try:
            page = await self.context.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            response = await page.goto(url, timeout=timeout)
            await page.wait_for_load_state("networkidle", timeout=network_timeout)
            await asyncio.sleep(0.3)

            test.js_errors = errors

            if errors:
                test.result = TestResult.FAIL
                test.message = f"JS Error: {errors[0][:60]}"
            elif response and response.status >= 400:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response.status}"
            else:
                test.result = TestResult.PASS
                test.message = "No JS errors"

            await page.close()
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]

        test.duration_ms = (time.time() - start) * 1000
        return test
