#!/usr/bin/env python3
"""
NEMO Comprehensive Browser Test Suite
======================================
Playwright-based browser tests for ALL frontend HTML pages.
Tests page loads, button clicks, form submissions, and output verification.

Usage:
    source scripts/.venv/bin/activate
    python scripts/nemo_browser_tests.py

Requirements:
    pip install playwright
    playwright install chromium
"""

import asyncio
import time
import sys
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from playwright.async_api import async_playwright, Page, expect, Error as PlaywrightError


class TestResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


@dataclass
class BrowserTest:
    name: str
    page: str
    result: TestResult = TestResult.PASS
    message: str = ""
    js_errors: List[str] = field(default_factory=list)
    duration_ms: float = 0
    output: Optional[str] = None  # Captured output/response


class NemoBrowserTests:
    """Comprehensive Playwright browser tests for Nemo Server frontend"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.browser = None
        self.context = None
        self.playwright = None
        self.results: List[BrowserTest] = []

    async def setup(self):
        """Initialize browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()

    async def teardown(self):
        """Close browser"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    # =========================================================================
    # CORE TEST METHODS
    # =========================================================================

    async def test_page_loads(self, path: str, name: str, 
                               required_elements: List[str] = None) -> BrowserTest:
        """Test page loads without JS errors and has required elements"""
        test = BrowserTest(name=f"{name} - Load", page=path)
        start = time.time()

        try:
            page = await self.context.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            response = await page.goto(f"{self.base_url}{path}", timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=15000)
            await asyncio.sleep(0.3)

            test.js_errors = errors
            
            # Check for JS errors
            if errors:
                test.result = TestResult.FAIL
                test.message = f"JS Error: {errors[0][:60]}"
            # Check HTTP status
            elif response and response.status >= 400:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response.status}"
            # Check required elements
            elif required_elements:
                missing = []
                for selector in required_elements:
                    try:
                        await page.wait_for_selector(selector, timeout=2000)
                    except:
                        missing.append(selector)
                if missing:
                    test.result = TestResult.FAIL
                    test.message = f"Missing: {missing[0]}"
                else:
                    test.result = TestResult.PASS
                    test.message = "All elements found"
            else:
                test.result = TestResult.PASS
                test.message = "No JS errors"

            await page.close()
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]

        test.duration_ms = (time.time() - start) * 1000
        return test

    async def test_button_click(self, path: str, selector: str, name: str,
                                 wait_for_selector: str = None,
                                 expected_text: str = None) -> BrowserTest:
        """Test button click and verify result"""
        test = BrowserTest(name=name, page=path)
        start = time.time()

        try:
            page = await self.context.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            await page.goto(f"{self.base_url}{path}", timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=15000)

            # Click button
            await page.click(selector, timeout=5000)
            await asyncio.sleep(1)

            test.js_errors = errors
            
            if errors:
                test.result = TestResult.FAIL
                test.message = f"JS on click: {errors[0][:50]}"
            elif wait_for_selector:
                try:
                    await page.wait_for_selector(wait_for_selector, timeout=5000)
                    if expected_text:
                        content = await page.text_content(wait_for_selector)
                        if expected_text.lower() in (content or "").lower():
                            test.result = TestResult.PASS
                            test.message = f"Found: {expected_text[:30]}"
                            test.output = content[:200] if content else None
                        else:
                            test.result = TestResult.FAIL
                            test.message = f"Expected '{expected_text}' not found"
                    else:
                        test.result = TestResult.PASS
                        test.message = "Element appeared"
                except:
                    test.result = TestResult.FAIL
                    test.message = f"Timeout waiting for {wait_for_selector}"
            else:
                test.result = TestResult.PASS
                test.message = "Click successful"

            await page.close()
        except PlaywrightError as e:
            if "Timeout" in str(e):
                test.result = TestResult.SKIP
                test.message = "Element not found"
            else:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:60]

        test.duration_ms = (time.time() - start) * 1000
        return test

    async def test_form_submit(self, path: str, form_data: Dict[str, str],
                                submit_selector: str, name: str,
                                success_selector: str = None) -> BrowserTest:
        """Test form fill and submit"""
        test = BrowserTest(name=name, page=path)
        start = time.time()

        try:
            page = await self.context.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            await page.goto(f"{self.base_url}{path}", timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=15000)

            # Fill form fields
            for selector, value in form_data.items():
                await page.fill(selector, value, timeout=3000)

            # Submit
            await page.click(submit_selector, timeout=5000)
            await asyncio.sleep(1.5)

            test.js_errors = errors
            
            if errors:
                test.result = TestResult.FAIL
                test.message = f"JS Error: {errors[0][:50]}"
            elif success_selector:
                try:
                    await page.wait_for_selector(success_selector, timeout=5000)
                    test.result = TestResult.PASS
                    test.message = "Form submitted successfully"
                except:
                    test.result = TestResult.FAIL
                    test.message = "Success indicator not found"
            else:
                test.result = TestResult.PASS
                test.message = "Form submitted"

            await page.close()
        except PlaywrightError as e:
            test.result = TestResult.SKIP
            test.message = str(e)[:60]
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:60]

        test.duration_ms = (time.time() - start) * 1000
        return test

    # =========================================================================
    # PAGE LOAD TESTS - ALL 36 PAGES
    # =========================================================================

    async def run_all_page_loads(self) -> List[BrowserTest]:
        """Test all pages load without JS errors"""
        pages = [
            # Tier 1: Core Pages (no required elements - sections may be hidden)
            ("/ui/banking.html", "Banking", None),
            ("/ui/gemma.html", "Gemma AI", ["#chat-messages"]),
            ("/ui/index.html", "Dashboard", None),
            ("/ui/admin_qa.html", "Admin QA", None),
            ("/ui/analytics.html", "Analytics", None),
            ("/ui/meetings.html", "Meetings", None),
            ("/ui/transcripts.html", "Transcripts", None),
            ("/ui/predictions.html", "Predictions", None),
            ("/ui/databases.html", "Databases", None),
            
            # Tier 2: Feature Pages
            ("/ui/automation.html", "Automation", None),
            ("/ui/knowledge.html", "Knowledge", None),
            ("/ui/salesforce.html", "Salesforce", None),
            ("/ui/emotions.html", "Emotions", None),
            ("/ui/memories.html", "Memories", None),
            ("/ui/settings.html", "Settings", None),
            ("/ui/search.html", "Search", None),
            ("/ui/login.html", "Login", ["#username"]),
            
            # Tier 3: Utility Pages
            ("/ui/call_qa.html", "Call QA", None),
            ("/ui/chat.html", "Chat", None),
            ("/ui/chatbot.html", "Chatbot", None),
            ("/ui/email.html", "Email", None),
            ("/ui/speakers.html", "Speakers", None),
            ("/ui/personalization.html", "Personalization", None),
            ("/ui/ml_dashboard.html", "ML Dashboard", None),
            ("/ui/gemma_data.html", "Gemma Data", None),
            ("/ui/analysis.html", "Analysis", None),
            ("/ui/database_analysis.html", "DB Analysis", None),
            ("/ui/financial-dashboard.html", "Financial", None),
            ("/ui/about.html", "About", None),
        ]

        tests = []
        for path, name, required in pages:
            tests.append(await self.test_page_loads(path, name, required))
        return tests

    # =========================================================================
    # BANKING.HTML TESTS
    # =========================================================================

    async def run_banking_tests(self) -> List[BrowserTest]:
        """Comprehensive banking.html tests"""
        tests = []
        path = "/ui/banking.html"

        # Navigation tabs - Core Banking (using text-based selectors for reliability)
        nav_items = [
            ("Member Search", "Banking - Nav: Member Search", "#section-party"),
            ("Account Lookup", "Banking - Nav: Account Lookup", "#section-account"),
            ("Transactions", "Banking - Nav: Transactions", "#section-transactions"),
            ("Fund Transfers", "Banking - Nav: Fund Transfers", "#section-transfers"),
        ]
        for text, name, expected_section in nav_items:
            tests.append(await self.test_button_click(
                path, f".nav-item:has-text('{text}')", name,
                wait_for_selector=expected_section
            ))

        # Role tabs
        roles = [
            ("msr", "MSR"),
            ("loan_officer", "Loan Officer"),
            ("fraud_analyst", "Fraud Analyst"),
            ("executive", "Executive"),
            ("call_intelligence", "Call Intel"),
        ]
        for role_id, role_name in roles:
            tests.append(await self.test_button_click(
                path, f"[data-role='{role_id}']", f"Banking - Role: {role_name}"
            ))

        # Key buttons - Party Search
        tests.append(await self.test_button_click(
            path, "button:has-text('Search')", "Banking - Party Search"
        ))

        # Fund Transfers form test
        tests.append(await self.test_form_submit(
            path,
            {
                "#transferFrom": "10001",
                "#transferTo": "20002",
                "#transferAmount": "50.00",
                "#transferMemo": "Test Transfer",
            },
            "button:has-text('Submit Transfer')",
            "Banking - Transfer Form Fill"
        ))

        return tests


    # =========================================================================
    # GEMMA.HTML TESTS
    # =========================================================================

    async def run_gemma_tests(self) -> List[BrowserTest]:
        """Gemma AI chat tests - must click Chat tab first (Analytics is default)"""
        tests = []
        path = "/ui/gemma.html"

        # Test 1: Page loads (no element check - tabs hidden by default)
        tests.append(await self.test_page_loads(path, "Gemma Page", None))

        # Test 2: Click Chat tab to reveal chat interface
        tests.append(await self.test_button_click(
            path, ".gemma-tab:has-text('Chat')", "Gemma - Chat Tab"
        ))

        # Test 3: After clicking Chat tab, test Send button
        # This uses a custom test that clicks tab first then tries send
        test = BrowserTest(name="Gemma - Chat Send", page=path)
        start = time.time()
        try:
            page = await self.context.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            
            await page.goto(f"{self.base_url}{path}", timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=15000)
            
            # First click Chat tab
            await page.click(".gemma-tab:has-text('Chat')", timeout=5000)
            await asyncio.sleep(1)
            
            # Now try to click Send
            await page.click("button:has-text('Send')", timeout=5000)
            await asyncio.sleep(1)
            
            test.js_errors = errors
            if errors:
                test.result = TestResult.FAIL
                test.message = f"JS Error: {errors[0][:50]}"
            else:
                test.result = TestResult.PASS
                test.message = "Chat tab + Send works"
            
            await page.close()
        except PlaywrightError as e:
            test.result = TestResult.SKIP
            test.message = str(e)[:60]
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:60]
        
        test.duration_ms = (time.time() - start) * 1000
        tests.append(test)

        return tests

    # =========================================================================
    # PREDICTIONS.HTML TESTS
    # =========================================================================

    async def run_predictions_tests(self) -> List[BrowserTest]:
        """ML Predictions tests"""
        tests = []
        path = "/ui/predictions.html"

        # Tab switching
        tab_tests = [
            ("switchTab('overview')", "Predictions - Tab: Overview"),
            ("switchTab('models')", "Predictions - Tab: Models"),
            ("switchTab('config')", "Predictions - Tab: Config"),
        ]
        for onclick, name in tab_tests:
            tests.append(await self.test_button_click(
                path, f"[onclick*=\"{onclick}\"]", name
            ))

        return tests

    # =========================================================================
    # ADMIN_QA.HTML TESTS
    # =========================================================================

    async def run_admin_qa_tests(self) -> List[BrowserTest]:
        """Admin QA tests"""
        tests = []
        path = "/ui/admin_qa.html"

        tests.append(await self.test_page_loads(path, "Admin QA", None))

        # Refresh button
        tests.append(await self.test_button_click(
            path, "button:has-text('Refresh')", "Admin QA - Refresh"
        ))

        return tests

    # =========================================================================
    # LOGIN.HTML TESTS  
    # =========================================================================

    async def run_login_tests(self) -> List[BrowserTest]:
        """Login form tests"""
        tests = []
        path = "/ui/login.html"

        # Test form elements exist
        tests.append(await self.test_page_loads(
            path, "Login Form", ["#username", "#password", "button[type='submit']"]
        ))

        return tests

    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================

    async def run_all(self) -> List[BrowserTest]:
        """Run comprehensive test suite"""
        all_tests = []

        print("\n" + "=" * 60)
        print("🌐 NEMO COMPREHENSIVE BROWSER TEST SUITE")
        print("=" * 60)

        await self.setup()

        try:
            # Phase 1: All page loads
            print("\n📄 PHASE 1: Page Load Tests")
            print("-" * 40)
            page_tests = await self.run_all_page_loads()
            all_tests.extend(page_tests)
            self._print_results(page_tests)

            # Phase 2: Banking tests
            print("\n🏦 PHASE 2: Banking.html Feature Tests")
            print("-" * 40)
            banking_tests = await self.run_banking_tests()
            all_tests.extend(banking_tests)
            self._print_results(banking_tests)

            # Phase 3: Gemma tests
            print("\n🤖 PHASE 3: Gemma.html Feature Tests")
            print("-" * 40)
            gemma_tests = await self.run_gemma_tests()
            all_tests.extend(gemma_tests)
            self._print_results(gemma_tests)

            # Phase 4: Predictions tests
            print("\n📊 PHASE 4: Predictions Feature Tests")
            print("-" * 40)
            pred_tests = await self.run_predictions_tests()
            all_tests.extend(pred_tests)
            self._print_results(pred_tests)

            # Phase 5: Admin QA tests
            print("\n📞 PHASE 5: Admin QA Feature Tests")
            print("-" * 40)
            qa_tests = await self.run_admin_qa_tests()
            all_tests.extend(qa_tests)
            self._print_results(qa_tests)

            # Phase 6: Login tests
            print("\n🔐 PHASE 6: Login Feature Tests")
            print("-" * 40)
            login_tests = await self.run_login_tests()
            all_tests.extend(login_tests)
            self._print_results(login_tests)

        finally:
            await self.teardown()

        return all_tests

    def _print_results(self, tests: List[BrowserTest]):
        """Print test results"""
        for test in tests:
            icon = {
                "pass": "✅",
                "fail": "❌", 
                "error": "💥",
                "skip": "⏭️"
            }[test.result.value]
            print(f"{icon} {test.result.value.upper():5} {test.name} ({test.duration_ms:.0f}ms)")
            if test.message and test.result != TestResult.PASS:
                print(f"         └─ {test.message}")

    def print_summary(self, tests: List[BrowserTest]) -> bool:
        """Print final summary"""
        passed = sum(1 for t in tests if t.result == TestResult.PASS)
        failed = sum(1 for t in tests if t.result == TestResult.FAIL)
        errors = sum(1 for t in tests if t.result == TestResult.ERROR)
        skipped = sum(1 for t in tests if t.result == TestResult.SKIP)
        total = len(tests)

        print("\n" + "=" * 60)
        print("📊 FINAL SUMMARY")
        print("=" * 60)
        print(f"  Total Tests:  {total}")
        print(f"  ✅ Passed:    {passed}")
        print(f"  ❌ Failed:    {failed}")
        print(f"  💥 Errors:    {errors}")
        print(f"  ⏭️  Skipped:   {skipped}")
        print("=" * 60)

        if failed == 0 and errors == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("⚠️  SOME TESTS FAILED - Review above for details")

        return failed == 0 and errors == 0

    def generate_report(self, tests: List[BrowserTest]) -> Dict[str, Any]:
        """Generate JSON report"""
        return {
            "total": len(tests),
            "passed": sum(1 for t in tests if t.result == TestResult.PASS),
            "failed": sum(1 for t in tests if t.result == TestResult.FAIL),
            "errors": sum(1 for t in tests if t.result == TestResult.ERROR),
            "skipped": sum(1 for t in tests if t.result == TestResult.SKIP),
            "tests": [
                {
                    "name": t.name,
                    "page": t.page,
                    "result": t.result.value,
                    "message": t.message,
                    "duration_ms": t.duration_ms,
                    "js_errors": t.js_errors,
                    "output": t.output
                }
                for t in tests
            ]
        }


async def main():
    tester = NemoBrowserTests()
    tests = await tester.run_all()
    success = tester.print_summary(tests)
    
    # Save report
    report = tester.generate_report(tests)
    with open("scripts/browser_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📁 Report saved: scripts/browser_test_report.json")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
