"""
Test Runner
============
Discovers and runs all page tests, aggregates results.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Type

from playwright.async_api import async_playwright

from .base import BasePageTest, BrowserTest, TestResult
from .config import Config, DEFAULT_CONFIG
from .pages import (
    BankingTests,
    GemmaTests,
    DashboardTests,
    PredictionsTests,
    AdminQATests,
    CommonPageTests,
    BankingEnterpriseTests,
    SalesforceTests,
)


class TestRunner:
    """
    Main test runner that discovers and executes all page tests.
    """

    # All test classes to run
    TEST_CLASSES: List[Type[BasePageTest]] = [
        BankingTests,
        BankingEnterpriseTests,
        GemmaTests,
        DashboardTests,
        PredictionsTests,
        AdminQATests,
        CommonPageTests,
        SalesforceTests,
    ]

    def __init__(self, config: Config = None):
        self.config = config or DEFAULT_CONFIG
        self.results: List[BrowserTest] = []
        self.start_time: float = 0
        self.end_time: float = 0

    async def run_all(self) -> List[BrowserTest]:
        """Run all test classes"""
        self.results = []
        self.start_time = time.time()

        print("\n" + "=" * 70)
        print("🌐 NEMO MODULAR BROWSER TEST SUITE")
        print("=" * 70)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔗 Base URL: {self.config.base_url}")
        print("=" * 70)

        # Initialize Playwright
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo
        )
        context = await browser.new_context()

        try:
            # Track progress across all tests
            current_test_num = 0

            for TestClass in self.TEST_CLASSES:
                # Create test instance
                test_instance = TestClass(self.config)
                test_instance.context = context  # Share browser context

                # Get display name
                class_name = TestClass.__name__.replace("Tests", "")
                print(f"\n📄 {class_name} Tests")
                print("-" * 50)

                # Run tests
                tests = await test_instance.run_tests()
                self.results.extend(tests)

                # Print results with progress
                for test in tests:
                    current_test_num += 1
                    self._print_result(test, current_test_num)

        finally:
            await context.close()
            await browser.close()
            await playwright.stop()

        self.end_time = time.time()
        return self.results

    def _print_result(self, test: BrowserTest, test_num: int):
        """Print a single test result with running progress"""
        icons = {
            "pass": "✅",
            "fail": "❌",
            "error": "💥",
            "skip": "⏭️",
        }
        icon = icons.get(test.result.value, "❓")
        status = test.result.value.upper()

        # Calculate running totals
        passed = sum(1 for t in self.results if t.result == TestResult.PASS)
        failed = sum(1 for t in self.results if t.result in (TestResult.FAIL, TestResult.ERROR))

        # Print progress line
        print(f"[{test_num}] {icon} {status:5} {test.name} ({test.duration_ms:.0f}ms)")
        print(f"       └─ Running: {passed} passed, {failed} failed")
        if test.message and test.result != TestResult.PASS:
            print(f"       └─ Error: {test.message}")

    def print_summary(self) -> bool:
        """Print final summary, return True if all passed"""
        passed = sum(1 for t in self.results if t.result == TestResult.PASS)
        failed = sum(1 for t in self.results if t.result == TestResult.FAIL)
        errors = sum(1 for t in self.results if t.result == TestResult.ERROR)
        skipped = sum(1 for t in self.results if t.result == TestResult.SKIP)
        total = len(self.results)
        duration = self.end_time - self.start_time

        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)
        print(f"  Total Tests:   {total}")
        print(f"  ✅ Passed:     {passed} ({passed/total*100:.0f}%)" if total > 0 else "")
        print(f"  ❌ Failed:     {failed}")
        print(f"  💥 Errors:     {errors}")
        print(f"  ⏭️  Skipped:    {skipped}")
        print(f"  ⏱️  Duration:   {duration:.1f}s")
        print("=" * 70)

        if failed == 0 and errors == 0:
            print("🎉 ALL TESTS PASSED!")
            return True
        else:
            print("⚠️  SOME TESTS FAILED")
            return False

    def generate_report(self, output_path: str = None) -> Dict[str, Any]:
        """Generate JSON report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "base_url": self.config.base_url,
                "headless": self.config.headless,
            },
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for t in self.results if t.result == TestResult.PASS),
                "failed": sum(1 for t in self.results if t.result == TestResult.FAIL),
                "errors": sum(1 for t in self.results if t.result == TestResult.ERROR),
                "skipped": sum(1 for t in self.results if t.result == TestResult.SKIP),
                "duration_seconds": self.end_time - self.start_time,
            },
            "tests": [
                {
                    "name": t.name,
                    "page": t.page,
                    "result": t.result.value,
                    "message": t.message,
                    "duration_ms": t.duration_ms,
                    "js_errors": t.js_errors,
                }
                for t in self.results
            ]
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n📁 Report saved: {output_path}")

        return report
