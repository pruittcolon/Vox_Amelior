#!/usr/bin/env python3
"""
Run Salesforce Tests Only
Quick script to run just the Salesforce test suite.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from browser_tests.base import BasePageTest, TestResult
from browser_tests.config import DEFAULT_CONFIG
from browser_tests.pages.salesforce import SalesforceTests
from playwright.async_api import async_playwright


async def main():
    print("\n" + "=" * 70)
    print("🔷 SALESFORCE PLAYWRIGHT TEST SUITE")
    print("=" * 70)
    
    # Initialize Playwright
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True, slow_mo=100)
    context = await browser.new_context()
    
    try:
        # Create test instance
        test_instance = SalesforceTests(DEFAULT_CONFIG)
        test_instance.context = context
        
        # Run tests
        tests = await test_instance.run_tests()
        
        # Print results
        passed = 0
        failed = 0
        errors = 0
        
        for test in tests:
            if test.result == TestResult.PASS:
                icon = "✅"
                passed += 1
            elif test.result == TestResult.FAIL:
                icon = "❌"
                failed += 1
            elif test.result == TestResult.SKIP:
                icon = "⏭️"
            else:
                icon = "💥"
                errors += 1
            
            print(f"{icon} {test.result.value.upper():5} {test.name}")
            if test.message:
                print(f"         └─ {test.message[:80]}")
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        print(f"  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  💥 Errors: {errors}")
        print(f"  Total:    {len(tests)}")
        
        if failed == 0 and errors == 0:
            print("\n🎉 ALL SALESFORCE TESTS PASSED!")
            return 0
        else:
            print("\n⚠️  SOME TESTS FAILED")
            return 1
            
    finally:
        await context.close()
        await browser.close()
        await playwright.stop()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
