"""
Phase 1 Banking Tests - MSR Core Operations
============================================
Playwright tests for verifying Phase 1 implementation:
- Member Search Enhancement
- Member 360 Panel
- Recent Members
- Transaction Filters
- Quick Notes
- CSV Export

Dependencies: base.py, config.py
"""

import asyncio
from .base import BasePageTest, BrowserTest, TestResult


class BankingPhase1Tests(BasePageTest):
    """Phase 1: MSR Core Operations Tests"""

    def page_name(self) -> str:
        return "Banking Phase 1"

    async def run_tests(self, page, config) -> list:
        """Run all Phase 1 tests"""
        tests = []
        base_url = f"{config.base_url}/ui/banking.html"

        try:
            # Navigate to banking page
            await page.goto(base_url, wait_until="networkidle")
            await asyncio.sleep(1)

            # ============================================================
            # Test 1: Page loads with Member Search visible
            # ============================================================
            test1 = BrowserTest(
                name="Phase1 - Banking Page Load",
                passed=False,
                result=TestResult.PENDING,
                error=None
            )
            try:
                title_elem = await page.query_selector("h1.page-title")
                if title_elem:
                    text = await title_elem.text_content()
                    test1.passed = "Member Search" in text
                    test1.result = TestResult.PASS if test1.passed else TestResult.FAIL
                    test1.details = f"Title: {text}"
            except Exception as e:
                test1.error = str(e)
                test1.result = TestResult.FAIL
            tests.append(test1)

            # ============================================================
            # Test 2: Recent Members Panel Exists
            # ============================================================
            test2 = BrowserTest(
                name="Phase1 - Recent Members Panel",
                passed=False,
                result=TestResult.PENDING,
                error=None
            )
            try:
                panel = await page.query_selector("#recentMembersPanel")
                test2.passed = panel is not None
                test2.result = TestResult.PASS if test2.passed else TestResult.FAIL
                test2.details = "Panel found" if test2.passed else "Panel not found"
            except Exception as e:
                test2.error = str(e)
                test2.result = TestResult.FAIL
            tests.append(test2)

            # ============================================================
            # Test 3: Member Search Form
            # ============================================================
            test3 = BrowserTest(
                name="Phase1 - Member Search Form",
                passed=False,
                result=TestResult.PENDING,
                error=None
            )
            try:
                name_input = await page.query_selector("#partyName")
                phone_input = await page.query_selector("#partyPhone")
                email_input = await page.query_selector("#partyEmail")
                search_btn = await page.query_selector("button:has-text('Search Member')")
                
                test3.passed = all([name_input, phone_input, email_input, search_btn])
                test3.result = TestResult.PASS if test3.passed else TestResult.FAIL
                test3.details = "All search form elements present"
            except Exception as e:
                test3.error = str(e)
                test3.result = TestResult.FAIL
            tests.append(test3)

            # ============================================================
            # Test 4: Member Search Submission
            # ============================================================
            test4 = BrowserTest(
                name="Phase1 - Member Search Submit",
                passed=False,
                result=TestResult.PENDING,
                error=None
            )
            try:
                await page.fill("#partyName", "Smith")
                await page.click("button:has-text('Search Member')")
                await asyncio.sleep(2)  # Wait for API response
                
                results = await page.query_selector("#partyResults")
                test4.passed = results is not None
                test4.result = TestResult.PASS if test4.passed else TestResult.FAIL
                test4.details = "Search submission completed"
            except Exception as e:
                test4.error = str(e)
                test4.result = TestResult.FAIL
            tests.append(test4)

            # ============================================================
            # Test 5: Transactions Section Navigation
            # ============================================================
            test5 = BrowserTest(
                name="Phase1 - Transactions Navigation",
                passed=False,
                result=TestResult.PENDING,
                error=None
            )
            try:
                tx_nav = await page.query_selector(".nav-item:has-text('Transactions')")
                if tx_nav:
                    await tx_nav.click()
                    await asyncio.sleep(0.5)
                    
                    tx_section = await page.query_selector("#section-transactions")
                    is_visible = await tx_section.is_visible() if tx_section else False
                    test5.passed = is_visible
                    test5.result = TestResult.PASS if test5.passed else TestResult.FAIL
                    test5.details = "Transactions section visible"
            except Exception as e:
                test5.error = str(e)
                test5.result = TestResult.FAIL
            tests.append(test5)

            # ============================================================
            # Test 6: banking_member.js Module Loaded
            # ============================================================
            test6 = BrowserTest(
                name="Phase1 - Member JS Module Loaded",
                passed=False,
                result=TestResult.PENDING,
                error=None
            )
            try:
                # Check if MemberState global exists
                has_member_state = await page.evaluate("typeof window.MemberState !== 'undefined'")
                has_export_fn = await page.evaluate("typeof window.exportTransactionsCSV === 'function'")
                
                test6.passed = has_member_state and has_export_fn
                test6.result = TestResult.PASS if test6.passed else TestResult.FAIL
                test6.details = f"MemberState: {has_member_state}, exportCSV: {has_export_fn}"
            except Exception as e:
                test6.error = str(e)
                test6.result = TestResult.FAIL
            tests.append(test6)

            # ============================================================
            # Test 7: Recent Members Functions Available
            # ============================================================
            test7 = BrowserTest(
                name="Phase1 - Recent Members Functions",
                passed=False,
                result=TestResult.PENDING,
                error=None
            )
            try:
                has_add_fn = await page.evaluate("typeof window.addToRecentMembers === 'function'")
                has_render_fn = await page.evaluate("typeof window.renderRecentMembers === 'function'")
                
                test7.passed = has_add_fn and has_render_fn
                test7.result = TestResult.PASS if test7.passed else TestResult.FAIL
                test7.details = f"addToRecentMembers: {has_add_fn}, renderRecentMembers: {has_render_fn}"
            except Exception as e:
                test7.error = str(e)
                test7.result = TestResult.FAIL
            tests.append(test7)

            # ============================================================
            # Test 8: Notes Functions Available
            # ============================================================
            test8 = BrowserTest(
                name="Phase1 - Notes Functions",
                passed=False,
                result=TestResult.PENDING,
                error=None
            )
            try:
                has_get_notes = await page.evaluate("typeof window.getMemberNotes === 'function'")
                has_save_note = await page.evaluate("typeof window.saveMemberNote === 'function'")
                has_show_notes = await page.evaluate("typeof window.showMemberNotes === 'function'")
                
                test8.passed = all([has_get_notes, has_save_note, has_show_notes])
                test8.result = TestResult.PASS if test8.passed else TestResult.FAIL
                test8.details = f"All notes functions available"
            except Exception as e:
                test8.error = str(e)
                test8.result = TestResult.FAIL
            tests.append(test8)

        except Exception as e:
            # Create a failed test for any unexpected error
            tests.append(BrowserTest(
                name="Phase1 - Unexpected Error",
                passed=False,
                result=TestResult.FAIL,
                error=str(e)
            ))

        return tests


async def run_phase1_tests(config=None):
    """Run Phase 1 tests standalone"""
    from .config import Config
    
    if config is None:
        config = Config()
    
    tester = BankingPhase1Tests()
    
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        tests = await tester.run_tests(page, config)
        
        await browser.close()
        
        # Print results
        passed = sum(1 for t in tests if t.passed)
        total = len(tests)
        print(f"\n{'='*60}")
        print(f"Phase 1 Test Results: {passed}/{total} passed")
        print(f"{'='*60}")
        for t in tests:
            status = "✅" if t.passed else "❌"
            print(f"  {status} {t.name}")
            if t.error:
                print(f"      Error: {t.error}")
        
        return tests


if __name__ == "__main__":
    asyncio.run(run_phase1_tests())
