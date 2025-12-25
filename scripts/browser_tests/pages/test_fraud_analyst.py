"""
Fraud Analyst Workstation Tests
===============================
Authenticated Playwright tests for the AI-powered Fraud Analyst Workstation.
Tests navigation, alert queue, transaction analysis, investigation panel, and SAR filing.

Author: Antigravity AI
Version: 1.0.0
"""

import asyncio
from playwright.async_api import async_playwright


class FraudAnalystTests:
    """Comprehensive tests for Fraud Analyst Workstation"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000/ui"
        self.results = []
        
    async def login(self, page):
        """Login with credentials"""
        await page.goto(f"{self.base_url}/login.html", timeout=15000)
        await asyncio.sleep(1)
        await page.fill('#username', 'pruittcolon')
        await page.fill('#password', 'Pruitt12!')
        await page.click('#login-button')
        await asyncio.sleep(3)
        return True
        
    async def navigate_to_fraud_analyst(self, page):
        """Navigate to Fraud Analyst section"""
        await page.goto(f"{self.base_url}/banking.html", wait_until="networkidle", timeout=15000)
        await asyncio.sleep(2)
        
        # Click Fraud Analyst role tab
        await page.evaluate("""
            document.querySelector('[data-role="fraud_analyst"]').click();
        """)
        await asyncio.sleep(1)
        return True
    
    async def test_navigation_works(self, page):
        """Test: Clicking Fraud Analyst shows correct section"""
        try:
            # Check page title
            title = await page.evaluate("document.querySelector('#section-anomaly h1')?.textContent || 'Not found'")
            assert 'Fraud Analyst' in title, f"Expected Fraud Analyst title, got: {title}"
            
            self.results.append(('Navigation to Fraud Analyst', True, title))
            return True
        except Exception as e:
            self.results.append(('Navigation to Fraud Analyst', False, str(e)))
            return False
    
    async def test_metrics_dashboard(self, page):
        """Test: Fraud metrics dashboard shows all cards"""
        try:
            # Check critical alerts card
            critical = await page.evaluate("document.getElementById('fraudCriticalCount')?.textContent || '-'")
            assert critical != '-', "Critical alerts not displayed"
            
            # Check high priority card
            high = await page.evaluate("document.getElementById('fraudHighCount')?.textContent || '-'")
            assert high != '-', "High priority not displayed"
            
            # Check resolved today
            resolved = await page.evaluate("document.getElementById('fraudResolvedCount')?.textContent || '-'")
            assert resolved != '-', "Resolved count not displayed"
            
            # Check SAR count
            sar = await page.evaluate("document.getElementById('fraudSARCount')?.textContent || '-'")
            assert sar != '-', "SAR count not displayed"
            
            self.results.append(('Metrics Dashboard', True, f"Critical:{critical} High:{high} Resolved:{resolved} SARs:{sar}"))
            return True
        except Exception as e:
            self.results.append(('Metrics Dashboard', False, str(e)))
            return False
    
    async def test_tab_navigation(self, page):
        """Test: All 5 tabs work correctly"""
        try:
            tabs = ['queue', 'analyze', 'investigate', 'sar', 'trends']
            for tab in tabs:
                await page.evaluate(f"showFraudTab('{tab}')")
                await asyncio.sleep(0.3)
                
                is_visible = await page.evaluate(f"document.getElementById('fraud-tab-{tab}')?.style.display !== 'none'")
                assert is_visible, f"Tab {tab} not visible"
            
            self.results.append(('Tab Navigation (5 tabs)', True, ''))
            return True
        except Exception as e:
            self.results.append(('Tab Navigation (5 tabs)', False, str(e)))
            return False
    
    async def test_alert_queue(self, page):
        """Test: Alert queue displays alerts"""
        try:
            await page.evaluate("showFraudTab('queue')")
            await asyncio.sleep(0.3)
            
            # Check for alert items
            alert_count = await page.evaluate("document.querySelectorAll('.alert-item').length")
            assert alert_count >= 1, f"Expected alerts in queue, got {alert_count}"
            
            # Check for critical indicator
            has_critical = await page.evaluate("document.querySelector('.alert-item')?.innerHTML.includes('CRITICAL') || false")
            
            self.results.append(('Alert Queue', True, f"Found {alert_count} alerts"))
            return True
        except Exception as e:
            self.results.append(('Alert Queue', False, str(e)))
            return False
    
    async def test_transaction_analysis(self, page):
        """Test: Transaction analysis calculates fraud score"""
        try:
            await page.evaluate("showFraudTab('analyze')")
            await asyncio.sleep(0.3)
            
            # Fill form
            await page.fill('#anomalyAmount', '50000')
            await page.fill('#anomalyAvg', '500')
            await page.select_option('#anomalyPayee', 'new')
            
            # Run analysis
            await page.evaluate("runFraudAnalysis()")
            await asyncio.sleep(0.5)
            
            # Check results
            results_html = await page.evaluate("document.getElementById('fraudScoreResults')?.innerHTML || ''")
            has_score = 'RISK' in results_html
            assert has_score, "Fraud score not calculated"
            
            self.results.append(('Transaction Analysis', True, ''))
            return True
        except Exception as e:
            self.results.append(('Transaction Analysis', False, str(e)))
            return False
    
    async def test_investigation_panel(self, page):
        """Test: Investigation panel loads with member context"""
        try:
            # Click on an alert to open investigation
            await page.evaluate("openInvestigation('ALT-001')")
            await asyncio.sleep(0.5)
            
            # Check investigation panel has content
            panel_html = await page.evaluate("document.getElementById('investigationPanel')?.innerHTML || ''")
            has_member = 'Member Profile' in panel_html
            has_timeline = 'Activity Timeline' in panel_html
            
            assert has_member, "Member profile not shown"
            assert has_timeline, "Activity timeline not shown"
            
            self.results.append(('Investigation Panel', True, ''))
            return True
        except Exception as e:
            self.results.append(('Investigation Panel', False, str(e)))
            return False
    
    async def test_sar_filing_form(self, page):
        """Test: SAR filing form is present and functional"""
        try:
            await page.evaluate("showFraudTab('sar')")
            await asyncio.sleep(0.3)
            
            # Check form fields exist
            subject_name = await page.query_selector('#sarSubjectName')
            assert subject_name is not None, "Subject name field not found"
            
            activity_type = await page.query_selector('#sarActivityType')
            assert activity_type is not None, "Activity type field not found"
            
            narrative = await page.query_selector('#sarNarrative')
            assert narrative is not None, "Narrative field not found"
            
            # Check buttons
            submit_btn = await page.evaluate("document.querySelector('button[onclick*=\"submitSAR\"]') !== null")
            assert submit_btn, "Submit SAR button not found"
            
            self.results.append(('SAR Filing Form', True, ''))
            return True
        except Exception as e:
            self.results.append(('SAR Filing Form', False, str(e)))
            return False
    
    async def test_portfolio_trends(self, page):
        """Test: Portfolio trends tab shows population analytics"""
        try:
            await page.evaluate("showFraudTab('trends')")
            await asyncio.sleep(0.3)
            
            # Check for trends content
            trends_html = await page.evaluate("document.getElementById('fraud-tab-trends')?.innerHTML || ''")
            
            has_trends = 'Fraud Trends' in trends_html or 'Wire Transfer' in trends_html
            has_distribution = 'Risk Distribution' in trends_html
            has_threats = 'Emerging Threats' in trends_html
            
            assert has_trends, "Population trends not displayed"
            assert has_distribution, "Risk distribution not displayed"
            
            self.results.append(('Portfolio Trends', True, ''))
            return True
        except Exception as e:
            self.results.append(('Portfolio Trends', False, str(e)))
            return False
    
    async def run_all_tests(self):
        """Run all Fraud Analyst tests"""
        print("="*60)
        print("🔍 FRAUD ANALYST WORKSTATION - AUTHENTICATED TESTS")
        print("="*60)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # Login
                print("\n📝 Logging in...")
                await self.login(page)
                
                # Navigate to Fraud Analyst section
                print("🔀 Navigating to Fraud Analyst section...")
                await self.navigate_to_fraud_analyst(page)
                
                # Run tests
                print("\n🧪 Running tests...\n")
                
                await self.test_navigation_works(page)
                await self.test_metrics_dashboard(page)
                await self.test_tab_navigation(page)
                await self.test_alert_queue(page)
                await self.test_transaction_analysis(page)
                await self.test_investigation_panel(page)
                await self.test_sar_filing_form(page)
                await self.test_portfolio_trends(page)
                
            except Exception as e:
                print(f"❌ Test suite error: {e}")
            finally:
                await browser.close()
        
        # Print results
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        
        passed = 0
        failed = 0
        for test_name, success, detail in self.results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} | {test_name}")
            if detail and success:
                print(f"       └── {detail}")
            elif detail and not success:
                print(f"       └── ERROR: {detail}")
            if success:
                passed += 1
            else:
                failed += 1
        
        print("\n" + "-"*60)
        print(f"TOTAL: {passed} passed, {failed} failed out of {len(self.results)} tests")
        print("="*60)
        
        return passed, failed


async def main():
    """Main entry point"""
    tester = FraudAnalystTests()
    passed, failed = await tester.run_all_tests()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
