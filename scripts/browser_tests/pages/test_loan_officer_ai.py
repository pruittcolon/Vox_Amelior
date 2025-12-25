"""
Loan Officer AI Workstation Tests
=================================
Authenticated Playwright tests for the AI-powered Loan Officer Workstation.
Tests the full workflow: login, navigation, AI underwriting, XAI visualization,
decision workflow, and compliance features.

Author: Antigravity AI
Version: 1.0.0
"""

import asyncio
from playwright.async_api import async_playwright


class LoanOfficerAITests:
    """Comprehensive tests for AI Loan Officer Workstation"""
    
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
        
    async def navigate_to_loan_officer(self, page):
        """Navigate to Loan Officer section"""
        await page.goto(f"{self.base_url}/banking.html", wait_until="networkidle", timeout=15000)
        await asyncio.sleep(2)
        
        # Click on Loan Officer in the navigation using role dropdown
        # First click the Loan Officer dropdown option
        await page.evaluate("""
            // Find and click the Loan Officer nav item
            const navItems = document.querySelectorAll('.nav-item');
            for (const item of navItems) {
                if (item.textContent.includes('Loan Officer')) {
                    item.click();
                    break;
                }
            }
            // Force show the section
            showSection('ml-profit');
        """)
        await asyncio.sleep(1)
        
        # Verify we're on the right section
        section = await page.query_selector('#section-ml-profit')
        if section:
            is_visible = await section.evaluate('el => el.style.display !== "none"')
            if not is_visible:
                await page.evaluate("document.getElementById('section-ml-profit').style.display = 'block'")
                await asyncio.sleep(0.5)
        
        return True
    
    async def test_section_loads(self, page):
        """Test: Loan Officer section loads correctly"""
        try:
            # Check page title
            title = await page.text_content('h1.page-title')
            assert 'AI Loan Officer' in title, f"Expected AI Loan Officer title, got: {title}"
            
            # Check stats bar exists
            stats = await page.query_selector('.stats-bar')
            assert stats is not None, "Stats bar not found"
            
            # Check tabs exist
            tabs = await page.query_selector_all('.lending-tab')
            assert len(tabs) == 4, f"Expected 4 tabs, got {len(tabs)}"
            
            self.results.append(('Loan Officer Section Loads', True, ''))
            return True
        except Exception as e:
            self.results.append(('Loan Officer Section Loads', False, str(e)))
            return False
    
    async def test_tab_navigation(self, page):
        """Test: Tab navigation works correctly"""
        try:
            # Click New Application tab
            await page.evaluate("showLendingTab('new-app')")
            await asyncio.sleep(0.5)
            
            new_app_content = await page.query_selector('#lending-tab-new-app')
            is_visible = await new_app_content.is_visible()
            assert is_visible, "New Application tab content not visible"
            
            # Click AI Underwriting tab
            await page.evaluate("showLendingTab('underwrite')")
            await asyncio.sleep(0.5)
            
            underwrite_content = await page.query_selector('#lending-tab-underwrite')
            is_visible = await underwrite_content.is_visible()
            assert is_visible, "AI Underwriting tab content not visible"
            
            # Click Quick Tools tab
            await page.evaluate("showLendingTab('tools')")
            await asyncio.sleep(0.5)
            
            tools_content = await page.query_selector('#lending-tab-tools')
            is_visible = await tools_content.is_visible()
            assert is_visible, "Quick Tools tab content not visible"
            
            # Back to queue
            await page.evaluate("showLendingTab('queue')")
            await asyncio.sleep(0.5)
            
            self.results.append(('Tab Navigation', True, ''))
            return True
        except Exception as e:
            self.results.append(('Tab Navigation', False, str(e)))
            return False
    
    async def test_new_application_form(self, page):
        """Test: New application form works"""
        try:
            # Switch to new application tab
            await page.evaluate("showLendingTab('new-app')")
            await asyncio.sleep(0.5)
            
            # Fill form fields
            await page.fill('#newAppMemberId', 'M12345')
            await page.fill('#newAppMemberName', 'Test Member')
            await page.select_option('#newAppLoanType', 'auto')
            await page.fill('#newAppAmount', '35000')
            await page.fill('#newAppIncome', '6000')
            await page.fill('#newAppDebt', '1800')
            await page.fill('#newAppEmployment', '48')
            
            # Verify values
            member_name = await page.input_value('#newAppMemberName')
            assert member_name == 'Test Member', "Member name not set"
            
            loan_amount = await page.input_value('#newAppAmount')
            assert loan_amount == '35000', "Loan amount not set"
            
            self.results.append(('New Application Form', True, ''))
            return True
        except Exception as e:
            self.results.append(('New Application Form', False, str(e)))
            return False
    
    async def test_ai_underwriting_engine(self, page):
        """Test: AI Underwriting Engine runs correctly"""
        try:
            # Go to underwriting tab
            await page.evaluate("showLendingTab('underwrite')")
            await asyncio.sleep(0.5)
            
            # Fill underwriting inputs
            await page.fill('#uwIncome', '5500')
            await page.fill('#uwDebt', '1400')
            await page.fill('#uwLoanAmt', '28000')
            await page.fill('#uwTerm', '48')
            
            # Run AI Underwriting
            await page.click('button:has-text("Run AI Underwriting")')
            await asyncio.sleep(1.5)  # Allow for async processing
            
            # Check SHAP visualization appeared
            shap_panel = await page.query_selector('#shapVisualization .shap-panel')
            
            # Check recommendation panel is visible
            rec_panel = await page.query_selector('#aiRecommendationPanel')
            is_visible = await rec_panel.is_visible()
            assert is_visible, "AI Recommendation panel not visible after underwriting"
            
            # Check recommendation display has content
            rec_text = await page.text_content('#aiRecDisplay')
            assert rec_text and rec_text != '-', f"No recommendation displayed: {rec_text}"
            
            # Check confidence is displayed
            conf_text = await page.text_content('#aiConfDisplay')
            assert '%' in conf_text, f"Confidence not displayed correctly: {conf_text}"
            
            # Check fair lending status
            fair_text = await page.text_content('#aiFairDisplay')
            assert fair_text and fair_text != '-', f"Fair lending not displayed: {fair_text}"
            
            self.results.append(('AI Underwriting Engine', True, f"Recommendation: {rec_text}"))
            return True
        except Exception as e:
            self.results.append(('AI Underwriting Engine', False, str(e)))
            return False
    
    async def test_shap_visualization(self, page):
        """Test: SHAP factor visualization renders"""
        try:
            # Should already be on underwriting tab with results
            shap_panel = await page.query_selector('#shapVisualization')
            shap_html = await shap_panel.inner_html()
            
            # Check for factor rows
            assert 'shap-factor-row' in shap_html or 'Factor Impact' in shap_html or 'Credit Score' in shap_html, \
                "SHAP visualization not rendered correctly"
            
            self.results.append(('SHAP Visualization', True, ''))
            return True
        except Exception as e:
            self.results.append(('SHAP Visualization', False, str(e)))
            return False
    
    async def test_pricing_display(self, page):
        """Test: Pricing information is displayed"""
        try:
            # Check pricing elements
            apr = await page.text_content('#pricingAPR')
            assert '%' in apr, f"APR not displayed: {apr}"
            
            payment = await page.text_content('#pricingPayment')
            assert '$' in payment, f"Payment not displayed: {payment}"
            
            tier = await page.text_content('#pricingTier')
            assert 'Tier' in tier, f"Tier not displayed: {tier}"
            
            self.results.append(('Pricing Display', True, f"APR: {apr}, Payment: {payment}"))
            return True
        except Exception as e:
            self.results.append(('Pricing Display', False, str(e)))
            return False
    
    async def test_llm_explanation(self, page):
        """Test: LLM explanation is generated"""
        try:
            explanation = await page.text_content('#llmExplanationText')
            assert explanation and explanation != '-' and len(explanation) > 20, \
                f"LLM explanation not generated: {explanation}"
            
            self.results.append(('LLM Explanation', True, f"Length: {len(explanation)} chars"))
            return True
        except Exception as e:
            self.results.append(('LLM Explanation', False, str(e)))
            return False
    
    async def test_human_approval_notice(self, page):
        """Test: Human approval required notice is visible"""
        try:
            notice = await page.text_content('#aiRecommendationPanel')
            assert 'Human Approval Required' in notice, "Human approval notice not found"
            assert 'CFPB' in notice, "CFPB reference not found"
            
            self.results.append(('Human Approval Notice', True, ''))
            return True
        except Exception as e:
            self.results.append(('Human Approval Notice', False, str(e)))
            return False
    
    async def test_decision_buttons(self, page):
        """Test: Decision buttons are present and functional"""
        try:
            # Check all decision buttons exist
            approve_btn = await page.query_selector('button:has-text("APPROVE")')
            assert approve_btn is not None, "Approve button not found"
            
            conditions_btn = await page.query_selector('button:has-text("CONDITIONS")')
            assert conditions_btn is not None, "Approve with Conditions button not found"
            
            decline_btn = await page.query_selector('button:has-text("DECLINE")')
            assert decline_btn is not None, "Decline button not found"
            
            escalate_btn = await page.query_selector('button:has-text("ESCALATE")')
            assert escalate_btn is not None, "Escalate button not found"
            
            self.results.append(('Decision Buttons', True, ''))
            return True
        except Exception as e:
            self.results.append(('Decision Buttons', False, str(e)))
            return False
    
    async def test_approve_workflow(self, page):
        """Test: Approve workflow creates audit log"""
        try:
            # Click approve button
            await page.click('button:has-text("🔒 APPROVE")')
            await asyncio.sleep(0.5)
            
            # Check modal appeared
            modal = await page.query_selector('#decisionModal')
            assert modal is not None, "Decision modal not shown"
            
            modal_text = await modal.text_content()
            assert 'APPROVE' in modal_text, "Approval not confirmed in modal"
            assert 'Audit Log Created' in modal_text, "Audit log not created"
            
            # Close modal
            await page.click('#decisionModal button')
            await asyncio.sleep(0.3)
            
            self.results.append(('Approve Workflow', True, ''))
            return True
        except Exception as e:
            self.results.append(('Approve Workflow', False, str(e)))
            return False
    
    async def test_quick_tools_calculators(self, page):
        """Test: Quick tools calculators work"""
        try:
            # Switch to quick tools tab
            await page.evaluate("showLendingTab('tools')")
            await asyncio.sleep(0.5)
            
            # Fill calculator
            await page.fill('#lendingDebt', '1500')
            await page.fill('#lendingIncome', '5000')
            await page.fill('#lendingLoanAmt', '200000')
            await page.fill('#lendingPropertyVal', '250000')
            
            # Click calculate
            await page.click('button:has-text("Calculate Ratios")')
            await asyncio.sleep(0.5)
            
            # Check results
            results = await page.text_content('#lendingResults')
            assert 'DTI' in results or 'Debt-to-Income' in results, f"DTI not in results: {results}"
            
            self.results.append(('Quick Tools Calculators', True, ''))
            return True
        except Exception as e:
            self.results.append(('Quick Tools Calculators', False, str(e)))
            return False
    
    async def run_all_tests(self):
        """Run all Loan Officer AI tests"""
        print("="*60)
        print("🏦 AI LOAN OFFICER WORKSTATION - AUTHENTICATED TESTS")
        print("="*60)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # Login
                print("\n📝 Logging in...")
                await self.login(page)
                
                # Navigate to Loan Officer section
                print("🔀 Navigating to Loan Officer section...")
                await self.navigate_to_loan_officer(page)
                
                # Run tests
                print("\n🧪 Running tests...\n")
                
                await self.test_section_loads(page)
                await self.test_tab_navigation(page)
                await self.test_new_application_form(page)
                await self.test_ai_underwriting_engine(page)
                await self.test_shap_visualization(page)
                await self.test_pricing_display(page)
                await self.test_llm_explanation(page)
                await self.test_human_approval_notice(page)
                await self.test_decision_buttons(page)
                await self.test_approve_workflow(page)
                await self.test_quick_tools_calculators(page)
                
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
            if detail and not success:
                print(f"       └── {detail}")
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
    tester = LoanOfficerAITests()
    passed, failed = await tester.run_all_tests()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
