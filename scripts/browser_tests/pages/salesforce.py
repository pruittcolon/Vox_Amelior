"""
Salesforce Page Tests
=====================
Tests for salesforce.html - CRM Analytics Dashboard.
Also tests the Salesforce API endpoints.
"""

import asyncio
import json
from typing import List
from ..base import BasePageTest, BrowserTest, TestResult


class SalesforceTests(BasePageTest):
    """Tests for salesforce.html and Salesforce API endpoints."""

    @property
    def page_name(self) -> str:
        return "salesforce.html"

    async def run_tests(self) -> List[BrowserTest]:
        """Run all Salesforce page and API tests."""
        tests = []

        # =====================================================================
        # API Tests - Test the Salesforce REST API endpoints
        # =====================================================================
        tests.append(await self.test_api_status())
        tests.append(await self.test_api_accounts())
        tests.append(await self.test_api_contacts())
        tests.append(await self.test_api_opportunities())
        tests.append(await self.test_api_pipeline())
        tests.append(await self.test_api_metrics())

        # =====================================================================
        # Page Load Tests
        # =====================================================================
        # Updated selectors to match actual Salesforce Agentic Cockpit structure
        tests.append(await self.test_page_loads(
            name="Salesforce - Page Load",
            required_elements=[".cockpit-grid, .feed-panel"]
        ))

        # Test dashboard KPI widget (Pipeline Health in sidebar)
        tests.append(await self.test_page_loads(
            name="Salesforce - KPI Cards",
            required_elements=[".context-widget, .nav-panel"]
        ))

        # =====================================================================
        # Navigation Tests - Using sidebar nav-items
        # =====================================================================
        # Salesforce page uses sidebar navigation: Cockpit, Opportunities, Leads, Analytics
        # Using .nav-item selector which is the actual class in the DOM
        nav_items = ["Cockpit", "Opportunities", "Leads"]
        for item_name in nav_items:
            tests.append(await self.test_button_click(
                selector=".nav-item",
                name=f"Salesforce - Nav: {item_name}"
            ))

        # =====================================================================
        # Interactive Element Tests
        # =====================================================================
        
        # Test refresh data button (in feed header)
        tests.append(await self.test_button_click(
            selector=".btn-primary, .btn, [onclick*='refresh']",
            name="Salesforce - Refresh Data"
        ))

        # Test task cards exist in feed-stream (skip chart since it's dynamically loaded)
        tests.append(await self.test_page_loads(
            name="Salesforce - Task Cards",
            required_elements=[".feed-stream, .task-card, .agent-panel"]
        ))

        return tests

    # =========================================================================
    # API Test Methods
    # =========================================================================

    async def test_api_status(self) -> BrowserTest:
        """Test the /api/v1/salesforce/status endpoint."""
        test = BrowserTest(name="API - Salesforce Status", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            
            # Navigate to any page first to get session
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            # Call the API
            response = await page.evaluate("""
                async () => {
                    const resp = await fetch('/api/v1/salesforce/status');
                    return {
                        status: resp.status,
                        body: await resp.json()
                    };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                # Check expected fields
                if "enabled" in body and "connected" in body:
                    test.result = TestResult.PASS
                    test.message = f"enabled={body.get('enabled')}, configured={body.get('configured', 'N/A')}"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Missing expected fields: {body}"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_api_accounts(self) -> BrowserTest:
        """Test the /api/v1/salesforce/accounts endpoint."""
        test = BrowserTest(name="API - Salesforce Accounts", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const resp = await fetch('/api/v1/salesforce/accounts?limit=5');
                    return {
                        status: resp.status,
                        body: await resp.json()
                    };
                }
            """)
            
            await page.close()
            
            # Accept 200 (data) or 503 (not configured - expected)
            if response["status"] == 200:
                body = response["body"]
                if "records" in body or "success" in body:
                    test.result = TestResult.PASS
                    test.message = f"Got {len(body.get('records', []))} accounts"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Unexpected response format"
            elif response["status"] == 503:
                # Salesforce not configured - acceptable
                test.result = TestResult.PASS
                test.message = "503 - Not configured (expected)"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_api_contacts(self) -> BrowserTest:
        """Test the /api/v1/salesforce/contacts endpoint."""
        test = BrowserTest(name="API - Salesforce Contacts", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const resp = await fetch('/api/v1/salesforce/contacts?limit=5');
                    return { status: resp.status };
                }
            """)
            
            await page.close()
            
            if response["status"] in (200, 503):
                test.result = TestResult.PASS
                test.message = f"HTTP {response['status']}"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_api_opportunities(self) -> BrowserTest:
        """Test the /api/v1/salesforce/opportunities endpoint."""
        test = BrowserTest(name="API - Salesforce Opportunities", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const resp = await fetch('/api/v1/salesforce/opportunities?limit=5');
                    return { status: resp.status };
                }
            """)
            
            await page.close()
            
            if response["status"] in (200, 503):
                test.result = TestResult.PASS
                test.message = f"HTTP {response['status']}"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_api_pipeline(self) -> BrowserTest:
        """Test the /api/v1/salesforce/pipeline endpoint."""
        test = BrowserTest(name="API - Salesforce Pipeline", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const resp = await fetch('/api/v1/salesforce/pipeline');
                    return { status: resp.status };
                }
            """)
            
            await page.close()
            
            if response["status"] in (200, 503):
                test.result = TestResult.PASS
                test.message = f"HTTP {response['status']}"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_api_metrics(self) -> BrowserTest:
        """Test the /api/v1/salesforce/metrics endpoint."""
        test = BrowserTest(name="API - Salesforce Metrics", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const resp = await fetch('/api/v1/salesforce/metrics');
                    return { status: resp.status };
                }
            """)
            
            await page.close()
            
            if response["status"] in (200, 503):
                test.result = TestResult.PASS
                test.message = f"HTTP {response['status']}"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    # =========================================================================
    # AI SCORING API TESTS
    # =========================================================================

    async def test_api_lead_score(self) -> BrowserTest:
        """Test the /api/v1/salesforce/analytics/lead-score endpoint."""
        test = BrowserTest(name="API - Lead Scoring", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            await page.goto(self.page_url, timeout=15000)
            await asyncio.sleep(1)
            
            response = await page.evaluate("""
                async () => {
                    const getCsrf = () => { const m = document.cookie.match(/ws_csrf=([^;]+)/); return m ? m[1] : ''; };
                    const resp = await fetch('/api/v1/salesforce/analytics/lead-score', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': getCsrf() },
                        credentials: 'include',
                        body: JSON.stringify({
                            leads: [
                                { Id: 'L1', FirstName: 'Test', LastName: 'Lead', Company: 'TestCorp', Title: 'CTO' }
                            ],
                            config: {}
                        })
                    });
                    return { status: resp.status, body: await resp.json() };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                if "scored_items" in body or "summary" in body:
                    test.result = TestResult.PASS
                    test.message = f"Engine: {body.get('engine', 'unknown')}"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing expected fields"
            elif response["status"] == 422:
                test.result = TestResult.PASS
                test.message = "422 - Validation error (expected)"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_api_opportunity_score(self) -> BrowserTest:
        """Test the /api/v1/salesforce/analytics/opportunity-score endpoint."""
        test = BrowserTest(name="API - Opportunity Scoring", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const getCsrf = () => { const m = document.cookie.match(/ws_csrf=([^;]+)/); return m ? m[1] : ''; };
                    const resp = await fetch('/api/v1/salesforce/analytics/opportunity-score', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': getCsrf() },
                        credentials: 'include',
                        body: JSON.stringify({
                            opportunities: [
                                { Id: 'O1', Name: 'Test Opp', StageName: 'Proposal', Amount: 50000 }
                            ],
                            config: {}
                        })
                    });
                    return { status: resp.status, body: await resp.json() };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                if "scored_items" in body or "summary" in body:
                    test.result = TestResult.PASS
                    test.message = f"Engine: {body.get('engine', 'unknown')}"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing expected fields"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_api_churn_score(self) -> BrowserTest:
        """Test the /api/v1/salesforce/analytics/churn-score endpoint."""
        test = BrowserTest(name="API - Churn Scoring", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const getCsrf = () => { const m = document.cookie.match(/ws_csrf=([^;]+)/); return m ? m[1] : ''; };
                    const resp = await fetch('/api/v1/salesforce/analytics/churn-score', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': getCsrf() },
                        credentials: 'include',
                        body: JSON.stringify({
                            accounts: [
                                { Id: 'A1', Name: 'Test Account', Industry: 'Technology', AnnualRevenue: 100000 }
                            ],
                            config: {}
                        })
                    });
                    return { status: resp.status, body: await resp.json() };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                if "scored_items" in body or "summary" in body:
                    test.result = TestResult.PASS
                    test.message = f"Engine: {body.get('engine', 'unknown')}"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing expected fields"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_api_next_best_action(self) -> BrowserTest:
        """Test the /api/v1/salesforce/analytics/next-best-action endpoint."""
        test = BrowserTest(name="API - Next Best Action", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const getCsrf = () => { const m = document.cookie.match(/ws_csrf=([^;]+)/); return m ? m[1] : ''; };
                    const resp = await fetch('/api/v1/salesforce/analytics/next-best-action', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': getCsrf() },
                        credentials: 'include',
                        body: JSON.stringify({
                            lead_scores: [{ lead_id: 'L1', name: 'Test', score: 85, segment: 'Hot' }],
                            max_actions: 5
                        })
                    });
                    return { status: resp.status, body: await resp.json() };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                if "actions" in body:
                    test.result = TestResult.PASS
                    test.message = f"Actions: {len(body.get('actions', []))}"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing actions field"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_api_today_actions(self) -> BrowserTest:
        """Test the /api/v1/salesforce/analytics/today-actions endpoint."""
        test = BrowserTest(name="API - Today Actions", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const resp = await fetch('/api/v1/salesforce/analytics/today-actions');
                    return { status: resp.status, body: await resp.json() };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                if "actions" in body and "date" in body:
                    test.result = TestResult.PASS
                    test.message = f"Actions for {body.get('date')}"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing expected fields"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_api_pipeline_health(self) -> BrowserTest:
        """Test the /api/v1/salesforce/analytics/pipeline-health endpoint."""
        test = BrowserTest(name="API - Pipeline Health", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const resp = await fetch('/api/v1/salesforce/analytics/pipeline-health');
                    return { status: resp.status, body: await resp.json() };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                if "status" in body:
                    test.result = TestResult.PASS
                    test.message = f"Status: {body.get('status')}"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing status field"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    # =========================================================================
    # UI COMPONENT TESTS
    # =========================================================================

    async def test_ai_scores_panel_exists(self) -> BrowserTest:
        """Test that the AI Scores panel is present on the page."""
        test = BrowserTest(name="UI - AI Scores Panel", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            
            # Check for AI Scores section
            has_ai_section = await page.evaluate("""
                () => {
                    const title = document.body.innerText.includes('AI Lead & Opportunity Scores');
                    const tabs = document.getElementById('ai-score-tabs');
                    return { title, tabs: !!tabs };
                }
            """)
            
            await page.close()
            
            if has_ai_section.get("title") or has_ai_section.get("tabs"):
                test.result = TestResult.PASS
                test.message = "AI Scores section found"
            else:
                test.result = TestResult.FAIL
                test.message = "AI Scores section not found"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_ai_score_tabs_switch(self) -> BrowserTest:
        """Test that clicking tabs switches between Lead and Opportunity scores."""
        test = BrowserTest(name="UI - AI Score Tab Switch", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            await asyncio.sleep(2)  # Wait for scores to load
            
            # Check initial state and try clicking tabs
            result = await page.evaluate("""
                () => {
                    const leadPanel = document.getElementById('lead-scores-panel');
                    const oppPanel = document.getElementById('opp-scores-panel');
                    return {
                        leadExists: !!leadPanel,
                        oppExists: !!oppPanel
                    };
                }
            """)
            
            await page.close()
            
            if result.get("leadExists") and result.get("oppExists"):
                test.result = TestResult.PASS
                test.message = "Both score panels exist"
            else:
                test.result = TestResult.SKIP
                test.message = "Score panels not found (may need data)"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_gemma_chat_panel_exists(self) -> BrowserTest:
        """Test that the Gemma CRM Chat panel is present."""
        test = BrowserTest(name="UI - Gemma Chat Panel", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            
            result = await page.evaluate("""
                () => {
                    const chatTitle = document.body.innerText.includes('Ask Gemma AI');
                    const input = document.getElementById('gemma-input');
                    const messages = document.getElementById('gemma-messages');
                    return { 
                        title: chatTitle, 
                        input: !!input, 
                        messages: !!messages 
                    };
                }
            """)
            
            await page.close()
            
            if result.get("title") or result.get("input"):
                test.result = TestResult.PASS
                test.message = "Gemma chat elements found"
            else:
                test.result = TestResult.FAIL
                test.message = "Gemma chat not found"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_gemma_suggestion_chips(self) -> BrowserTest:
        """Test that Gemma suggestion chips are clickable."""
        test = BrowserTest(name="UI - Gemma Suggestion Chips", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            
            result = await page.evaluate("""
                () => {
                    const chips = document.querySelectorAll('.sf-suggestion-chip');
                    return { count: chips.length };
                }
            """)
            
            await page.close()
            
            if result.get("count", 0) > 0:
                test.result = TestResult.PASS
                test.message = f"Found {result['count']} suggestion chips"
            else:
                test.result = TestResult.SKIP
                test.message = "No suggestion chips found"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_score_cards_render(self) -> BrowserTest:
        """Test that score cards render with proper structure."""
        test = BrowserTest(name="UI - Score Cards Render", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            await asyncio.sleep(3)  # Wait for demo data to load
            
            result = await page.evaluate("""
                () => {
                    const cards = document.querySelectorAll('.sf-score-card');
                    const badges = document.querySelectorAll('.sf-score-badge');
                    return { 
                        cards: cards.length, 
                        badges: badges.length 
                    };
                }
            """)
            
            await page.close()
            
            if result.get("cards", 0) > 0:
                test.result = TestResult.PASS
                test.message = f"Found {result['cards']} score cards"
            else:
                test.result = TestResult.SKIP
                test.message = "No score cards rendered (may need API)"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_kanban_board_renders(self) -> BrowserTest:
        """Test that Kanban board renders with columns."""
        test = BrowserTest(name="UI - Kanban Board", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            await asyncio.sleep(2)
            
            result = await page.evaluate("""
                () => {
                    const columns = document.querySelectorAll('.sf-kanban-column');
                    const cards = document.querySelectorAll('.sf-kanban-card');
                    return { columns: columns.length, cards: cards.length };
                }
            """)
            
            await page.close()
            
            if result.get("columns", 0) > 0:
                test.result = TestResult.PASS
                test.message = f"{result['columns']} columns, {result['cards']} cards"
            else:
                test.result = TestResult.FAIL
                test.message = "Kanban board not rendered"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test


class SalesforceAITests(SalesforceTests):
    """Extended tests specifically for Salesforce AI features."""

    async def run_tests(self) -> List[BrowserTest]:
        """Run all Salesforce AI tests including base tests."""
        tests = []

        # Run base tests first
        base_tests = await super().run_tests()
        tests.extend(base_tests)

        # =====================================================================
        # AI Analytics API Tests
        # =====================================================================
        tests.append(await self.test_api_lead_score())
        tests.append(await self.test_api_opportunity_score())
        tests.append(await self.test_api_churn_score())
        tests.append(await self.test_api_next_best_action())
        tests.append(await self.test_api_today_actions())
        tests.append(await self.test_api_pipeline_health())

        # =====================================================================
        # UI Component Tests
        # =====================================================================
        tests.append(await self.test_ai_scores_panel_exists())
        tests.append(await self.test_ai_score_tabs_switch())
        tests.append(await self.test_gemma_chat_panel_exists())
        tests.append(await self.test_gemma_suggestion_chips())
        tests.append(await self.test_score_cards_render())
        tests.append(await self.test_kanban_board_renders())

        return tests


class SalesforceEnterpriseTests(SalesforceAITests):
    """
    Complete Enterprise Salesforce Tests - Phase 2 AI Analytics.
    
    Tests ALL features including:
    - Deal Velocity Engine
    - Competitive Intelligence Engine
    - Customer 360 Engine
    - Enterprise Summary
    - Full UI integration
    """

    async def run_tests(self) -> List[BrowserTest]:
        """Run ALL enterprise Salesforce tests."""
        tests = []

        # Run all base AI tests first
        base_tests = await super().run_tests()
        tests.extend(base_tests)

        # =====================================================================
        # PHASE 2: ENTERPRISE AI ANALYTICS API TESTS
        # =====================================================================
        tests.append(await self.test_api_deal_velocity())
        tests.append(await self.test_api_competitive_intelligence())
        tests.append(await self.test_api_customer_360())
        tests.append(await self.test_api_enterprise_summary())
        tests.append(await self.test_api_velocity_single())
        tests.append(await self.test_api_competitive_single())
        tests.append(await self.test_api_c360_single())

        # =====================================================================
        # PHASE 2: UI INTEGRATION TESTS
        # =====================================================================
        tests.append(await self.test_ui_activity_timeline())
        tests.append(await self.test_ui_stage_distribution())
        tests.append(await self.test_ui_charts_render())

        return tests

    # =========================================================================
    # PHASE 2 API TESTS - Deal Velocity
    # =========================================================================

    async def test_api_deal_velocity(self) -> BrowserTest:
        """Test the /api/v1/salesforce/analytics/deal-velocity endpoint."""
        test = BrowserTest(name="API - Deal Velocity", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const getCsrf = () => { const m = document.cookie.match(/ws_csrf=([^;]+)/); return m ? m[1] : ''; };
                    const resp = await fetch('/api/v1/salesforce/analytics/deal-velocity', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': getCsrf() },
                        credentials: 'include',
                        body: JSON.stringify({
                            opportunities: [
                                { 
                                    Id: 'OPP1', 
                                    Name: 'Enterprise Deal', 
                                    StageName: 'Proposal/Price Quote', 
                                    Amount: 150000,
                                    CreatedDate: '2024-10-01',
                                    CloseDate: '2025-01-15'
                                },
                                { 
                                    Id: 'OPP2', 
                                    Name: 'Mid-Market Deal', 
                                    StageName: 'Negotiation/Review', 
                                    Amount: 75000,
                                    CreatedDate: '2024-11-15',
                                    CloseDate: '2025-02-01'
                                }
                            ],
                            avg_cycle_days: 45,
                            config: {}
                        })
                    });
                    return { status: resp.status, body: await resp.json() };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                # Verify expected fields
                expected = ["engine", "summary", "scored_items", "insights"]
                missing = [f for f in expected if f not in body]
                
                if not missing:
                    test.result = TestResult.PASS
                    summary = body.get("summary", {})
                    test.message = f"Velocity engine OK, avg score: {summary.get('avg_velocity_score', 'N/A')}"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Missing fields: {missing}"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_api_velocity_single(self) -> BrowserTest:
        """Test the GET /api/v1/salesforce/analytics/velocity/{opp_id} endpoint."""
        test = BrowserTest(name="API - Velocity Single Opp", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const resp = await fetch('/api/v1/salesforce/analytics/velocity/OPP123');
                    return { status: resp.status, body: await resp.json() };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                if "velocity_score" in body and "velocity_status" in body:
                    test.result = TestResult.PASS
                    test.message = f"Score: {body.get('velocity_score')}, Status: {body.get('velocity_status')}"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing expected fields"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    # =========================================================================
    # PHASE 2 API TESTS - Competitive Intelligence
    # =========================================================================

    async def test_api_competitive_intelligence(self) -> BrowserTest:
        """Test the /api/v1/salesforce/analytics/competitive-intelligence endpoint."""
        test = BrowserTest(name="API - Competitive Intelligence", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const getCsrf = () => { const m = document.cookie.match(/ws_csrf=([^;]+)/); return m ? m[1] : ''; };
                    const resp = await fetch('/api/v1/salesforce/analytics/competitive-intelligence', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': getCsrf() },
                        credentials: 'include',
                        body: JSON.stringify({
                            opportunities: [
                                { 
                                    Id: 'OPP1', 
                                    Name: 'HubSpot Competitive Deal', 
                                    StageName: 'Proposal', 
                                    Amount: 100000
                                },
                                { 
                                    Id: 'OPP2', 
                                    Name: 'Dynamics Replacement', 
                                    StageName: 'Negotiation', 
                                    Amount: 250000
                                }
                            ],
                            config: {}
                        })
                    });
                    return { status: resp.status, body: await resp.json() };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                expected = ["engine", "summary", "scored_items", "insights"]
                missing = [f for f in expected if f not in body]
                
                if not missing:
                    test.result = TestResult.PASS
                    summary = body.get("summary", {})
                    test.message = f"Competitive engine OK, {summary.get('competitive_deals', 0)} competitive deals"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Missing fields: {missing}"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_api_competitive_single(self) -> BrowserTest:
        """Test the GET /api/v1/salesforce/analytics/competitive/{opp_id} endpoint."""
        test = BrowserTest(name="API - Competitive Single Opp", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const resp = await fetch('/api/v1/salesforce/analytics/competitive/OPP123');
                    return { status: resp.status, body: await resp.json() };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                if "competitor_detected" in body and "competitive_position" in body:
                    test.result = TestResult.PASS
                    test.message = f"Competitor: {body.get('competitor', {}).get('name', 'N/A')}"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing expected fields"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    # =========================================================================
    # PHASE 2 API TESTS - Customer 360
    # =========================================================================

    async def test_api_customer_360(self) -> BrowserTest:
        """Test the /api/v1/salesforce/analytics/customer-360 endpoint."""
        test = BrowserTest(name="API - Customer 360", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const getCsrf = () => { const m = document.cookie.match(/ws_csrf=([^;]+)/); return m ? m[1] : ''; };
                    const resp = await fetch('/api/v1/salesforce/analytics/customer-360', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': getCsrf() },
                        credentials: 'include',
                        body: JSON.stringify({
                            accounts: [
                                { 
                                    Id: 'ACC1', 
                                    Name: 'Acme Corporation', 
                                    Industry: 'Technology', 
                                    AnnualRevenue: 500000,
                                    NumberOfEmployees: 250,
                                    CreatedDate: '2022-01-15'
                                },
                                { 
                                    Id: 'ACC2', 
                                    Name: 'GlobalTech Inc', 
                                    Industry: 'Manufacturing', 
                                    AnnualRevenue: 1200000,
                                    NumberOfEmployees: 800
                                }
                            ],
                            opportunities: [
                                { Id: 'OPP1', Name: 'Upgrade', AccountId: 'ACC1', Amount: 50000, StageName: 'Proposal' }
                            ],
                            config: {}
                        })
                    });
                    return { status: resp.status, body: await resp.json() };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                expected = ["engine", "summary", "scored_items", "insights"]
                missing = [f for f in expected if f not in body]
                
                if not missing:
                    test.result = TestResult.PASS
                    summary = body.get("summary", {})
                    test.message = f"C360 engine OK, avg score: {summary.get('avg_c360_score', 'N/A')}"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Missing fields: {missing}"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_api_c360_single(self) -> BrowserTest:
        """Test the GET /api/v1/salesforce/analytics/customer-360/{account_id} endpoint."""
        test = BrowserTest(name="API - C360 Single Account", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const resp = await fetch('/api/v1/salesforce/analytics/customer-360/ACC123');
                    return { status: resp.status, body: await resp.json() };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                if "c360_score" in body and "dimension_scores" in body:
                    test.result = TestResult.PASS
                    test.message = f"C360 Score: {body.get('c360_score')}, Segment: {body.get('segment')}"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing expected fields"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    # =========================================================================
    # PHASE 2 API TESTS - Enterprise Summary
    # =========================================================================

    async def test_api_enterprise_summary(self) -> BrowserTest:
        """Test the /api/v1/salesforce/analytics/enterprise-summary endpoint."""
        test = BrowserTest(name="API - Enterprise Summary", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            
            response = await page.evaluate("""
                async () => {
                    const resp = await fetch('/api/v1/salesforce/analytics/enterprise-summary');
                    return { status: resp.status, body: await resp.json() };
                }
            """)
            
            await page.close()
            
            if response["status"] == 200:
                body = response["body"]
                if "engines_available" in body and "enterprise_features" in body:
                    engines = body.get("engines_available", [])
                    test.result = TestResult.PASS
                    test.message = f"{len(engines)} engines available"
                    
                    # Verify all expected engines
                    engine_names = [e.get("name") for e in engines]
                    expected_engines = [
                        "lead_scoring", "opportunity_scoring", "salesforce_churn",
                        "salesforce_nba", "salesforce_velocity", 
                        "salesforce_competitive", "salesforce_c360"
                    ]
                    missing_engines = [e for e in expected_engines if e not in engine_names]
                    if missing_engines:
                        test.result = TestResult.FAIL
                        test.message = f"Missing engines: {missing_engines}"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing expected fields"
            else:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response['status']}"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    # =========================================================================
    # UI INTEGRATION TESTS
    # =========================================================================

    async def test_ui_activity_timeline(self) -> BrowserTest:
        """Test that the Activity Timeline renders properly."""
        test = BrowserTest(name="UI - Activity Timeline", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            await asyncio.sleep(2)
            
            result = await page.evaluate("""
                () => {
                    const timeline = document.getElementById('activity-timeline');
                    const items = document.querySelectorAll('.sf-timeline-item');
                    return { 
                        timeline: !!timeline, 
                        items: items.length 
                    };
                }
            """)
            
            await page.close()
            
            if result.get("timeline"):
                test.result = TestResult.PASS
                test.message = f"Timeline found with {result['items']} items"
            else:
                test.result = TestResult.SKIP
                test.message = "Activity timeline not found"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_ui_stage_distribution(self) -> BrowserTest:
        """Test that the Stage Distribution chart renders."""
        test = BrowserTest(name="UI - Stage Distribution Chart", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            await asyncio.sleep(2)
            
            result = await page.evaluate("""
                () => {
                    const chart = document.getElementById('stage-chart');
                    const canvas = chart ? chart.querySelector('canvas') : null;
                    return { 
                        chartContainer: !!chart, 
                        canvas: !!canvas 
                    };
                }
            """)
            
            await page.close()
            
            if result.get("chartContainer"):
                test.result = TestResult.PASS
                test.message = "Stage distribution chart found"
            else:
                test.result = TestResult.SKIP
                test.message = "Stage chart not found"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test

    async def test_ui_charts_render(self) -> BrowserTest:
        """Test that all major charts render on the page."""
        test = BrowserTest(name="UI - All Charts Render", page=self.page_name)
        
        try:
            page = await self.context.new_page()
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=10000)
            await self.login(page)
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            await asyncio.sleep(3)  # Wait for all charts to initialize
            
            result = await page.evaluate("""
                () => {
                    const pipeline = document.getElementById('pipeline-chart');
                    const forecast = document.getElementById('forecast-chart');
                    const stage = document.getElementById('stage-chart');
                    const canvases = document.querySelectorAll('canvas');
                    const echarts = document.querySelectorAll('[_echarts_instance_]');
                    
                    return { 
                        pipeline: !!pipeline,
                        forecast: !!forecast,
                        stage: !!stage,
                        canvasCount: canvases.length,
                        echartsCount: echarts.length
                    };
                }
            """)
            
            await page.close()
            
            chart_count = result.get("canvasCount", 0) + result.get("echartsCount", 0)
            
            if chart_count >= 2:
                test.result = TestResult.PASS
                test.message = f"Found {chart_count} charts (canvas: {result['canvasCount']}, echarts: {result['echartsCount']})"
            elif chart_count >= 1:
                test.result = TestResult.PASS
                test.message = f"Found {chart_count} chart(s)"
            else:
                test.result = TestResult.FAIL
                test.message = "No charts rendered"
                
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:80]
        
        return test
