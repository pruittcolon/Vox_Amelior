"""
Banking Enterprise Tests - All 5 Phases
========================================
Comprehensive test suite for the banking.html enterprise features.
Covers MSR Core, Cross-Sell, Lending, Security, and Reporting.
"""

import asyncio
from typing import List
from ..base import BasePageTest, BrowserTest, TestResult


class BankingEnterpriseTests(BasePageTest):
    """Test all 5 phases of the banking enterprise platform"""
    
    @property
    def page_name(self) -> str:
        return "banking.html"
    
    async def run_tests(self) -> List[BrowserTest]:
        """Run all banking enterprise tests - WITH AUTHENTICATION"""
        tests = []
        
        # Login first to get authenticated session
        login_page = await self.context.new_page()
        logged_in = await self.login(login_page)
        if not logged_in:
            # Add a failed test if login fails
            tests.append(BrowserTest(
                name="Authentication Login",
                page=self.page_name,
                result=TestResult.FAIL,
                message="Login failed - cannot run authenticated tests"
            ))
            await login_page.close()
            return tests
        
        # Close login page - session cookies are in context
        await login_page.close()
        
        # Now run all tests with authenticated context
        
        # API Integration Tests
        tests.extend(await self._test_fiserv_api())
        
        # Phase 1: MSR Core
        tests.extend(await self._test_phase1_msr())
        
        # Phase 2: Cross-Sell
        tests.extend(await self._test_phase2_crosssell())
        
        # Phase 3: Lending
        tests.extend(await self._test_phase3_lending())
        
        # Phase 4: Security
        tests.extend(await self._test_phase4_security())
        
        # Phase 5: Reporting
        tests.extend(await self._test_phase5_reporting())
        
        return tests
    
    # =========================================================================
    # FISERV API INTEGRATION
    # =========================================================================
    
    async def _test_fiserv_api(self) -> List[BrowserTest]:
        """Test Fiserv API integration"""
        tests = []
        page = await self.context.new_page()
        
        try:
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            await asyncio.sleep(1)
            
            # Test 1: API Usage
            test = BrowserTest(name="API Usage Tracking", page=self.page_name)
            try:
                result = await page.evaluate('fetch("/fiserv/api/v1/usage").then(r=>r.json())')
                if 'calls_remaining' in result:
                    test.result = TestResult.PASS
                    test.message = f"{result['calls_remaining']} calls remaining"
                    test.output = str(result)
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing calls_remaining field"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test 2: Token Status
            test = BrowserTest(name="OAuth Token Status", page=self.page_name)
            try:
                result = await page.evaluate('fetch("/fiserv/api/v1/token").then(r=>r.json())')
                if 'status' in result:
                    test.result = TestResult.PASS
                    test.message = f"Status: {result['status']}"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing status field"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
        finally:
            await page.close()
        
        return tests
    
    # =========================================================================
    # PHASE 1: MSR CORE OPERATIONS
    # =========================================================================
    
    async def _test_phase1_msr(self) -> List[BrowserTest]:
        """Test Phase 1: MSR Core Operations"""
        tests = []
        page = await self.context.new_page()
        
        try:
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            await asyncio.sleep(1)
            
            # Test: MemberState Object
            test = BrowserTest(name="P1: MemberState Object", page=self.page_name)
            try:
                result = await page.evaluate('typeof window.MemberState === "object" && "currentMember" in window.MemberState')
                test.result = TestResult.PASS if result else TestResult.FAIL
                test.message = "State initialized" if result else "MemberState missing"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Notes Functions
            test = BrowserTest(name="P1: Notes Save/Retrieve", page=self.page_name)
            try:
                await page.evaluate('window.saveMemberNote("TEST001", "Test note", "general")')
                notes = await page.evaluate('window.getMemberNotes("TEST001")')
                if len(notes) >= 1:
                    test.result = TestResult.PASS
                    test.message = f"Saved and retrieved {len(notes)} note(s)"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Notes not persisted"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Transaction Filters
            test = BrowserTest(name="P1: Transaction Filters", page=self.page_name)
            try:
                filter_fn = await page.evaluate('typeof window.applyTransactionFilters === "function"')
                clear_fn = await page.evaluate('typeof window.clearTransactionFilters === "function"')
                if filter_fn and clear_fn:
                    test.result = TestResult.PASS
                    test.message = "Filter functions available"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing filter functions"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: CSV Export
            test = BrowserTest(name="P1: CSV Export Function", page=self.page_name)
            try:
                result = await page.evaluate('typeof window.exportTransactionsCSV === "function"')
                test.result = TestResult.PASS if result else TestResult.FAIL
                test.message = "Export function available" if result else "Missing export"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
        finally:
            await page.close()
        
        return tests
    
    # =========================================================================
    # PHASE 2: CROSS-SELL & PRODUCTIVITY
    # =========================================================================
    
    async def _test_phase2_crosssell(self) -> List[BrowserTest]:
        """Test Phase 2: Cross-Sell & Productivity"""
        tests = []
        page = await self.context.new_page()
        
        try:
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            await asyncio.sleep(1)
            
            # Test: CrossSellState
            test = BrowserTest(name="P2: CrossSellState Object", page=self.page_name)
            try:
                result = await page.evaluate('typeof window.CrossSellState === "object"')
                test.result = TestResult.PASS if result else TestResult.FAIL
                test.message = "State initialized" if result else "CrossSellState missing"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Next Best Action
            test = BrowserTest(name="P2: NBA Generation", page=self.page_name)
            try:
                offers = await page.evaluate('window.generateNextBestActions({name: "Test"})')
                if len(offers) >= 1 and 'propensity' in offers[0]:
                    test.result = TestResult.PASS
                    test.message = f"Generated {len(offers)} offers"
                    test.output = f"Top: {offers[0]['name']} ({int(offers[0]['propensity']*100)}%)"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Invalid NBA output"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Risk Flags
            test = BrowserTest(name="P2: Risk Flag Check", page=self.page_name)
            try:
                flags = await page.evaluate('window.checkRiskFlags({name: "Test"})')
                if isinstance(flags, list):
                    test.result = TestResult.PASS
                    test.message = f"Returned {len(flags)} flag(s)"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Invalid flag output"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Quick Actions
            test = BrowserTest(name="P2: Quick Actions Constant", page=self.page_name)
            try:
                actions = await page.evaluate('window.QUICK_ACTIONS')
                if len(actions) >= 5:
                    test.result = TestResult.PASS
                    test.message = f"{len(actions)} quick actions"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Missing quick actions"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
        finally:
            await page.close()
        
        return tests
    
    # =========================================================================
    # PHASE 3: LENDING INTELLIGENCE
    # =========================================================================
    
    async def _test_phase3_lending(self) -> List[BrowserTest]:
        """Test Phase 3: Lending Intelligence"""
        tests = []
        page = await self.context.new_page()
        
        try:
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            await asyncio.sleep(1)
            
            # Test: DTI Calculation
            test = BrowserTest(name="P3: DTI Calculator", page=self.page_name)
            try:
                dti = await page.evaluate('window.calculateDTI(1500, 5000)')
                if dti['ratio'] == 30.0 and dti['status'] == 'good':
                    test.result = TestResult.PASS
                    test.message = f"DTI={dti['ratio']}% ({dti['status']})"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Expected 30% good, got {dti}"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: LTV Calculation
            test = BrowserTest(name="P3: LTV Calculator", page=self.page_name)
            try:
                ltv = await page.evaluate('window.calculateLTV(200000, 250000)')
                if ltv['ratio'] == 80.0:
                    test.result = TestResult.PASS
                    test.message = f"LTV={ltv['ratio']}%, PMI={ltv['pmiRequired']}"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Expected 80%, got {ltv['ratio']}"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: DSCR Calculation
            test = BrowserTest(name="P3: DSCR Calculator", page=self.page_name)
            try:
                dscr = await page.evaluate('window.calculateDSCR(50000, 35000)')
                if abs(dscr['ratio'] - 1.43) < 0.01:
                    test.result = TestResult.PASS
                    test.message = f"DSCR={dscr['ratio']}x"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Expected 1.43x, got {dscr['ratio']}"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Nemo Score
            test = BrowserTest(name="P3: Nemo Score", page=self.page_name)
            try:
                score = await page.evaluate('''
                    window.calculateNemoScore({
                        onTimePaymentRate: 0.95,
                        accountAgeMonths: 24,
                        averageBalance: 2500,
                        nsfCount: 0
                    })
                ''')
                if 600 <= score['score'] <= 850 and len(score['factors']) >= 3:
                    test.result = TestResult.PASS
                    test.message = f"Score={score['score']} ({score['tier']})"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Invalid score: {score}"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Pricing Engine
            test = BrowserTest(name="P3: Loan Pricing Engine", page=self.page_name)
            try:
                pricing = await page.evaluate('window.calculateLoanPricing(720, 25000, 36)')
                if pricing['tier'] == 'B' and pricing['monthlyPayment'] > 0:
                    test.result = TestResult.PASS
                    test.message = f"Tier={pricing['tier']}, APR={pricing['finalRate']}%"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Invalid pricing: {pricing}"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Document Checklist
            test = BrowserTest(name="P3: Document Checklist", page=self.page_name)
            try:
                docs = await page.evaluate('window.getDocumentChecklist("mortgage")')
                if len(docs) >= 6:
                    test.result = TestResult.PASS
                    test.message = f"{len(docs)} docs for mortgage"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Expected 6+ docs, got {len(docs)}"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
        finally:
            await page.close()
        
        return tests
    
    # =========================================================================
    # PHASE 4: FRAUD & SECURITY
    # =========================================================================
    
    async def _test_phase4_security(self) -> List[BrowserTest]:
        """Test Phase 4: Fraud & Security"""
        tests = []
        page = await self.context.new_page()
        
        try:
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            await asyncio.sleep(1)
            
            # Test: Anomaly Detection
            test = BrowserTest(name="P4: Anomaly Detection", page=self.page_name)
            try:
                anomalies = await page.evaluate('''
                    window.detectAnomalies(
                        { amount: 15000, timestamp: new Date().toISOString() },
                        { averageTransaction: 500 }
                    )
                ''')
                high_value = any(a['code'] == 'HIGH_VALUE' for a in anomalies)
                if high_value:
                    test.result = TestResult.PASS
                    test.message = f"{len(anomalies)} anomaly(s): HIGH_VALUE detected"
                else:
                    test.result = TestResult.FAIL
                    test.message = "HIGH_VALUE not detected"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Velocity Check
            test = BrowserTest(name="P4: Velocity Check", page=self.page_name)
            try:
                velocity = await page.evaluate('''
                    const now = Date.now();
                    window.checkVelocity([
                        { timestamp: new Date(now - 10*60*1000).toISOString() },
                        { timestamp: new Date(now - 20*60*1000).toISOString() },
                        { timestamp: new Date(now - 30*60*1000).toISOString() },
                        { timestamp: new Date(now - 40*60*1000).toISOString() },
                        { timestamp: new Date(now - 50*60*1000).toISOString() },
                        { timestamp: new Date(now - 55*60*1000).toISOString() },
                    ], 60, 5)
                ''')
                if velocity['isViolation'] and velocity['count'] == 6:
                    test.result = TestResult.PASS
                    test.message = f"Violation detected: {velocity['count']}/5"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Expected violation, got {velocity}"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Fraud Score
            test = BrowserTest(name="P4: Fraud Score", page=self.page_name)
            try:
                fraud = await page.evaluate('''
                    window.calculateFraudScore(
                        { amount: 12000, timestamp: new Date().toISOString(), isNewPayee: true },
                        { averageTransaction: 500, tenureMonths: 6 }
                    )
                ''')
                if 0 <= fraud['score'] <= 100 and fraud['action'] in ['ALLOW', 'REVIEW', 'BLOCK']:
                    test.result = TestResult.PASS
                    test.message = f"Score={fraud['score']}, Action={fraud['action']}"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Invalid fraud score: {fraud}"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Alert Queue
            test = BrowserTest(name="P4: Alert Queue", page=self.page_name)
            try:
                await page.evaluate('window.SecurityState.alertQueue = []')
                alert = await page.evaluate('''
                    window.createAlert(
                        { id: "TX999", memberId: "M456", amount: 8000 },
                        { score: 72, riskLevel: "high", action: "BLOCK", factors: [] },
                        "CRITICAL"
                    )
                ''')
                if alert['status'] == 'OPEN' and alert['priority']['label'] == 'Critical':
                    test.result = TestResult.PASS
                    test.message = f"Alert {alert['id']}: {alert['priority']['label']}"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Invalid alert: {alert}"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: SAR Generation
            test = BrowserTest(name="P4: SAR Generation", page=self.page_name)
            try:
                sar = await page.evaluate('''
                    window.generateSARData(
                        window.SecurityState.alertQueue[0],
                        { id: "M456", name: "Jane Smith" }
                    )
                ''')
                if 'reportId' in sar and 'subject' in sar:
                    test.result = TestResult.PASS
                    test.message = f"SAR {sar['reportId']}: {sar['status']}"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Invalid SAR output"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
        finally:
            await page.close()
        
        return tests
    
    # =========================================================================
    # PHASE 5: REPORTING & COMPLIANCE
    # =========================================================================
    
    async def _test_phase5_reporting(self) -> List[BrowserTest]:
        """Test Phase 5: Reporting & Compliance"""
        tests = []
        page = await self.context.new_page()
        
        try:
            await page.goto(self.page_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            await asyncio.sleep(1)
            
            # Test: Member Activity Report
            test = BrowserTest(name="P5: Member Activity Report", page=self.page_name)
            try:
                report = await page.evaluate('window.generateMemberActivityReport("M12345", {dateRange: 30})')
                if 'activitySummary' in report and 'transactionSummary' in report:
                    test.result = TestResult.PASS
                    test.message = f"{report['reportId']}: {report['transactionSummary']['totalTransactions']} tx"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Invalid report structure"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Transaction Summary
            test = BrowserTest(name="P5: Transaction Summary", page=self.page_name)
            try:
                report = await page.evaluate('window.generateTransactionSummary({dateRange: 30})')
                if 'volumeMetrics' in report and 'byType' in report:
                    test.result = TestResult.PASS
                    test.message = f"{report['volumeMetrics']['totalCount']} transactions"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Invalid summary structure"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: CTR Report (BSA)
            test = BrowserTest(name="P5: CTR Report (BSA)", page=self.page_name)
            try:
                report = await page.evaluate('''
                    window.generateCTR([
                        { id: "TX1", amount: 15000, memberId: "M001" },
                        { id: "TX2", amount: 8000, memberId: "M002" },
                        { id: "TX3", amount: 12000, memberId: "M003" }
                    ])
                ''')
                if report['summary']['totalTransactions'] == 2:  # 2 over $10k
                    test.result = TestResult.PASS
                    test.message = f"{report['summary']['totalTransactions']} reportable (>${report['compliance']['threshold']:,})"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Expected 2, got {report['summary']['totalTransactions']}"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Audit Trail
            test = BrowserTest(name="P5: Audit Trail", page=self.page_name)
            try:
                await page.evaluate('window.ReportingState.auditLog = []')
                await page.evaluate('window.logAuditEvent("TEST_EVENT", {data: "test"})')
                entries = await page.evaluate('window.getAuditLog({})')
                if len(entries) >= 1:
                    test.result = TestResult.PASS
                    test.message = f"{len(entries)} audit entries"
                else:
                    test.result = TestResult.FAIL
                    test.message = "No audit entries"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Export Functions
            test = BrowserTest(name="P5: Export CSV/JSON", page=self.page_name)
            try:
                csv = await page.evaluate('window.exportToCSV(window.ReportingState.generatedReports[0])')
                json_str = await page.evaluate('window.exportToJSON(window.ReportingState.generatedReports[0])')
                if len(csv) > 50 and len(json_str) > 100:
                    test.result = TestResult.PASS
                    test.message = f"CSV:{len(csv)}c, JSON:{len(json_str)}c"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Export output too short"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
            # Test: Scheduled Reports
            test = BrowserTest(name="P5: Scheduled Reports", page=self.page_name)
            try:
                await page.evaluate('window.ReportingState.scheduledReports = []')
                schedule = await page.evaluate('window.scheduleReport("TRANSACTION_SUMMARY", "weekly", {})')
                if schedule['frequency'] == 'weekly' and schedule['active']:
                    test.result = TestResult.PASS
                    test.message = f"Scheduled: {schedule['nextRun'][:10]}"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Invalid schedule: {schedule}"
            except Exception as e:
                test.result = TestResult.ERROR
                test.message = str(e)[:60]
            tests.append(test)
            
        finally:
            await page.close()
        
        return tests
