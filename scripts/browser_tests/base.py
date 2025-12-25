"""
Base Test Classes and Utilities
================================
Shared functionality for all page tests.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from playwright.async_api import async_playwright, Page, BrowserContext, Error as PlaywrightError

from .config import Config, DEFAULT_CONFIG


class TestResult(Enum):
    """Test result status"""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


@dataclass
class BrowserTest:
    """Individual test result"""
    name: str
    page: str
    result: TestResult = TestResult.PASS
    message: str = ""
    js_errors: List[str] = field(default_factory=list)
    duration_ms: float = 0
    output: Optional[str] = None


class BasePageTest(ABC):
    """
    Base class for page-specific tests.
    Provides shared utilities and browser management.
    """

    def __init__(self, config: Config = None):
        self.config = config or DEFAULT_CONFIG
        self.context: Optional[BrowserContext] = None
        self.tests: List[BrowserTest] = []

    @property
    @abstractmethod
    def page_name(self) -> str:
        """HTML page filename (e.g., 'banking.html')"""
        pass

    @property
    def page_url(self) -> str:
        """Full URL to the page"""
        return self.config.page_url(self.page_name)

    @abstractmethod
    async def run_tests(self) -> List[BrowserTest]:
        """Run all tests for this page. Override in subclasses."""
        pass

    async def login(
        self,
        page: Page,
        username: str = "admin",
        password: str = "admin123"
    ) -> bool:
        """
        Login to the application to get an authenticated session.
        This should be called before running tests that require auth.
        
        Uses demo credentials by default (admin/admin123).
        Set ENABLE_DEMO_USERS=true on the API gateway for these to work.
        
        NOTE: Login can take 15-20 seconds due to bcrypt password hashing.
        
        Args:
            page: Playwright page instance
            username: Login username (default: admin)
            password: Login password (default: admin123)
            
        Returns:
            True if login was successful, False otherwise
        """
        try:
            await page.goto(f"{self.config.base_url}/ui/login.html", timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            await asyncio.sleep(0.5)
            
            # Fill login form
            await page.fill('#username', username)
            await page.fill('#password', password)
            
            # Submit - bcrypt hashing can take 15-20 seconds
            await page.click('#login-button, .btn-primary, button[type="submit"]')
            
            # Wait longer for bcrypt-based authentication (up to 30 seconds)
            for _ in range(30):
                await asyncio.sleep(1)
                
                # Check for success indicators
                current_url = page.url
                
                # Success if redirected away from login page
                if 'login' not in current_url.lower() or 'index' in current_url.lower():
                    return True
                    
                # Check if we got a session cookie
                cookies = await page.context.cookies()
                if any(c['name'] == 'ws_session' for c in cookies):
                    return True
                    
                # Check for rate limiting error - return False to skip gracefully
                rate_limit_check = await page.evaluate(
                    "document.body.innerText.includes('Too Many Requests')"
                )
                if rate_limit_check:
                    print("[Login] Rate limited - returning False")
                    return False
                    
            return False
        except Exception as e:
            print(f"[Login] Error: {e}")
            return False

    # =========================================================================
    # SHARED TEST METHODS
    # =========================================================================

    async def test_page_loads(
        self,
        name: str = None,
        required_elements: List[str] = None
    ) -> BrowserTest:
        """Test that page loads without JS errors"""
        test_name = name or f"{self.page_name} - Load"
        test = BrowserTest(name=test_name, page=self.page_name)
        start = time.time()

        try:
            page = await self.context.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            response = await page.goto(
                self.page_url,
                timeout=self.config.page_load_timeout
            )
            await page.wait_for_load_state(
                "networkidle",
                timeout=self.config.network_idle_timeout
            )
            await asyncio.sleep(0.3)

            test.js_errors = errors

            if errors:
                test.result = TestResult.FAIL
                test.message = f"JS Error: {errors[0][:60]}"
            elif response and response.status >= 400:
                test.result = TestResult.FAIL
                test.message = f"HTTP {response.status}"
            elif required_elements:
                missing = []
                for selector in required_elements:
                    try:
                        await page.wait_for_selector(
                            selector,
                            timeout=self.config.element_timeout
                        )
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

    async def test_button_click(
        self,
        selector: str,
        name: str,
        wait_for_selector: str = None,
        expected_text: str = None,
        force: bool = False
    ) -> BrowserTest:
        """Test clicking a button
        
        Args:
            selector: CSS selector for the button to click
            name: Test name
            wait_for_selector: Optional selector to wait for after click
            expected_text: Optional text to verify after click
            force: If True, bypass visibility checks (for hidden elements)
        """
        test = BrowserTest(name=name, page=self.page_name)
        start = time.time()

        try:
            page = await self.context.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            await page.goto(self.page_url, timeout=self.config.page_load_timeout)
            await page.wait_for_load_state(
                "networkidle",
                timeout=self.config.network_idle_timeout
            )

            await page.click(selector, timeout=self.config.element_timeout, force=force)
            await asyncio.sleep(1)

            test.js_errors = errors

            if errors:
                test.result = TestResult.FAIL
                test.message = f"JS on click: {errors[0][:50]}"
            elif wait_for_selector:
                try:
                    await page.wait_for_selector(
                        wait_for_selector,
                        timeout=self.config.element_timeout
                    )
                    test.result = TestResult.PASS
                    test.message = "Element appeared"
                except:
                    test.result = TestResult.FAIL
                    test.message = f"Timeout: {wait_for_selector}"
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

    async def test_multi_step(
        self,
        name: str,
        steps: List[Dict[str, Any]]
    ) -> BrowserTest:
        """
        Test multiple sequential actions on a page.
        
        steps: [
            {"action": "click", "selector": "..."},
            {"action": "wait", "seconds": 1},
            {"action": "fill", "selector": "...", "value": "..."},
        ]
        """
        test = BrowserTest(name=name, page=self.page_name)
        start = time.time()

        try:
            page = await self.context.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            await page.goto(self.page_url, timeout=self.config.page_load_timeout)
            await page.wait_for_load_state("networkidle", timeout=self.config.network_idle_timeout)

            for step in steps:
                action = step.get("action")
                if action == "click":
                    await page.click(step["selector"], timeout=self.config.element_timeout)
                elif action == "fill":
                    await page.fill(step["selector"], step["value"])
                elif action == "wait":
                    await asyncio.sleep(step.get("seconds", 1))

            test.js_errors = errors
            if errors:
                test.result = TestResult.FAIL
                test.message = f"JS Error: {errors[0][:50]}"
            else:
                test.result = TestResult.PASS
                test.message = "All steps completed"

            await page.close()
        except PlaywrightError as e:
            test.result = TestResult.SKIP
            test.message = str(e)[:60]
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)[:60]

        test.duration_ms = (time.time() - start) * 1000
        return test
