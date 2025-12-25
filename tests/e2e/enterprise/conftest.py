"""
Enterprise E2E Test Suite - Shared Fixtures

Provides authentication, page setup, and utility fixtures
for all enterprise Playwright tests.
"""

import pytest
from playwright.async_api import Page, Browser, async_playwright
import os


# Base URL for tests
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:5005")


@pytest.fixture(scope="session")
def base_url():
    """Base URL for API Gateway."""
    return BASE_URL


@pytest.fixture
async def browser():
    """Create a browser instance."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()


@pytest.fixture
async def page(browser: Browser):
    """Create a new page for each test."""
    context = await browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent="Playwright Test Agent"
    )
    page = await context.new_page()
    yield page
    await context.close()


@pytest.fixture
async def authenticated_page(page: Page, base_url: str):
    """
    Login and establish session before test.
    Per user rules: Full E2E flow - Login → Cookie/Token → Feature Action
    """
    await page.goto(f"{base_url}/login.html")
    
    # Check if login form exists
    login_form = page.locator('form, [data-testid="login-form"]')
    if await login_form.count() > 0:
        # Fill credentials
        email_input = page.locator('[type="email"], [name="email"], [data-testid="email"]')
        password_input = page.locator('[type="password"], [name="password"], [data-testid="password"]')
        
        if await email_input.count() > 0:
            await email_input.fill("test@voxamelior.com")
        if await password_input.count() > 0:
            await password_input.fill("testpassword")
        
        # Submit
        submit_btn = page.locator('[type="submit"], button:has-text("Sign In"), button:has-text("Login")').first
        if await submit_btn.count() > 0:
            await submit_btn.click()
            await page.wait_for_timeout(1000)
    
    return page


@pytest.fixture
async def enterprise_page(authenticated_page: Page, base_url: str):
    """Navigate to enterprise landing page."""
    await authenticated_page.goto(f"{base_url}/enterprise/")
    await authenticated_page.wait_for_load_state("networkidle")
    return authenticated_page


@pytest.fixture
async def salesforce_page(authenticated_page: Page, base_url: str):
    """Navigate to Salesforce cockpit page."""
    await authenticated_page.goto(f"{base_url}/enterprise/salesforce/")
    await authenticated_page.wait_for_load_state("networkidle")
    return authenticated_page


@pytest.fixture
async def fiserv_page(authenticated_page: Page, base_url: str):
    """Navigate to Fiserv banking page."""
    await authenticated_page.goto(f"{base_url}/enterprise/fiserv/")
    await authenticated_page.wait_for_load_state("networkidle")
    return authenticated_page


# --- Utility Functions ---

async def wait_for_api_response(page: Page, url_pattern: str, timeout: int = 10000):
    """Wait for a specific API response."""
    async with page.expect_response(
        lambda resp: url_pattern in resp.url,
        timeout=timeout
    ) as response_info:
        return await response_info.value


async def get_computed_style(page: Page, selector: str, property: str) -> str:
    """Get computed CSS style for an element."""
    element = page.locator(selector).first
    return await element.evaluate(f'el => getComputedStyle(el).{property}')


async def has_animation(page: Page, selector: str) -> bool:
    """Check if element has CSS animation."""
    animation = await get_computed_style(page, selector, 'animation')
    return animation != 'none' and animation != ''
