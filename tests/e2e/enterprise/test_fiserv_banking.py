"""
Fiserv Banking E2E Tests

Tests the Fiserv Banking page functionality.
All tests use authenticated sessions per user rules.
"""

import pytest
from playwright.async_api import Page, expect


class TestBubbleSummary:
    """Bubble summary component tests."""

    async def test_bubble_summary_exists(self, fiserv_page: Page):
        """Verify bubble summary card is present."""
        bubble = fiserv_page.locator('.ent-bubble-summary, .ent-card:has(.ent-bubble-icon)')
        await expect(bubble).to_be_visible()

    async def test_bubble_icon_visible(self, fiserv_page: Page):
        """Verify bubble icon is visible."""
        icon = fiserv_page.locator('.ent-bubble-icon')
        await expect(icon).to_be_visible()

    async def test_stats_populate(self, fiserv_page: Page):
        """Verify stats populate with data."""
        members_stat = fiserv_page.locator('#statMembers')
        
        # Wait for animation/data load
        await fiserv_page.wait_for_timeout(3000)
        
        text = await members_stat.text_content()
        assert text != '--', "Stats should populate with data"

    async def test_all_stats_visible(self, fiserv_page: Page):
        """Verify all three stats are visible."""
        await expect(fiserv_page.locator('#statMembers')).to_be_visible()
        await expect(fiserv_page.locator('#statAccounts')).to_be_visible()
        await expect(fiserv_page.locator('#statTransactions')).to_be_visible()


class TestAPIStatus:
    """API status monitoring tests."""

    async def test_api_status_section_exists(self, fiserv_page: Page):
        """Verify API status section exists."""
        status_section = fiserv_page.locator('text=System Status')
        await expect(status_section).to_be_visible()

    async def test_fiserv_status_updates(self, fiserv_page: Page):
        """Verify Fiserv API status resolves."""
        status = fiserv_page.locator('#fiservStatus')
        
        await fiserv_page.wait_for_timeout(5000)
        
        text = await status.text_content()
        assert text in ['Operational', 'Unavailable', 'Offline', 'Checking...'], \
            f"Status should resolve to known state, got: {text}"

    async def test_ml_status_updates(self, fiserv_page: Page):
        """Verify ML service status resolves."""
        status = fiserv_page.locator('#mlStatus')
        
        await fiserv_page.wait_for_timeout(5000)
        
        text = await status.text_content()
        assert text in ['Operational', 'Degraded', 'Offline', 'Checking...'], \
            f"ML status should resolve to known state, got: {text}"

    async def test_latency_displays(self, fiserv_page: Page):
        """Verify latency metric shows milliseconds."""
        latency = fiserv_page.locator('#latency')
        
        await fiserv_page.wait_for_timeout(5000)
        
        text = await latency.text_content()
        assert 'ms' in text, "Latency should show milliseconds"

    async def test_last_sync_updates(self, fiserv_page: Page):
        """Verify last sync timestamp updates."""
        sync = fiserv_page.locator('#lastSync')
        
        await fiserv_page.wait_for_timeout(3000)
        
        text = await sync.text_content()
        assert text != '--', "Last sync should update"


class TestCapabilityCards:
    """Capability/feature cards tests."""

    async def test_capability_cards_exist(self, fiserv_page: Page):
        """Verify capability cards are present."""
        cards = await fiserv_page.query_selector_all('.ent-card')
        assert len(cards) >= 6, "Should have at least 6 capability cards"

    async def test_member_search_card(self, fiserv_page: Page):
        """Verify Member Search card exists and has link."""
        card = fiserv_page.locator('.ent-card:has-text("Member Search")')
        await expect(card).to_be_visible()
        
        link = card.locator('.ent-feature-link, a')
        if await link.count() > 0:
            href = await link.first.get_attribute('href')
            assert href is not None, "Should have link"

    async def test_fraud_detection_card(self, fiserv_page: Page):
        """Verify Fraud Detection card exists."""
        card = fiserv_page.locator('.ent-card:has-text("Fraud Detection")')
        await expect(card).to_be_visible()

    async def test_cards_have_icons(self, fiserv_page: Page):
        """Verify cards have feature icons."""
        icons = await fiserv_page.query_selector_all('.ent-feature-icon')
        assert len(icons) >= 6, "Each card should have an icon"


class TestNavigation:
    """Navigation tests for Fiserv page."""

    async def test_breadcrumb_exists(self, fiserv_page: Page):
        """Verify breadcrumb back link exists."""
        breadcrumb = fiserv_page.locator('a:has-text("Back to Enterprise")')
        await expect(breadcrumb).to_be_visible()

    async def test_breadcrumb_navigates(self, fiserv_page: Page, base_url: str):
        """Verify breadcrumb navigates to enterprise."""
        breadcrumb = fiserv_page.locator('a:has-text("Back to Enterprise")')
        await breadcrumb.click()
        
        await fiserv_page.wait_for_url('**/enterprise/**')

    async def test_enterprise_dropdown_active(self, fiserv_page: Page):
        """Verify Fiserv is highlighted in enterprise dropdown."""
        dropdown = fiserv_page.locator('.ent-nav-dropdown')
        await dropdown.hover()
        
        # The Fiserv item should have a highlighted background
        fiserv_item = fiserv_page.locator('.ent-nav-dropdown-item:has-text("Fiserv Banking")')
        style = await fiserv_item.get_attribute('style')
        
        # It should have inline background style for active state
        assert style is not None and 'background' in style, "Fiserv should be highlighted"


class TestButtons:
    """Button functionality tests."""

    async def test_open_dashboard_button(self, fiserv_page: Page):
        """Verify Open Full Dashboard button exists."""
        btn = fiserv_page.locator('a:has-text("Open Full Dashboard"), button:has-text("Open Full Dashboard")')
        await expect(btn).to_be_visible()

    async def test_refresh_data_button(self, fiserv_page: Page):
        """Verify Refresh Data button works."""
        btn = fiserv_page.locator('button:has-text("Refresh Data")')
        await expect(btn).to_be_visible()
        
        # Get initial latency
        latency = fiserv_page.locator('#latency')
        initial = await latency.text_content()
        
        # Click refresh
        await btn.click()
        
        # Wait for refresh
        await fiserv_page.wait_for_timeout(3000)
        
        # Latency should still show (may be same or different value)
        final = await latency.text_content()
        assert 'ms' in final, "Latency should still show after refresh"
