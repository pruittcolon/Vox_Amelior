"""
Enterprise Landing Page E2E Tests

Tests the Enterprise landing/overview page functionality.
All tests use authenticated sessions per user rules.
"""

import pytest
from playwright.async_api import Page, expect


class TestHeroSection:
    """Hero section tests."""

    async def test_hero_exists(self, enterprise_page: Page):
        """Verify hero section is present."""
        hero = enterprise_page.locator('.ent-hero, section:has(.ent-hero-title)')
        await expect(hero).to_be_visible()

    async def test_hero_badge_visible(self, enterprise_page: Page):
        """Verify hero badge is visible."""
        badge = enterprise_page.locator('.ent-hero-badge')
        await expect(badge).to_be_visible()

    async def test_hero_badge_has_pulse(self, enterprise_page: Page):
        """Verify hero badge dot has animation."""
        dot = enterprise_page.locator('.ent-hero-badge-dot')
        
        if await dot.count() > 0:
            animation = await dot.evaluate('el => getComputedStyle(el).animation')
            assert animation != 'none', "Badge dot should have pulse animation"

    async def test_hero_title_has_gradient(self, enterprise_page: Page):
        """Verify gradient text effect on title."""
        accent = enterprise_page.locator('.ent-hero-title-accent')
        
        if await accent.count() > 0:
            fill = await accent.evaluate('el => getComputedStyle(el).webkitTextFillColor')
            # Gradient text typically has transparent fill
            assert 'transparent' in fill or fill == 'transparent', "Should use gradient text"

    async def test_cta_buttons_exist(self, enterprise_page: Page):
        """Verify CTA buttons are present."""
        buttons = await enterprise_page.query_selector_all('.ent-btn')
        assert len(buttons) >= 1, "Should have at least one CTA button"


class TestFeatureCards:
    """Feature card tests."""

    async def test_feature_cards_exist(self, enterprise_page: Page):
        """Verify feature cards are present."""
        cards = await enterprise_page.query_selector_all('.ent-card, .ent-feature-card')
        assert len(cards) >= 2, "Should have at least 2 feature cards"

    async def test_salesforce_card_exists(self, enterprise_page: Page):
        """Verify Salesforce integration card exists."""
        card = enterprise_page.locator('.ent-card:has-text("Salesforce"), .ent-feature-card:has-text("Salesforce")')
        await expect(card).to_be_visible()

    async def test_fiserv_card_exists(self, enterprise_page: Page):
        """Verify Fiserv integration card exists."""
        card = enterprise_page.locator('.ent-card:has-text("Fiserv"), .ent-feature-card:has-text("Banking")')
        await expect(card).to_be_visible()

    async def test_card_hover_effect(self, enterprise_page: Page):
        """Verify hover transforms cards."""
        card = enterprise_page.locator('.ent-card, .ent-feature-card').first
        
        # Get initial state
        initial_transform = await card.evaluate('el => getComputedStyle(el).transform')
        initial_shadow = await card.evaluate('el => getComputedStyle(el).boxShadow')
        
        # Hover
        await card.hover()
        await enterprise_page.wait_for_timeout(400)
        
        # Get hover state
        hover_transform = await card.evaluate('el => getComputedStyle(el).transform')
        hover_shadow = await card.evaluate('el => getComputedStyle(el).boxShadow')
        
        # Either transform or shadow should change
        changed = (initial_transform != hover_transform) or (initial_shadow != hover_shadow)
        assert changed, "Card should have hover effect"

    async def test_salesforce_card_navigates(self, enterprise_page: Page, base_url: str):
        """Verify Salesforce card navigates correctly."""
        card = enterprise_page.locator('.ent-card:has-text("Salesforce") a, .ent-feature-card:has-text("Salesforce") a').first
        
        if await card.count() > 0:
            await card.click()
            await enterprise_page.wait_for_url('**/salesforce/**')


class TestFloatingBlobs:
    """Floating blob animation tests."""

    async def test_blobs_exist(self, enterprise_page: Page):
        """Verify floating blobs are present."""
        blobs = await enterprise_page.query_selector_all('.ent-blob')
        assert len(blobs) >= 2, "Should have at least 2 floating blobs"

    async def test_blob_layer_exists(self, enterprise_page: Page):
        """Verify floating layer container exists."""
        layer = enterprise_page.locator('.ent-floating-layer')
        await expect(layer).to_be_visible()

    async def test_blobs_have_animation(self, enterprise_page: Page):
        """Verify blobs have float animation."""
        blob = enterprise_page.locator('.ent-blob-1')
        
        if await blob.count() > 0:
            animation = await blob.evaluate('el => getComputedStyle(el).animation')
            assert animation != 'none', "Blobs should have animation"


class TestNavigation:
    """Navigation tests."""

    async def test_vox_nav_exists(self, enterprise_page: Page):
        """Verify main navigation exists."""
        nav = enterprise_page.locator('.vox-nav')
        await expect(nav).to_be_visible()

    async def test_logo_links_home(self, enterprise_page: Page):
        """Verify logo links to home."""
        logo = enterprise_page.locator('.vox-logo')
        href = await logo.get_attribute('href')
        assert 'index.html' in href or href == '/', "Logo should link to home"

    async def test_enterprise_dropdown_exists(self, enterprise_page: Page):
        """Verify enterprise dropdown is present."""
        dropdown = enterprise_page.locator('.ent-nav-dropdown')
        await expect(dropdown).to_be_visible()

    async def test_enterprise_dropdown_opens(self, enterprise_page: Page):
        """Verify enterprise dropdown opens on hover."""
        dropdown = enterprise_page.locator('.ent-nav-dropdown')
        await dropdown.hover()
        
        menu = enterprise_page.locator('.ent-nav-dropdown-menu')
        await expect(menu).to_be_visible()


class TestCapabilitiesSection:
    """Capabilities/features section tests."""

    async def test_capabilities_heading(self, enterprise_page: Page):
        """Verify capabilities section has heading."""
        heading = enterprise_page.locator('h2:has-text("Capabilit"), h2:has-text("Feature")')
        if await heading.count() > 0:
            await expect(heading.first).to_be_visible()

    async def test_capability_icons(self, enterprise_page: Page):
        """Verify capability items have icons."""
        icons = await enterprise_page.query_selector_all('.ent-capability-icon, .ent-feature-icon, [data-lucide]')
        # Should have several icons
        assert len(icons) >= 3, "Should have capability icons"


class TestResponsiveness:
    """Responsive design tests."""

    async def test_renders_on_mobile_viewport(self, browser, base_url: str):
        """Test page renders on mobile viewport."""
        context = await browser.new_context(
            viewport={"width": 375, "height": 812}
        )
        page = await context.new_page()
        await page.goto(f"{base_url}/enterprise/")
        
        # Should still show hero
        hero = page.locator('.ent-hero, .ent-hero-title')
        if await hero.count() > 0:
            await expect(hero.first).to_be_visible()
        
        await context.close()
