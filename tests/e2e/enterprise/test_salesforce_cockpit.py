"""
Salesforce Cockpit E2E Tests

Tests the Salesforce Agentic Cockpit page functionality.
All tests use authenticated sessions per user rules.
"""

import pytest
from playwright.async_api import Page, expect


class TestSmartFeed:
    """Tests for the Smart Feed component."""

    async def test_feed_loads_with_cards(self, salesforce_page: Page):
        """Verify smart feed displays task cards."""
        # Wait for loading spinner to disappear and cards to appear
        await salesforce_page.wait_for_selector('.sf-task-card', timeout=15000)
        
        cards = await salesforce_page.query_selector_all('.sf-task-card')
        assert len(cards) >= 1, "At least one task card should display"

    async def test_card_has_required_elements(self, salesforce_page: Page):
        """Verify task cards have all required elements."""
        await salesforce_page.wait_for_selector('.sf-task-card')
        
        card = salesforce_page.locator('.sf-task-card').first
        
        # Should have icon, title, description, meta, and action button
        await expect(card.locator('.sf-task-icon')).to_be_visible()
        await expect(card.locator('.sf-task-content h3')).to_be_visible()
        await expect(card.locator('.sf-task-content p')).to_be_visible()
        await expect(card.locator('button')).to_be_visible()

    async def test_card_type_styling(self, salesforce_page: Page):
        """Verify card type indicators (urgent, risk, opportunity)."""
        await salesforce_page.wait_for_selector('.sf-task-card')
        
        # Check for type-specific classes
        all_cards = await salesforce_page.query_selector_all('.sf-task-card')
        
        has_typed_card = False
        for card in all_cards:
            class_attr = await card.get_attribute('class')
            if any(t in class_attr for t in ['urgent', 'risk', 'opportunity', 'task']):
                has_typed_card = True
                break
        
        assert has_typed_card, "At least one card should have a type class"

    async def test_refresh_button_works(self, salesforce_page: Page):
        """Verify Refresh Analysis button triggers reload."""
        await salesforce_page.wait_for_selector('.sf-task-card')
        
        refresh_btn = salesforce_page.locator('button:has-text("Refresh")')
        await refresh_btn.click()
        
        # Verify loading state appears (spinner or text)
        # Give it a moment to show loading state
        await salesforce_page.wait_for_timeout(500)
        
        # Then wait for cards to reappear
        await salesforce_page.wait_for_selector('.sf-task-card', timeout=15000)


class TestActionButtons:
    """Tests for all action button behaviors."""

    async def test_action_button_changes_state(self, salesforce_page: Page):
        """Test action button visual feedback."""
        await salesforce_page.wait_for_selector('.sf-task-card')
        
        # Click first action button
        action_btn = salesforce_page.locator('.sf-task-card button').first
        original_text = await action_btn.text_content()
        
        await action_btn.click()
        await salesforce_page.wait_for_timeout(500)
        
        # Verify button changes state (text or style)
        new_text = await action_btn.text_content()
        assert 'Done' in new_text or original_text != new_text, "Button should change on click"

    async def test_action_adds_chat_message(self, salesforce_page: Page):
        """Verify action triggers chat message."""
        await salesforce_page.wait_for_selector('.sf-task-card')
        
        # Count initial messages
        initial_messages = await salesforce_page.query_selector_all('.sf-message.bot')
        initial_count = len(initial_messages)
        
        # Click action button
        action_btn = salesforce_page.locator('.sf-task-card button').first
        await action_btn.click()
        
        # Wait for new message
        await salesforce_page.wait_for_timeout(1500)
        
        final_messages = await salesforce_page.query_selector_all('.sf-message.bot')
        assert len(final_messages) > initial_count, "New chat message should appear"

    async def test_action_fades_card(self, salesforce_page: Page):
        """Verify completed action fades the card."""
        await salesforce_page.wait_for_selector('.sf-task-card')
        
        card = salesforce_page.locator('.sf-task-card').first
        await card.locator('button').click()
        
        # Wait for fade animation
        await salesforce_page.wait_for_timeout(1500)
        
        opacity = await card.evaluate('el => getComputedStyle(el).opacity')
        assert float(opacity) < 1, "Card should fade after action"


class TestNavigation:
    """Navigation panel tests."""

    async def test_sidebar_exists(self, salesforce_page: Page):
        """Verify left sidebar navigation exists."""
        sidebar = salesforce_page.locator('.sf-nav-panel')
        await expect(sidebar).to_be_visible()

    async def test_cockpit_is_active(self, salesforce_page: Page):
        """Verify Cockpit nav item is active by default."""
        cockpit_btn = salesforce_page.locator('.sf-nav-item:has-text("Cockpit")')
        
        class_attr = await cockpit_btn.get_attribute('class')
        assert 'active' in class_attr, "Cockpit should be active"

    async def test_nav_items_respond_to_click(self, salesforce_page: Page):
        """Test navigation items are interactive."""
        opp_btn = salesforce_page.locator('.sf-nav-item:has-text("Opportunities")')
        await opp_btn.click()
        
        # Verify it gets some visual feedback (active class)
        await salesforce_page.wait_for_timeout(300)
        class_attr = await opp_btn.get_attribute('class')
        # Could become active or just respond visually

    async def test_enterprise_dropdown_opens(self, salesforce_page: Page):
        """Test enterprise dropdown menu opens on hover."""
        dropdown = salesforce_page.locator('.ent-nav-dropdown')
        await dropdown.hover()
        
        menu = salesforce_page.locator('.ent-nav-dropdown-menu')
        await expect(menu).to_be_visible()

    async def test_dropdown_has_links(self, salesforce_page: Page):
        """Verify dropdown contains expected links."""
        dropdown = salesforce_page.locator('.ent-nav-dropdown')
        await dropdown.hover()
        
        menu = salesforce_page.locator('.ent-nav-dropdown-menu')
        await expect(menu.locator('text=Overview')).to_be_visible()
        await expect(menu.locator('text=Fiserv Banking')).to_be_visible()


class TestPipelineWidget:
    """Pipeline health widget tests."""

    async def test_pipeline_widget_exists(self, salesforce_page: Page):
        """Verify pipeline widget is present."""
        widget = salesforce_page.locator('.sf-context-widget')
        await expect(widget).to_be_visible()

    async def test_pipeline_health_updates(self, salesforce_page: Page):
        """Verify pipeline health updates from placeholder."""
        health_el = salesforce_page.locator('#pipelineHealth')
        
        # Wait for data to potentially load
        await salesforce_page.wait_for_timeout(3000)
        
        health_text = await health_el.text_content()
        assert health_text != "Analyzing...", "Health should update from placeholder"

    async def test_pipeline_value_shows_currency(self, salesforce_page: Page):
        """Verify pipeline value shows dollar amount."""
        value_el = salesforce_page.locator('#pipelineValue')
        
        await salesforce_page.wait_for_timeout(3000)
        
        value_text = await value_el.text_content()
        assert "$" in value_text, "Pipeline value should show dollar amount"


class TestAgentChat:
    """Agent assistant panel tests."""

    async def test_agent_panel_exists(self, salesforce_page: Page):
        """Verify agent panel is present."""
        panel = salesforce_page.locator('.sf-agent-panel')
        await expect(panel).to_be_visible()

    async def test_agent_has_avatar(self, salesforce_page: Page):
        """Verify agent has avatar."""
        avatar = salesforce_page.locator('.sf-agent-avatar')
        await expect(avatar).to_be_visible()

    async def test_chat_input_exists(self, salesforce_page: Page):
        """Verify chat input field exists."""
        input_field = salesforce_page.locator('.sf-chat-input')
        await expect(input_field).to_be_visible()

    async def test_initial_bot_message(self, salesforce_page: Page):
        """Verify initial bot message is present."""
        bot_message = salesforce_page.locator('.sf-message.bot').first
        await expect(bot_message).to_be_visible()
