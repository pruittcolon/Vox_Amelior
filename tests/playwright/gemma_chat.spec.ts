import { test, expect } from '@playwright/test';

test.describe('Gemma Feature Verification', () => {

    test.beforeEach(async ({ page }) => {
        // 1. Login
        await page.goto('/');
        await page.fill('#username', 'demo');
        await page.fill('#password', 'demo123');
        await page.click('button[type="submit"]');
        await expect(page).toHaveURL(/nexus.html/, { timeout: 30000 });

        // 2. Navigate to Gemma
        await page.goto('/gemma.html');
        // Wait for initial load
        await expect(page.locator('.gemma-shell')).toBeVisible({ timeout: 30000 });
    });

    test('Gemma Chat /api/public/chat', async ({ page }) => {
        // 1. Wait for warmup / GPU stats
        await expect(page.locator('#model-status-indicator')).toBeVisible();

        // 2. Type message
        const chatInput = page.locator('#chat-input');
        await chatInput.fill('Hello, are you working?');
        await page.keyboard.press('Enter');

        // 3. Verify user message appears
        await expect(page.locator('.user-message').last()).toContainText('Hello, are you working?');

        // 4. Verify AI thinking/response
        // Wait for response (can take 10s for first token on cold boot)
        await expect(page.locator('.ai-message').last()).toBeVisible({ timeout: 45000 });
    });

    test('Transcripts Tab', async ({ page }) => {
        // 1. Switch to transcripts
        await page.click('button[data-tab="transcripts"]');

        // 2. Verify list loads
        const list = page.locator('#transcript-list');
        await expect(list).toBeVisible();
    });

    test('Emotions Tab Analytics', async ({ page }) => {
        // 1. Switch to emotions
        await page.click('button[data-tab="emotions"]');

        // 2. Verify dashboard loads
        await expect(page.locator('#emotion-distribution-chart')).toBeVisible();
        await expect(page.locator('#stat-joy')).toBeVisible();
    });

    test('Databases Tab Upload & List', async ({ page }) => {
        // 1. Switch to databases
        await page.click('button[data-tab="databases"]');

        // 2. Verify existing databases list
        await expect(page.locator('#existing-databases')).toBeVisible();

        // 3. Verify file upload area
        await expect(page.locator('#db-upload-zone')).toBeVisible();
    });

    test('Search Tab', async ({ page }) => {
        // 1. Switch to search
        await page.click('button[data-tab="search"]');

        // 2. Perform search
        await page.fill('#semantic-search-input', 'test query');
        await page.click('#search-btn');

        // 3. Check for results or no results message
        await expect(page.locator('#search-results')).toBeVisible();
    });

    test('Quality Intelligence Injection', async ({ page }) => {
        // This dashboard is injected dynamically
        // We can simulate having a database selected or check if the valid HTML container exists
        await page.click('button[data-tab="databases"]');

        // Check if the quality module code is at least present/loaded
        const qualityModule = await page.evaluate(() => typeof window.QualityIntelligence);
        expect(qualityModule).toBe('object');
    });
});
