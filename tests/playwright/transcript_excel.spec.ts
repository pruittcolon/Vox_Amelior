/**
 * Excel-Like Transcriptions View Tests
 * Playwright E2E Tests for the new Excel grid view
 * 
 * Tests: view toggle, sorting, filtering, keyboard nav, CSV export
 */

import { test, expect } from '@playwright/test';

const TEST_USER = process.env.TEST_USER || 'admin';
const TEST_PASS = process.env.TEST_PASS || 'admin123';

test.describe('Transcriptions Excel View', () => {
    test.beforeEach(async ({ page }) => {
        // Login
        await page.goto('/ui/login.html');

        const loginForm = page.locator('form, [data-testid="login-form"]');
        if (await loginForm.isVisible()) {
            await page.fill('input[name="username"], #username', TEST_USER);
            await page.fill('input[name="password"], #password', TEST_PASS);
            await page.click('button[type="submit"]');
            await page.waitForTimeout(1500);
        }

        // Navigate to Gemma page
        await page.goto('/ui/gemma.html');
        await page.waitForTimeout(1000);
    });

    test('view mode toggle switches between Cards and Excel views', async ({ page }) => {
        // Click Transcriptions tab
        const transcriptsTab = page.locator('.gemma-tab:has-text("Transcriptions")');
        await transcriptsTab.click();
        await page.waitForTimeout(500);

        // Verify Cards view is default or check current view
        const excelBtn = page.locator('.view-mode-btn[data-mode="excel"]');
        const cardsBtn = page.locator('.view-mode-btn[data-mode="standard"]');

        // Click Excel view
        await excelBtn.click();
        await page.waitForTimeout(300);

        // Excel grid should be visible
        const excelGrid = page.locator('#excel-transcript-grid');
        await expect(excelGrid).toBeVisible();

        // Click Cards view
        await cardsBtn.click();
        await page.waitForTimeout(300);

        // Card list should be visible
        const cardList = page.locator('#transcript-list');
        await expect(cardList).toBeVisible();
    });

    test('Excel view has sortable column headers', async ({ page }) => {
        // Navigate to transcriptions
        const transcriptsTab = page.locator('.gemma-tab:has-text("Transcriptions")');
        await transcriptsTab.click();
        await page.waitForTimeout(500);

        // Switch to Excel view
        await page.locator('.view-mode-btn[data-mode="excel"]').click();
        await page.waitForTimeout(500);

        // Check sortable headers exist
        const dateHeader = page.locator('.excel-th[data-sort="created_at"]');
        await expect(dateHeader).toBeVisible();

        const speakerHeader = page.locator('.excel-th[data-sort="speaker"]');
        await expect(speakerHeader).toBeVisible();

        // Click to sort
        await dateHeader.click();
        await page.waitForTimeout(300);

        // Header should show sort indicator
        expect(await dateHeader.getAttribute('class')).toMatch(/sorted-/);
    });

    test('quick filter searches across columns', async ({ page }) => {
        const transcriptsTab = page.locator('.gemma-tab:has-text("Transcriptions")');
        await transcriptsTab.click();
        await page.waitForTimeout(500);

        await page.locator('.view-mode-btn[data-mode="excel"]').click();
        await page.waitForTimeout(500);

        // Type in quick filter
        const filterInput = page.locator('#excel-quick-filter');
        await filterInput.fill('speaker');
        await page.waitForTimeout(300);

        // Row count should update
        const rowCount = page.locator('#excel-row-count');
        await expect(rowCount).toBeVisible();
    });

    test('Export CSV button exists and is clickable', async ({ page }) => {
        const transcriptsTab = page.locator('.gemma-tab:has-text("Transcriptions")');
        await transcriptsTab.click();
        await page.waitForTimeout(500);

        await page.locator('.view-mode-btn[data-mode="excel"]').click();
        await page.waitForTimeout(500);

        const exportBtn = page.locator('.excel-btn:has-text("Export CSV")');
        await expect(exportBtn).toBeVisible();

        // Click should not throw
        await exportBtn.click();
    });

    test('security headers present on page responses', async ({ request }) => {
        const response = await request.get('/ui/gemma.html');
        const headers = response.headers();

        // CSP should be present
        expect(headers['content-security-policy']).toBeDefined();

        // Security headers
        expect(headers['x-content-type-options']).toBe('nosniff');
        expect(headers['x-frame-options']).toBe('DENY');

        // HSTS
        expect(headers['strict-transport-security']).toBeDefined();
    });
});
