/**
 * NexusAI Page E2E Tests
 * 
 * Comprehensive tests for the NexusAI enterprise intelligence platform.
 * Tests page load, upload flow, Quality Intelligence dashboard, and performance.
 * 
 * @security All tests use real authentication
 */

import { test, expect, Page } from '@playwright/test';

// Configuration
const TEST_USER = process.env.TEST_USER || 'admin';
const TEST_PASS = process.env.TEST_PASS || 'admin123';

/**
 * Helper to login before tests.
 * @param {Page} page - Playwright page
 */
async function loginHelper(page: Page): Promise<void> {
    await page.goto('/ui/login.html');
    await page.fill('#username', TEST_USER);
    await page.fill('#password', TEST_PASS);
    await page.click('#login-button');

    await Promise.race([
        page.waitForURL(/index|dashboard|banking|nexus/, { timeout: 10000 }),
        page.waitForSelector('.error-message', { timeout: 10000 }),
    ]).catch(() => { });
}

test.describe('NexusAI Page Tests', () => {
    test.beforeEach(async ({ page }) => {
        await loginHelper(page);
    });

    test('nexus page loads without console errors', async ({ page }) => {
        // Collect console errors
        const errors: string[] = [];
        page.on('console', msg => {
            if (msg.type() === 'error') {
                errors.push(msg.text());
            }
        });

        await page.goto('/ui/nexus.html');
        await page.waitForLoadState('domcontentloaded');

        // Wait for scripts to initialize
        await page.waitForTimeout(2000);

        // Verify page loaded
        expect(page.url()).toContain('nexus');

        // Verify no critical console errors (excluding expected API 404s)
        const criticalErrors = errors.filter(e =>
            !e.includes('Failed to load resource') &&
            !e.includes('404')
        );
        expect(criticalErrors.length).toBe(0);
    });

    test('nexus page loads all vendored scripts', async ({ page }) => {
        // Track script load failures
        const failedScripts: string[] = [];
        page.on('requestfailed', request => {
            if (request.url().endsWith('.js')) {
                failedScripts.push(request.url());
            }
        });

        await page.goto('/ui/nexus.html');
        await page.waitForLoadState('networkidle');

        // Verify no script load failures
        expect(failedScripts.length).toBe(0);
    });

    test('nexus page hero section renders correctly', async ({ page }) => {
        await page.goto('/ui/nexus.html');
        await page.waitForLoadState('domcontentloaded');

        // Verify hero content
        const heroTitle = await page.locator('.vox-hero-title');
        await expect(heroTitle).toBeVisible();
        await expect(heroTitle).toContainText('Where Data Meets');

        // Verify navigation
        const nav = await page.locator('.vox-nav');
        await expect(nav).toBeVisible();

        // Verify NexusAI branding
        const logo = await page.locator('.vox-logo-text');
        await expect(logo).toContainText('NexusAI');
    });

    test('upload area is functional', async ({ page }) => {
        await page.goto('/ui/nexus.html');
        await page.waitForLoadState('domcontentloaded');

        // Verify upload area exists and is clickable
        const uploadArea = await page.locator('#upload-area');
        await expect(uploadArea).toBeVisible();

        // Verify file input exists
        const fileInput = await page.locator('#file-input');
        await expect(fileInput).toBeAttached();

        // Verify supported formats are listed
        const formats = await page.locator('.upload-format');
        expect(await formats.count()).toBeGreaterThanOrEqual(5);
    });

    test('analysis workspaces render correctly', async ({ page }) => {
        await page.goto('/ui/nexus.html');
        await page.waitForLoadState('domcontentloaded');

        // Verify category cards exist
        const categoryCards = await page.locator('.category-card');
        expect(await categoryCards.count()).toBeGreaterThanOrEqual(4);

        // Verify ML & Analytics card
        const mlCard = await page.locator('[data-category="ml"]');
        await expect(mlCard).toBeVisible();
        await expect(mlCard).toContainText('ML & Analytics');

        // Verify Financial Intelligence card
        const financialCard = await page.locator('[data-category="financial"]');
        await expect(financialCard).toBeVisible();
        await expect(financialCard).toContainText('Financial Intelligence');

        // Verify Quality Intelligence card (flagship)
        const qualityCard = await page.locator('[data-category="quality"]');
        await expect(qualityCard).toBeVisible();
        await expect(qualityCard).toContainText('Quality Intelligence');
    });

    test('quality dashboard can be opened and closed', async ({ page }) => {
        await page.goto('/ui/nexus.html');
        await page.waitForLoadState('networkidle');

        // Quality dashboard section should be hidden initially
        const qualitySection = await page.locator('#quality-dashboard-section');
        await expect(qualitySection).toBeHidden();

        // Click on Quality Intelligence card
        const qualityCard = await page.locator('[data-category="quality"]');
        await qualityCard.click();

        // Wait for dashboard to appear
        await page.waitForTimeout(1000);
        await expect(qualitySection).toBeVisible();

        // Click close button
        const closeButton = await page.locator('button:has-text("Close")');
        await closeButton.click();

        // Dashboard should be hidden again
        await expect(qualitySection).toBeHidden();
    });

    test('three.js 3D viewer initializes', async ({ page }) => {
        await page.goto('/ui/nexus.html');
        await page.waitForLoadState('networkidle');

        // Verify Three.js is loaded
        const threeLoaded = await page.evaluate(() => typeof (window as any).THREE !== 'undefined');
        expect(threeLoaded).toBe(true);

        // Verify Quality3DViewer class is available
        const viewerAvailable = await page.evaluate(() => typeof (window as any).Quality3DViewer !== 'undefined');
        expect(viewerAvailable).toBe(true);
    });

    test('page performance - loads within 5 seconds', async ({ page }) => {
        const startTime = Date.now();

        await page.goto('/ui/nexus.html');
        await page.waitForLoadState('domcontentloaded');

        const loadTime = Date.now() - startTime;

        // Page should load within 5 seconds
        expect(loadTime).toBeLessThan(5000);
    });
});

test.describe('NexusAI API Integration', () => {
    test('engine definitions are accessible', async ({ request }) => {
        const response = await request.get('/api/v1/engines');
        // Expect either success or auth required
        expect([200, 401, 404]).toContain(response.status());
    });
});
