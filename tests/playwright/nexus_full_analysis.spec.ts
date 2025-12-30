/**
 * NexusAI Full Analysis E2E Test
 *
 * Comprehensive tests covering the complete analysis flow from login to results.
 * Verifies:
 * - Authentication flow
 * - File upload functionality
 * - Analysis execution
 * - Engine card auto-expansion
 * - Gemma text readability (dark on light)
 * - Clear/Reset button functionality
 *
 * @security All tests use real authentication via cookies
 * @author NemoServer Team
 */

import { test, expect, Page } from '@playwright/test';
import * as path from 'path';
import * as fs from 'fs';

// Configuration
const TEST_USER = process.env.TEST_USER || 'admin';
const TEST_PASS = process.env.TEST_PASS || 'admin123';
const BASE_URL = process.env.BASE_URL || 'http://localhost:8000';

/**
 * Helper to login before tests.
 * @param {Page} page - Playwright page
 */
async function loginHelper(page: Page): Promise<void> {
    await page.goto(`${BASE_URL}/ui/login.html`);

    // Fill login form
    await page.fill('#username', TEST_USER);
    await page.fill('#password', TEST_PASS);
    await page.click('#login-button');

    // Wait for redirect or error
    await Promise.race([
        page.waitForURL(/index|dashboard|banking|nexus/, { timeout: 15000 }),
        page.waitForSelector('.error-message', { timeout: 15000 }),
    ]).catch(() => { });

    // Verify login succeeded by checking cookies
    const cookies = await page.context().cookies();
    const sessionCookie = cookies.find(c => c.name === 'ws_session');
    if (!sessionCookie) {
        throw new Error('Login failed: No session cookie found');
    }
}

/**
 * Create a test CSV file for upload testing.
 * @returns {string} Path to the test file
 */
function createTestCSV(): string {
    const testDir = path.join(__dirname, 'test_data');
    if (!fs.existsSync(testDir)) {
        fs.mkdirSync(testDir, { recursive: true });
    }

    const testFile = path.join(testDir, 'test_analysis.csv');

    // Sample dataset with various column types for ML testing
    const csvContent = `id,date,amount,category,score,status
1,2024-01-01,1500.50,A,85,active
2,2024-01-02,2300.75,B,72,active
3,2024-01-03,890.00,A,91,inactive
4,2024-01-04,3200.25,C,68,active
5,2024-01-05,1750.00,B,88,active
6,2024-01-06,420.50,A,45,inactive
7,2024-01-07,5600.00,C,95,active
8,2024-01-08,980.25,B,77,active
9,2024-01-09,2100.00,A,82,inactive
10,2024-01-10,1350.75,C,69,active`;

    fs.writeFileSync(testFile, csvContent);
    return testFile;
}

test.describe('NexusAI Full Analysis Flow', () => {
    let testFile: string;

    test.beforeAll(() => {
        testFile = createTestCSV();
    });

    test.beforeEach(async ({ page }) => {
        await loginHelper(page);
    });

    test.afterAll(() => {
        // Cleanup test file
        if (fs.existsSync(testFile)) {
            fs.unlinkSync(testFile);
        }
    });

    test('complete analysis flow: upload, analyze, verify readability', async ({ page }) => {
        // Navigate to nexus page
        await page.goto(`${BASE_URL}/ui/nexus.html`);
        await page.waitForLoadState('domcontentloaded');

        // Verify page loaded with correct styling
        const heroTitle = page.locator('.vox-hero-title');
        await expect(heroTitle).toBeVisible({ timeout: 10000 });

        // Verify upload area exists
        const uploadArea = page.locator('#upload-area');
        await expect(uploadArea).toBeVisible();

        // Upload test file
        const fileInput = page.locator('#file-input');
        await fileInput.setInputFiles(testFile);

        // Wait for file processing
        await page.waitForTimeout(2000);

        // Verify analyze button is enabled
        const analyzeBtn = page.locator('#analyze-btn');
        await expect(analyzeBtn).toBeEnabled({ timeout: 10000 });

        // Start analysis
        await analyzeBtn.click();

        // Wait for first engine to complete (up to 60 seconds for Gemma summary)
        const successStatus = page.locator('.engine-status.success').first();
        await expect(successStatus).toBeVisible({ timeout: 60000 });

        // Verify engine card auto-expanded
        const expandedCard = page.locator('.engine-result-card.expanded').first();
        await expect(expandedCard).toBeVisible();

        // CRITICAL: Verify Gemma text is dark and readable
        const gemmaText = page.locator('.gemma-text').first();
        await expect(gemmaText).toBeVisible();

        const textColor = await gemmaText.evaluate(el => {
            return window.getComputedStyle(el).color;
        });

        // Parse RGB value - should be dark (r, g, b all < 150)
        const colorMatch = textColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (colorMatch) {
            const [, r, g, b] = colorMatch.map(Number);
            expect(r).toBeLessThan(150);
            expect(g).toBeLessThan(150);
            expect(b).toBeLessThan(150);
        }

        // Verify Clear/Reset button is visible after analysis starts
        const clearBtn = page.locator('button:has-text("Clear"), button:has-text("Reset")').first();
        await expect(clearBtn).toBeVisible();
    });

    test('engine card background is light theme', async ({ page }) => {
        await page.goto(`${BASE_URL}/ui/nexus.html`);
        await page.waitForLoadState('domcontentloaded');

        // Upload and start analysis
        const fileInput = page.locator('#file-input');
        await fileInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        const analyzeBtn = page.locator('#analyze-btn');
        await expect(analyzeBtn).toBeEnabled({ timeout: 10000 });
        await analyzeBtn.click();

        // Wait for a card to appear
        const engineCard = page.locator('.engine-result-card').first();
        await expect(engineCard).toBeVisible({ timeout: 30000 });

        // Verify background is white/light
        const bgColor = await engineCard.evaluate(el => {
            return window.getComputedStyle(el).backgroundColor;
        });

        // Should be white or very light (RGB values > 200)
        const bgMatch = bgColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (bgMatch) {
            const [, r, g, b] = bgMatch.map(Number);
            expect(r).toBeGreaterThan(200);
            expect(g).toBeGreaterThan(200);
            expect(b).toBeGreaterThan(200);
        }
    });

    test('gemma summary section has proper contrast', async ({ page }) => {
        await page.goto(`${BASE_URL}/ui/nexus.html`);
        await page.waitForLoadState('domcontentloaded');

        // Upload and start analysis
        const fileInput = page.locator('#file-input');
        await fileInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        const analyzeBtn = page.locator('#analyze-btn');
        await expect(analyzeBtn).toBeEnabled({ timeout: 10000 });
        await analyzeBtn.click();

        // Wait for Gemma summary to appear
        const gemmaSummary = page.locator('.gemma-summary').first();
        await expect(gemmaSummary).toBeVisible({ timeout: 60000 });

        // Verify border-left exists (design element)
        const borderLeft = await gemmaSummary.evaluate(el => {
            return window.getComputedStyle(el).borderLeftWidth;
        });
        expect(parseInt(borderLeft)).toBeGreaterThan(0);
    });

    test('clear/reset button clears analysis', async ({ page }) => {
        await page.goto(`${BASE_URL}/ui/nexus.html`);
        await page.waitForLoadState('domcontentloaded');

        // Upload and start analysis
        const fileInput = page.locator('#file-input');
        await fileInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        const analyzeBtn = page.locator('#analyze-btn');
        await expect(analyzeBtn).toBeEnabled({ timeout: 10000 });
        await analyzeBtn.click();

        // Wait for at least one result
        await expect(page.locator('.engine-status.success').first()).toBeVisible({ timeout: 60000 });

        // Find and click Clear/Reset button
        const clearBtn = page.locator('button:has-text("Clear"), button:has-text("Reset")').first();
        await expect(clearBtn).toBeVisible();

        // Accept confirmation dialog
        page.on('dialog', dialog => dialog.accept());
        await clearBtn.click();

        // Wait for clear to complete
        await page.waitForTimeout(1000);

        // Verify engines section is cleared/hidden
        const enginesSection = page.locator('#all-engines-section');
        // Should either be hidden or have no results
        const isHidden = await enginesSection.evaluate(el => {
            const style = window.getComputedStyle(el);
            return style.display === 'none' || el.innerHTML.includes('0/');
        });
        expect(isHidden).toBeTruthy();
    });
});

test.describe('NexusAI UI Components', () => {
    test.beforeEach(async ({ page }) => {
        await loginHelper(page);
    });

    test('navigation links work correctly', async ({ page }) => {
        await page.goto(`${BASE_URL}/ui/nexus.html`);
        await page.waitForLoadState('domcontentloaded');

        // Verify nav exists
        const nav = page.locator('.vox-nav');
        await expect(nav).toBeVisible();

        // Verify key nav links
        const homeLink = page.locator('.vox-nav-link:has-text("Home")');
        const settingsLink = page.locator('.vox-nav-link:has-text("Settings")');

        await expect(homeLink).toBeVisible();
        await expect(settingsLink).toBeVisible();
    });

    test('category cards display correctly', async ({ page }) => {
        await page.goto(`${BASE_URL}/ui/nexus.html`);
        await page.waitForLoadState('domcontentloaded');

        // Verify category tabs/cards exist
        const categoryBtns = page.locator('.engine-category-btn');
        const count = await categoryBtns.count();

        // Should have multiple category buttons (All, ML, Financial, etc.)
        expect(count).toBeGreaterThanOrEqual(3);
    });

    test('no console errors on page load', async ({ page }) => {
        const consoleErrors: string[] = [];

        page.on('console', msg => {
            if (msg.type() === 'error') {
                // Ignore expected API 404s
                if (!msg.text().includes('404') && !msg.text().includes('Failed to load resource')) {
                    consoleErrors.push(msg.text());
                }
            }
        });

        await page.goto(`${BASE_URL}/ui/nexus.html`);
        await page.waitForLoadState('networkidle');

        // Allow some time for async errors
        await page.waitForTimeout(2000);

        expect(consoleErrors.length).toBe(0);
    });
});

test.describe('NexusAI Performance', () => {
    test.beforeEach(async ({ page }) => {
        await loginHelper(page);
    });

    test('page loads within 5 seconds', async ({ page }) => {
        const startTime = Date.now();

        await page.goto(`${BASE_URL}/ui/nexus.html`);
        await page.waitForLoadState('domcontentloaded');

        const loadTime = Date.now() - startTime;

        // Page should load within 5 seconds
        expect(loadTime).toBeLessThan(5000);
    });

    test('all CSS and JS assets load successfully', async ({ page }) => {
        const failedAssets: string[] = [];

        page.on('requestfailed', request => {
            if (request.url().endsWith('.js') || request.url().endsWith('.css')) {
                failedAssets.push(request.url());
            }
        });

        await page.goto(`${BASE_URL}/ui/nexus.html`);
        await page.waitForLoadState('networkidle');

        expect(failedAssets.length).toBe(0);
    });
});
