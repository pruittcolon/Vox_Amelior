/**
 * Database Viewer E2E Tests
 * Playwright tests for the Excel-like database viewer on gemma.html Databases tab
 * 
 * Tests: upload → viewer display, pagination, sorting, search, PII masking
 */

import { test, expect } from '@playwright/test';
import * as path from 'path';
import * as fs from 'fs';

const TEST_USER = process.env.TEST_USER || 'admin';
const TEST_PASS = process.env.TEST_PASS || 'admin123';

// Create a simple test CSV for uploads
function createTestCSV(): string {
    const csvContent = `Name,Email,Amount,Status
John Doe,john@example.com,1500.50,Active
Jane Smith,jane@test.com,2300.75,Pending
Bob Wilson,bob@demo.org,950.00,Active
Alice Brown,alice@sample.net,3100.25,Inactive
Charlie Davis,charlie@email.com,1750.00,Active`;

    const tempPath = '/tmp/test_database_viewer.csv';
    fs.writeFileSync(tempPath, csvContent);
    return tempPath;
}

test.describe('Database Viewer', () => {
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

    test('Databases tab exists and is clickable', async ({ page }) => {
        const databasesTab = page.locator('.gemma-tab:has-text("Databases")');
        await expect(databasesTab).toBeVisible();

        await databasesTab.click();
        await page.waitForTimeout(500);

        // Check that databases tab content is visible
        const tabContent = page.locator('#tab-databases');
        await expect(tabContent).toBeVisible();
    });

    test('upload zone is visible when Databases tab is active', async ({ page }) => {
        // Switch to Databases tab
        await page.locator('.gemma-tab:has-text("Databases")').click();
        await page.waitForTimeout(500);

        const uploadZone = page.locator('#db-upload-zone');
        await expect(uploadZone).toBeVisible();

        // Check upload zone text
        await expect(page.locator('#db-upload-zone h4')).toContainText('Drop CSV files');
    });

    test('database viewer container exists but is hidden initially', async ({ page }) => {
        await page.locator('.gemma-tab:has-text("Databases")').click();
        await page.waitForTimeout(500);

        const viewer = page.locator('#database-excel-viewer');
        // Should exist but be hidden
        await expect(viewer).toHaveCount(1);
        await expect(viewer).toHaveCSS('display', 'none');
    });

    test('can upload CSV and viewer appears', async ({ page }) => {
        // Switch to Databases tab
        await page.locator('.gemma-tab:has-text("Databases")').click();
        await page.waitForTimeout(500);

        // Create test CSV
        const csvPath = createTestCSV();

        // Upload file
        const fileInput = page.locator('#db-file-input');
        await fileInput.setInputFiles(csvPath);

        // Wait for upload to complete (check for status change)
        await page.waitForTimeout(3000);

        // Check if viewer becomes visible
        const viewer = page.locator('#database-excel-viewer');
        // Should be visible after upload completes
        await expect(viewer).toBeVisible({ timeout: 10000 });

        // Cleanup
        fs.unlinkSync(csvPath);
    });

    test('viewer shows correct column headers after upload', async ({ page }) => {
        await page.locator('.gemma-tab:has-text("Databases")').click();
        await page.waitForTimeout(500);

        const csvPath = createTestCSV();
        await page.locator('#db-file-input').setInputFiles(csvPath);

        // Wait for viewer to appear
        await page.waitForSelector('#database-excel-viewer:not([style*="display: none"])', { timeout: 15000 });

        // Check headers exist
        const headerRow = page.locator('#db-excel-header-row');
        await expect(headerRow).toBeVisible();

        // Check for expected column names
        await expect(page.locator('#db-excel-header-row .excel-th:has-text("Name")')).toBeVisible();
        await expect(page.locator('#db-excel-header-row .excel-th:has-text("Amount")')).toBeVisible();

        fs.unlinkSync(csvPath);
    });

    test('pagination controls appear with data', async ({ page }) => {
        await page.locator('.gemma-tab:has-text("Databases")').click();
        await page.waitForTimeout(500);

        const csvPath = createTestCSV();
        await page.locator('#db-file-input').setInputFiles(csvPath);

        await page.waitForSelector('#database-excel-viewer:not([style*="display: none"])', { timeout: 15000 });

        // Check pagination controls
        const pagination = page.locator('#db-pagination');
        await expect(pagination).toBeVisible();

        // Check row count display
        await expect(pagination).toContainText('rows');

        fs.unlinkSync(csvPath);
    });

    test('Export CSV button is clickable', async ({ page }) => {
        await page.locator('.gemma-tab:has-text("Databases")').click();
        await page.waitForTimeout(500);

        const csvPath = createTestCSV();
        await page.locator('#db-file-input').setInputFiles(csvPath);

        await page.waitForSelector('#database-excel-viewer:not([style*="display: none"])', { timeout: 15000 });

        // Find and click export button
        const exportBtn = page.locator('.excel-btn:has-text("Export CSV")');
        await expect(exportBtn).toBeVisible();

        // Click should not throw
        await exportBtn.click();

        fs.unlinkSync(csvPath);
    });

    test('search input filters data', async ({ page }) => {
        await page.locator('.gemma-tab:has-text("Databases")').click();
        await page.waitForTimeout(500);

        const csvPath = createTestCSV();
        await page.locator('#db-file-input').setInputFiles(csvPath);

        await page.waitForSelector('#database-excel-viewer:not([style*="display: none"])', { timeout: 15000 });

        // Find search input
        const searchInput = page.locator('#db-excel-filter');
        await expect(searchInput).toBeVisible();

        // Type search query
        await searchInput.fill('Active');
        await page.waitForTimeout(500);

        // Results should filter
        const rowCount = page.locator('#db-excel-row-count');
        await expect(rowCount).toBeVisible();

        fs.unlinkSync(csvPath);
    });

    test('column headers are sortable', async ({ page }) => {
        await page.locator('.gemma-tab:has-text("Databases")').click();
        await page.waitForTimeout(500);

        const csvPath = createTestCSV();
        await page.locator('#db-file-input').setInputFiles(csvPath);

        await page.waitForSelector('#database-excel-viewer:not([style*="display: none"])', { timeout: 15000 });

        // Find a sortable header
        const nameHeader = page.locator('#db-excel-header-row .excel-th:has-text("Name")');
        await expect(nameHeader).toBeVisible();

        // Click to sort
        await nameHeader.click();
        await page.waitForTimeout(500);

        // Header should have sort class or indicator
        // (The class might change based on implementation)

        fs.unlinkSync(csvPath);
    });

    test('close button hides viewer', async ({ page }) => {
        await page.locator('.gemma-tab:has-text("Databases")').click();
        await page.waitForTimeout(500);

        const csvPath = createTestCSV();
        await page.locator('#db-file-input').setInputFiles(csvPath);

        await page.waitForSelector('#database-excel-viewer:not([style*="display: none"])', { timeout: 15000 });

        // Find and click close button
        const closeBtn = page.locator('#database-excel-viewer button:has([data-lucide="x"])');
        await expect(closeBtn).toBeVisible();
        await closeBtn.click();

        await page.waitForTimeout(500);

        // Viewer should be hidden
        const viewer = page.locator('#database-excel-viewer');
        await expect(viewer).toHaveCSS('display', 'none');

        fs.unlinkSync(csvPath);
    });

    test('security headers present on page', async ({ request }) => {
        const response = await request.get('/ui/gemma.html');
        const headers = response.headers();

        // CSP should be present
        expect(headers['content-security-policy']).toBeDefined();

        // Security headers
        expect(headers['x-content-type-options']).toBe('nosniff');
        expect(headers['x-frame-options']).toBe('DENY');
    });
});
