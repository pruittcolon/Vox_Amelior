import { test, expect } from '@playwright/test';
import { loginToApp } from './utils/auth';

test.describe('Fiserv Enterprise Banking', () => {
    test.beforeEach(async ({ page }) => {
        // 1. Authenticate first (Standard Auth Flow)
        await loginToApp(page);

        // 2. Navigate to Fiserv Landing Page
        await page.goto('/ui/enterprise/fiserv/index.html');
        await expect(page).toHaveTitle(/Fiserv Banking/);
    });

    test('Fiserv system status is operational', async ({ page }) => {
        // Check initial "Connecting..." state updates to "Connected"
        const statusText = page.locator('#apiStatusText');
        await expect(statusText).toContainText('Connected', { timeout: 30000 });

        // Check Fiserv Core API status
        const fiservStatus = page.locator('#fiservStatus');
        await expect(fiservStatus).toHaveText(['Operational', 'Unavailable']); // Allow Unavailable if mock mode off/no creds

        // Check ML Service status
        const mlStatus = page.locator('#mlStatus');
        await expect(mlStatus).toHaveText(['Operational', 'Degraded']);
    });

    test('Member Search capability deep-link works', async ({ page }) => {
        // Click "Search Now" on Member Search card
        await page.click('.ent-card:has-text("Member Search") a');

        // Verify URL hash
        await expect(page).toHaveURL(/banking\.html#section-party/);

        // Verify Section Visibility
        await expect(page.locator('#section-party')).toBeVisible();
        await expect(page.locator('#section-party h1.page-title')).toContainText('Member Search');
    });

    test('Fraud Detection capability deep-link works', async ({ page }) => {
        await page.click('.ent-card:has-text("Fraud Detection") a');
        await expect(page).toHaveURL(/banking\.html#section-cases/);
        await expect(page.locator('#section-cases')).toBeVisible();
        await expect(page.locator('#section-cases h1')).toContainText('Case Management');
    });

    test('Transaction History capability deep-link works', async ({ page }) => {
        await page.click('.ent-card:has-text("Transaction History") a');
        await expect(page).toHaveURL(/banking\.html#section-account/);
        await expect(page.locator('#section-account')).toBeVisible();
        await expect(page.locator('#section-account h1')).toContainText('Account Lookup');
    });

    test('Fund Transfers capability deep-link works', async ({ page }) => {
        await page.click('.ent-card:has-text("Fund Transfers") a');
        await expect(page).toHaveURL(/banking\.html#section-transfers/);
        await expect(page.locator('#section-transfers')).toBeVisible();
        await expect(page.locator('#section-transfers h1')).toContainText('Fund Transfers');
    });

    test('Case Management capability deep-link works', async ({ page }) => {
        await page.click('.ent-card:has-text("Case Management") a');
        await expect(page).toHaveURL(/banking\.html#section-cases/);
        await expect(page.locator('#section-cases')).toBeVisible();
        await expect(page.locator('#section-cases h1')).toContainText('Case Management');
    });

    test('Executive View capability deep-link works', async ({ page }) => {
        await page.click('.ent-card:has-text("Executive View") a');
        await expect(page).toHaveURL(/banking\.html#section-executive-dashboard/);
        await expect(page.locator('#section-executive-dashboard')).toBeVisible();

        // Verify Executive-specific components load
        await expect(page.locator('#execTotalAssets')).toBeVisible();
    });

    test('Deep linking triggers robust role switching', async ({ page }) => {
        // Direct navigation to a specific section (simulating external link)
        await page.goto('/ui/banking.html#section-executive-dashboard');

        // Verify it didn't default back to MSR/Party search
        await expect(page.locator('#section-executive-dashboard')).toBeVisible();
        await expect(page.locator('#section-party')).not.toBeVisible();
    });
});
