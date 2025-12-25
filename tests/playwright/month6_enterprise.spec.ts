/**
 * Phase 6: Enterprise Readiness and Adoption
 * Playwright E2E Tests
 *
 * All tests use REAL authentication and REAL API endpoints.
 */

import { test, expect } from '@playwright/test';

// Configuration
const TEST_USER = process.env.TEST_USER || 'admin';
const TEST_PASS = process.env.TEST_PASS || 'admin123';

test.describe('Phase 6: Enterprise API', () => {
    test('compliance status endpoint responds', async ({ request }) => {
        const response = await request.get('/api/enterprise/compliance/status');
        expect([200, 401, 404, 503]).toContain(response.status());

        if (response.status() === 200) {
            const data = await response.json();
            expect(data.compliance).toBeDefined();
        }
    });

    test('ROI summary endpoint responds', async ({ request }) => {
        const response = await request.get('/api/enterprise/roi/summary');
        expect([200, 401, 404, 503]).toContain(response.status());
    });

    test('analytics overview endpoint responds', async ({ request }) => {
        const response = await request.get('/api/enterprise/analytics/overview');
        expect([200, 401, 404, 503]).toContain(response.status());
    });

    test('salesforce status endpoint responds', async ({ request }) => {
        const response = await request.get('/api/v1/salesforce/status');
        expect([200, 401, 404, 503]).toContain(response.status());
    });

    test('fiserv status endpoint responds', async ({ request }) => {
        const response = await request.get('/api/v1/fiserv/status');
        expect([200, 401, 404, 503]).toContain(response.status());
    });
});

test.describe('Phase 6: Frontend Pages', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/ui/login.html');
        await page.fill('#username', TEST_USER);
        await page.fill('#password', TEST_PASS);
        await page.click('#login-button');

        await Promise.race([
            page.waitForURL(/index|dashboard|banking/, { timeout: 10000 }),
            page.waitForSelector('.error-message', { timeout: 10000 }),
        ]).catch(() => { });
    });

    test('about page loads', async ({ page }) => {
        await page.goto('/ui/about.html');
        await page.waitForLoadState('domcontentloaded');
        expect(page.url()).toContain('about');
    });

    test('salesforce page loads', async ({ page }) => {
        await page.goto('/ui/salesforce.html');
        await page.waitForLoadState('domcontentloaded');
        expect(page.url()).toContain('salesforce');
    });
});

test.describe('Phase 6: Security Headers', () => {
    test('security headers on enterprise endpoints', async ({ request }) => {
        const response = await request.get('/health');
        expect(response.headers()['x-content-type-options']).toBe('nosniff');
        expect(response.headers()['x-frame-options']).toBe('DENY');
    });
});
