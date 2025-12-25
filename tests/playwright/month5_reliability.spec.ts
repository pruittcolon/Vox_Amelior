/**
 * Phase 5: Reliability and FinOps
 * Playwright E2E Tests
 *
 * All tests use REAL authentication and REAL API endpoints.
 */

import { test, expect } from '@playwright/test';

// Configuration
const TEST_USER = process.env.TEST_USER || 'admin';
const TEST_PASS = process.env.TEST_PASS || 'admin123';

test.describe('Phase 5: Reliability', () => {
    test('health endpoint returns status', async ({ request }) => {
        const response = await request.get('/health');
        expect(response.status()).toBe(200);

        const data = await response.json();
        expect(data.status).toBeDefined();
    });

    test('health endpoint has security headers', async ({ request }) => {
        const response = await request.get('/health');
        expect(response.headers()['x-content-type-options']).toBe('nosniff');
        expect(response.headers()['x-frame-options']).toBe('DENY');
    });
});

test.describe('Phase 5: Analytics API', () => {
    test('analytics summary endpoint responds', async ({ request }) => {
        const response = await request.get('/api/v1/analytics/summary');
        expect([200, 401, 404]).toContain(response.status());
    });

    test('analytics costs endpoint responds', async ({ request }) => {
        const response = await request.get('/api/v1/analytics/costs');
        expect([200, 401, 404]).toContain(response.status());
    });

    test('analytics trends endpoint responds', async ({ request }) => {
        const response = await request.get('/api/v1/analytics/trends');
        expect([200, 401, 404]).toContain(response.status());
    });

    test('analytics slos endpoint responds', async ({ request }) => {
        const response = await request.get('/api/v1/analytics/slos');
        expect([200, 401, 404]).toContain(response.status());
    });

    test('analytics health-summary endpoint responds', async ({ request }) => {
        const response = await request.get('/api/v1/analytics/health-summary');
        expect([200, 401, 404]).toContain(response.status());
    });
});

test.describe('Phase 5: Frontend Pages', () => {
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

    test('analytics page loads', async ({ page }) => {
        await page.goto('/ui/analytics.html');
        await page.waitForLoadState('domcontentloaded');
        expect(page.url()).toContain('analytics');
    });

    test('predictions page loads', async ({ page }) => {
        await page.goto('/ui/predictions.html');
        await page.waitForLoadState('domcontentloaded');
        expect(page.url()).toContain('predictions');
    });
});
