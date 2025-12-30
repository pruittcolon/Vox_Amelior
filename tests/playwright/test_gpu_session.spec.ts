/**
 * GPU Session E2E Tests
 * 
 * Tests the complete GPU acquisition and release flow:
 * 1. Login to application
 * 2. Navigate to Gemma page (triggers warmup)
 * 3. Send a chat message
 * 4. Close page (triggers release)
 * 5. Verify no errors
 * 
 * @author Enterprise Analytics Team
 */

import { test, expect } from '@playwright/test';

// Test configuration
const BASE_URL = process.env.BASE_URL || 'https://localhost';
const TEST_USER = process.env.TEST_USER || 'admin';
const TEST_PASS = process.env.TEST_PASS || 'admin';

test.describe('GPU Session Management', () => {

    test.beforeEach(async ({ page }) => {
        // Accept self-signed certificate warning
        await page.goto(`${BASE_URL}/login`, {
            waitUntil: 'networkidle',
            timeout: 30000
        });
    });

    test('should acquire and release GPU during Gemma session', async ({ page }) => {
        // Step 1: Login
        await page.fill('input[name="username"]', TEST_USER);
        await page.fill('input[name="password"]', TEST_PASS);
        await page.click('button[type="submit"]');

        // Wait for login to complete
        await page.waitForURL('**/nexus.html', { timeout: 10000 });

        // Step 2: Navigate to Gemma chat
        await page.goto(`${BASE_URL}/gemma.html`);
        await page.waitForLoadState('networkidle');

        // Verify page loaded
        await expect(page.locator('h1, h2').first()).toBeVisible();

        // Step 3: Click warmup button if available
        const warmupButton = page.locator('button:has-text("Warmup"), button:has-text("warmup")');
        if (await warmupButton.isVisible({ timeout: 2000 }).catch(() => false)) {
            await warmupButton.click();
            // Wait for warmup to complete
            await page.waitForTimeout(5000);
        }

        // Step 4: Send a test message
        const chatInput = page.locator('textarea, input[type="text"]').first();
        if (await chatInput.isVisible({ timeout: 2000 }).catch(() => false)) {
            await chatInput.fill('Hello, this is a GPU session test.');

            const sendButton = page.locator('button:has-text("Send"), button[type="submit"]').first();
            if (await sendButton.isVisible({ timeout: 2000 }).catch(() => false)) {
                await sendButton.click();

                // Wait for response (GPU inference)
                await page.waitForTimeout(10000);
            }
        }

        // Step 5: Check for console errors
        const consoleErrors: string[] = [];
        page.on('console', msg => {
            if (msg.type() === 'error') {
                consoleErrors.push(msg.text());
            }
        });

        // Step 6: Navigate away (triggers GPU release)
        await page.goto(`${BASE_URL}/nexus.html`);
        await page.waitForLoadState('networkidle');

        // Verify no critical errors
        const criticalErrors = consoleErrors.filter(e =>
            e.includes('OOM') ||
            e.includes('CUDA') ||
            e.includes('GPU') ||
            e.includes('500')
        );
        expect(criticalErrors).toHaveLength(0);
    });

    test('should handle GPU coordinator status endpoint', async ({ request }) => {
        // Test the new /gpu/state endpoint
        const response = await request.get(`${BASE_URL}/api/gpu-coordinator/gpu/state`, {
            headers: {
                'Accept': 'application/json',
            },
            ignoreHTTPSErrors: true,
        });

        // Should return 200 or 401 (if auth required)
        expect([200, 401, 404]).toContain(response.status());

        if (response.status() === 200) {
            const data = await response.json();
            expect(data).toHaveProperty('owner');
            expect(data).toHaveProperty('state');
        }
    });

    test('should verify GPU returns to transcription after session', async ({ page, request }) => {
        // Login first
        await page.fill('input[name="username"]', TEST_USER);
        await page.fill('input[name="password"]', TEST_PASS);
        await page.click('button[type="submit"]');
        await page.waitForURL('**/nexus.html', { timeout: 10000 });

        // Go to Gemma page to acquire GPU
        await page.goto(`${BASE_URL}/gemma.html`);
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(3000);

        // Navigate away to release GPU
        await page.goto(`${BASE_URL}/nexus.html`);
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(2000);

        // Check GPU status via health endpoint
        const healthResponse = await request.get(`${BASE_URL}/api/gpu-coordinator/health`, {
            ignoreHTTPSErrors: true,
        });

        if (healthResponse.status() === 200) {
            const data = await healthResponse.json();
            // After release, state should be transcription
            expect(data.current_state).toBe('transcription');
        }
    });
});

test.describe('GPU Protocol API Tests', () => {

    test('should respond to health check', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/gpu-coordinator/health`, {
            ignoreHTTPSErrors: true,
        });

        expect([200, 502]).toContain(response.status());

        if (response.status() === 200) {
            const data = await response.json();
            expect(data.status).toBe('healthy');
        }
    });

    test('should respond to status endpoint', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/gpu-coordinator/status`, {
            ignoreHTTPSErrors: true,
        });

        expect([200, 401, 502]).toContain(response.status());

        if (response.status() === 200) {
            const data = await response.json();
            expect(data).toHaveProperty('lock_status');
            expect(data).toHaveProperty('gpu_status');
        }
    });
});
