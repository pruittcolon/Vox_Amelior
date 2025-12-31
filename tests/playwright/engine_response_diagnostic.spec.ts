/**
 * Playwright Test: Engine Backend Response Diagnostic
 * 
 * This test captures the exact backend response for multiple engines
 * and compares with what the frontend visualization expects.
 */

import { test, expect, request } from '@playwright/test';

const BASE_URL = 'http://localhost:8000';

test.describe('Engine Backend Response Diagnostic', () => {
    let apiContext;
    let authCookie;

    test.beforeAll(async () => {
        // Create request context and login
        apiContext = await request.newContext({
            baseURL: BASE_URL,
        });

        // Login to get auth cookie
        const loginResponse = await apiContext.post('/ui/api/login', {
            data: {
                username: 'admin',
                password: 'admin123'
            }
        });

        console.log('[AUTH] Login status:', loginResponse.status());

        const cookies = loginResponse.headers()['set-cookie'];
        if (cookies) {
            authCookie = cookies;
            console.log('[AUTH] Got auth cookie');
        }
    });

    test('Capture Cost Optimization backend response', async ({ page }) => {
        // Login via browser
        await page.goto(`${BASE_URL}/ui/login.html`);
        await page.fill('#username', 'admin');
        await page.fill('#password', 'admin123');
        await page.click('#login-button');
        await page.waitForURL('**/predictions.html');
        console.log('[LOGIN] Logged in successfully');

        // Navigate to nexus
        await page.goto(`${BASE_URL}/ui/nexus.html`);
        await page.waitForLoadState('networkidle');

        // Intercept API responses
        const responses = {};

        page.on('response', async (response) => {
            const url = response.url();
            if (url.includes('/analytics/run-engine/cost')) {
                const data = await response.json().catch(() => null);
                responses['cost'] = data;
                console.log('\n========== COST OPTIMIZATION RESPONSE ==========');
                console.log('Status:', response.status());
                console.log('Keys:', data ? Object.keys(data) : 'null');
                console.log('cost_breakdown:', JSON.stringify(data?.cost_breakdown)?.substring(0, 500));
                console.log('pareto:', JSON.stringify(data?.pareto)?.substring(0, 500));
                console.log('categories:', JSON.stringify(data?.categories)?.substring(0, 500));
                console.log('breakdown:', JSON.stringify(data?.breakdown)?.substring(0, 500));
                console.log('=================================================\n');
            }
            if (url.includes('/analytics/run-engine/resource')) {
                const data = await response.json().catch(() => null);
                responses['resource'] = data;
                console.log('\n========== RESOURCE UTILIZATION RESPONSE ==========');
                console.log('Status:', response.status());
                console.log('Keys:', data ? Object.keys(data) : 'null');
                console.log('summary:', JSON.stringify(data?.summary));
                console.log('utilization:', data?.utilization);
                console.log('avg_utilization:', data?.avg_utilization);
                console.log('resources:', JSON.stringify(data?.resources)?.substring(0, 500));
                console.log('====================================================\n');
            }
            if (url.includes('/analytics/run-engine/rag')) {
                const data = await response.json().catch(() => null);
                responses['rag'] = data;
                console.log('\n========== RAG EVALUATION RESPONSE ==========');
                console.log('Status:', response.status());
                console.log('Keys:', data ? Object.keys(data) : 'null');
                console.log('precision:', data?.precision);
                console.log('recall:', data?.recall);
                console.log('f1:', data?.f1 || data?.f1_score);
                console.log('accuracy:', data?.accuracy);
                console.log('retrieval_precision:', data?.retrieval_precision);
                console.log('==============================================\n');
            }
        });

        // Try running just cost engine directly
        console.log('\n[TEST] Clicking Cost Optimization engine button...');

        // Find and click cost engine test button
        const costBtn = page.locator('button:has-text("Cost"), [data-engine="cost"]').first();
        if (await costBtn.isVisible()) {
            await costBtn.click();
            await page.waitForTimeout(5000);
        } else {
            console.log('[TEST] Cost button not found, trying to start full analysis...');

            // Upload file first
            const fileInput = page.locator('input[type="file"]').first();
            if (await fileInput.isVisible()) {
                await fileInput.setInputFiles('/home/pruittcolon/Desktop/Nemo_Server/CASchools_synthetic.csv');
                await page.waitForTimeout(2000);
            }

            // Start analysis
            const analyzeBtn = page.locator('#analyze-btn, button:has-text("Start")').first();
            if (await analyzeBtn.isVisible()) {
                await analyzeBtn.click();
                // Wait for cost engine to complete
                await page.waitForTimeout(30000);
            }
        }

        // Log final captured responses
        console.log('\n\n========== FINAL SUMMARY ==========');
        console.log('Captured responses for:', Object.keys(responses));

        for (const [engine, data] of Object.entries(responses)) {
            console.log(`\n--- ${engine.toUpperCase()} ---`);
            console.log('Full response:', JSON.stringify(data, null, 2).substring(0, 2000));
        }
        console.log('====================================\n');

        // Assertions to fail with useful info
        expect(Object.keys(responses).length).toBeGreaterThan(0);
    });

    test('Direct API call to Cost engine', async () => {
        // First login
        const loginResp = await apiContext.post('/ui/api/login', {
            data: { username: 'admin', password: 'admin123' }
        });

        const cookies = loginResp.headers()['set-cookie'] || '';
        console.log('[API] Login status:', loginResp.status());

        // Try to call cost engine directly
        const costResp = await apiContext.post('/api/analytics/run-engine/cost', {
            headers: {
                'Content-Type': 'application/json',
                'Cookie': cookies
            },
            data: {
                filename: 'CASchools_synthetic.csv'
            }
        });

        console.log('\n========== DIRECT API: COST ENGINE ==========');
        console.log('Status:', costResp.status());

        if (costResp.ok()) {
            const data = await costResp.json();
            console.log('Response keys:', Object.keys(data));
            console.log('Full response:', JSON.stringify(data, null, 2));

            // Check what the frontend expects
            console.log('\n--- Frontend expects ---');
            console.log('cost_breakdown or breakdown array:', data.cost_breakdown || data.breakdown);
            console.log('pareto array:', data.pareto);
            console.log('categories:', data.categories);
            console.log('total_cost:', data.total_cost);
        } else {
            console.log('Error:', await costResp.text());
        }
        console.log('==============================================\n');
    });
});
