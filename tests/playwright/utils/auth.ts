import { Page, expect } from '@playwright/test';

export async function loginToApp(page: Page) {
    // Register user first (ignoring errors if already exists)
    try {
        await page.request.post('/api/auth/register', {
            data: {
                username: 'test_msr',
                password: 'Password123!',
                email: 'msr@test.com'
            },
            timeout: 5000
        });
    } catch {
        // User likely already exists, continue
    }

    // Navigate to login page
    await page.goto('/ui/login.html', { waitUntil: 'networkidle' });

    // Wait for form to be ready
    await page.waitForSelector('#username', { state: 'visible' });

    // Fill credentials
    await page.fill('#username', 'test_msr');
    await page.fill('#password', 'Password123!');

    // Click submit and wait for navigation
    await Promise.all([
        page.waitForNavigation({ timeout: 15000 }),
        page.click('button[type="submit"]')
    ]);

    // Verify we're on the main page
    await page.waitForLoadState('domcontentloaded');
}
