import { test, expect } from '@playwright/test';
import { loginToApp } from './utils/auth';

test.describe('Fiserv Member Search Intelligence', () => {
    test.beforeEach(async ({ page }) => {
        await loginToApp(page);
        await page.goto('/ui/banking.html#section-party');
        await expect(page.locator('#section-party')).toBeVisible();
    });

    test('Typeahead search triggers on input', async ({ page }) => {
        const nameInput = page.locator('#partyName');

        // Type a search query
        await nameInput.fill('Sm');

        // Wait for debounce and results (or hint message if no results)
        await page.waitForTimeout(500); // Wait for debounce

        // Should show either results or a hint message
        const resultsDiv = page.locator('#partyResults');
        await expect(resultsDiv).toBeVisible();
    });

    test('Search button triggers member search', async ({ page }) => {
        // Fill in search criteria
        await page.fill('#partyName', 'Smith');

        // Click search button
        await page.click('button:has-text("Search Member")');

        // Wait for results
        await page.waitForTimeout(2000);

        // Check results panel is updated
        const resultCount = page.locator('#partyResultCount');
        await expect(resultCount).toBeVisible();
    });

    test('Member 360 panel renders on member selection', async ({ page }) => {
        // First search for a member
        await page.fill('#partyName', 'Test');
        await page.click('button:has-text("Search Member")');
        await page.waitForTimeout(2000);

        // Check if we have any results
        const resultsDiv = page.locator('#partyResults');
        const resultItems = resultsDiv.locator('.result-item');

        // If there are results, click the first one
        const count = await resultItems.count();
        if (count > 0) {
            await resultItems.first().click();

            // Verify Member 360 panel appears
            const member360Panel = page.locator('#member360Panel');
            await expect(member360Panel).toBeVisible();

            // Verify it has the glassmorphism container
            await expect(member360Panel.locator('.member-360-container')).toBeVisible();
        }
    });

    test('Clear button resets search form', async ({ page }) => {
        // Fill fields
        await page.fill('#partyName', 'Test Name');
        await page.fill('#partyPhone', '555-1234');

        // Click clear
        await page.click('button:has-text("Clear")');

        // Verify fields are empty
        await expect(page.locator('#partyName')).toHaveValue('');
        await expect(page.locator('#partyPhone')).toHaveValue('');
    });
});
