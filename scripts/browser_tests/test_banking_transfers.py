import pytest
from playwright.sync_api import Page, expect

def test_banking_transfer_flow(page: Page):
    # 1. Login (Mock or direct navigation if auth disabled/mocked)
    # Assuming direct navigation for now or simple bypass
    page.goto("http://localhost:8000/ui/banking.html")
    
    # 2. Verify Page Matches Corporate Standard (Simple Check)
    expect(page).to_have_title("Nemo Banking | Enterprise AI Platform")
    
    # Check for specific elements that indicate the new "Simple" design
    # e.g., the specific color header
    header = page.locator("header")
    expect(header).to_be_visible()
    # Check for enterprise CSS load
    # (Playwright can verify computed styles if strictly needed, but presence of .header is good enough for structure)

    # 3. Navigate to Transfers
    # Click sidebar item ("Transfers" with icon)
    # Using text locator with cleaner match
    page.click("div.nav-item:has-text('Transfers')")
    
    # Verify Transfer Section is visible
    transfer_section = page.locator("#section-transfers")
    expect(transfer_section).to_be_visible()
    
    # 4. Fill form
    page.fill("#transferFrom", "10001")
    page.fill("#transferTo", "20002")
    page.fill("#transferAmount", "50.00")
    page.fill("#transferMemo", "Test Playwright Transfer")
    
    # 5. Handle Confirm Dialog
    page.on("dialog", lambda dialog: dialog.accept())
    
    # 6. Submit
    page.click("button:has-text('Submit Transfer')")
    
    # 7. Verify Success Alert
    # Since we use window.alert, we need to catch the next dialog too
    # But for simplicity, we mock window.alert to standard console log or similar in advanced tests
    # However, 'page.on("dialog")' handles all dialogs.
    # The first was confirm(), the second is alert(Success) or alert(Fail).
    
    # We can check specific mocks if we want, but checking that the button returns to normal state implies completion
    
    # Wait for response (simulated by button disabling)
    # The button text changes to 'Processing...' then back.
    submit_btn = page.locator("button:has-text('Submit Transfer')")
    expect(submit_btn).not_to_be_disabled() 
    
    # If successful, fields might be cleared - let's check Amount field
    expect(page.locator("#transferAmount")).to_have_value("")

    print("Test Complete: Transfer Flow Executed")

if __name__ == "__main__":
    # Self-running for quick debug if needed
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        test_banking_transfer_flow(page)
        browser.close()
