
import pytest
from playwright.sync_api import Page, expect
import os

# List of all HTML pages found in frontend
PAGES = [
    "about.html", "admin_qa.html", "all_engines_activity_log_demo.html", 
    "analysis.html", "analytics.html", "automation.html", "banking.html",
    "chatbot.html", "chat.html", "database_analysis.html", "databases.html",
    "dev_auto_login.html", "emotions.html", "financial-dashboard.html",
    "gemma_data.html", "gemma.html", "graph_preview_static.html",
    "index.html", "knowledge.html", "login.html", "meetings.html",
    "memories.html", "mirror_gemma.html", "ml_dashboard.html",
    "personalization.html", "predictions.html", "predictions-modular.html",
    "premium_engine_test.html", "quick_test.html", "salesforce.html",
    "settings.html", "speakers.html", "transcripts.html"
]

@pytest.mark.parametrize("page_name", PAGES)
def test_page_loads_successfully(page: Page, page_name: str):
    """
    Smoke test to verify every HTML page loads with 200 OK (implied by no error)
    and has a title.
    """
    url = f"http://localhost:8000/ui/{page_name}"
    print(f"Testing {url}...")
    
    response = page.goto(url)
    
    # Check for HTTP 200 OK (if applicable, file:// won't have it, but http://localhost will)
    assert response.ok, f"Failed to load {page_name}: Status {response.status}"
    
    # Check title is not empty and not "Error"
    title = page.title()
    assert title, f"Page {page_name} has no title"
    assert "Error" not in title, f"Page {page_name} loaded with Error title"
    assert "404" not in title, f"Page {page_name} likely 404"

if __name__ == "__main__":
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for page_name in PAGES:
            page = browser.new_page()
            try:
                # Wrap in simple runner logic
                url = f"http://localhost:8000/ui/{page_name}"
                res = page.goto(url)
                if not res.ok:
                    print(f"❌ {page_name} FAILED: Status {res.status}")
                else:
                    t = page.title()
                    if not t or "Error" in t or "404" in t:
                        print(f"❌ {page_name} FAILED: Invalid Title '{t}'")
                    else:
                        print(f"✅ {page_name} PASSED")
            except Exception as e:
                print(f"❌ {page_name} ERROR: {e}")
            finally:
                page.close()
        browser.close()
