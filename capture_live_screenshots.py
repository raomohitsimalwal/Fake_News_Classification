"""
Capture Live Screenshots from running Streamlit on http://127.0.0.1:8501
Captures both Project 6 (Fake News & Claims) and Project 15 (Domain Document Classifier).
"""

import os
import time
from playwright.sync_api import sync_playwright

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def capture():
    print("Connecting Playwright to http://127.0.0.1:8501 ...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 950})
        page.goto("http://127.0.0.1:8501", timeout=30000)
        time.sleep(4)

        # 1. Claim Verifier - Analyze Credible Claim
        print("[1/5] Running Claim Analysis & Capturing...")
        try:
            tab1 = page.get_by_text("Claim Verifier (Project 6)")
            tab1.click()
            time.sleep(1)
            btn_cred = page.locator("button:has-text('Sample Credible Claim')").first
            btn_cred.click()
            time.sleep(1)
            btn_analyze = page.locator("button:has-text('Analyze Claim Credibility')").first
            btn_analyze.click()
            time.sleep(3)
            page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "gui_claim_verifier_result.png"))
            print("Captured gui_claim_verifier_result.png!")
        except Exception as e:
            print("Error on claim verifier:", e)

        # 2. News Classifier Tab
        print("[2/5] Switching to News Classifier Tab...")
        try:
            tab2 = page.get_by_text("News Article Classifier (Project 6)")
            tab2.click()
            time.sleep(1)
            btn_real = page.locator("button:has-text('Load Sample Real News')").first
            btn_real.click()
            time.sleep(1)
            btn_verify = page.locator("button:has-text('Verify News Article')").first
            btn_verify.click()
            time.sleep(3)
            page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "gui_news_article_result.png"))
            print("Captured gui_news_article_result.png!")
        except Exception as e:
            print("Error on news classifier:", e)

        # 3. Domain Document Classifier Tab (Project 15)
        print("[3/5] Switching to Domain Document Classifier Tab (Project 15)...")
        try:
            tab3 = page.get_by_text("Domain Document Classifier (Project 15)")
            tab3.click()
            time.sleep(1)
            btn_doc = page.locator("button:has-text('Classify Document & Generate Summary')").first
            btn_doc.click()
            time.sleep(3)
            page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "gui_domain_classifier_result.png"))
            print("Captured gui_domain_classifier_result.png!")
        except Exception as e:
            print("Error on domain classifier:", e)

        # 4. Model Benchmarks & Metrics Tab
        print("[4/5] Switching to Model Benchmark Tab...")
        try:
            tab4 = page.get_by_text("Model Benchmarks & Metrics")
            tab4.click()
            time.sleep(2)
            page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "gui_model_benchmark_tab.png"))
            print("Captured gui_model_benchmark_tab.png!")
        except Exception as e:
            print("Error on model benchmark:", e)

        # 5. Dataset & EDA Explorer Tab
        print("[5/5] Switching to Dataset Explorer Tab...")
        try:
            tab5 = page.get_by_text("Dataset & EDA Explorer")
            tab5.click()
            time.sleep(2)
            page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "gui_eda_explorer_tab.png"))
            print("Captured gui_eda_explorer_tab.png!")
        except Exception as e:
            print("Error on dataset explorer:", e)

        browser.close()
        print("All live screenshots captured successfully in screenshots/!")


if __name__ == "__main__":
    capture()
