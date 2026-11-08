"""Open the Streamlit app in a headless browser so Community Cloud counts a visit.

Streamlit Community Cloud puts apps to sleep after 12 hours without traffic,
and a plain HTTP request does not count as traffic: the app only registers a
visit when a browser loads it. If the app is already asleep, this script also
clicks the "Yes, get this app back up!" button.

Usage:
    STREAMLIT_APP_URL=https://<your-app>.streamlit.app python scripts/wake_streamlit.py
"""
import os
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get(
    "STREAMLIT_APP_URL",
    "https://credit-risk-tool-g4qccubtezupxgueqw3y7k.streamlit.app/",
)
WAKE_BUTTON = "Yes, get this app back up!"
APP_READY_TEXT = "Credit Risk Assessment"  # heading rendered by app.py


def app_is_rendered(page) -> bool:
    # Community Cloud renders the app inside an iframe, so check every frame.
    for frame in page.frames:
        try:
            if frame.get_by_text(APP_READY_TEXT).first.is_visible():
                return True
        except Exception:
            continue
    return False


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)

        wake_button = page.get_by_role("button", name=WAKE_BUTTON)
        try:
            wake_button.wait_for(timeout=10_000)
            print("App was asleep, clicking the wake-up button")
            wake_button.click()
        except PlaywrightTimeout:
            print("App was awake")

        # Give the app (and its websocket session) time to fully load.
        for _ in range(36):  # up to 3 minutes
            if app_is_rendered(page):
                print("App rendered successfully")
                page.wait_for_timeout(15_000)  # stay connected briefly
                browser.close()
                return 0
            page.wait_for_timeout(5_000)

        page.screenshot(path="streamlit-wake-failure.png", full_page=True)
        print("App did not render in time", file=sys.stderr)
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
