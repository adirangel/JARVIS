"""Playwright-based browser automation for JARVIS.

Manages a persistent browser context for page interactions.
"""

from __future__ import annotations

import asyncio
from loguru import logger

# Module-level state
_browser = None
_page = None
_playwright = None


def _ensure_sync():
    """Get or create a browser + page (sync API)."""
    global _browser, _page, _playwright
    if _page and not _page.is_closed():
        return _page
    try:
        from playwright.sync_api import sync_playwright
        if _playwright is None:
            _playwright = sync_playwright().start()
        _browser = _playwright.chromium.launch(headless=False, channel="msedge")
        _page = _browser.new_page()
        _page.set_default_timeout(15000)
        return _page
    except Exception as e:
        logger.error(f"Browser launch error: {e}")
        raise


def browser_action(
    action: str,
    url: str = "",
    query: str = "",
    selector: str = "",
    text: str = "",
    direction: str = "down",
) -> str:
    """Execute a browser action. Returns result string."""
    try:
        page = _ensure_sync()

        if action == "go_to":
            if not url.startswith("http"):
                url = "https://" + url
            page.goto(url, wait_until="domcontentloaded")
            return f"Navigated to {url}"

        elif action == "search":
            page.goto(f"https://www.google.com/search?q={query}", wait_until="domcontentloaded")
            # Try to extract top results
            try:
                results = page.query_selector_all("h3")[:5]
                texts = [r.inner_text() for r in results if r.inner_text().strip()]
                return "Search results:\n" + "\n".join(f"- {t}" for t in texts) if texts else f"Searched for: {query}"
            except Exception:
                return f"Searched for: {query}"

        elif action == "click":
            if selector:
                try:
                    page.click(selector, timeout=5000)
                except Exception:
                    # Try finding by text
                    page.get_by_text(selector, exact=False).first.click(timeout=5000)
            return f"Clicked: {selector}"

        elif action == "type":
            if selector:
                try:
                    page.fill(selector, text, timeout=5000)
                except Exception:
                    page.get_by_role("textbox").first.fill(text)
            else:
                page.keyboard.type(text)
            return f"Typed text"

        elif action == "scroll":
            amount = -500 if direction == "up" else 500
            page.mouse.wheel(0, amount)
            return f"Scrolled {direction}"

        elif action == "get_text":
            body = page.inner_text("body")
            return body[:3000] if body else "(no text)"

        elif action == "fill_form":
            # selector should be like "input[name=email]" - text is the value
            page.fill(selector, text, timeout=5000)
            return f"Filled form field: {selector}"

        elif action == "back":
            page.go_back()
            return "Went back"

        elif action == "forward":
            page.go_forward()
            return "Went forward"

        elif action == "refresh":
            page.reload()
            return "Refreshed page"

        elif action == "close":
            global _browser
            if _browser:
                _browser.close()
                _browser = None
                _page = None
            return "Browser closed"

        else:
            return f"Unknown browser action: {action}"

    except Exception as e:
        logger.error(f"Browser error: {e}")
        return f"Browser error: {e}"


def close_browser():
    """Clean shutdown."""
    global _browser, _page, _playwright
    try:
        if _browser:
            _browser.close()
        if _playwright:
            _playwright.stop()
    except Exception:
        pass
    _browser = None
    _page = None
    _playwright = None
