"""Playwright executor — turns Claude's `computer` tool actions into real browser actions.

This module is imported lazily — Playwright is an optional dep. CI / mock
mode / Streamlit Cloud don't need it.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutorResult:
    screenshot_b64: str  # PNG base64
    note: str = ""


class PlaywrightExecutor:
    """Wraps a Playwright Chromium page and exposes the actions Claude's
    `computer` tool emits.

    Action types Claude emits (Computer Use v20250124):
      screenshot, left_click, right_click, middle_click, double_click,
      type, key, mouse_move, left_click_drag, scroll, wait, cursor_position
    """

    def __init__(self, display_width: int = 1280, display_height: int = 800):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=False)
        self._context = self._browser.new_context(
            viewport={"width": display_width, "height": display_height},
            screen={"width": display_width, "height": display_height},
        )
        self.page = self._context.new_page()
        self.display_width = display_width
        self.display_height = display_height

    def goto(self, url: str) -> None:
        self.page.goto(url, wait_until="domcontentloaded")

    def screenshot(self) -> ExecutorResult:
        png = self.page.screenshot(full_page=False)
        return ExecutorResult(screenshot_b64=base64.b64encode(png).decode())

    def execute(self, action: str, params: dict[str, Any]) -> ExecutorResult:
        """Dispatch a Claude `computer` tool action."""
        if action == "screenshot":
            return self.screenshot()
        if action in ("left_click", "right_click", "middle_click", "double_click"):
            x, y = params.get("coordinate", (0, 0))
            button = {"left_click": "left", "right_click": "right",
                      "middle_click": "middle", "double_click": "left"}[action]
            click_count = 2 if action == "double_click" else 1
            self.page.mouse.click(x, y, button=button, click_count=click_count)
            return self.screenshot()
        if action == "type":
            self.page.keyboard.type(params.get("text", ""))
            return self.screenshot()
        if action == "key":
            self.page.keyboard.press(params.get("text", ""))
            return self.screenshot()
        if action == "mouse_move":
            x, y = params.get("coordinate", (0, 0))
            self.page.mouse.move(x, y)
            return self.screenshot()
        if action == "left_click_drag":
            x1, y1 = params.get("start_coordinate", (0, 0))
            x2, y2 = params.get("coordinate", (0, 0))
            self.page.mouse.move(x1, y1)
            self.page.mouse.down()
            self.page.mouse.move(x2, y2)
            self.page.mouse.up()
            return self.screenshot()
        if action == "scroll":
            delta = params.get("scroll_amount", 3) * 100
            direction = params.get("scroll_direction", "down")
            sign = 1 if direction == "down" else -1
            self.page.mouse.wheel(0, sign * delta)
            return self.screenshot()
        if action == "wait":
            self.page.wait_for_timeout(params.get("duration", 1) * 1000)
            return self.screenshot()
        if action == "cursor_position":
            # Playwright doesn't directly expose cursor position; return screenshot
            return self.screenshot()
        return ExecutorResult(screenshot_b64="", note=f"unsupported action: {action}")

    def close(self) -> None:
        try:
            self._context.close()
            self._browser.close()
            self._pw.stop()
        except Exception:
            pass
