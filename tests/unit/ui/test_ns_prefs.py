# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``src/aiperf/operator/ui/lib/ns-prefs.js``.

Loads the JS module into a route-intercepted blank page (real ``http://``
origin so ``localStorage`` is actually writable — Chromium denies
``localStorage`` access on ``about:blank`` and ``data:`` URLs), rewrites
its ``export`` declarations into assignments on a local ``exports``
object, and surfaces the resulting bag under ``window.M`` so each test
can drive the helper through ``page.evaluate``.

Skips gracefully when Playwright or Chromium is unavailable, matching
the pattern already in use under ``tests/unit/api/test_dashboard_js.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

NS_PREFS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "lib"
    / "ns-prefs.js"
)


def _playwright_ready() -> tuple[bool, str]:
    """Return ``(available, reason)`` — mirrors ``test_dashboard_js.py``."""
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except ImportError:
        return (
            False,
            "playwright not installed (`uv pip install playwright pytest-playwright`)",
        )
    return (True, "")


_PLAYWRIGHT_AVAILABLE, _PLAYWRIGHT_REASON = _playwright_ready()

pytestmark = pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)


def _shim(src: str) -> str:
    """Rewrite ESM ``export`` declarations to ``exports.NAME = ...`` form.

    Handles ``export function NAME(`` and ``export const NAME =``; both are
    the only forms used by ``ns-prefs.js``.
    """
    src = re.sub(
        r"export\s+function\s+([A-Za-z_$][\w$]*)\s*\(",
        r"exports.\1 = function \1(",
        src,
    )
    src = re.sub(
        r"export\s+const\s+([A-Za-z_$][\w$]*)\s*=",
        r"exports.\1 =",
        src,
    )
    return src


async def _eval_with_module(script: str) -> object:
    """Run ``script`` in a route-served blank page after exposing ns-prefs.js as ``window.M``.

    Uses ``page.route`` to fulfill any URL with an empty HTML document so the
    page has a real ``http://`` origin (Chromium blocks ``localStorage`` on
    ``about:blank`` and ``data:`` URLs).
    """
    from playwright.async_api import async_playwright

    src = NS_PREFS_PATH.read_text()
    shimmed = _shim(src)
    bootstrap = (
        "const __m = (() => { const exports = {}; "
        + shimmed
        + " ; return exports; })(); window.M = __m;"
    )
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        try:
            ctx = await browser.new_context()
            page = await ctx.new_page()
            await page.route(
                "**/*",
                lambda route: route.fulfill(
                    status=200,
                    content_type="text/html",
                    body="<html><body></body></html>",
                ),
            )
            await page.goto("http://aiperf.test/blank")
            await page.add_script_tag(content=bootstrap)
            return await page.evaluate(script)
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_get_ns_pref_missing_returns_default():
    out = await _eval_with_module("M.getNsPref('foo', 'pinnedRunNames', ['fallback'])")
    assert out == ["fallback"]


@pytest.mark.asyncio
async def test_set_then_get_round_trip():
    out = await _eval_with_module(
        "(() => { M.setNsPref('foo', 'pinnedRunNames', ['a','b']); "
        "return M.getNsPref('foo', 'pinnedRunNames', []); })()"
    )
    assert out == ["a", "b"]


@pytest.mark.asyncio
async def test_last_namespace_round_trip():
    out = await _eval_with_module(
        "(() => { M.setLastNamespace('team-llama'); return M.getLastNamespace(); })()"
    )
    assert out == "team-llama"


@pytest.mark.asyncio
async def test_get_last_namespace_missing_returns_null():
    out = await _eval_with_module("M.getLastNamespace()")
    assert out is None


@pytest.mark.asyncio
async def test_set_pref_quota_error_swallowed():
    """Force a throw on setItem; the helper must not propagate."""
    out = await _eval_with_module(
        "(() => { const orig = Storage.prototype.setItem; "
        "Storage.prototype.setItem = () => { throw new Error('quota'); }; "
        "try { M.setNsPref('foo', 'k', 'v'); return 'ok'; } "
        "finally { Storage.prototype.setItem = orig; } })()"
    )
    assert out == "ok"
