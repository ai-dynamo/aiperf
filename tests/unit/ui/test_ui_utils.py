# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from pathlib import Path

import pytest
from pytest import param

from aiperf.ui.utils import format_bytes, format_elapsed_time, format_eta

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.parametrize(
    "bytes, expected",
    [
        (None, "--"),
        (0, "0 B"),
        (1, "1 B"),
        (999, "999 B"),
        (1000, "1.0 KB"),  # 0.976 rounded to 1.0
        (1023, "1.0 KB"),
        (1024, "1.0 KB"),
        (1024**2, "1.0 MB"),
        (1024**3, "1.0 GB"),
        (1024**4, "1.0 TB"),
        (1024**5, "1.0 PB"),
        (1024**6, "1.0 EB"),
        (1024**7, "1.0 ZB"),
        (1024**8, "1.0 YB"),
    ],
)
def test_format_bytes(bytes, expected):
    assert format_bytes(bytes) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (None, "--"),
        (0, "0.0s"),
        (1, "1.0s"),
        (0.5, "0.5s"),
        (0.9, "0.9s"),
        (0.999, "1.0s"),
        (1.001, "1.0s"),
        (1.5, "1.5s"),
        (1.999, "2.0s"),
        (2.001, "2.0s"),
        (60, "1m"),
        (60 * 60 - 1, "59m 59s"),
        (60 * 60, "1h"),
        (60 * 60 + 1, "1h"),
        (60 * 60 + 69, "1h 1m"),
        (60 * 60 * 24, "1d"),
        (60 * 60 * 24 * 365, "365d"),
    ],
)
def test_format_eta(seconds, expected) -> None:
    assert format_eta(seconds) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (None, "--"),
        (0, "0.0s"),
        (1, "1.0s"),
        (0.5, "0.5s"),
        (0.9, "0.9s"),
        (0.999, "1.0s"),
        (1.001, "1.0s"),
        (1.5, "1.5s"),
        (1.999, "2.0s"),
        (2.001, "2.0s"),
        (60, "1m"),
        (60 * 60 - 1, "59m 59s"),
        (60 * 60, "1h"),
        (60 * 60 + 1, "1h 1s"),
        (60 * 60 + 69, "1h 1m 9s"),
        (60 * 60 * 24, "1d"),
        (60 * 60 * 24 * 365, "365d"),
    ],
)
def test_format_elapsed_time(seconds, expected) -> None:
    assert format_elapsed_time(seconds) == expected


# ---------------------------------------------------------------------------
# extractNamespaceField (lib/yaml-namespace.js)
# ---------------------------------------------------------------------------

YAML_NAMESPACE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "lib"
    / "yaml-namespace.js"
)


def _playwright_ready() -> tuple[bool, str]:
    """Return ``(available, reason)`` — mirrors ``test_ns_prefs.py``."""
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except ImportError:
        return (
            False,
            "playwright not installed (`uv pip install playwright pytest-playwright`)",
        )
    return (True, "")


_PLAYWRIGHT_AVAILABLE, _PLAYWRIGHT_REASON = _playwright_ready()


def _shim(src: str) -> str:
    """Rewrite ESM ``export`` declarations to ``exports.NAME = ...`` form."""
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


async def _extract(yaml_text: str) -> object:
    """Load ``yaml-namespace.js`` into a route-served blank page and call ``extractNamespaceField``."""
    from playwright.async_api import async_playwright

    src = YAML_NAMESPACE_PATH.read_text()
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
            return await page.evaluate(
                "(yamlText) => M.extractNamespaceField(yamlText)",
                yaml_text,
            )
        finally:
            await browser.close()


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "yaml_text, expected",
    [
        param("namespace: foo\n", "foo", id="bare-unquoted"),
        param("namespace: 'team-llama'\n", "team-llama", id="single-quoted"),
        param('namespace: "team-llama"\n', "team-llama", id="double-quoted"),
        param(
            "apiVersion: v1\nnamespace: bar\nkind: AIPerfJob\n",
            "bar",
            id="among-other-top-level-keys",
        ),
        param("metadata:\n  namespace: indented\n", None, id="indented-not-top-level"),
        param("# namespace: commented\n", None, id="commented-out"),
        param("", None, id="empty-string"),
        param("namespace:\n", None, id="empty-value"),
        param("namespace: foo  # trailing comment\n", "foo", id="trailing-comment"),
    ],
)  # fmt: skip
async def test_extract_namespace_field(yaml_text, expected) -> None:
    assert await _extract(yaml_text) == expected
