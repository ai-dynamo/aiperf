# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Browser guidance for static Plotly exports."""

from kaleido.errors import BrowserFailedError, ChromeNotFoundError

from aiperf.common.constants import IS_LINUX_ARM64


def is_linux_arm64() -> bool:
    """Return whether this host is running Linux on ARM64."""
    return IS_LINUX_ARM64


def browser_installation_guidance() -> str:
    """Return browser installation guidance for the current platform."""
    if is_linux_arm64():
        return (
            "Install a system Chromium for Linux ARM64. Set BROWSER_PATH to its "
            "executable if it is not on PATH; correct or unset a stale BROWSER_PATH. "
            "A Chrome downloaded by Choreographer may take precedence; remove "
            "that download if it is incompatible with this system."
        )
    return "Run plotly_get_chrome to install Chrome."


def png_fallback_readme() -> str:
    """Describe the HTML fallback and browser installation options."""
    return f"""AIPerf Dashboard Plots - Exported as HTML

Chrome rendering failed, so PNG export was unavailable.
Plots have been exported as interactive HTML files instead.

If Chrome or Chromium is missing or cannot start:
    {browser_installation_guidance()}

HTML files can be opened in any web browser and are fully interactive.
"""


def _has_cause(error: BaseException, error_type: type[BaseException]) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, error_type):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def is_browser_not_found(error: BaseException) -> bool:
    """Detect a missing browser in a chained Kaleido exception."""
    return _has_cause(error, ChromeNotFoundError)


def is_browser_start_failure(error: BaseException) -> bool:
    """Detect a browser startup failure in a chained Kaleido exception."""
    return _has_cause(error, BrowserFailedError)
