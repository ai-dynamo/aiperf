# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Browser guidance for static Plotly exports."""

import platform

from kaleido.errors import ChromeNotFoundError

from aiperf.common.constants import IS_LINUX


def is_linux_arm64() -> bool:
    """Return whether this host is running Linux on ARM64."""
    return IS_LINUX and platform.machine().lower() in {"aarch64", "arm64"}


def browser_installation_guidance() -> str:
    """Return browser installation guidance for the current platform."""
    if is_linux_arm64():
        return (
            "Install a system Chromium for Linux ARM64. If it is not on PATH, "
            "set BROWSER_PATH to the Chromium executable."
        )
    return "Run plotly_get_chrome to install Chrome."


def png_fallback_readme() -> str:
    """Describe the HTML fallback and browser installation options."""
    return f"""AIPerf Dashboard Plots - Exported as HTML

Chrome rendering failed, so PNG export was unavailable.
Plots have been exported as interactive HTML files instead.

If Chrome or Chromium is missing:
    {browser_installation_guidance()}

HTML files can be opened in any web browser and are fully interactive.
"""


def is_browser_not_found(error: BaseException) -> bool:
    """Detect a missing browser in a chained Kaleido exception."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, ChromeNotFoundError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False
