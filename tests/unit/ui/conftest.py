# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""pytest configuration for operator UI unit tests.

All tests in this directory require Node.js (``node`` on PATH) because they
run JavaScript modules under ``node --input-type=module``. The entire package
is skipped on runners that don't have Node.js installed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).resolve().parent
_NODE_AVAILABLE = shutil.which("node") is not None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if _NODE_AVAILABLE:
        return
    skip = pytest.mark.skip(
        reason="node not found on PATH; install Node.js to run UI tests"
    )
    for item in items:
        if Path(item.fspath).resolve().is_relative_to(_THIS_DIR):
            item.add_marker(skip)
