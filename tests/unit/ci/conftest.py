# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared import containment for the docs-e2e harness tests.

The harness imports its own modules flatly (``from parser import ...``), so
its directory has to go on ``sys.path`` for these tests to exercise it. The
names it brings in -- ``main``, ``parser``, ``constants``, ``data_types``,
``test_runner``, ``utils`` -- are generic enough to shadow unrelated modules
for the rest of an xdist worker, which outlives any one test file. Containing
that here rather than per file keeps every module in this directory on the
same rule, and keeps the harness imports inside the test bodies where the
fixture is active.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS = REPO_ROOT / "tests/ci/test_docs_end_to_end"


@pytest.fixture(autouse=True)
def _contain_import_side_effects():
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)
    sys.path.insert(0, str(HARNESS))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name in set(sys.modules) - saved_modules:
            del sys.modules[name]
