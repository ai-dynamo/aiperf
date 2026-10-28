# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The harness's flat module names must not outlive the test that imported them.

The harness imports its own modules flatly, so its directory goes on
``sys.path`` and it binds generic top-level names -- ``main``, ``parser``,
``constants``, ``data_types``, ``test_runner``, ``utils``. An xdist worker
outlives any one test file, so leaving those in ``sys.modules`` shadows the
same names for every later test the worker happens to pick up.
"""

from __future__ import annotations

import sys

GENERIC_NAMES = frozenset(
    {"main", "parser", "constants", "data_types", "test_runner", "utils"}
)


def test_no_harness_module_is_resident_at_test_start() -> None:
    """Every sibling module in this directory imports the harness, so by the
    time this runs the names would be resident if nothing removed them."""
    resident = sorted(GENERIC_NAMES & set(sys.modules))
    assert resident == [], f"harness modules outlived their test: {resident}"


def test_importing_the_harness_inside_a_test_works() -> None:
    """The containment must not cost the tests their imports: the fixture puts
    the harness on sys.path for the duration of each test."""
    from parser import MarkdownParser

    assert MarkdownParser is not None
    assert "parser" in sys.modules
