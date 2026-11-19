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

import importlib.util
import sys
from pathlib import Path

GENERIC_NAMES = frozenset(
    {"main", "parser", "constants", "data_types", "test_runner", "utils"}
)


def _conftest():
    """Load this directory's conftest by path.

    ``import conftest`` does not resolve: pytest loads conftest modules itself
    and never puts them on ``sys.path``.
    """
    path = Path(__file__).with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("_ci_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_fixture_purges_what_a_test_imported() -> None:
    """Drives the fixture directly, so this holds under any test ordering.

    Asserting only that the names are absent right now passes trivially when
    this file runs alone -- nothing would have imported them yet -- so it
    could never fail in that case. Stepping the fixture proves the contract
    itself: it admits the harness during a test and withdraws it after.
    """
    assert "parser" not in sys.modules, "a prior test leaked the harness"
    before_path = list(sys.path)

    gen = _conftest()._contain_import_side_effects.__wrapped__()
    next(gen)
    try:
        import parser  # noqa: F401

        assert "parser" in sys.modules, "the fixture did not admit the harness"
    finally:
        next(gen, None)

    assert "parser" not in sys.modules, "the fixture did not purge the harness"
    assert sys.path == before_path, "the fixture did not restore sys.path"


def test_no_harness_module_is_resident_at_test_start() -> None:
    """The whole-suite view of the same contract: every sibling file in this
    directory imports the harness, so after a full run these names would be
    resident if nothing removed them."""
    resident = sorted(GENERIC_NAMES & set(sys.modules))
    assert resident == [], f"harness modules outlived their test: {resident}"
