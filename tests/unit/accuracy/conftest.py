# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy-scoped fixtures.

Carries the fake-deepeval wiring used by the BigBench loader tests so they
can run in CI without the ``[accuracy]`` extras (which add ~1 GiB and are
not installed in the default unit-test job).

Two pieces:

- ``_patch_bigbench_deepeval_names`` is an autouse fixture that swaps the
  bigbench loader's deepeval-imported module attributes for the fake
  stand-ins. Active only when the real deepeval isn't importable, so the
  real install wins locally / in any job that opts into ``[accuracy]``.
  Scoped per-test (function-scope ``monkeypatch``) so it doesn't leak
  into adjacent tests like HellaSwag, which still use the existing
  ``pytest.importorskip("deepeval")`` skip mechanism.
- ``pytest_collection_modifyitems`` skips tests tagged
  ``@pytest.mark.requires_deepeval`` when only the fake is available —
  used for byte-equal-prompt assertions that depend on deepeval's
  bundled ``.txt`` prompt files which the fake doesn't reproduce.
"""

from __future__ import annotations

import pytest

from tests.harness import fake_deepeval


def _real_deepeval_available() -> bool:
    """True iff the real deepeval (with bundled CoT/shot prompt files) is
    importable. The fake harness does not satisfy this — it lives under
    ``tests.harness``."""
    try:
        import deepeval.benchmarks.big_bench_hard.template as _t  # noqa: F401

        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.requires_deepeval`` items when the real
    deepeval install isn't available."""
    if _real_deepeval_available():
        return
    skip_mark = pytest.mark.skip(
        reason="requires the real deepeval install ([accuracy] extras); "
        "the fake-deepeval harness cannot reproduce upstream prompt bytes."
    )
    for item in items:
        if "requires_deepeval" in item.keywords:
            item.add_marker(skip_mark)


@pytest.fixture(autouse=True)
def _patch_bigbench_deepeval_names(request, monkeypatch):
    """Swap ``bigbench.py``'s deepeval-imported names for the fake when
    the real install isn't present.

    ``bigbench.py``'s top-level ``try / except ImportError`` already
    binds the four affected names (``_HAS_DEEPEVAL``, ``BigBenchHardTask``,
    ``BigBenchHardTemplate``, ``bbh_confinement_statements_dict``) to
    ``False`` / ``None`` when deepeval is missing. This fixture patches
    them per-test to the harness fakes so loader tests can run.

    Skipped (no patching) when the real deepeval is importable so the
    real upstream behavior is exercised end-to-end in ``[accuracy]``
    environments.
    """
    if _real_deepeval_available():
        return
    try:
        import aiperf.accuracy.benchmarks.bigbench as bigbench_mod
    except ImportError:
        # bigbench.py couldn't load at all — nothing to patch. Tests
        # that need it will fail loudly on import, which is what we
        # want.
        return
    monkeypatch.setattr(bigbench_mod, "_HAS_DEEPEVAL", True)
    monkeypatch.setattr(
        bigbench_mod, "BigBenchHardTask", fake_deepeval.BigBenchHardTask
    )
    monkeypatch.setattr(
        bigbench_mod, "BigBenchHardTemplate", fake_deepeval.BigBenchHardTemplate
    )
    monkeypatch.setattr(
        bigbench_mod,
        "bbh_confinement_statements_dict",
        fake_deepeval.bbh_confinement_statements_dict,
    )
