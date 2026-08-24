# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy-scoped fixtures.

Carries the fake-dependency wiring that lets optional-extra-backed accuracy
tests run in CI without those extras installed: ``[accuracy]`` (deepeval,
lighteval, torch, ...) adds roughly 1 GiB, and ``[bfcl]`` cannot even co-exist
with it (bfcl-eval pins ``numpy==1.26.4`` against lighteval's ``numpy>=2``).

Pieces:

- ``_patch_bigbench_deepeval_names`` is an autouse fixture that swaps the
  bigbench loader's deepeval-imported module attributes for the fake
  stand-ins. Active only when the real deepeval isn't importable, so the
  real install wins locally / in any job that opts into ``[accuracy]``.
  Scoped per-test (function-scope ``monkeypatch``) so it doesn't leak
  into adjacent tests like HellaSwag, which still use the existing
  ``pytest.importorskip("deepeval")`` skip mechanism.
- ``_patch_bfcl_compat_names`` does the same for the BFCL loader/grader,
  patching the resolver functions in ``aiperf.accuracy.graders._bfcl_compat``
  (the single seam through which all bfcl-eval access flows).
- ``pytest_collection_modifyitems`` skips tests tagged
  ``@pytest.mark.requires_deepeval`` / ``@pytest.mark.requires_bfcl`` when only
  the fakes are available — used for the parity assertions that depend on
  upstream bytes and semantics the fakes deliberately don't reproduce.
"""

from __future__ import annotations

import pytest

from tests.harness import fake_bfcl, fake_deepeval


def _real_bfcl_available() -> bool:
    """True iff the real ``bfcl_eval`` (with its bundled dataset and AST
    checker) is importable. The fake harness does not satisfy this — it lives
    under ``tests.harness``."""
    try:
        import importlib.util

        return importlib.util.find_spec("bfcl_eval") is not None
    except (ImportError, ValueError):
        return False


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
    """Skip optional-dependency-gated items when only the fakes are available.

    ``requires_deepeval`` and ``requires_bfcl`` are handled independently:
    the two extras are mutually exclusive (bfcl-eval pins ``numpy==1.26.4``
    against lighteval's ``numpy>=2``), so no environment has both.
    """
    if not _real_deepeval_available():
        skip_deepeval = pytest.mark.skip(
            reason="requires the real deepeval install ([accuracy] extras); "
            "the fake-deepeval harness cannot reproduce upstream prompt bytes."
        )
        for item in items:
            if "requires_deepeval" in item.keywords:
                item.add_marker(skip_deepeval)

    if not _real_bfcl_available():
        skip_bfcl = pytest.mark.skip(
            reason="requires the real bfcl-eval install ([bfcl] extra); the "
            "fake-bfcl harness is not a parity oracle for the AST checker or "
            "the system-prompt bytes."
        )
        for item in items:
            if "requires_bfcl" in item.keywords:
                item.add_marker(skip_bfcl)


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


@pytest.fixture(autouse=True)
def _patch_bfcl_compat_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap ``_bfcl_compat``'s upstream resolvers for the fake when the real
    ``bfcl_eval`` isn't present.

    Every call aiperf makes into bfcl-eval routes through ``_bfcl_compat``
    precisely so the fake can be installed at this single seam - the loader and
    grader never import ``bfcl_eval`` themselves.

    Skipped (no patching) when the real ``bfcl_eval`` is importable, so a
    ``[bfcl]`` environment exercises the real checker, decoder and prompt
    builder end to end. ``data_dir``/``possible_answer_dir`` are deliberately
    left alone: loader tests point them at tmp fixture files themselves, since
    the right directory is per-test.
    """
    if _real_bfcl_available():
        return
    from aiperf.accuracy.graders import _bfcl_compat

    monkeypatch.setattr(_bfcl_compat, "bfcl_available", lambda: True)
    monkeypatch.setattr(_bfcl_compat, "require_bfcl", lambda: None)
    monkeypatch.setattr(_bfcl_compat, "check_version_pin", lambda: None)
    monkeypatch.setattr(_bfcl_compat, "installed_version", lambda: "fake")
    monkeypatch.setattr(
        _bfcl_compat, "version_prefix", lambda: fake_bfcl.VERSION_PREFIX
    )
    monkeypatch.setattr(
        _bfcl_compat,
        "single_turn_categories",
        lambda: tuple(fake_bfcl.NON_LIVE_CATEGORY) + tuple(fake_bfcl.LIVE_CATEGORY),
    )
    monkeypatch.setattr(
        _bfcl_compat, "build_chat_messages", fake_bfcl.build_chat_messages
    )

    monkeypatch.setattr(_bfcl_compat, "decode_calls", fake_bfcl.decode_calls)
    monkeypatch.setattr(
        _bfcl_compat,
        "ast_check",
        lambda **kwargs: fake_bfcl.ast_checker(**kwargs),
    )
