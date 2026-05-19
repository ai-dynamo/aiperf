# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy-import regression tests for ``aiperf.orchestrator.search_planner``.

The smooth-isotonic planner stack (``smooth_isotonic.py`` plus
``_smooth_isotonic_fit``, ``_smooth_isotonic_phases``, ``_replicate_budget``)
imports ``scipy.{interpolate,optimize,stats}`` at module top. Users who pick
the bayesian / monotonic / optuna planner should not pay that cost. The
package ``__init__`` exposes ``SmoothIsotonicSLAPlanner`` via ``__getattr__``
(same shape as ``OptunaSearchPlanner``); these tests pin that contract.

Each ``sys.modules``-introspecting test runs in a fresh subprocess so unrelated
imports in the surrounding test session can't pollute the assertion.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(snippet: str) -> subprocess.CompletedProcess[str]:
    """Run ``snippet`` in a fresh interpreter and return the completed process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_importing_package_does_not_import_scipy() -> None:
    """Bare ``import aiperf.orchestrator.search_planner`` must not pull scipy."""
    result = _run(
        """
        import sys
        import aiperf.orchestrator.search_planner  # noqa: F401
        scipy_mods = [m for m in sys.modules if m == "scipy" or m.startswith("scipy.")]
        assert scipy_mods == [], f"scipy leaked into sys.modules: {scipy_mods}"
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_importing_monotonic_planner_does_not_import_scipy() -> None:
    """Choosing ``MonotonicSLASearchPlanner`` must not pay the scipy cost."""
    result = _run(
        """
        import sys
        from aiperf.orchestrator.search_planner import MonotonicSLASearchPlanner  # noqa: F401
        scipy_mods = [m for m in sys.modules if m == "scipy" or m.startswith("scipy.")]
        assert scipy_mods == [], f"scipy leaked into sys.modules: {scipy_mods}"
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_smooth_isotonic_planner_accessible_via_package() -> None:
    """``SmoothIsotonicSLAPlanner`` must still resolve through the package."""
    result = _run(
        """
        from aiperf.orchestrator.search_planner import SmoothIsotonicSLAPlanner
        from aiperf.orchestrator.search_planner.smooth_isotonic import (
            SmoothIsotonicSLAPlanner as Direct,
        )
        assert SmoothIsotonicSLAPlanner is Direct, "lazy import resolved a different class"
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_smooth_isotonic_planner_pulls_scipy_when_requested() -> None:
    """Sanity check: importing ``SmoothIsotonicSLAPlanner`` *does* pull scipy.

    Confirms the lazy gate is the only thing keeping scipy out — i.e. the
    earlier "no scipy" assertions aren't passing because scipy was secretly
    removed from the planner stack.
    """
    result = _run(
        """
        import sys
        from aiperf.orchestrator.search_planner import SmoothIsotonicSLAPlanner  # noqa: F401
        scipy_mods = [m for m in sys.modules if m == "scipy" or m.startswith("scipy.")]
        assert scipy_mods, "smooth-isotonic should import scipy"
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_unknown_attribute_raises_attribute_error() -> None:
    """``__getattr__`` must reject unknown names rather than silently returning None."""
    import aiperf.orchestrator.search_planner as planner_pkg

    try:
        planner_pkg.NotARealPlanner  # type: ignore[attr-defined]  # noqa: B018
    except AttributeError as exc:
        assert "NotARealPlanner" in str(exc)
    else:
        raise AssertionError("expected AttributeError for unknown attribute")
