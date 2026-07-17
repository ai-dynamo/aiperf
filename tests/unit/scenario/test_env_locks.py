# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trajectory-window apply-or-lock (config-native) and ScenarioSpec doc contract (C3).

``apply_trajectory_ratios`` operates purely on the run config — no
process-global state: unset ``cfg.trajectory_start_min/max_ratio`` fields are
auto-applied from the spec; user-explicit values are locked with a violation
on mismatch. These tests build a REAL ``BenchmarkConfig`` (no MagicMock) so
the ``model_fields_set``-based explicitness check is exercised for real.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pytest import param

from aiperf.common.scenario import get_scenario
from aiperf.common.scenario._env_locks import apply_trajectory_ratios
from aiperf.common.scenario.base import ScenarioSpec, ScenarioViolation
from aiperf.config import BenchmarkConfig
from aiperf.plugin.enums import TimingMode


def _cfg(**explicit: Any) -> BenchmarkConfig:
    """Minimal real BenchmarkConfig; ``explicit`` kwargs mark fields user-set."""
    return BenchmarkConfig(
        models=["m"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets=[
            {
                "name": "profiling",
                "type": "synthetic",
                "entries": 5,
                "prompts": {"isl": 32},
            }
        ],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "sessions": 5,
            }
        ],
        **explicit,
    )


def _run(**explicit: Any) -> SimpleNamespace:
    # apply_trajectory_ratios reads only ``run.cfg``.
    return SimpleNamespace(cfg=_cfg(**explicit))


def _spec() -> ScenarioSpec:
    """Minimal real spec carrying a t* window to auto-apply (no MagicMock)."""
    return ScenarioSpec(
        name="lock-test",
        timing_mode=TimingMode.GRAPH_IR,
        require_ignore_eos=False,
        forbid_input_truncation=False,
        require_loader="weka_trace",
        min_benchmark_duration_seconds=1,
        default_trajectory_start_min_ratio=0.0,
        default_trajectory_start_max_ratio=1.0,
    )


def test_trajectory_ratios_auto_applied_when_unset() -> None:
    """Unset window fields are auto-applied onto the live config."""
    run = _run()
    violations: list[ScenarioViolation] = []
    applied: list[str] = []
    apply_trajectory_ratios(run, _spec(), violations, applied)
    assert violations == []
    assert "trajectory_start_ratios" in applied
    assert run.cfg.trajectory_start_min_ratio == 0.0
    assert run.cfg.trajectory_start_max_ratio == 1.0


def test_trajectory_ratios_mvp_spec_values_via_registry() -> None:
    """The real inferencex-agentx-mvp spec applies its 0.0..1.0 window."""
    run = _run()
    violations: list[ScenarioViolation] = []
    applied: list[str] = []
    apply_trajectory_ratios(
        run, get_scenario("inferencex-agentx-mvp"), violations, applied
    )
    assert violations == []
    assert (run.cfg.trajectory_start_min_ratio, run.cfg.trajectory_start_max_ratio) == (
        0.0,
        1.0,
    )


def test_trajectory_ratios_explicit_conflict_violates() -> None:
    """A user-explicit flag value differing from the spec is a violation and
    is never overwritten; the matching explicit sibling passes."""
    run = _run(trajectory_start_min_ratio=0.10, trajectory_start_max_ratio=1.0)
    violations: list[ScenarioViolation] = []
    applied: list[str] = []
    apply_trajectory_ratios(run, _spec(), violations, applied)
    assert [v.flag for v in violations] == ["--trajectory-start-min-ratio"]
    assert "trajectory_start_ratios" not in applied
    assert run.cfg.trajectory_start_min_ratio == 0.10


def test_trajectory_ratio_violation_still_applies_unset_sibling() -> None:
    """A violated bound never blocks its unset sibling's auto-apply: under
    --unsafe-override the run proceeds with the mixed window, so the sibling
    write is load-bearing to pin."""
    run = _run(trajectory_start_max_ratio=0.9)
    violations: list[ScenarioViolation] = []
    applied: list[str] = []
    apply_trajectory_ratios(run, _spec(), violations, applied)
    assert [v.flag for v in violations] == ["--trajectory-start-max-ratio"]
    assert run.cfg.trajectory_start_max_ratio == 0.9  # explicit, never rewritten
    assert run.cfg.trajectory_start_min_ratio == 0.0  # unset sibling applied


def test_trajectory_ratios_explicit_match_passes() -> None:
    """User-explicit values equal to the spec pass as applied."""
    run = _run(trajectory_start_min_ratio=0.0, trajectory_start_max_ratio=1.0)
    violations: list[ScenarioViolation] = []
    applied: list[str] = []
    apply_trajectory_ratios(run, _spec(), violations, applied)
    assert violations == []
    assert "trajectory_start_ratios" in applied


def test_inverted_window_rejected_at_config_validation() -> None:
    """min > max is a config-level error, before any scenario lock runs."""
    with pytest.raises(ValueError, match="trajectory_start_min_ratio"):
        _cfg(trajectory_start_min_ratio=0.5)


# ---------------------------------------------------------------------------
# C3: trajectory-ratio Field descriptions must state the real lock contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name, flag",
    [
        param(
            "default_trajectory_start_min_ratio",
            "--trajectory-start-min-ratio",
            id="min_ratio",
        ),
        param(
            "default_trajectory_start_max_ratio",
            "--trajectory-start-max-ratio",
            id="max_ratio",
        ),
    ],
)  # fmt: skip
def test_trajectory_ratio_description_states_lock_contract(
    field_name: str, flag: str
) -> None:
    """The docs must name the enforced flag and the lock error, and must not
    reference the retired AIPERF_GRAPH_START_* env vars."""
    description = ScenarioSpec.model_fields[field_name].description
    assert description is not None
    assert flag in description
    assert "ScenarioLockError" in description
    assert "AIPERF_GRAPH_START" not in description
