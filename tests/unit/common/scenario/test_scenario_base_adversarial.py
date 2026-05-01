# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for `ScenarioSpec`, the scenario registry, and scenario error types.

Covers edge cases not exercised by `test_scenario_base.py` /
`test_scenario_registry.py`:

- Frozen-spec attempted mutation surfaces `pydantic.ValidationError` (not AttributeError).
- `extra="forbid"` on `ScenarioSpec` rejects unknown kwargs.
- Required-field omission raises `ValidationError`.
- `ScenarioLockError` message pluralization for 1 vs many violations.
- `ScenarioLockError.violations` round-trips the input list.
- `ScenarioViolation.__str__` renders all four fields.
- `TrajectoryWarmupFailedError` plural/singular phrasing, many ids, non-ASCII safety.
- `UnknownScenarioError` is a `ValueError` subclass.
- Registry lookup is case-sensitive and does not strip whitespace.
- `SCENARIOS` is keyed by `spec.name`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.common.scenario import (
    SCENARIOS,
    ScenarioLockError,
    ScenarioSpec,
    ScenarioViolation,
    TrajectoryWarmupFailedError,
    UnknownScenarioError,
    get_scenario,
)
from aiperf.common.scenario.inferencex_agentx_mvp import INFERENCEX_AGENTX_MVP
from aiperf.plugin.enums import TimingMode


def _minimal_spec_kwargs() -> dict:
    """Return a complete kwargs dict for `ScenarioSpec(...)`."""
    return {
        "name": "test-scenario",
        "timing_mode": TimingMode.AGENTIC_REPLAY,
        "require_ignore_eos": True,
        "require_use_think_time_only": True,
        "forbid_input_truncation": True,
        "require_loader": "weka_trace",
        "min_benchmark_duration_seconds": 900,
        "inter_turn_delay_cap_seconds": 60.0,
    }


def _make_violation(flag: str = "--foo") -> ScenarioViolation:
    return ScenarioViolation(
        flag=flag,
        current_value=3,
        required_value=4,
        message="bad",
    )


# ---------------------------------------------------------------------------
# ScenarioSpec frozen / forbid / required-field behavior.
# ---------------------------------------------------------------------------
def test_scenario_spec_frozen_raises_validation_error_not_attribute_error() -> None:
    """Pydantic v2 frozen=True surfaces ValidationError on assignment, not AttributeError."""
    spec = ScenarioSpec(**_minimal_spec_kwargs())
    with pytest.raises(ValidationError) as exc_info:
        spec.name = "mutated"
    # Confirm it's ValidationError (subclass of ValueError), not AttributeError.
    assert not isinstance(exc_info.value, AttributeError)
    assert "frozen" in str(exc_info.value).lower()


def test_scenario_spec_extra_fields_forbidden() -> None:
    kwargs = _minimal_spec_kwargs()
    kwargs["extra_garbage"] = True
    with pytest.raises(ValidationError) as exc_info:
        ScenarioSpec(**kwargs)
    assert "extra_garbage" in str(exc_info.value)


def test_scenario_spec_required_field_omitted_raises() -> None:
    kwargs = _minimal_spec_kwargs()
    del kwargs["name"]
    with pytest.raises(ValidationError) as exc_info:
        ScenarioSpec(**kwargs)
    assert "name" in str(exc_info.value)


# ---------------------------------------------------------------------------
# ScenarioLockError pluralization + violation round-trip.
# ---------------------------------------------------------------------------
def test_scenario_lock_error_singular_pluralization_one_violation() -> None:
    err = ScenarioLockError([_make_violation()])
    assert "(1 conflict):" in str(err)
    assert "(1 conflicts):" not in str(err)


def test_scenario_lock_error_pluralization_multiple_violations() -> None:
    violations = [_make_violation(f"--flag-{i}") for i in range(3)]
    err = ScenarioLockError(violations)
    assert "(3 conflicts):" in str(err)


def test_scenario_lock_error_carries_violations_list() -> None:
    violations = [_make_violation("--a"), _make_violation("--b")]
    err = ScenarioLockError(violations)
    assert err.violations == violations


def test_scenario_lock_error_zero_violations_uses_plural_form() -> None:
    """Pin degenerate case: empty list renders as '(0 conflicts):' (plural branch)."""
    err = ScenarioLockError([])
    assert "(0 conflicts):" in str(err)


# ---------------------------------------------------------------------------
# ScenarioViolation __str__ rendering.
# ---------------------------------------------------------------------------
def test_scenario_violation_str_renders_all_fields() -> None:
    violation = ScenarioViolation(
        flag="--foo",
        current_value=3,
        required_value=4,
        message="bad",
    )
    rendered = str(violation)
    assert "--foo" in rendered
    assert "3" in rendered
    assert "4" in rendered
    assert "bad" in rendered


# ---------------------------------------------------------------------------
# TrajectoryWarmupFailedError formatting edge cases.
# ---------------------------------------------------------------------------
def test_trajectory_warmup_failed_error_singular_trace_count() -> None:
    err = TrajectoryWarmupFailedError(["trace_a"])
    msg = str(err)
    assert "1 trace" in msg
    assert "trace_a" in msg
    assert err.failed_trace_ids == ["trace_a"]


def test_trajectory_warmup_failed_error_many_trace_ids_all_present() -> None:
    ids = [f"trace_{i}" for i in range(5)]
    err = TrajectoryWarmupFailedError(ids)
    msg = str(err)
    assert "5 trace" in msg
    for trace_id in ids:
        assert trace_id in msg


def test_trajectory_warmup_failed_error_non_ascii_trace_ids() -> None:
    ids = ["traçe_α", "trace_β"]
    err = TrajectoryWarmupFailedError(ids)
    msg = str(err)
    assert "traçe_α" in msg
    assert "trace_β" in msg


# ---------------------------------------------------------------------------
# Error type hierarchy.
# ---------------------------------------------------------------------------
def test_unknown_scenario_error_is_value_error_subclass() -> None:
    assert issubclass(UnknownScenarioError, ValueError)


# ---------------------------------------------------------------------------
# Registry lookup edge cases.
# ---------------------------------------------------------------------------
def test_get_scenario_returns_singleton_identity() -> None:
    """The registry returns the exact INFERENCEX_AGENTX_MVP singleton, not a copy."""
    assert get_scenario("inferencex-agentx-mvp") is INFERENCEX_AGENTX_MVP


def test_get_scenario_is_case_sensitive() -> None:
    with pytest.raises(UnknownScenarioError) as exc_info:
        get_scenario("INFERENCEX-AGENTX-MVP")
    assert "INFERENCEX-AGENTX-MVP" in str(exc_info.value)


def test_get_scenario_does_not_strip_whitespace() -> None:
    with pytest.raises(UnknownScenarioError):
        get_scenario(" inferencex-agentx-mvp ")


def test_scenarios_dict_keyed_by_spec_name_attribute() -> None:
    """SCENARIOS dict keys must match the `name` field of the contained spec."""
    for key, spec in SCENARIOS.items():
        assert key == spec.name
