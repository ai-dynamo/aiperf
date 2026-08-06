# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advanced adversarial edge cases for the v2 scenario resolver ``apply_scenario``: ignore_eos coercion variants, clean-config override, undetectable loader, and duration auto-fill boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aiperf.common.scenario import ScenarioLockError, apply_scenario
from aiperf.config.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkRun

_WEKA_LOADER = "semianalysis_cc_traces_weka_with_subagents"


def _build_run(
    *,
    scenario: str | None = "inferencex-agentx-mvp",
    unsafe_override: bool = False,
    dataset: dict[str, Any] | None = None,
    streaming: bool | None = None,
    extra: dict[str, Any] | None = None,
    duration: Any = 1800,
    profiling_overrides: dict[str, Any] | None = None,
) -> BenchmarkRun:
    """Construct a BenchmarkRun for a weka public dataset under the scenario."""
    if dataset is None:
        dataset = {
            "name": "main",
            "type": "public",
            "dataset": _WEKA_LOADER,
        }
    endpoint: dict[str, Any] = {
        "urls": ["http://localhost:8000/v1/chat/completions"],
        "type": "chat",
    }
    if streaming is not None:
        endpoint["streaming"] = streaming
    if extra is not None:
        endpoint["extra"] = extra

    profiling: dict[str, Any] = {
        "name": "profiling",
        "type": "concurrency",
        "concurrency": 8,
        "duration": duration,
    }
    if profiling_overrides:
        profiling.update(profiling_overrides)

    body: dict[str, Any] = {
        "models": ["my-model"],
        "endpoint": endpoint,
        "datasets": [dataset],
        "phases": [profiling],
    }
    if scenario is not None:
        body["scenario"] = scenario
    if unsafe_override:
        body["unsafe_override"] = True

    cfg = BenchmarkConfig.model_validate(body)
    return BenchmarkRun(
        benchmark_id="test-run",
        cfg=cfg,
        artifact_dir=Path("/tmp/aiperf-scenario-test"),
    )


def test_ignore_eos_truthy_string_yes_passes() -> None:
    """'yes' is not falsy, so it passes clean."""
    run = _build_run(streaming=True, extra={"ignore_eos": "yes"})
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_ignore_eos_truthy_string_one_passes() -> None:
    """The string '1' is not falsy and produces no violation."""
    run = _build_run(streaming=True, extra={"ignore_eos": "1"})
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_ignore_eos_uppercase_true_treated_as_truthy() -> None:
    """``_is_falsy_extra_input`` lower-cases — 'TRUE' is not falsy."""
    run = _build_run(streaming=True, extra={"ignore_eos": "TRUE"})
    outcome = apply_scenario(run)
    assert outcome.violations == []


def test_ignore_eos_padded_yes_treated_as_truthy() -> None:
    """``_is_falsy_extra_input`` strips whitespace before lower-casing."""
    run = _build_run(streaming=True, extra={"ignore_eos": "  yes  "})
    outcome = apply_scenario(run)
    assert outcome.violations == []


def test_ignore_eos_unknown_string_not_falsy_does_not_violate() -> None:
    """A string outside the falsy reject-list ('maybe') is NOT falsy, so no
    violation. Only explicit falsy strings trip the lock."""
    run = _build_run(streaming=True, extra={"ignore_eos": "maybe"})
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_ignore_eos_falsy_string_no_violates() -> None:
    """'no' is in ``_is_falsy_extra_input``'s reject list."""
    run = _build_run(streaming=True, extra={"ignore_eos": "no"})
    with pytest.raises(ScenarioLockError) as exc_info:
        apply_scenario(run)
    assert any(v.flag == "extra_inputs.ignore_eos" for v in exc_info.value.violations)


def test_ignore_eos_falsy_string_zero_violates() -> None:
    """The string '0' is falsy."""
    run = _build_run(streaming=True, extra={"ignore_eos": "0"})
    with pytest.raises(ScenarioLockError) as exc_info:
        apply_scenario(run)
    assert any(v.flag == "extra_inputs.ignore_eos" for v in exc_info.value.violations)


def test_trace_idle_gap_cap_explicit_old_default_is_allowed() -> None:
    """The prior 10-second per-trace cap remains a valid explicit choice."""
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset={
            "name": "main",
            "type": "public",
            "dataset": _WEKA_LOADER,
            "trace_idle_gap_cap_seconds": 10.0,
        },
    )
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is True
    assert run.cfg.get_default_dataset().trace_idle_gap_cap_seconds == 10.0


def test_unsafe_override_with_no_violations_returns_submission_valid_true() -> None:
    """unsafe_override on a clean config keeps submission_valid=True with no invalid reasons."""
    run = _build_run(streaming=True, extra={"ignore_eos": True}, unsafe_override=True)
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is True
    assert outcome.submission_invalid_reasons == []


def test_detected_loader_none_violates_when_loader_required() -> None:
    """An undetectable loader (None) is not in the allowed tuple, so the lock fires the --input-file (loader) violation."""
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset={
            "name": "main",
            "type": "file",
            "format": "mooncake_trace",
            "records": [
                {
                    "timestamp": 0,
                    "input_length": 10,
                    "output_length": 5,
                    "hash_ids": [1],
                }
            ],
            "cache_bust": {"target": "first_turn_prefix"},
        },
    )
    # No run.resolved.dataset_types entry -> _detect_loader returns None.
    with pytest.raises(ScenarioLockError) as exc_info:
        apply_scenario(run)
    assert any(v.flag == "--input-file (loader)" for v in exc_info.value.violations)


def test_benchmark_duration_zero_violates() -> None:
    """duration=0 is treated like unset via ``duration or 0.0`` and violates the 900s floor."""
    # phase.duration has gt=0 at config-build; build valid then set 0 directly
    # on the phase to reach the lock's ``duration or 0.0`` short-circuit.
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    run.cfg.get_profiling_phases()[0].duration = 0
    with pytest.raises(ScenarioLockError) as exc_info:
        apply_scenario(run)
    assert any(v.flag == "--benchmark-duration" for v in exc_info.value.violations)


def test_benchmark_duration_none_auto_fills_scenario_default() -> None:
    """None duration is auto-filled from the scenario's default_benchmark_duration_seconds (1800) instead of violating."""
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    run.cfg.get_profiling_phases()[0].duration = None
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert run.cfg.get_profiling_phases()[0].duration == 1800.0
