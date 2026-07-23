# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the v2 scenario resolver (``apply_scenario``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aiperf.common.scenario import (
    ScenarioLockError,
    UnknownScenarioError,
    apply_scenario,
)
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
    concurrency: int = 8,
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
        "concurrency": concurrency,
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


def _file_dataset_run(
    *,
    synthesis: dict[str, Any] | None = None,
    detected_loader: str | None = "weka_trace",
) -> BenchmarkRun:
    """Build a clean FileDataset (mooncake_trace) run under the scenario."""
    dataset: dict[str, Any] = {
        "name": "main",
        "type": "file",
        "format": "mooncake_trace",
        "records": [
            {"timestamp": 0, "input_length": 10, "output_length": 5, "hash_ids": [1]}
        ],
        "cache_bust": {"target": "first_turn_prefix"},
    }
    if synthesis is not None:
        dataset["synthesis"] = synthesis
    run = _build_run(streaming=True, extra={"ignore_eos": True}, dataset=dataset)
    if detected_loader is not None:
        run.resolved.dataset_types = {"main": detected_loader}
    return run


def test_scenario_set_twice_validator_uses_resolved_value() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    outcome = apply_scenario(run)
    assert outcome.submission_valid is True


def test_unsafe_override_without_scenario_is_noop() -> None:
    run = _build_run(scenario=None, unsafe_override=True)
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is None
    assert outcome.submission_invalid_reasons == []


def test_unknown_scenario_name_raises_unknown_scenario_error() -> None:
    run = _build_run(
        scenario="not-a-real-scenario", streaming=True, extra={"ignore_eos": True}
    )
    with pytest.raises(UnknownScenarioError) as exc:
        apply_scenario(run)
    msg = str(exc.value)
    assert "not-a-real-scenario" in msg
    assert "inferencex-agentx-mvp" in msg


def test_ignore_eos_string_true_treated_as_truthy() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": "true"})
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_ignore_eos_string_false_treated_as_falsy_violation() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": "false"})
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert any(
        "ignore_eos" in v.flag or "ignore_eos" in v.message
        for v in exc.value.violations
    )


def test_ignore_eos_int_one_treated_as_truthy() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": 1})
    outcome = apply_scenario(run)
    assert outcome.violations == []


def test_ignore_eos_int_zero_treated_as_falsy_violation() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": 0})
    with pytest.raises(ScenarioLockError):
        apply_scenario(run)


def test_ignore_eos_none_is_treated_as_absent_and_injected() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": None})
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert run.cfg.endpoint.extra["ignore_eos"] is True


def test_ignore_trace_delays_rejected_for_agentx() -> None:
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset={
            "name": "main",
            "type": "public",
            "dataset": _WEKA_LOADER,
            "ignore_trace_delays": True,
        },
    )
    with pytest.raises(ScenarioLockError, match="ignore-trace-delays"):
        apply_scenario(run)


def test_ignore_trace_delays_with_unsafe_override_marks_submission_invalid() -> None:
    run = _build_run(
        streaming=True,
        unsafe_override=True,
        extra={"ignore_eos": True},
        dataset={
            "name": "main",
            "type": "public",
            "dataset": _WEKA_LOADER,
            "ignore_trace_delays": True,
        },
    )
    outcome = apply_scenario(run)
    assert outcome.submission_valid is False
    assert any(v.flag == "--ignore-trace-delays" for v in outcome.violations)


def test_validator_idempotent_under_reentry() -> None:
    run = _build_run(streaming=True)
    first = apply_scenario(run)
    assert run.cfg.endpoint.extra["ignore_eos"] is True
    second = apply_scenario(run)
    assert first.violations == []
    assert second.violations == []
    assert run.cfg.endpoint.extra["ignore_eos"] is True
    assert second.submission_valid is True


@pytest.mark.parametrize(
    "duration,should_pass",
    [
        (900.0, True),
        (899.999, False),
        (900.0001, True),
    ],
)
def test_benchmark_duration_boundary(duration: float, should_pass: bool) -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": True}, duration=duration)
    if should_pass:
        outcome = apply_scenario(run)
        assert outcome.violations == []
    else:
        with pytest.raises(ScenarioLockError):
            apply_scenario(run)


def test_synthesis_max_isl_zero_rejected_under_lock() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _file_dataset_run(synthesis={"max_isl": 0})


def test_synthesis_max_isl_very_high_rejected_under_lock() -> None:
    run = _file_dataset_run(synthesis={"max_isl": 10**9})
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert any(v.flag == "--synthesis-max-isl" for v in exc.value.violations)


def _multi_violation_run(*, unsafe_override: bool) -> BenchmarkRun:
    return _build_run(
        streaming=False,
        unsafe_override=unsafe_override,
        extra={"ignore_eos": False},
        dataset={
            "name": "main",
            "type": "public",
            "dataset": "sharegpt",
            "cache_bust": {"target": "none"},
        },
        duration=60,
    )


def test_all_five_invariants_lock_raises_with_multiple_violations() -> None:
    run = _multi_violation_run(unsafe_override=False)
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert len(exc.value.violations) == 5


def test_all_five_invariants_unsafe_override_warns_and_invalidates() -> None:
    run = _multi_violation_run(unsafe_override=True)
    outcome = apply_scenario(run)
    assert outcome.submission_valid is False
    assert len(outcome.violations) == 5
    assert "unsafe_override" in outcome.submission_invalid_reasons


def test_int_concurrency_passes_lock() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": True}, concurrency=10)
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is True
