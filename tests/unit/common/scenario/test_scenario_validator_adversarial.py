# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the v2 scenario resolver (``apply_scenario``).

Rebased from the v1 ``validate_scenario(MagicMock UserConfig)`` suite onto the
v2 ``apply_scenario(run)`` resolver. Every test builds a REAL
``BenchmarkConfig`` + ``BenchmarkRun`` (no MagicMock) so attribute-path drift
fails loudly. Each test attacks a specific edge case in the AgentX scenario
lock.

DROP/PORT NOTES (v1 behaviors with no v2 analog):
* ``_extract_extra_inputs`` fallback chain (parsed -> extra -> dict(raw)) is
  gone: v2 reads ``endpoint.extra`` (a real dict) directly. The coercion
  semantics themselves are still asserted via the ignore_eos value tests below.
* random_seed auto-set: v2 ``apply_scenario`` stamps a fresh
  ``secrets.randbits(63)`` onto ``run.random_seed`` (the operative per-run seed
  field) when the user left it unset, mirroring the v1 ``validate_scenario``
  auto-fill. See ``test_random_seed_unset_auto_filled`` /
  ``test_random_seed_explicit_preserved`` in ``test_scenario_validator.py``.
  The v1 ``random_seed=0 not-injected`` edge does not apply: v2 keys on
  ``run.random_seed is None`` (an explicit 0 is preserved), and v2 has no
  ``input.random_seed`` whose falsy-0 ambiguity the v1 test guarded against.
* list-concurrency sweep rejection: ``phase.concurrency`` is a scalar ``int``
  that rejects lists at config-build time, and the sweep-vs-scenario
  interaction moved to the sweep layer (see the concurrency-sweep PORT-NOTE at
  the bottom of ``src/aiperf/common/scenario/validator.py``). The validator no
  longer checks concurrency, so the two list-concurrency tests are dropped; a
  rebased scalar-concurrency clean-run test is kept.
"""

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
    """Construct a BenchmarkRun for a weka public dataset under the scenario.

    Mirrors the canonical ``_build_run`` in ``test_scenario_validator.py``.
    ``dataset`` overrides the default public-weka dataset dict; ``streaming`` /
    ``extra`` configure the endpoint; ``profiling_overrides`` merge onto the
    profiling phase.
    """
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
    """Build a clean FileDataset (mooncake_trace) run under the scenario.

    The FileDataset resolves to ``detected_loader`` via
    ``run.resolved.dataset_types`` so the require_loader check passes and the
    only violation that can fire is ``--synthesis-max-isl``.
    """
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


# ---------------------------------------------------------------------------
# Test 1: --scenario resolved value pins to a clean outcome.
# The "resolved value" / config-file precedence concept is upstream of the v2
# resolver; here we just pin that a clean run lands submission_valid=True.
# ---------------------------------------------------------------------------
def test_scenario_set_twice_validator_uses_resolved_value() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    outcome = apply_scenario(run)
    assert outcome.submission_valid is True


# ---------------------------------------------------------------------------
# Test 2: unsafe_override without a scenario is a no-op.
# ---------------------------------------------------------------------------
def test_unsafe_override_without_scenario_is_noop() -> None:
    run = _build_run(scenario=None, unsafe_override=True)
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is None
    assert outcome.submission_invalid_reasons == []


# ---------------------------------------------------------------------------
# Test 3: Unknown scenario name raises UnknownScenarioError listing valid set.
# ---------------------------------------------------------------------------
def test_unknown_scenario_name_raises_unknown_scenario_error() -> None:
    run = _build_run(
        scenario="not-a-real-scenario", streaming=True, extra={"ignore_eos": True}
    )
    with pytest.raises(UnknownScenarioError) as exc:
        apply_scenario(run)
    msg = str(exc.value)
    assert "not-a-real-scenario" in msg
    assert "inferencex-agentx-mvp" in msg


# ---------------------------------------------------------------------------
# Test 4a: ignore_eos string "true" treated as truthy (clean).
# ---------------------------------------------------------------------------
def test_ignore_eos_string_true_treated_as_truthy() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": "true"})
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is True


# ---------------------------------------------------------------------------
# Test 4b: ignore_eos string "false" treated as falsy (violation).
# ---------------------------------------------------------------------------
def test_ignore_eos_string_false_treated_as_falsy_violation() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": "false"})
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert any(
        "ignore_eos" in v.flag or "ignore_eos" in v.message
        for v in exc.value.violations
    )


# ---------------------------------------------------------------------------
# Test 5: ignore_eos numeric / null coercion behavior.
# 1 -> truthy (clean); 0 -> falsy (violation); None/absent -> injected to True.
# ---------------------------------------------------------------------------
def test_ignore_eos_int_one_treated_as_truthy() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": 1})
    outcome = apply_scenario(run)
    assert outcome.violations == []


def test_ignore_eos_int_zero_treated_as_falsy_violation() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": 0})
    with pytest.raises(ScenarioLockError):
        apply_scenario(run)


def test_ignore_eos_none_is_treated_as_absent_and_injected() -> None:
    # A null ignore_eos is treated as "absent" and auto-injected to True.
    run = _build_run(streaming=True, extra={"ignore_eos": None})
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert run.cfg.endpoint.extra["ignore_eos"] is True


# ---------------------------------------------------------------------------
# Test 7: --ignore-trace-delays is REJECTED for AgentX MVP. The scenario
# replays recorded trace timing; --ignore-trace-delays nulls every per-turn
# timestamp/delay and dispatches all turns back-to-back, falsifying the
# workload. The v2 lock gates on spec.forbid_ignore_trace_delays.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Test 8 (lightened): apply_scenario is idempotent across two calls on a clean
# run. v1 asserted model_post_init re-entry + single injection log; v2
# apply_scenario is a plain function. Calling it twice must leave the run clean
# and not corrupt the auto-injected ignore_eos.
# ---------------------------------------------------------------------------
def test_validator_idempotent_under_reentry() -> None:
    run = _build_run(streaming=True)  # ignore_eos absent -> injected on first call
    first = apply_scenario(run)
    assert run.cfg.endpoint.extra["ignore_eos"] is True
    second = apply_scenario(run)
    assert first.violations == []
    assert second.violations == []
    assert run.cfg.endpoint.extra["ignore_eos"] is True
    assert second.submission_valid is True


# ---------------------------------------------------------------------------
# Test 9: --benchmark-duration boundary behavior (lock at 900s floor).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Test 10: --synthesis-max-isl edge values (forbid_input_truncation).
# v2 moved the floor to SynthesisConfig.max_isl (ge=1), so max_isl=0 is
# rejected at CONFIG-BUILD time by pydantic, NOT by the scenario lock. A very
# high value (10**9) is a valid config and is rejected by the scenario lock.
# This lock only bites on FILE datasets with a synthesis block.
# ---------------------------------------------------------------------------
def test_synthesis_max_isl_zero_rejected_under_lock() -> None:
    from pydantic import ValidationError

    # v2: the floor moved to the field constraint (ge=1), so 0 never reaches
    # the scenario lock -- it fails at BenchmarkConfig construction.
    with pytest.raises(ValidationError):
        _file_dataset_run(synthesis={"max_isl": 0})


def test_synthesis_max_isl_very_high_rejected_under_lock() -> None:
    run = _file_dataset_run(synthesis={"max_isl": 10**9})
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert any(v.flag == "--synthesis-max-isl" for v in exc.value.violations)


# ---------------------------------------------------------------------------
# Test 12 (rebased): multiple invariants violated simultaneously under AgentX
# MVP. The default concurrency phase here accepts the timing_mode stamp
# (an explicit rate / user-centric / fixed-schedule phase would add a sixth
# violation — see test_scenario_validator.py::
# test_explicit_scheduling_phase_conflicts_with_scenario). The
# simultaneously-firing v2 violations are:
#   1) streaming=False explicit
#   2) ignore_eos=False explicit
#   3) wrong loader (sharegpt)
#   4) cache_bust=none explicit
#   5) duration below floor
# We count what v2 actually emits (5 here) and assert that count; the
# non-override variant raises ScenarioLockError, mirroring
# test_scenario_validator.py::test_unsafe_override_converts_errors_to_warnings.
# ---------------------------------------------------------------------------
def _multi_violation_run(*, unsafe_override: bool) -> BenchmarkRun:
    return _build_run(
        streaming=False,  # --streaming violation
        unsafe_override=unsafe_override,
        extra={"ignore_eos": False},  # ignore_eos violation
        dataset={
            "name": "main",
            "type": "public",
            "dataset": "sharegpt",  # loader violation
            "cache_bust": {"target": "none"},  # cache_bust violation
        },
        duration=60,  # duration-floor violation
    )


def test_all_five_invariants_lock_raises_with_multiple_violations() -> None:
    run = _multi_violation_run(unsafe_override=False)
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    # v2 emits 5 simultaneous violations for this config (streaming, ignore_eos,
    # loader, cache_bust, duration); timing_mode is stamped, not violated.
    assert len(exc.value.violations) == 5


def test_all_five_invariants_unsafe_override_warns_and_invalidates() -> None:
    run = _multi_violation_run(unsafe_override=True)
    outcome = apply_scenario(run)
    assert outcome.submission_valid is False
    assert len(outcome.violations) == 5
    assert "unsafe_override" in outcome.submission_invalid_reasons


# ---------------------------------------------------------------------------
# Scalar concurrency passes the lock (rebased from test_int_concurrency).
# v2 no longer checks concurrency in the scenario lock (see the
# concurrency-sweep PORT-NOTE in validator.py); a scalar concurrency phase is
# simply a clean run.
# ---------------------------------------------------------------------------
def test_int_concurrency_passes_lock() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": True}, concurrency=10)
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is True
