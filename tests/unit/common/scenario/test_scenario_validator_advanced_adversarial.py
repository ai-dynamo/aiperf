# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advanced adversarial tests for the v2 scenario resolver (``apply_scenario``).

Rebased from the v1 ``validate_scenario(MagicMock UserConfig)`` suite onto the
v2 ``apply_scenario(run)`` resolver with REAL ``BenchmarkConfig`` /
``BenchmarkRun`` objects. Picks up where
``test_scenario_validator_adversarial.py`` leaves off, pinning edge cases not
covered by the basic or first-round adversarial suites:

* truthy/falsy coercion variants for ``endpoint.extra['ignore_eos']`` beyond
  the canonical "true"/"false" strings
* ``--unsafe-override`` interaction with a clean config (no violations)
* detected-loader None (no recognized loader)
* benchmark-duration 0 / None auto-fill boundaries

DROP NOTES (v1 behaviors with no v2 analog):
* ``_extract_extra_inputs`` fallback paths (parsed -> extra -> dict(raw)) are
  gone: v2 reads ``endpoint.extra`` (a real dict) directly. The two fallback
  tests (``..._falls_back_to_extra_attribute_when_parsed_is_none``,
  ``..._non_coercible_raw_treated_as_empty``) are dropped. The coercion
  semantics themselves are still exercised by the ignore_eos value tests here
  and in the sibling adversarial module.
"""

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


# ---------------------------------------------------------------------------
# ignore_eos truthy-string variants beyond "true"
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ignore_eos falsy variants beyond "false"
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# trace_idle_gap_cap_seconds: explicit-and-matching path
# ---------------------------------------------------------------------------
def test_trace_idle_gap_cap_explicit_matching_no_violation() -> None:
    """When the user explicitly sets the cap to the spec value (10.0), no
    violation fires."""
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


# ---------------------------------------------------------------------------
# unsafe_override + clean config: must NOT flip submission_valid to False
# ---------------------------------------------------------------------------
def test_unsafe_override_with_no_violations_returns_submission_valid_true() -> None:
    """``unsafe_override=True`` only flips ``submission_valid`` to False when
    there are violations. A clean config under override still returns
    ``submission_valid=True`` and ``submission_invalid_reasons=[]``."""
    run = _build_run(streaming=True, extra={"ignore_eos": True}, unsafe_override=True)
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is True
    assert outcome.submission_invalid_reasons == []


# ---------------------------------------------------------------------------
# detected_loader=None: when scenario requires a loader, an undetectable loader
# IS a violation. A FileDataset with no recognized loader in
# run.resolved.dataset_types yields detected=None.
# ---------------------------------------------------------------------------
def test_detected_loader_none_violates_when_loader_required() -> None:
    """``_detect_loader`` returns None for a FileDataset with no resolved
    dataset_types entry; None is not in the allowed loader tuple, so the lock
    fires the ``--input-file (loader)`` violation."""
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


# ---------------------------------------------------------------------------
# benchmark_duration=0 still violates (0 < 900 floor; ``duration or 0.0``
# short-circuits identically to None).
# ---------------------------------------------------------------------------
def test_benchmark_duration_zero_violates() -> None:
    """0 < 900 produces a duration violation; 0 is treated like 'unset'
    through ``duration or 0.0`` rather than 'unlimited'."""
    # phase.duration has gt=0 at config-build; build valid then set 0 directly
    # on the phase to reach the lock's ``duration or 0.0`` short-circuit.
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    run.cfg.get_profiling_phases()[0].duration = 0
    with pytest.raises(ScenarioLockError) as exc_info:
        apply_scenario(run)
    assert any(v.flag == "--benchmark-duration" for v in exc_info.value.violations)


def test_benchmark_duration_none_auto_fills_scenario_default() -> None:
    """``None`` duration is auto-filled from the scenario's
    ``default_benchmark_duration_seconds`` (1800) instead of violating. Build
    then clear the phase duration to simulate 'unset' (duration is required at
    config-build time as a stop condition)."""
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    run.cfg.get_profiling_phases()[0].duration = None
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert run.cfg.get_profiling_phases()[0].duration == 1800.0
