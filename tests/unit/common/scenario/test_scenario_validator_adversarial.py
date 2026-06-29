# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for `validate_scenario`.

Each test attacks a specific edge case in the AgentX scenario validator.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario import (
    ScenarioLockError,
    UnknownScenarioError,
    validate_scenario,
)
from aiperf.plugin.enums import TimingMode


def _user_config(
    *,
    scenario: str | None = "inferencex-agentx-mvp",
    timing_mode: TimingMode | str = TimingMode.AGENTIC_REPLAY,
    extra_inputs: dict | None = None,
    use_think_time_only: bool = True,
    ignore_trace_delays: bool = False,
    synthesis_max_isl: int | None = None,
    loader: str | None = "semianalysis_cc_traces_weka_with_subagents",
    benchmark_duration: float | None = 900.0,
    inter_turn_delay_cap_seconds: float | None = None,
    trace_idle_gap_cap_seconds: float | None = 10.0,
    random_seed: int | None = 42,
    unsafe_override: bool = False,
    cache_bust_target: CacheBustTarget = CacheBustTarget.FIRST_TURN_PREFIX,
) -> MagicMock:
    """Build a MagicMock UserConfig pre-shaped for the scenario validator."""
    cfg = MagicMock()
    cfg.scenario = scenario
    cfg.unsafe_override = unsafe_override
    cfg.timing_mode = timing_mode
    cfg.input.extra_inputs_parsed = extra_inputs if extra_inputs is not None else {}
    cfg.input.use_think_time_only = use_think_time_only
    cfg.input.ignore_trace_delays = ignore_trace_delays
    cfg.input.random_seed = random_seed
    cfg.input.synthesis.max_isl = synthesis_max_isl
    cfg.input.detected_loader = loader
    cfg.loadgen.benchmark_duration = benchmark_duration
    cfg.loadgen.inter_turn_delay_cap_seconds = inter_turn_delay_cap_seconds
    cfg.loadgen.trace_idle_gap_cap_seconds = trace_idle_gap_cap_seconds
    cfg.input.prompt.cache_bust.target = cache_bust_target
    cfg.input._use_think_time_only_explicitly_set = False
    cfg.loadgen._inter_turn_delay_cap_explicitly_set = False
    cfg.loadgen._trace_idle_gap_cap_explicitly_set = False
    cfg.input.prompt.cache_bust._target_explicitly_set = False
    return cfg


# ---------------------------------------------------------------------------
# Test 1: --scenario set twice via config-file precedence (pin behavior).
# The validator only sees the *resolved* `scenario` attribute, so config-file
# precedence is upstream of validation. Pin: whatever value lands on
# `cfg.scenario` is the one validated; the validator does not double-validate
# a list of scenario names.
# ---------------------------------------------------------------------------
def test_scenario_set_twice_validator_uses_resolved_value() -> None:
    cfg = _user_config(
        scenario="inferencex-agentx-mvp", extra_inputs={"ignore_eos": True}
    )
    outcome = validate_scenario(cfg)
    assert outcome.submission_valid is True


# ---------------------------------------------------------------------------
# Test 2: --unsafe-override without --scenario is a no-op.
# ---------------------------------------------------------------------------
def test_unsafe_override_without_scenario_is_noop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _user_config(scenario=None, unsafe_override=True)
    with caplog.at_level("WARNING"):
        outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is None
    assert outcome.submission_invalid_reasons == []
    assert not any(
        "scenario" in r.message.lower() or "override" in r.message.lower()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Test 3: Unknown scenario name raises UnknownScenarioError listing valid set.
# ---------------------------------------------------------------------------
def test_unknown_scenario_name_raises_unknown_scenario_error() -> None:
    cfg = _user_config(scenario="not-a-real-scenario")
    with pytest.raises(UnknownScenarioError) as exc:
        validate_scenario(cfg)
    msg = str(exc.value)
    assert "not-a-real-scenario" in msg
    assert "inferencex-agentx-mvp" in msg


# ---------------------------------------------------------------------------
# Test 4a: extra_inputs.ignore_eos string "true" treated as truthy.
# ---------------------------------------------------------------------------
def test_ignore_eos_string_true_treated_as_truthy() -> None:
    cfg = _user_config(extra_inputs={"ignore_eos": "true"})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True


# ---------------------------------------------------------------------------
# Test 4b: extra_inputs.ignore_eos string "false" treated as falsy (violation).
# ---------------------------------------------------------------------------
def test_ignore_eos_string_false_treated_as_falsy_violation() -> None:
    cfg = _user_config(extra_inputs={"ignore_eos": "false"})
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    assert any(
        "ignore_eos" in v.flag or "ignore_eos" in v.message
        for v in exc.value.violations
    )


# ---------------------------------------------------------------------------
# Test 5: extra_inputs.ignore_eos numeric / null coercion behavior.
# Pinned: 1 -> truthy (clean); 0 -> falsy (violation); None/null -> absent
# (auto-injected to True).
# ---------------------------------------------------------------------------
def test_ignore_eos_int_one_treated_as_truthy() -> None:
    cfg = _user_config(extra_inputs={"ignore_eos": 1})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []


def test_ignore_eos_int_zero_treated_as_falsy_violation() -> None:
    cfg = _user_config(extra_inputs={"ignore_eos": 0})
    with pytest.raises(ScenarioLockError):
        validate_scenario(cfg)


def test_ignore_eos_none_is_treated_as_absent_and_injected() -> None:
    # Pinned: a parsed JSON null becomes Python None; the validator treats
    # this as "absent" (the same as if the key weren't provided at all) and
    # injects ignore_eos=True. Documented as the explicit precedence: only
    # `is None` qualifies as absent for injection purposes.
    cfg = _user_config(extra_inputs={"ignore_eos": None})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert cfg.input.extra_inputs_parsed["ignore_eos"] is True


# ---------------------------------------------------------------------------
# Test 6: extra_inputs as JSON string vs dict at validator entry.
# The validator is documented to run AFTER extra_inputs parsing — its
# contract is that `extra_inputs_parsed` is already a dict. Passing a raw
# JSON string falls back to `{}` (treated as absent), which then triggers
# the ignore_eos injection path. The post-parsed dict is the supported
# shape and produces the canonical clean outcome. Both shapes succeed
# without violations because injection happens for the absent case.
# ---------------------------------------------------------------------------
def test_extra_inputs_json_string_vs_dict_identical_clean_outcome() -> None:
    cfg_dict = _user_config(extra_inputs={"ignore_eos": True})
    cfg_str = _user_config()
    # Simulate an unparsed string surviving to the validator: extract path
    # cannot coerce it, so it appears absent and `ignore_eos` is injected.
    cfg_str.input.extra_inputs_parsed = '{"ignore_eos": true}'
    out_dict = validate_scenario(cfg_dict)
    out_str = validate_scenario(cfg_str)
    assert out_dict.violations == []
    assert out_str.violations == []
    assert out_dict.submission_valid is True
    assert out_str.submission_valid is True


# ---------------------------------------------------------------------------
# Test 7: --ignore-trace-delays is REJECTED for AgentX MVP. The scenario replays
# recorded trace timing (timing_mode=AGENTIC_REPLAY); --ignore-trace-delays nulls
# every per-turn timestamp/delay in the loader and dispatches all turns
# back-to-back, falsifying the workload. The validator gates on the dedicated
# spec.forbid_ignore_trace_delays invariant (decoupled from the now-unset
# require_use_think_time_only), so it raises without --unsafe-override and stamps
# submission_valid=false with it.
# ---------------------------------------------------------------------------
def test_ignore_trace_delays_rejected_for_agentx() -> None:
    cfg = _user_config(
        ignore_trace_delays=True,
        use_think_time_only=False,
        extra_inputs={"ignore_eos": True},
    )
    with pytest.raises(ScenarioLockError, match="ignore-trace-delays"):
        validate_scenario(cfg)


def test_ignore_trace_delays_with_unsafe_override_marks_submission_invalid() -> None:
    cfg = _user_config(
        ignore_trace_delays=True,
        use_think_time_only=False,
        extra_inputs={"ignore_eos": True},
        unsafe_override=True,
    )
    outcome = validate_scenario(cfg)
    assert outcome.submission_valid is False
    assert any(v.flag == "--ignore-trace-delays" for v in outcome.violations)


# ---------------------------------------------------------------------------
# Test 8: Validator invoked twice on the same UserConfig is idempotent.
# Re-running model_post_init must not double-inject ignore_eos, must not
# re-log injection notices, and must not auto-set random_seed twice.
# ---------------------------------------------------------------------------
def test_validator_idempotent_under_reentry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _user_config(extra_inputs={}, random_seed=None)
    with caplog.at_level("INFO"):
        first = validate_scenario(cfg)
        seed_after_first = cfg.input.random_seed
        injected_after_first = cfg.input.extra_inputs_parsed["ignore_eos"]
        first_log_count = sum(1 for r in caplog.records if "ignore_eos" in r.message)
        caplog.clear()
        second = validate_scenario(cfg)
        second_log_count = sum(1 for r in caplog.records if "ignore_eos" in r.message)
    assert first.violations == []
    assert second.violations == []
    assert cfg.input.random_seed == seed_after_first
    assert cfg.input.extra_inputs_parsed["ignore_eos"] == injected_after_first
    assert first_log_count == 1
    assert second_log_count == 0


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
    cfg = _user_config(benchmark_duration=duration, extra_inputs={"ignore_eos": True})
    if should_pass:
        outcome = validate_scenario(cfg)
        assert outcome.violations == []
    else:
        with pytest.raises(ScenarioLockError):
            validate_scenario(cfg)


# ---------------------------------------------------------------------------
# Test 10: --synthesis-max-isl edge values.
# Pinned (current behavior): the validator rejects ANY non-None
# synthesis.max_isl, including 0. The spec hints 0 might semantically mean
# "no truncation"; we pin the strict behavior here. A very high value
# (10**9) is also rejected — there is no warn-only middle ground today.
# ---------------------------------------------------------------------------
def test_synthesis_max_isl_zero_rejected_under_lock() -> None:
    cfg = _user_config(synthesis_max_isl=0, extra_inputs={"ignore_eos": True})
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    assert any(v.flag == "--synthesis-max-isl" for v in exc.value.violations)


def test_synthesis_max_isl_very_high_rejected_under_lock() -> None:
    cfg = _user_config(synthesis_max_isl=10**9, extra_inputs={"ignore_eos": True})
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    assert any(v.flag == "--synthesis-max-isl" for v in exc.value.violations)


# ---------------------------------------------------------------------------
# Test 11: random_seed=0 is treated as set (falsy but not None).
# ---------------------------------------------------------------------------
def test_random_seed_zero_not_auto_injected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _user_config(random_seed=0, extra_inputs={"ignore_eos": True})
    with caplog.at_level("INFO"):
        outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert cfg.input.random_seed == 0
    assert not any("auto-set random_seed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Test 12: All 5 invariants violated simultaneously under AgentX MVP.
# 1) timing_mode mismatch
# 2) ignore_eos=false explicit
# 3) synthesis.max_isl set
# 4) wrong loader
# 5) duration below floor
#
# Note: use_think_time_only and ignore_trace_delays no longer surface as
# violations — the scenario dropped require_use_think_time_only=True in favor
# of trace_idle_gap_cap_seconds.
# ---------------------------------------------------------------------------
def _five_violations_config(*, unsafe_override: bool) -> MagicMock:
    cfg = _user_config(
        timing_mode=TimingMode.REQUEST_RATE,
        extra_inputs={"ignore_eos": False},
        synthesis_max_isl=4096,
        loader="dag_jsonl",
        benchmark_duration=60.0,
        unsafe_override=unsafe_override,
    )
    cfg._timing_mode_explicitly_set = True
    return cfg


def test_all_five_invariants_lock_raises_with_five_violations() -> None:
    cfg = _five_violations_config(unsafe_override=False)
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    assert len(exc.value.violations) == 5


def test_all_five_invariants_unsafe_override_warns_and_invalidates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _five_violations_config(unsafe_override=True)
    with caplog.at_level("WARNING"):
        outcome = validate_scenario(cfg)
    assert outcome.submission_valid is False
    assert len(outcome.violations) == 5
    warning_count = sum(
        1
        for r in caplog.records
        if r.levelname == "WARNING" and "Scenario violation" in r.message
    )
    assert warning_count == 5
    assert "unsafe_override" in outcome.submission_invalid_reasons


# ---------------------------------------------------------------------------
# Test: list-shape --concurrency (parameter sweep) is rejected by lock.
# A locked scenario describes one fixed configuration; sweeping concurrency
# would multiply it into N runs with diverging settings, which violates the
# "one scenario = one spec" contract.
# ---------------------------------------------------------------------------
def test_list_concurrency_rejected_as_sweep_violation() -> None:
    cfg = _user_config(extra_inputs={"ignore_eos": True})
    cfg.prompt.cache_bust.target = "first_turn_prefix"
    cfg.loadgen.concurrency = [10, 20, 30]
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    assert any(
        v.flag == "--concurrency" and "sweep" in v.message for v in exc.value.violations
    ), f"expected sweep violation, got: {[str(v) for v in exc.value.violations]}"


def test_int_concurrency_passes_lock() -> None:
    cfg = _user_config(extra_inputs={"ignore_eos": True})
    cfg.prompt.cache_bust.target = "first_turn_prefix"
    cfg.loadgen.concurrency = 10
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_list_concurrency_with_unsafe_override_warns_only() -> None:
    cfg = _user_config(extra_inputs={"ignore_eos": True}, unsafe_override=True)
    cfg.prompt.cache_bust.target = "first_turn_prefix"
    cfg.loadgen.concurrency = [10, 20, 30]
    outcome = validate_scenario(cfg)
    assert outcome.submission_valid is False
    assert "unsafe_override" in outcome.submission_invalid_reasons
    assert any(v.flag == "--concurrency" for v in outcome.violations)
