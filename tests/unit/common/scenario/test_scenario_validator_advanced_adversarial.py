# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advanced adversarial tests for `validate_scenario`.

Picks up where `test_scenario_validator_adversarial.py` leaves off; each test
pins behavior on edge cases not covered by the basic or first-round
adversarial suites:

* truthy/falsy coercion variants for `extra_inputs.ignore_eos` beyond the
  canonical "true"/"false" strings already exercised
* the inter-turn-delay-cap explicit-but-matching path
* `--unsafe-override` interaction with a clean config (no violations)
* `detected_loader=None` (loader auto-detection unset)
* `_extract_extra_inputs` fallback paths (the `extra` attribute and
  non-coercible raw values)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario import (
    ScenarioLockError,
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
# ignore_eos truthy-string variants beyond "true"
# ---------------------------------------------------------------------------
def test_ignore_eos_truthy_string_yes_passes() -> None:
    """'yes' is in `_is_truthy_extra_input`'s allow-list AND is not falsy."""
    cfg = _user_config(extra_inputs={"ignore_eos": "yes"})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_ignore_eos_truthy_string_one_passes() -> None:
    """The string '1' is recognized as truthy and produces no violation."""
    cfg = _user_config(extra_inputs={"ignore_eos": "1"})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_ignore_eos_uppercase_true_treated_as_truthy() -> None:
    """`_is_truthy_extra_input` lower-cases — 'TRUE' / 'YES' must pass."""
    cfg = _user_config(extra_inputs={"ignore_eos": "TRUE"})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []


def test_ignore_eos_padded_yes_treated_as_truthy() -> None:
    """`_is_truthy_extra_input` strips whitespace before lower-casing."""
    cfg = _user_config(extra_inputs={"ignore_eos": "  yes  "})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []


def test_ignore_eos_unknown_string_not_falsy_does_not_violate() -> None:
    """A string outside both allow-lists ('maybe') is NOT falsy, so no
    violation. Pin: only explicit falsy strings trigger
    `--scenario` lock; everything else passes through."""
    cfg = _user_config(extra_inputs={"ignore_eos": "maybe"})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True


# ---------------------------------------------------------------------------
# ignore_eos falsy variants beyond "false"
# ---------------------------------------------------------------------------
def test_ignore_eos_falsy_string_no_violates() -> None:
    """'no' is in `_is_falsy_extra_input`'s reject list."""
    cfg = _user_config(extra_inputs={"ignore_eos": "no"})
    with pytest.raises(ScenarioLockError) as exc_info:
        validate_scenario(cfg)
    assert any(v.flag == "extra_inputs.ignore_eos" for v in exc_info.value.violations)


def test_ignore_eos_falsy_string_zero_violates() -> None:
    """The string '0' is falsy."""
    cfg = _user_config(extra_inputs={"ignore_eos": "0"})
    with pytest.raises(ScenarioLockError) as exc_info:
        validate_scenario(cfg)
    assert any(v.flag == "extra_inputs.ignore_eos" for v in exc_info.value.violations)


# ---------------------------------------------------------------------------
# trace_idle_gap_cap_seconds: explicit-and-matching path
# ---------------------------------------------------------------------------
def test_trace_idle_gap_cap_explicit_matching_no_violation() -> None:
    """When the user explicitly sets the cap to the spec value, no violation
    fires and no auto-fill log line is emitted."""
    cfg = _user_config(
        extra_inputs={"ignore_eos": True},
        trace_idle_gap_cap_seconds=10.0,
    )
    cfg.loadgen._trace_idle_gap_cap_explicitly_set = True
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True
    assert cfg.loadgen.trace_idle_gap_cap_seconds == 10.0


# ---------------------------------------------------------------------------
# unsafe_override + clean config: must NOT flip submission_valid to False
# ---------------------------------------------------------------------------
def test_unsafe_override_with_no_violations_returns_submission_valid_true() -> None:
    """Pin: `unsafe_override=True` only flips `submission_valid` to False
    when there are violations. A clean config under override still returns
    `submission_valid=True` and `submission_invalid_reasons=[]`."""
    cfg = _user_config(extra_inputs={"ignore_eos": True}, unsafe_override=True)
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True
    assert outcome.submission_invalid_reasons == []


# ---------------------------------------------------------------------------
# detected_loader=None: when scenario requires a loader, an unset detection
# IS a violation (None != "semianalysis_cc_traces_weka_with_subagents"). Loader auto-detection runs before
# scenario validation in production; if it produced None, the user gave us
# something we couldn't classify as the required loader.
# ---------------------------------------------------------------------------
def test_detected_loader_none_violates_when_loader_required() -> None:
    """Pin: `spec.require_loader is not None and detected != spec.require_loader`
    fires for None. Treating None as 'not yet detected' silently accepted
    runs that bypassed the loader entirely."""
    cfg = _user_config(extra_inputs={"ignore_eos": True}, loader=None)
    with pytest.raises(ScenarioLockError) as exc_info:
        validate_scenario(cfg)
    assert any(v.flag == "--input-file (loader)" for v in exc_info.value.violations)


# ---------------------------------------------------------------------------
# benchmark_duration=0 still violates (zero falls below the 900s floor and
# the `or 0.0` short-circuits identically to None).
# ---------------------------------------------------------------------------
def test_benchmark_duration_zero_violates() -> None:
    """0 < 900 produces a duration violation; pin that 0 is treated like
    'unset' through `duration or 0.0` rather than as 'unlimited'."""
    cfg = _user_config(extra_inputs={"ignore_eos": True}, benchmark_duration=0)
    with pytest.raises(ScenarioLockError) as exc_info:
        validate_scenario(cfg)
    assert any(v.flag == "--benchmark-duration" for v in exc_info.value.violations)


def test_benchmark_duration_none_auto_fills_scenario_default() -> None:
    """`None` benchmark_duration is auto-filled from the scenario's
    default_benchmark_duration_seconds (1800) instead of violating."""
    cfg = _user_config(extra_inputs={"ignore_eos": True}, benchmark_duration=None)
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert cfg.loadgen.benchmark_duration == 1800.0


# ---------------------------------------------------------------------------
# `_extract_extra_inputs` fallback paths
# ---------------------------------------------------------------------------
def test_extra_inputs_falls_back_to_extra_attribute_when_parsed_is_none() -> None:
    """Pin: if `extra_inputs_parsed` is None, the helper falls through to
    `cfg.input.extra`. A dict on `extra` containing falsy ignore_eos must
    still surface as a violation."""
    cfg = _user_config()
    cfg.input.extra_inputs_parsed = None
    cfg.input.extra = {"ignore_eos": False}
    with pytest.raises(ScenarioLockError) as exc_info:
        validate_scenario(cfg)
    assert any(v.flag == "extra_inputs.ignore_eos" for v in exc_info.value.violations)


def test_extra_inputs_non_coercible_raw_treated_as_empty() -> None:
    """Pin: a raw value that is neither dict nor None and that `dict(raw)`
    cannot coerce (e.g. an int) lands in the `except (TypeError, ValueError)`
    branch and yields `{}`. The validator then injects `ignore_eos=True`
    into `extra_inputs_parsed` and runs clean."""
    cfg = _user_config()
    # Override both lookup attributes so neither yields a usable mapping.
    cfg.input.extra_inputs_parsed = 42
    cfg.input.extra = 42
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True
