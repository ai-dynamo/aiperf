# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario import ScenarioLockError
from aiperf.common.scenario.validator import (
    ValidationOutcome,
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
    public_dataset: str | None = None,
    hf_weka_dataset: str | None = None,
    benchmark_duration: float | None = 900.0,
    inter_turn_delay_cap_seconds: float | None = None,
    trace_idle_gap_cap_seconds: float | None = 60.0,
    random_seed: int | None = 42,
    unsafe_override: bool = False,
    cache_bust_target: CacheBustTarget = CacheBustTarget.FIRST_TURN_PREFIX,
    streaming: bool = True,
    streaming_explicitly_set: bool = False,
) -> MagicMock:
    cfg = MagicMock()
    cfg.scenario = scenario
    cfg.unsafe_override = unsafe_override
    cfg.timing_mode = timing_mode
    cfg.endpoint.streaming = streaming
    cfg.endpoint._streaming_explicitly_set = streaming_explicitly_set
    cfg.input.extra_inputs_parsed = extra_inputs if extra_inputs is not None else {}
    cfg.input.use_think_time_only = use_think_time_only
    cfg.input.ignore_trace_delays = ignore_trace_delays
    cfg.input.random_seed = random_seed
    cfg.input.synthesis.max_isl = synthesis_max_isl
    cfg.input.detected_loader = loader
    cfg.input.public_dataset = public_dataset
    cfg.input.hf_weka_dataset = hf_weka_dataset
    cfg.loadgen.benchmark_duration = benchmark_duration
    cfg.loadgen.inter_turn_delay_cap_seconds = inter_turn_delay_cap_seconds
    cfg.loadgen.trace_idle_gap_cap_seconds = trace_idle_gap_cap_seconds
    cfg.input.prompt.cache_bust.target = cache_bust_target
    # Default: explicit-set flags off, so auto-injection paths are exercised
    # unless a test overrides them.
    cfg.input._use_think_time_only_explicitly_set = False
    cfg.loadgen._inter_turn_delay_cap_explicitly_set = False
    cfg.loadgen._trace_idle_gap_cap_explicitly_set = False
    cfg.input.prompt.cache_bust._target_explicitly_set = False
    return cfg


def test_no_scenario_returns_noop() -> None:
    cfg = _user_config(scenario=None)
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is None


def test_clean_config_no_violations() -> None:
    cfg = _user_config(extra_inputs={"ignore_eos": True})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_wrong_timing_mode_raises_under_lock() -> None:
    cfg = _user_config(
        timing_mode=TimingMode.REQUEST_RATE, extra_inputs={"ignore_eos": True}
    )
    cfg._timing_mode_explicitly_set = True
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    assert "--request-rate" in str(exc.value)


def test_default_timing_mode_auto_set_under_scenario(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _user_config(
        timing_mode=TimingMode.REQUEST_RATE, extra_inputs={"ignore_eos": True}
    )
    cfg._timing_mode_explicitly_set = False
    with caplog.at_level("INFO"):
        outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert cfg._timing_mode == TimingMode.AGENTIC_REPLAY
    assert any("timing_mode" in r.message for r in caplog.records)


def test_explicit_ignore_eos_false_raises() -> None:
    cfg = _user_config(extra_inputs={"ignore_eos": False})
    with pytest.raises(ScenarioLockError):
        validate_scenario(cfg)


def test_absent_ignore_eos_injects_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _user_config(extra_inputs={})
    with caplog.at_level("INFO"):
        outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert cfg.input.extra_inputs_parsed["ignore_eos"] is True
    assert any("ignore_eos" in r.message for r in caplog.records)


def test_explicit_no_streaming_raises() -> None:
    cfg = _user_config(
        streaming=False,
        streaming_explicitly_set=True,
        extra_inputs={"ignore_eos": True},
    )
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    assert "--streaming" in str(exc.value)


def test_absent_streaming_auto_enabled_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _user_config(
        streaming=False,
        streaming_explicitly_set=False,
        extra_inputs={"ignore_eos": True},
    )
    with caplog.at_level("INFO"):
        outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert cfg.endpoint.streaming is True
    assert any("--streaming" in r.message for r in caplog.records)


def test_streaming_on_no_violation() -> None:
    cfg = _user_config(streaming=True, extra_inputs={"ignore_eos": True})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_require_streaming_against_real_endpoint_config() -> None:
    """Run validate_scenario with a real ``EndpointConfig`` (not a MagicMock) at
    ``user_config.endpoint``. An explicit ``--no-streaming`` must surface as a
    ``--streaming`` violation, confirming the validator reads a path that exists
    on the production config — if ``EndpointConfig.streaming`` or its
    explicit-set flag is renamed/relocated, this fails loudly."""
    from aiperf.common.config.endpoint_config import EndpointConfig

    cfg = _user_config(extra_inputs={"ignore_eos": True}, unsafe_override=True)
    # streaming=False explicitly provided -> chat supports streaming, so it is
    # left False and recorded as explicitly set.
    cfg.endpoint = EndpointConfig(model_names=["m"], type="chat", streaming=False)
    assert cfg.endpoint.streaming is False
    assert cfg.endpoint._streaming_explicitly_set is True

    outcome = validate_scenario(cfg)
    flags = [v.flag for v in outcome.violations]
    assert "--streaming" in flags, (
        "validator did not flag --streaming on a real EndpointConfig "
        "— the attribute path likely drifted; check validator.py and "
        "EndpointConfig.streaming / _streaming_explicitly_set"
    )


def test_use_think_time_only_false_explicit_does_not_raise() -> None:
    """AgentX MVP no longer locks --use-think-time-only; trace_idle_gap_cap_seconds
    supersedes think-time-based delays in the weka loader."""
    cfg = _user_config(use_think_time_only=False, extra_inputs={"ignore_eos": True})
    cfg.input._use_think_time_only_explicitly_set = True
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_synthesis_max_isl_set_raises() -> None:
    cfg = _user_config(synthesis_max_isl=4096, extra_inputs={"ignore_eos": True})
    with pytest.raises(ScenarioLockError):
        validate_scenario(cfg)


def test_wrong_loader_raises() -> None:
    cfg = _user_config(loader="dag_jsonl", extra_inputs={"ignore_eos": True})
    with pytest.raises(ScenarioLockError):
        validate_scenario(cfg)


def test_agentx_allows_generic_weka_hf_loader_for_explicit_weka_repo() -> None:
    cfg = _user_config(
        loader="weka_hf",
        public_dataset="weka_hf",
        hf_weka_dataset="semianalysisai/cc-traces-weka-062126",
        extra_inputs={"ignore_eos": True},
    )

    outcome = validate_scenario(cfg)

    assert outcome.violations == []
    assert cfg.input.public_dataset == "weka_hf"


def test_agentx_rejects_generic_weka_hf_loader_without_repo() -> None:
    cfg = _user_config(
        loader="weka_hf",
        public_dataset="weka_hf",
        hf_weka_dataset=None,
        extra_inputs={"ignore_eos": True},
    )

    with pytest.raises(ScenarioLockError) as exc_info:
        validate_scenario(cfg)

    assert "hf_weka_dataset" in str(exc_info.value)


def test_agentx_rejects_generic_weka_hf_loader_for_arbitrary_repo() -> None:
    cfg = _user_config(
        loader="weka_hf",
        public_dataset="weka_hf",
        hf_weka_dataset="example/not-agentx-corpus",
        extra_inputs={"ignore_eos": True},
    )

    with pytest.raises(ScenarioLockError) as exc_info:
        validate_scenario(cfg)

    assert "semianalysisai/cc-traces-weka-062126" in str(exc_info.value)


@pytest.mark.parametrize(
    "loader",
    [
        "semianalysis_cc_traces_weka_061326",
        "semianalysis_cc_traces_weka_061326_256k",
        "semianalysis_cc_traces_weka_061526",
        "semianalysis_cc_traces_weka_061526_256k",
        "semianalysis_cc_traces_weka_062126",
        "semianalysis_cc_traces_weka_062126_256k",
        "semianalysis_cc_traces_weka_with_subagents",
        "semianalysis_cc_traces_weka_with_subagents_256k",
        "semianalysis_cc_traces_weka_with_subagents_060826",
        "weka_trace",
    ],
)  # fmt: skip
def test_agentx_accepts_allowed_weka_loaders(loader: str) -> None:
    """Every loader in the scenario allow-list validates without violations."""
    cfg = _user_config(loader=loader, extra_inputs={"ignore_eos": True})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []


def test_duration_below_floor_raises() -> None:
    cfg = _user_config(benchmark_duration=899.999, extra_inputs={"ignore_eos": True})
    with pytest.raises(ScenarioLockError):
        validate_scenario(cfg)


def test_duration_at_floor_ok() -> None:
    cfg = _user_config(benchmark_duration=900.0, extra_inputs={"ignore_eos": True})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []


def test_random_seed_unset_auto_injected_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _user_config(random_seed=None, extra_inputs={"ignore_eos": True})
    with caplog.at_level("INFO"):
        outcome = validate_scenario(cfg)
    assert cfg.input.random_seed is not None
    assert outcome.violations == []
    assert any("random_seed" in r.message for r in caplog.records)


def test_inter_turn_delay_cap_explicit_other_value_does_not_raise() -> None:
    """AgentX MVP no longer locks --inter-turn-delay-cap-seconds;
    trace_idle_gap_cap_seconds supersedes the per-turn cap in the weka loader."""
    cfg = _user_config(
        inter_turn_delay_cap_seconds=30.0, extra_inputs={"ignore_eos": True}
    )
    cfg.loadgen._inter_turn_delay_cap_explicitly_set = True
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_trace_idle_gap_cap_explicit_other_value_raises() -> None:
    cfg = _user_config(
        trace_idle_gap_cap_seconds=30.0, extra_inputs={"ignore_eos": True}
    )
    cfg.loadgen._trace_idle_gap_cap_explicitly_set = True
    with pytest.raises(ScenarioLockError):
        validate_scenario(cfg)


def test_trace_idle_gap_cap_unset_auto_filled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _user_config(
        trace_idle_gap_cap_seconds=None, extra_inputs={"ignore_eos": True}
    )
    cfg.loadgen._trace_idle_gap_cap_explicitly_set = False
    with caplog.at_level("INFO"):
        outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert cfg.loadgen.trace_idle_gap_cap_seconds == 60.0


def test_unsafe_override_converts_errors_to_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _user_config(
        timing_mode=TimingMode.REQUEST_RATE,
        synthesis_max_isl=4096,
        extra_inputs={"ignore_eos": True},
        unsafe_override=True,
    )
    cfg._timing_mode_explicitly_set = True
    with caplog.at_level("WARNING"):
        outcome = validate_scenario(cfg)
    assert outcome.submission_valid is False
    assert len(outcome.violations) == 2
    assert any("--request-rate" in r.message for r in caplog.records)
    assert any("--synthesis-max-isl" in r.message for r in caplog.records)


def test_all_violations_collected_in_one_pass() -> None:
    cfg = _user_config(
        timing_mode=TimingMode.REQUEST_RATE,
        extra_inputs={"ignore_eos": False},
        synthesis_max_isl=4096,
        loader="dag_jsonl",
        benchmark_duration=60.0,
    )
    cfg._timing_mode_explicitly_set = True
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    assert len(exc.value.violations) >= 5


def test_validation_outcome_dataclass_defaults() -> None:
    outcome = ValidationOutcome()
    assert outcome.violations == []
    assert outcome.submission_valid is None
    assert outcome.submission_invalid_reasons == []


# =============================================================================
# Default timing_mode must NOT raise — auto-injection covers it
# =============================================================================
#
# With --scenario alone (no explicit --timing-mode), user_config falls through
# to TimingMode.REQUEST_RATE default. The validator gates the violation on
# `_timing_mode_explicitly_set` and auto-injects spec.timing_mode when at
# default, so the run continues cleanly under the scenario's required mode.


def test_default_timing_mode_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Default REQUEST_RATE under --scenario auto-sets to AGENTIC_REPLAY
    rather than raising."""
    cfg = _user_config(
        timing_mode=TimingMode.REQUEST_RATE, extra_inputs={"ignore_eos": True}
    )
    cfg._timing_mode_explicitly_set = False
    with caplog.at_level("INFO"):
        outcome = validate_scenario(cfg)  # must NOT raise
    assert outcome.violations == []
    assert outcome.submission_valid is True
    assert cfg._timing_mode == TimingMode.AGENTIC_REPLAY
    # INFO log records the auto-set decision so users can see what changed.
    assert any(
        "timing_mode" in r.message and r.levelname == "INFO" for r in caplog.records
    )


# =============================================================================
# Read-only timing_mode property — write must reach _timing_mode storage
# =============================================================================
#
# UserConfig.timing_mode is a read-only @property; storage is `_timing_mode`.
# The validator's auto-injection writes through to `_timing_mode` via an
# AttributeError fallback so it works against the real config layout.


class _ReadOnlyTimingModeConfig:
    """Mimics UserConfig: timing_mode is a read-only property over _timing_mode."""

    def __init__(self, *, scenario: str, initial_timing_mode: TimingMode) -> None:
        self.scenario = scenario
        self.unsafe_override = False
        self._timing_mode = initial_timing_mode
        self._timing_mode_explicitly_set = False
        # All other attributes consulted by validate_scenario routed through
        # MagicMock for parity with the other tests in this file.
        self.input = MagicMock()
        self.input.extra_inputs_parsed = {"ignore_eos": True}
        self.input.use_think_time_only = True
        self.input.ignore_trace_delays = False
        self.input.random_seed = 42
        self.input.synthesis.max_isl = None
        self.input.detected_loader = "semianalysis_cc_traces_weka_with_subagents"
        self.input._use_think_time_only_explicitly_set = False
        self.loadgen = MagicMock()
        self.loadgen.benchmark_duration = 900.0
        self.loadgen.inter_turn_delay_cap_seconds = None
        self.loadgen._inter_turn_delay_cap_explicitly_set = False
        self.loadgen.trace_idle_gap_cap_seconds = 60.0
        self.loadgen._trace_idle_gap_cap_explicitly_set = False
        self.endpoint = MagicMock()
        self.endpoint.streaming = True
        self.endpoint._streaming_explicitly_set = False
        self.prompt = MagicMock()
        self.input.prompt.cache_bust.target = CacheBustTarget.FIRST_TURN_PREFIX

    @property
    def timing_mode(self) -> TimingMode:
        return self._timing_mode


def test_timing_mode_property_assignment(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Validator falls back to ``_timing_mode`` when ``timing_mode`` is a
    read-only property (real UserConfig shape)."""
    cfg = _ReadOnlyTimingModeConfig(
        scenario="inferencex-agentx-mvp",
        initial_timing_mode=TimingMode.REQUEST_RATE,
    )
    with caplog.at_level("INFO"):
        outcome = validate_scenario(cfg)  # must NOT raise AttributeError
    assert outcome.violations == []
    assert outcome.submission_valid is True
    # Underlying storage was updated through the AttributeError fallback path.
    assert cfg._timing_mode == TimingMode.AGENTIC_REPLAY
    assert cfg.timing_mode == TimingMode.AGENTIC_REPLAY


# =============================================================================
# Cache-bust enforcement under inferencex-agentx-mvp
# =============================================================================
#
# The scenario pins `require_cache_bust=CacheBustTarget.FIRST_TURN_PREFIX`. The
# validator auto-injects FIRST_TURN_PREFIX when the user didn't explicitly set
# --cache-bust (mirroring ignore_eos / use_think_time_only / cap auto-inject),
# rejects any other target value when explicitly set, and downgrades to a
# warning under --unsafe-override.


def test_agentx_mvp_unset_cache_bust_auto_injected_to_first_turn_prefix(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Default `target=NONE` with no explicit user opt-in is auto-set to
    FIRST_TURN_PREFIX (same auto-inject pattern as the other locked settings)."""
    cfg = _user_config(
        extra_inputs={"ignore_eos": True},
        cache_bust_target=CacheBustTarget.NONE,
    )
    cfg.input.prompt.cache_bust._target_explicitly_set = False
    with caplog.at_level("INFO"):
        outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True
    assert cfg.input.prompt.cache_bust.target == CacheBustTarget.FIRST_TURN_PREFIX
    assert any("cache-bust" in r.message.lower() for r in caplog.records)


def test_agentx_mvp_explicit_cache_bust_none_raises() -> None:
    """When the user explicitly passes `--cache-bust none`, the lock fires —
    auto-injection only applies to the unset/default path."""
    cfg = _user_config(
        extra_inputs={"ignore_eos": True},
        cache_bust_target=CacheBustTarget.NONE,
    )
    cfg.input.prompt.cache_bust._target_explicitly_set = True
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    assert "cache_bust" in str(exc.value).lower()
    assert len(exc.value.violations) == 1
    assert exc.value.violations[0].flag == "--cache-bust"
    assert exc.value.violations[0].current_value == str(CacheBustTarget.NONE)
    assert exc.value.violations[0].required_value == str(
        CacheBustTarget.FIRST_TURN_PREFIX
    )


def test_agentx_mvp_rejects_cache_bust_system_prefix() -> None:
    cfg = _user_config(
        extra_inputs={"ignore_eos": True},
        cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
    )
    cfg.input.prompt.cache_bust._target_explicitly_set = True
    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)
    assert "cache_bust" in str(exc.value).lower()
    assert len(exc.value.violations) == 1
    assert exc.value.violations[0].flag == "--cache-bust"
    assert exc.value.violations[0].current_value == str(CacheBustTarget.SYSTEM_PREFIX)
    assert exc.value.violations[0].required_value == str(
        CacheBustTarget.FIRST_TURN_PREFIX
    )


def test_agentx_mvp_accepts_cache_bust_first_turn_prefix() -> None:
    cfg = _user_config(
        extra_inputs={"ignore_eos": True},
        cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
    )
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_agentx_mvp_unsafe_override_allows_cache_bust_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _user_config(
        extra_inputs={"ignore_eos": True},
        cache_bust_target=CacheBustTarget.NONE,
        unsafe_override=True,
    )
    cfg.input.prompt.cache_bust._target_explicitly_set = True
    with caplog.at_level("WARNING"):
        outcome = validate_scenario(cfg)
    assert outcome.submission_valid is False
    assert len(outcome.violations) == 1
    assert any("cache_bust" in r.message.lower() for r in caplog.records)


# =============================================================================
# Path-drift guard: real Pydantic UserConfig (not MagicMock)
# =============================================================================
#
# Every test above uses MagicMock, which auto-creates whatever attribute path
# the validator reads. That hides path drift: if the validator reads
# `user_config.input.foo.bar` and the real config has no `foo`, the unit
# tests pass while the production guard silently no-ops. This test exercises
# the `forbid_input_truncation` branch against a *real* UserConfig so that
# any future rename of `synthesis.max_isl` will fail loudly here.


def test_forbid_input_truncation_against_real_user_config() -> None:
    """Run validate_scenario with a real ``SynthesisConfig`` (not a MagicMock)
    plumbed in at ``user_config.input.synthesis``. The violation must surface
    as ``--synthesis-max-isl``, confirming the validator reads a path that
    actually exists on the production config — if ``SynthesisConfig.max_isl``
    is ever renamed or relocated, this test fails loudly."""
    from aiperf.common.config.synthesis_config import SynthesisConfig

    cfg = _user_config(extra_inputs={"ignore_eos": True}, unsafe_override=True)
    cfg.input.synthesis = SynthesisConfig(max_isl=1024)  # real, not MagicMock
    assert cfg.input.synthesis.max_isl == 1024

    outcome = validate_scenario(cfg)
    flags = [v.flag for v in outcome.violations]
    assert "--synthesis-max-isl" in flags, (
        "validator did not flag --synthesis-max-isl on a real SynthesisConfig "
        "— the attribute path likely drifted; check validator.py and "
        "SynthesisConfig.max_isl"
    )


# =============================================================================
# Scenario defaults: --benchmark-duration and trajectory start ratios
# =============================================================================
#
# The agentx MVP scenario auto-fills --benchmark-duration=1800 when unset and
# pins the trajectory start ratios to [0.25, 0.75] unless the user explicitly
# overrides them. Explicit user values are honored without violations.


def test_benchmark_duration_unset_auto_fills_scenario_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _user_config(benchmark_duration=None, extra_inputs={"ignore_eos": True})
    with caplog.at_level("INFO"):
        outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert cfg.loadgen.benchmark_duration == 1800.0
    assert any("--benchmark-duration" in r.message for r in caplog.records)


def test_benchmark_duration_explicit_value_not_overridden() -> None:
    cfg = _user_config(benchmark_duration=900.0, extra_inputs={"ignore_eos": True})
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert cfg.loadgen.benchmark_duration == 900.0


def test_trajectory_start_ratios_auto_filled_when_unset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _user_config(extra_inputs={"ignore_eos": True})
    cfg.loadgen.trajectory_start_min_ratio = 0.0
    cfg.loadgen.trajectory_start_max_ratio = 0.7
    cfg.loadgen._trajectory_start_min_ratio_explicitly_set = False
    cfg.loadgen._trajectory_start_max_ratio_explicitly_set = False
    with caplog.at_level("INFO"):
        outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert cfg.loadgen.trajectory_start_min_ratio == 0.25
    assert cfg.loadgen.trajectory_start_max_ratio == 0.75
    assert any("--trajectory-start-min-ratio" in r.message for r in caplog.records)
    assert any("--trajectory-start-max-ratio" in r.message for r in caplog.records)


def test_trajectory_start_ratios_explicit_values_honored() -> None:
    cfg = _user_config(extra_inputs={"ignore_eos": True})
    cfg.loadgen.trajectory_start_min_ratio = 0.1
    cfg.loadgen.trajectory_start_max_ratio = 0.9
    cfg.loadgen._trajectory_start_min_ratio_explicitly_set = True
    cfg.loadgen._trajectory_start_max_ratio_explicitly_set = True
    outcome = validate_scenario(cfg)
    assert outcome.violations == []
    assert cfg.loadgen.trajectory_start_min_ratio == 0.1
    assert cfg.loadgen.trajectory_start_max_ratio == 0.9


def test_scenario_defaults_apply_to_real_loadgen_config() -> None:
    """Run validate_scenario against a real ``LoadGeneratorConfig`` (not a
    MagicMock) so a drifted attribute path in the validator cannot silently
    no-op. Duration auto-fill, ratio defaults, and the idle-gap cap must all
    land on the real Pydantic model."""
    from aiperf.common.config import LoadGeneratorConfig

    cfg = _user_config(extra_inputs={"ignore_eos": True})
    cfg.loadgen = LoadGeneratorConfig()

    outcome = validate_scenario(cfg)

    assert outcome.violations == []
    assert cfg.loadgen.benchmark_duration == 1800.0
    assert cfg.loadgen.trajectory_start_min_ratio == 0.25
    assert cfg.loadgen.trajectory_start_max_ratio == 0.75
    assert cfg.loadgen.trace_idle_gap_cap_seconds == 60.0


def test_real_loadgen_config_records_explicit_ratio_flags() -> None:
    from aiperf.common.config import LoadGeneratorConfig

    default = LoadGeneratorConfig()
    assert default._trajectory_start_min_ratio_explicitly_set is False
    assert default._trajectory_start_max_ratio_explicitly_set is False

    explicit = LoadGeneratorConfig(
        trajectory_start_min_ratio=0.1, trajectory_start_max_ratio=0.9
    )
    assert explicit._trajectory_start_min_ratio_explicitly_set is True
    assert explicit._trajectory_start_max_ratio_explicitly_set is True


def test_explicit_ratio_override_on_real_config_is_honored() -> None:
    from aiperf.common.config import LoadGeneratorConfig

    cfg = _user_config(extra_inputs={"ignore_eos": True})
    cfg.loadgen = LoadGeneratorConfig(
        trajectory_start_min_ratio=0.1, trajectory_start_max_ratio=0.9
    )

    outcome = validate_scenario(cfg)

    assert outcome.violations == []
    assert cfg.loadgen.trajectory_start_min_ratio == 0.1
    assert cfg.loadgen.trajectory_start_max_ratio == 0.9
