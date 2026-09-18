# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""v2 scenario-resolver tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario import ScenarioLockError, apply_scenario
from aiperf.common.scenario.base import ScenarioOutcome
from aiperf.config.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.config.resolution.resolvers import build_default_resolver_chain
from aiperf.plugin.enums import TimingMode

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


def test_no_scenario_returns_noop() -> None:
    run = _build_run(scenario=None)
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is None
    assert run.resolved.scenario_outcome is outcome


def test_clean_weka_public_dataset_through_resolver_chain() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    build_default_resolver_chain().resolve_all(run)

    phase = run.cfg.get_profiling_phases()[0]
    assert phase.timing_mode == TimingMode.AGENTIC_REPLAY
    assert run.cfg.endpoint.streaming is True
    assert run.cfg.get_cache_bust_target() == CacheBustTarget.FIRST_TURN_PREFIX
    assert run.cfg.endpoint.extra["ignore_eos"] is True
    assert run.cfg.get_default_dataset().trace_idle_gap_cap_seconds is None
    assert run.cfg.get_default_dataset().inter_turn_delay_cap_seconds is None
    assert phase.system_idle_gap_cap_seconds == 10.0
    outcome = run.resolved.scenario_outcome
    assert isinstance(outcome, ScenarioOutcome)
    assert outcome.submission_valid is True
    assert "timing_mode" in outcome.applied_locks


def test_timing_mode_stamped_on_profiling_phase() -> None:
    run = _build_run(extra={"ignore_eos": True})
    apply_scenario(run)
    assert run.cfg.get_profiling_phases()[0].timing_mode == TimingMode.AGENTIC_REPLAY


@pytest.mark.parametrize(
    "overrides",
    [
        param({"type": "poisson", "rate": 5.0}, id="request_rate"),
        param({"type": "gamma", "rate": 5.0, "smoothness": 2.0}, id="gamma"),
        param({"type": "user_centric", "rate": 1.0, "users": 4}, id="user_centric"),
    ],
)  # fmt: skip
def test_explicit_scheduling_phase_conflicts_with_scenario(
    overrides: dict[str, Any],
) -> None:
    """An explicitly-scheduled rate/user-centric/fixed-schedule phase is rejected rather than stamped over."""
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        profiling_overrides=overrides,
    )
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert "--request-rate / --user-centric-rate / --fixed-schedule" in str(exc.value)


def test_adaptive_scale_phase_conflicts_with_scenario() -> None:
    """--adaptive-scale flips a per-phase flag the phase-type gate cannot see, and the lock still rejects it."""
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        profiling_overrides={
            "adaptive_scale": True,
            "adaptive_sustain_duration": 120,
            "sla": [
                {
                    "metric_tag": "request_latency",
                    "stat": "p95",
                    "op": "le",
                    "threshold": 30000,
                }
            ],
        },
    )
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert "--adaptive-scale" in str(exc.value)


def test_explicit_scheduling_phase_unsafe_override_keeps_user_mode() -> None:
    """Under --unsafe-override the scheduling conflict downgrades to a warning and the user's mode is kept un-stamped."""
    run = _build_run(
        streaming=True,
        unsafe_override=True,
        extra={"ignore_eos": True},
        profiling_overrides={"type": "poisson", "rate": 5.0},
    )
    outcome = apply_scenario(run)
    assert outcome.submission_valid is False
    assert any(
        v.flag == "--request-rate / --user-centric-rate / --fixed-schedule"
        for v in outcome.violations
    )
    assert run.cfg.get_profiling_phases()[0].timing_mode != TimingMode.AGENTIC_REPLAY


def test_absent_streaming_auto_enabled() -> None:
    run = _build_run(extra={"ignore_eos": True})
    assert run.cfg.endpoint.streaming is False
    apply_scenario(run)
    assert run.cfg.endpoint.streaming is True


def test_explicit_no_streaming_raises() -> None:
    run = _build_run(streaming=False, extra={"ignore_eos": True})
    assert run.cfg.endpoint._streaming_explicitly_set is True
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert "--streaming" in str(exc.value)


def test_streaming_on_no_violation() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is True


def test_absent_ignore_eos_injected() -> None:
    run = _build_run(streaming=True)
    apply_scenario(run)
    assert run.cfg.endpoint.extra["ignore_eos"] is True


def test_explicit_ignore_eos_false_raises() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": False})
    with pytest.raises(ScenarioLockError):
        apply_scenario(run)


def test_ignore_trace_delays_raises() -> None:
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
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert "ignore-trace-delays" in str(exc.value)


def test_ignore_trace_delays_raises_through_resolver_chain() -> None:
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
    with pytest.raises(ScenarioLockError):
        build_default_resolver_chain().resolve_all(run)


def test_wrong_public_loader_raises() -> None:
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset={"name": "main", "type": "public", "dataset": "sharegpt"},
    )
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert "loader" in str(exc.value)


def test_synthetic_default_dataset_raises_clean_loader_lock_not_value_error() -> None:
    """A synthetic default dataset surfaces the --input-file (loader) ScenarioViolation, not a raw ValueError from the require_use_* auto-fill."""
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset={"name": "main", "type": "synthetic"},
    )
    dataset = run.cfg.get_default_dataset()
    assert not hasattr(dataset, "use_think_time_only")

    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    flags = [v.flag for v in exc.value.violations]
    assert "--input-file (loader)" in flags


def test_synthetic_loader_not_bypassable_via_unsafe_override() -> None:
    """--unsafe-override cannot bypass a synthetic loader (unlike an explicit wrong loader such as sharegpt)."""
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset={"name": "main", "type": "synthetic", "entries": 393},
        unsafe_override=True,
        duration=30,
    )
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert exc.value.bypassable is False
    assert "synthetic" in str(exc.value).lower()
    assert "--unsafe-override cannot bypass" in str(exc.value).lower() or (
        "cannot be bypassed with --unsafe-override" in str(exc.value)
    )
    loader_vs = [v for v in exc.value.violations if v.flag == "--input-file (loader)"]
    assert loader_vs
    assert loader_vs[0].current_value == "synthetic"


@pytest.mark.parametrize(
    "loader",
    [
        "semianalysis_cc_traces_weka_061326",
        "semianalysis_cc_traces_weka_061526",
        "semianalysis_cc_traces_weka_with_subagents",
        "semianalysis_cc_traces_weka_with_subagents_256k",
        "semianalysis_cc_traces_weka_with_subagents_060826",
    ],
)  # fmt: skip
def test_allowed_weka_public_loaders(loader: str) -> None:
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset={"name": "main", "type": "public", "dataset": loader},
    )
    outcome = apply_scenario(run)
    assert outcome.violations == []


def test_weka_hf_requires_pinned_repo() -> None:
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset={
            "name": "main",
            "type": "public",
            "dataset": "weka_hf",
            "hf_weka_dataset": "example/not-agentx-corpus",
        },
    )
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert "semianalysisai/cc-traces-weka-062126" in str(exc.value)


def test_weka_hf_with_pinned_repo_ok() -> None:
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset={
            "name": "main",
            "type": "public",
            "dataset": "weka_hf",
            "hf_weka_dataset": "semianalysisai/cc-traces-weka-062126",
        },
    )
    outcome = apply_scenario(run)
    assert outcome.violations == []


def _local_weka_trace_run(*, unsafe_override: bool = False) -> BenchmarkRun:
    """FileDataset whose resolver would detect ``weka_trace`` (local, unpinned)."""
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        unsafe_override=unsafe_override,
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
    run.resolved.dataset_types = {"main": "weka_trace"}
    return run


def test_local_weka_trace_raises_without_unsafe_override() -> None:
    """Local weka_trace is format-ok but unpinned — refuse submission_valid=true."""
    run = _local_weka_trace_run()
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert "cannot verify corpus identity" in str(exc.value)
    assert any(
        v.flag == "--custom-dataset-type / --input-file" for v in exc.value.violations
    )


def test_local_weka_trace_unsafe_override_marks_submission_invalid() -> None:
    """Offline smoke: local weka_trace + --unsafe-override -> submission_valid=false."""
    run = _local_weka_trace_run(unsafe_override=True)
    outcome = apply_scenario(run)
    assert outcome.submission_valid is False
    assert "unsafe_override" in outcome.submission_invalid_reasons
    assert any(
        v.flag == "--custom-dataset-type / --input-file" for v in outcome.violations
    )


def test_cache_bust_auto_filled_when_default() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    assert run.cfg.get_cache_bust_target() == CacheBustTarget.NONE
    apply_scenario(run)
    assert run.cfg.get_cache_bust_target() == CacheBustTarget.FIRST_TURN_PREFIX


def test_explicit_cache_bust_none_raises() -> None:
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset={
            "name": "main",
            "type": "public",
            "dataset": _WEKA_LOADER,
            "cache_bust": {"target": "none"},
        },
    )
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert "cache_bust" in str(exc.value).lower()


def test_explicit_cache_bust_first_turn_prefix_ok() -> None:
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset={
            "name": "main",
            "type": "public",
            "dataset": _WEKA_LOADER,
            "cache_bust": {"target": "first_turn_prefix"},
        },
    )
    outcome = apply_scenario(run)
    assert outcome.violations == []


def test_duration_below_floor_raises() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": True}, duration=300)
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert "--benchmark-duration" in str(exc.value)


def test_duration_at_floor_ok() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": True}, duration=900)
    outcome = apply_scenario(run)
    assert outcome.violations == []


def test_duration_unset_auto_filled_to_scenario_default() -> None:
    # duration must be provided at config-build time (stop condition); use the
    # warmup-style scalar then clear it to simulate "unset" before apply.
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    run.cfg.get_profiling_phases()[0].duration = None
    apply_scenario(run)
    assert run.cfg.get_profiling_phases()[0].duration == 1800.0


def test_trajectory_ratios_auto_filled_when_default() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    phase = run.cfg.get_profiling_phases()[0]
    # Defaults match the scenario; force off-default + unset-flag to exercise fill.
    phase.trajectory_start_min_ratio = 0.2
    phase.trajectory_start_max_ratio = 0.9
    phase._trajectory_start_min_ratio_explicitly_set = False
    phase._trajectory_start_max_ratio_explicitly_set = False
    apply_scenario(run)
    assert phase.trajectory_start_min_ratio == 0.0
    assert phase.trajectory_start_max_ratio == 1.0


def test_trajectory_ratios_explicit_honored() -> None:
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        profiling_overrides={
            "trajectory_start_min_ratio": 0.1,
            "trajectory_start_max_ratio": 0.9,
        },
    )
    phase = run.cfg.get_profiling_phases()[0]
    assert phase._trajectory_start_min_ratio_explicitly_set is True
    apply_scenario(run)
    assert phase.trajectory_start_min_ratio == 0.1
    assert phase.trajectory_start_max_ratio == 0.9


def test_system_idle_gap_cap_auto_filled_without_changing_trace_timing() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    phase = run.cfg.get_profiling_phases()[0]
    assert run.cfg.get_default_dataset().trace_idle_gap_cap_seconds is None
    assert phase.system_idle_gap_cap_seconds is None
    apply_scenario(run)
    assert run.cfg.get_default_dataset().trace_idle_gap_cap_seconds is None
    assert phase.system_idle_gap_cap_seconds == 10.0


def test_system_idle_gap_cap_explicit_other_value_raises() -> None:
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        profiling_overrides={"system_idle_gap_cap_seconds": 30.0},
    )
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert "system-idle-gap-cap-seconds" in str(exc.value)


def test_trace_idle_gap_cap_explicit_value_is_honored() -> None:
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset={
            "name": "main",
            "type": "public",
            "dataset": _WEKA_LOADER,
            "trace_idle_gap_cap_seconds": 30.0,
        },
    )
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert outcome.submission_valid is True
    assert run.cfg.get_default_dataset().trace_idle_gap_cap_seconds == 30.0


def test_inter_turn_delay_cap_shipped_scenario_forbids_value() -> None:
    """AgentX preserves raw trace think times instead of capping each turn."""
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset={
            "name": "main",
            "type": "public",
            "dataset": _WEKA_LOADER,
            "inter_turn_delay_cap_seconds": 30.0,
        },
    )
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert "inter-turn-delay-cap-seconds" in str(exc.value)


def _register_inter_turn_cap_scenario(
    monkeypatch: pytest.MonkeyPatch, cap: float
) -> str:
    """Register a scenario that locks inter_turn_delay_cap_seconds and return its name."""
    from aiperf.common.scenario import registry
    from aiperf.common.scenario.inferencex_agentx_mvp import INFERENCEX_AGENTX_MVP

    name = "test-inter-turn-cap-lock"
    spec = INFERENCEX_AGENTX_MVP.model_copy(
        update={
            "name": name,
            "trace_idle_gap_cap_seconds": None,
            "inter_turn_delay_cap_seconds": cap,
            "forbid_inter_turn_delay_cap": False,
        }
    )
    monkeypatch.setitem(registry.SCENARIOS, name, spec)
    return name


def test_inter_turn_delay_cap_auto_filled_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = _register_inter_turn_cap_scenario(monkeypatch, 45.0)
    run = _build_run(scenario=name, streaming=True, extra={"ignore_eos": True})
    assert run.cfg.get_default_dataset().inter_turn_delay_cap_seconds is None
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert run.cfg.get_default_dataset().inter_turn_delay_cap_seconds == 45.0
    assert "inter_turn_delay_cap" in outcome.applied_locks


def test_inter_turn_delay_cap_explicit_other_value_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = _register_inter_turn_cap_scenario(monkeypatch, 45.0)
    run = _build_run(
        scenario=name,
        streaming=True,
        extra={"ignore_eos": True},
        dataset={
            "name": "main",
            "type": "public",
            "dataset": _WEKA_LOADER,
            "inter_turn_delay_cap_seconds": 30.0,
        },
    )
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert "inter-turn-delay-cap-seconds" in str(exc.value)


def test_inter_turn_delay_cap_explicit_matching_value_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = _register_inter_turn_cap_scenario(monkeypatch, 45.0)
    run = _build_run(
        scenario=name,
        streaming=True,
        extra={"ignore_eos": True},
        dataset={
            "name": "main",
            "type": "public",
            "dataset": _WEKA_LOADER,
            "inter_turn_delay_cap_seconds": 45.0,
        },
    )
    outcome = apply_scenario(run)
    assert outcome.violations == []
    assert "inter_turn_delay_cap" in outcome.applied_locks


def test_unsafe_override_converts_errors_to_warnings() -> None:
    run = _build_run(
        streaming=False,  # would raise --streaming
        unsafe_override=True,
        extra={"ignore_eos": False},  # would raise ignore_eos
        dataset={
            "name": "main",
            "type": "public",
            "dataset": "sharegpt",  # would raise loader
            "cache_bust": {"target": "none"},  # would raise cache_bust
        },
        duration=300,  # would raise duration floor
    )
    outcome = apply_scenario(run)
    assert outcome.submission_valid is False
    assert "unsafe_override" in outcome.submission_invalid_reasons
    assert len(outcome.violations) >= 4


def test_outcome_stored_on_resolved() -> None:
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    outcome = apply_scenario(run)
    assert run.resolved.scenario_outcome is outcome
    assert outcome.scenario_name == "inferencex-agentx-mvp"


def test_random_seed_unset_auto_filled(caplog: pytest.LogCaptureFixture) -> None:
    """A scenario run with no seed gets a fresh per-run random_seed + info log."""
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    assert run.random_seed is None
    with caplog.at_level("INFO"):
        outcome = apply_scenario(run)
    assert run.random_seed is not None
    assert 0 <= run.random_seed < (1 << 63)
    assert outcome.violations == []
    assert "random_seed" in outcome.applied_locks
    assert any("random_seed" in record.message for record in caplog.records)


def test_random_seed_explicit_preserved(caplog: pytest.LogCaptureFixture) -> None:
    """An explicit run seed (including 0) is preserved, not overwritten."""
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    run.random_seed = 0
    with caplog.at_level("INFO"):
        apply_scenario(run)
    assert run.random_seed == 0
    assert not any(
        "auto-set random_seed" in record.message for record in caplog.records
    )


def test_random_seed_auto_filled_through_resolver_chain() -> None:
    """The seed is stamped when running the full resolver chain, not just the bare apply_scenario call."""
    run = _build_run(streaming=True, extra={"ignore_eos": True})
    assert run.random_seed is None
    build_default_resolver_chain().resolve_all(run)
    assert run.random_seed is not None
