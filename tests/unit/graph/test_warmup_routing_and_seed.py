# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warmup routing + seed-determinism + auto-detect precedence (Task F2 / M-seed / F3).

Covers the ``timing.config`` deliverables:

* a graph workload's WARMUP phase routes through ``TimingMode.GRAPH_IR`` (not the
  linear ``REQUEST_RATE`` path);
* ``resolve_graph_content_seed`` is the run ``--random-seed`` verbatim (None
  when unset -- no weka-specific fallback); threading the same explicit seed
  into two independent full parses of the same run synthesizes byte-identical
  weka content (the determinism the in-process build parse and any
  spawn-started pool worker rely on), and distinct seeds diverge;
* ``from_run`` honors an explicit, graph-incompatible ``--custom-dataset-type``
  over the file-sniff auto-detection (precedence), while a bare weka file with
  no pinned format still auto-routes to GRAPH_IR.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase, DatasetFormat
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.models import ParsedGraph
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.config import TimingConfig, resolve_graph_content_seed
from tests.unit.conftest import make_run_from_cli

WEKA_MIN = Path(__file__).parent / "fixtures" / "weka_min.json"


def _graph_run(**cli_overrides):
    """Build a ``BenchmarkRun`` pointing at the weka fixture, with warmup set."""
    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(WEKA_MIN),
        warmup_request_count=2,
        request_count=3,
        **cli_overrides,
    )
    return make_run_from_cli(cfg)


def _pool_contents(parsed: ParsedGraph) -> dict[str, tuple[str, str, str | None]]:
    """Materialized segment content keyed by content-addressed segment id."""
    pool = parsed.segment_pool
    assert pool is not None, "weka parse must surface the segment pool"
    return {sid: (s.role, s.content, s.parent_id) for sid, s in pool._by_id.items()}


def test_graph_warmup_phase_routes_to_graph_ir() -> None:
    run = _graph_run()
    tc = TimingConfig.from_run(run)
    warmup = [p for p in tc.phase_configs if p.phase == CreditPhase.WARMUP]
    profiling = [p for p in tc.phase_configs if p.phase == CreditPhase.PROFILING]
    assert warmup, "expected a warmup phase"
    assert all(p.timing_mode == TimingMode.GRAPH_IR for p in warmup), (
        "graph warmup must use GRAPH_IR, not the linear REQUEST_RATE strategy"
    )
    assert all(p.timing_mode == TimingMode.GRAPH_IR for p in profiling)


def test_agentx_window_injects_boundary_auto_warmup(monkeypatch) -> None:
    """An active AgentX t* window (no explicit warmup flags) gets the auto-warmup phase.

    AgentX parity: with the t* window active (trajectory_start_max_ratio > 0)
    the graph run ALWAYS primes each live chain's boundary turn before
    profiling. The injected phase must carry NO stop-condition counts -- the
    GraphIRReplayStrategy owns warmup completion (its ``rewrite_for_warmup``
    boundary graph fires exactly the live boundary turns, then drains).
    """
    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(WEKA_MIN),
        request_count=3,
        trajectory_start_min_ratio=0.25,
        trajectory_start_max_ratio=0.75,
    )
    run = make_run_from_cli(cfg)
    tc = TimingConfig.from_run(run)
    warmup = [p for p in tc.phase_configs if p.phase == CreditPhase.WARMUP]
    assert len(warmup) == 1, "expected exactly one auto-injected warmup phase"
    auto = warmup[0]
    assert auto.timing_mode == TimingMode.GRAPH_IR
    assert auto.total_expected_requests is None
    assert auto.expected_num_sessions is None
    assert auto.expected_duration_sec is None
    assert auto.grace_period_sec == float("inf")


def test_default_graph_run_injects_no_auto_warmup(monkeypatch) -> None:
    """Bare default = full replay: no t* window, no boundary auto-warmup.

    Pin the pressure duration to None explicitly: ``CLIConfig`` dual-writes it
    globally, so xdist ordering can otherwise leak a duration in and inject a
    (pressure-mode) warmup phase even with the t* window off.
    """
    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(WEKA_MIN),
        request_count=3,
    )
    run = make_run_from_cli(cfg)
    tc = TimingConfig.from_run(run)
    warmup = [p for p in tc.phase_configs if p.phase == CreditPhase.WARMUP]
    assert warmup == []


def test_pressure_duration_injects_warmup_even_at_tstar_zero(monkeypatch) -> None:
    """A pressure duration forces the auto-warmup phase even with t* inactive.

    The extended (cache-pressure) warmup runs inside the WARMUP phase, so a
    configured ``--agentic-cache-warmup-duration`` must inject the phase even
    when the t* window is closed (max ratio 0.0, the bare default => empty
    boundary priming). Mirrors the pressure gating in ``timing/config.py``.
    """
    cfg = CLIConfig(
        agentic_cache_warmup_duration=30.0,
        model_names=["test-model"],
        input_file=str(WEKA_MIN),
        request_count=3,
    )
    run = make_run_from_cli(cfg)
    tc = TimingConfig.from_run(run)
    warmup = [p for p in tc.phase_configs if p.phase == CreditPhase.WARMUP]
    assert len(warmup) == 1, "pressure duration must inject a warmup phase at t*=0"
    assert warmup[0].timing_mode == TimingMode.GRAPH_IR


def test_non_graph_warmup_phase_stays_request_rate() -> None:
    cfg = CLIConfig(
        model_names=["test-model"],
        warmup_request_count=2,
        request_count=3,
    )
    run = make_run_from_cli(cfg)
    tc = TimingConfig.from_run(run)
    warmup = [p for p in tc.phase_configs if p.phase == CreditPhase.WARMUP]
    assert warmup
    assert all(p.timing_mode == TimingMode.REQUEST_RATE for p in warmup)


def test_resolve_graph_content_seed_is_the_run_seed() -> None:
    run = _graph_run()
    # The content seed IS the AIPerf run seed -- no weka-specific fallback. A
    # default single run leaves --random-seed unset, so the content seed is None
    # (ambient global RNG), stable across calls for the same run.
    assert run.random_seed is None
    assert resolve_graph_content_seed(run) is None
    assert resolve_graph_content_seed(run) == resolve_graph_content_seed(run)


def test_resolve_graph_content_seed_honors_explicit_run_seed() -> None:
    run = _graph_run(random_seed=7)
    assert resolve_graph_content_seed(run) == 7


def test_same_seed_two_parses_synthesize_byte_identical_content() -> None:
    """Two independent full parses under the SAME explicit seed match byte-for-byte.

    Mirrors the content-determinism contract: the DatasetManager's in-process
    build parse and any spawn-started pool worker parse the weka file
    separately; threading the SAME seed makes the synthesized segment content
    byte-identical. Compares the segment pool's real
    materialized ``(role, content, parent_id)`` image -- NOT
    ``TraceRecord.replay_outputs``, which is always empty on the weka path.
    """
    parse_a = from_weka_trace(str(WEKA_MIN), content_root_seed=7)
    parse_b = from_weka_trace(str(WEKA_MIN), content_root_seed=7)
    assert parse_a.segment_pool is not None and parse_a.segment_pool._by_id
    assert _pool_contents(parse_a) == _pool_contents(parse_b)


def test_different_seed_parses_synthesize_different_content() -> None:
    """Distinct seeds synthesize distinct bytes (byte-identity is falsifiable)."""
    parse_a = from_weka_trace(str(WEKA_MIN), content_root_seed=7)
    parse_b = from_weka_trace(str(WEKA_MIN), content_root_seed=8)
    assert _pool_contents(parse_a) != _pool_contents(parse_b)


def test_explicit_incompatible_format_overrides_weka_sniff() -> None:
    """An explicit, graph-incompatible --custom-dataset-type beats the sniff.

    A user who pins ``--custom-dataset-type multi_turn`` on a file that happens
    to sniff as weka must NOT be silently rerouted to the graph pipeline.
    """
    run = _graph_run(custom_dataset_type=DatasetFormat.MULTI_TURN)
    tc = TimingConfig.from_run(run)
    profiling = [p for p in tc.phase_configs if p.phase == CreditPhase.PROFILING]
    assert profiling
    assert all(p.timing_mode != TimingMode.GRAPH_IR for p in profiling), (
        "an explicit incompatible --custom-dataset-type must suppress graph routing"
    )


def test_pressure_warmup_grace_is_min_of_duration_and_cap(monkeypatch) -> None:
    """Pressure warmup drains at min(duration, cap) -- agentx MAX_WARMUP_GRACE parity.

    A pressure-mode warmup pins ``expected_duration_sec=None`` (the drain is
    bounded by the finite grace, not a phase duration) and, absent an explicit
    ``--warmup-grace-period``, drains for ``min(pressure duration, cap)`` where
    ``cap`` is ``PRESSURE_DRAIN_GRACE_CAP`` (default 300).
    """
    # duration 30 < cap 300 -> grace 30
    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(WEKA_MIN),
        request_count=3,
        agentic_cache_warmup_duration=30.0,
    )
    warmup = next(
        p
        for p in TimingConfig.from_run(make_run_from_cli(cfg)).phase_configs
        if p.phase == CreditPhase.WARMUP
    )
    assert warmup.grace_period_sec == 30.0
    assert warmup.expected_duration_sec is None

    # duration 900 > cap 300 -> grace 300 (clamped to the cap)
    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(WEKA_MIN),
        request_count=3,
        agentic_cache_warmup_duration=900.0,
    )
    warmup = next(
        p
        for p in TimingConfig.from_run(make_run_from_cli(cfg)).phase_configs
        if p.phase == CreditPhase.WARMUP
    )
    assert warmup.grace_period_sec == 300.0


def test_no_pressure_auto_warmup_keeps_infinite_grace() -> None:
    """Without a pressure duration, the boundary-priming warmup still waits forever."""
    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(WEKA_MIN),
        request_count=3,
        trajectory_start_min_ratio=0.25,
        trajectory_start_max_ratio=0.75,
    )
    warmup = next(
        p
        for p in TimingConfig.from_run(make_run_from_cli(cfg)).phase_configs
        if p.phase == CreditPhase.WARMUP
    )
    assert warmup.grace_period_sec == float("inf")


def test_pressure_supersedes_user_warmup_phase(monkeypatch) -> None:
    """Graph + pressure: the warmup phase is MODE-OWNED (agentx parity).

    A user ``--warmup-request-count`` / warmup phase config is superseded by the
    auto boundary-priming + pressure shape; agentx pins expected_duration_sec=None
    unconditionally for its agentic warmup.
    """
    cfg = CLIConfig(
        agentic_cache_warmup_duration=30.0,
        model_names=["test-model"],
        input_file=str(WEKA_MIN),
        warmup_request_count=2,
        request_count=3,
    )
    warmups = [
        p
        for p in TimingConfig.from_run(make_run_from_cli(cfg)).phase_configs
        if p.phase == CreditPhase.WARMUP
    ]
    assert len(warmups) == 1
    assert warmups[0].expected_duration_sec is None
    assert warmups[0].total_expected_requests is None
    assert warmups[0].arrival_pattern == ArrivalPattern.CONCURRENCY_BURST


def test_pressure_supersede_carries_user_grace_verbatim(monkeypatch) -> None:
    """An explicit user warmup grace survives the supersede (agentx :227-229).

    The auto shape still owns duration (=None) and the burst pattern, but the
    operator's explicit grace is honored verbatim -- it is NOT re-derived as
    ``min(duration, cap)``.
    """
    cfg = CLIConfig(
        agentic_cache_warmup_duration=30.0,
        model_names=["test-model"],
        input_file=str(WEKA_MIN),
        warmup_duration=10.0,
        warmup_grace_period=45.0,
        request_count=3,
    )
    warmup = next(
        p
        for p in TimingConfig.from_run(make_run_from_cli(cfg)).phase_configs
        if p.phase == CreditPhase.WARMUP
    )
    assert warmup.grace_period_sec == 45.0  # NOT min(30, 300)
    assert warmup.expected_duration_sec is None  # still mode-owned (superseded)


def test_user_warmup_untouched_without_pressure() -> None:
    """No pressure duration: explicit user warmup phases build exactly as today."""
    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(WEKA_MIN),
        warmup_duration=10.0,
        warmup_grace_period=45.0,
        request_count=3,
    )
    warmup = next(
        p
        for p in TimingConfig.from_run(make_run_from_cli(cfg)).phase_configs
        if p.phase == CreditPhase.WARMUP
    )
    # Built verbatim from the user phase: duration and explicit grace preserved,
    # NOT nulled/superseded by the auto pressure shape.
    assert warmup.expected_duration_sec == 10.0
    assert warmup.grace_period_sec == 45.0


def test_adaptive_scale_on_graph_workload_raises() -> None:
    """An explicit adaptive_scale phase on a graph workload fails loudly.

    Adaptive scaling and the recorded graph replay both want to own pacing and
    concurrency; silently routing the phase to GRAPH_IR would discard the
    user's explicit adaptive_scale choice, so ``from_run`` must reject the
    combination up front.
    """
    run = _graph_run(
        adaptive_scale=True,
        concurrency=4,
        benchmark_duration=10.0,
        adaptive_sustain_duration=5.0,
        adaptive_scale_sla=["request_latency:p95:le:30000"],
    )
    with pytest.raises(
        ValueError, match="adaptive_scale is not supported for graph workloads"
    ):
        TimingConfig.from_run(run)


@pytest.mark.parametrize(
    "cli_overrides",
    [
        param({"request_rate": 5.0}, id="request_rate_poisson"),
        param(
            {"request_rate": 5.0, "arrival_pattern": ArrivalPattern.CONSTANT},
            id="request_rate_constant",
        ),
        param({"fixed_schedule": True}, id="fixed_schedule"),
        param({"warmup_request_rate": 2.0}, id="warmup_request_rate"),
    ],
)  # fmt: skip
def test_rate_or_schedule_phase_on_graph_workload_raises(cli_overrides) -> None:
    """A rate-controlled or fixed-schedule phase on a graph workload fails loudly.

    GRAPH_IR owns pacing (decision D2): forcing the graph strategy would
    silently discard the user's explicit --request-rate / --user-centric-rate /
    --fixed-schedule arrival timing, so ``from_run`` must reject the
    combination up front -- warmup phases included (--warmup-request-rate is
    equally discarded by the graph warmup). USER_CENTRIC is covered by the
    same ``type != CONCURRENCY`` check but is unreachable from the CLI with a
    file dataset (the converter rejects it earlier on the turn-mean floor),
    so it has no param here.
    """
    run = _graph_run(**cli_overrides)
    with pytest.raises(ValueError, match="is not supported for graph workloads"):
        TimingConfig.from_run(run)
