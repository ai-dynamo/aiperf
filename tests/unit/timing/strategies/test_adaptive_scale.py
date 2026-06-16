# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.enums import CreditPhase
from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies.adaptive_scale import (
    AdaptiveScaleStrategy,
    _percentile,
)


def _strategy(tmp_path, *, threshold: float = 100.0) -> AdaptiveScaleStrategy:
    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.ADAPTIVE_SCALE,
        expected_duration_sec=60.0,
        concurrency=10,
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        adaptive_sustain_duration_sec=10.0,
        adaptive_assessment_period_sec=1.0,
        adaptive_scale_min_concurrency=2,
        adaptive_sla_filters=[
            SLAFilter(
                metric_tag="request_latency",
                stat="p95",
                op="le",
                threshold=threshold,
            )
        ],
        artifact_dir=tmp_path,
    )
    return AdaptiveScaleStrategy(
        config=cfg,
        conversation_source=MagicMock(),
        scheduler=MagicMock(),
        stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
        credit_issuer=MagicMock(),
        lifecycle=MagicMock(),
        concurrency_manager=MagicMock(),
        progress=MagicMock(),
    )


def test_percentile_interpolates_p95() -> None:
    assert _percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48.0)


@pytest.mark.asyncio
async def test_handle_credit_result_buffers_latency(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    credit = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="c",
        x_correlation_id="x",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns() - 5_000_000,
    )

    await strategy.handle_credit_result(CreditReturn(credit=credit))
    stats = await strategy._take_window()

    assert len(stats.samples) == 1
    assert stats.samples[0] >= 0
    assert stats.errors == 0


@pytest.mark.asyncio
async def test_handle_credit_result_counts_cancelled_as_error(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    credit = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="c",
        x_correlation_id="x",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns() - 5_000_000,
    )

    await strategy.handle_credit_result(CreditReturn(credit=credit, cancelled=True))
    stats = await strategy._take_window()

    assert stats.samples == []
    assert stats.errors == 1


@pytest.mark.asyncio
async def test_handle_credit_return_does_not_record_success_sample(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    credit = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="c",
        x_correlation_id="x",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns() - 5_000_000,
    )

    await strategy.handle_credit_return(credit)
    stats = await strategy._take_window()

    assert stats.samples == []
    assert stats.errors == 0


def test_unsupported_sla_metric_fails_during_strategy_construction(tmp_path) -> None:
    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.ADAPTIVE_SCALE,
        expected_duration_sec=60.0,
        concurrency=10,
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        adaptive_sustain_duration_sec=10.0,
        adaptive_assessment_period_sec=1.0,
        adaptive_scale_min_concurrency=2,
        adaptive_sla_filters=[
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="le",
                threshold=100.0,
            )
        ],
        artifact_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="supports request_latency"):
        AdaptiveScaleStrategy(
            config=cfg,
            conversation_source=MagicMock(),
            scheduler=MagicMock(),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
            credit_issuer=MagicMock(),
            lifecycle=MagicMock(),
            concurrency_manager=MagicMock(),
            progress=MagicMock(),
        )


def test_discover_scales_up_and_writes_event(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    stats = MagicMock(samples=[10_000_000], errors=0, throughput=3.0)

    strategy._assess_discover(10.0, True, stats)

    strategy._concurrency_manager.set_session_limit.assert_called_with(
        CreditPhase.PROFILING, 10
    )
    events = [
        orjson.loads(line)
        for line in (tmp_path / "adaptive_scale_events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_decision"
    assert events[-1]["concurrency_before"] == 2
    assert events[-1]["concurrency_after"] == 10
    assert events[-1]["step_policy"] == "sla_margin"
    assert events[-1]["step_size"] == 8
    assert "timestamp" in events[-1]
    assert events[-1]["sla_value"] == 10.0
    assert events[-1]["sla_bound"] == 100.0


def test_percent_step_policy_uses_current_concurrency_percent(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    strategy._config = strategy._config.model_copy(
        update={
            "adaptive_scale_step_policy": "fixed_percent_step",
            "adaptive_scale_step_percent": 50.0,
        }
    )
    stats = MagicMock(samples=[10_000_000], errors=0, throughput=3.0)

    strategy._assess_discover(10.0, True, stats)

    strategy._concurrency_manager.set_session_limit.assert_called_with(
        CreditPhase.PROFILING, 3
    )


def test_margin_step_is_capped_by_max_multiplier(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    strategy._config = strategy._config.model_copy(
        update={
            "adaptive_scale_base_step": 10,
            "adaptive_scale_max_step_multiplier": 4,
        }
    )

    assert strategy._step_size(100, 0.0) == 40
    assert strategy._step_size(100, 90.0) == 10


def test_sla_margin_step_supports_higher_is_better_metrics(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    throughput_sla = SLAFilter(
        metric_tag="request_throughput",
        stat="avg",
        op="ge",
        threshold=1000.0,
    )
    strategy._sla_filters = [throughput_sla]
    strategy._primary_sla = throughput_sla
    strategy._config = strategy._config.model_copy(
        update={
            "adaptive_scale_base_step": 10,
            "adaptive_scale_max_step_multiplier": 4,
        }
    )

    assert strategy._step_size(100, {strategy._sla_key(throughput_sla): 2000.0}) == 40
    assert strategy._step_size(100, {strategy._sla_key(throughput_sla): 1100.0}) == 10


def test_sla_margin_uses_most_constrained_filter(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    latency_sla = strategy._primary_sla
    throughput_sla = SLAFilter(
        metric_tag="request_throughput",
        stat="avg",
        op="ge",
        threshold=1000.0,
    )
    strategy._sla_filters = [latency_sla, throughput_sla]
    strategy._config = strategy._config.model_copy(
        update={
            "adaptive_scale_base_step": 10,
            "adaptive_scale_max_step_multiplier": 4,
        }
    )

    observed = {
        strategy._sla_key(latency_sla): 10.0,
        strategy._sla_key(throughput_sla): 1100.0,
    }
    assert strategy._step_size(100, observed) == 10


def test_goodput_ratio_uses_successes_over_attempts(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    stats = MagicMock(samples=[10_000_000, 20_000_000], errors=1, total=3)
    sla = SLAFilter(
        metric_tag="goodput_ratio",
        stat="avg",
        op="ge",
        threshold=0.95,
    )

    assert strategy._sla_value(sla, stats) == pytest.approx(2 / 3)


def test_goodput_ratio_sla_participates_in_pass_fail(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    latency_sla = strategy._primary_sla
    goodput_sla = SLAFilter(
        metric_tag="goodput_ratio",
        stat="avg",
        op="ge",
        threshold=0.95,
    )
    strategy._sla_filters = [latency_sla, goodput_sla]
    observed = {
        strategy._sla_key(latency_sla): 10.0,
        strategy._sla_key(goodput_sla): 0.90,
    }

    assert strategy._passes_sla(observed) is False


def test_breach_enters_sustain_at_last_good_boundary(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    strategy._last_good_concurrency = 4
    strategy._current_concurrency = 5
    stats = MagicMock(samples=[150_000_000], errors=0, throughput=1.0)

    strategy._assess_discover(150.0, False, stats)

    assert strategy._controller_phase == "sustain"
    assert strategy._boundary_concurrency == 4
    strategy._concurrency_manager.set_session_limit.assert_called_with(
        CreditPhase.PROFILING, 4
    )
    events = [
        orjson.loads(line)
        for line in (tmp_path / "adaptive_scale_events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "sustain_started",
        "boundary_discovered",
    ]
    assert events[-1]["phase"] == "sustain"
    assert events[-1]["concurrency_after"] == 4


def test_sustain_completion_writes_complete_event_and_summary(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    strategy._controller_phase = "sustain"
    strategy._boundary_concurrency = 4
    strategy._last_good_concurrency = 4
    strategy._current_concurrency = 4
    strategy._sustain_started_at = time.perf_counter() - 20.0
    strategy._sustain_started_at_ns = 123
    stats = MagicMock(samples=[50_000_000], errors=0, throughput=2.0)

    strategy._assess_sustain(50.0, True, stats)

    events = [
        orjson.loads(line)
        for line in (tmp_path / "adaptive_scale_events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_complete"
    summary = orjson.loads((tmp_path / "adaptive_scale_summary.json").read_bytes())
    assert summary["boundary_concurrency"] == 4
    assert summary["last_good_concurrency"] == 4
    assert summary["sustain_started_at"] == 123
    assert summary["completed_reason"] == "sustain_duration_completed"
    assert summary["sla_passed_during_sustain"] is True


def test_execute_finalizer_writes_summary_when_phase_stops_before_boundary(
    tmp_path,
) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)

    strategy._complete_controller(reason="phase_stopped")

    events = [
        orjson.loads(line)
        for line in (tmp_path / "adaptive_scale_events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_complete"
    assert events[-1]["reason"] == "phase_stopped"
    summary = orjson.loads((tmp_path / "adaptive_scale_summary.json").read_bytes())
    assert summary["boundary_concurrency"] is None
    assert summary["completed_reason"] == "phase_stopped"


def test_all_failed_discover_window_enters_sustain(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    strategy._last_good_concurrency = 4
    strategy._current_concurrency = 6
    stats = MagicMock(samples=[], errors=3, throughput=0.0)

    strategy._assess_failed_window(stats)

    assert strategy._controller_phase == "sustain"
    assert strategy._boundary_concurrency == 4
    strategy._concurrency_manager.set_session_limit.assert_called_with(
        CreditPhase.PROFILING, 4
    )
    events = [
        orjson.loads(line)
        for line in (tmp_path / "adaptive_scale_events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "sustain_started",
        "boundary_discovered",
    ]
    assert events[-1]["reason"] == "all requests failed in assessment window"
    assert events[-1]["error_count"] == 3


def test_minimum_breach_fails_without_sustainable_concurrency(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    stats = MagicMock(samples=[150_000_000], errors=0, throughput=1.0)

    strategy._assess_discover(150.0, False, stats)

    events = [
        orjson.loads(line)
        for line in (tmp_path / "adaptive_scale_events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_failed"
    assert events[-1]["reason"] == "no_sustainable_concurrency_found"
    assert events[-1]["first_failing_value"] == 2


def test_max_concurrency_passing_is_incomplete_not_boundary(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    strategy._current_concurrency = 10
    stats = MagicMock(samples=[50_000_000], errors=0, throughput=1.0)

    strategy._assess_discover(50.0, True, stats)

    events = [
        orjson.loads(line)
        for line in (tmp_path / "adaptive_scale_events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_incomplete"
    assert events[-1]["reason"] == "max_concurrency_reached_without_saturation"
    assert events[-1]["last_passing_value"] == 10


@pytest.mark.asyncio
async def test_sparse_window_is_inconclusive(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    strategy._min_completed_requests = 2
    strategy._window_latency_ns = [10_000_000]

    await strategy._assess_window()

    events = [
        orjson.loads(line)
        for line in (tmp_path / "adaptive_scale_events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_window"
    assert events[-1]["sla_passed"] is None
    assert "inconclusive" in events[-1]["reason"]


def test_assessment_period_has_practical_lower_bound(tmp_path) -> None:
    with pytest.raises(ValueError, match="adaptive_assessment_period_sec"):
        CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.ADAPTIVE_SCALE,
            expected_duration_sec=60.0,
            concurrency=10,
            arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
            adaptive_sustain_duration_sec=10.0,
            adaptive_assessment_period_sec=0.1,
            adaptive_scale_min_concurrency=2,
            adaptive_sla_filters=[
                SLAFilter(
                    metric_tag="request_latency",
                    stat="p95",
                    op="le",
                    threshold=100.0,
                )
            ],
            artifact_dir=tmp_path,
        )
