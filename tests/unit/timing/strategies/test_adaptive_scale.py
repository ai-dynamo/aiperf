# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import re
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
from aiperf.timing.strategies.adaptive_scale_artifacts import (
    AdaptiveScaleArtifactWriter,
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
    lifecycle = MagicMock()
    lifecycle.is_sending_complete = False
    progress = MagicMock()
    progress.all_credits_sent_event = asyncio.Event()
    return AdaptiveScaleStrategy(
        config=cfg,
        conversation_source=MagicMock(),
        scheduler=MagicMock(),
        stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
        credit_issuer=MagicMock(),
        lifecycle=lifecycle,
        concurrency_manager=MagicMock(),
        progress=progress,
    )


def _assert_event_clock_fields(event: dict) -> None:
    assert event["schema_version"] == 1
    assert isinstance(event["timestamp"], int)
    assert event["timestamp_ns"] == event["timestamp"]
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
        event["timestamp_utc"],
    )


def test_percentile_interpolates_p95() -> None:
    assert _percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48.0)


@pytest.mark.asyncio
async def test_handle_credit_result_buffers_record_request_latency(tmp_path) -> None:
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

    await strategy.handle_credit_result(
        CreditReturn(credit=credit, request_latency_ns=123_000_000)
    )
    stats = await strategy._take_window()

    assert stats.samples == [123_000_000]
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
async def test_handle_credit_result_counts_missing_request_latency_as_error(
    tmp_path,
) -> None:
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

    assert stats.samples == []
    assert stats.errors == 1


@pytest.mark.asyncio
async def test_inherited_handle_credit_return_does_not_record_success_sample(
    tmp_path,
) -> None:
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
    _assert_event_clock_fields(events[-1])
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
    assert summary["schema_version"] == 1
    assert summary["status"] == "completed"
    assert summary["boundary_concurrency"] == 4
    assert summary["last_good_concurrency"] == 4
    assert summary["sustain_started_at"] == 123
    assert summary["completed_reason"] == "sustain_duration_completed"
    assert summary["sla"] == {
        "metric": "request_latency",
        "stat": "p95",
        "op": "le",
        "bound": 100.0,
    }
    assert summary["result"] == {
        "last_passing_value": 4,
        "first_failing_value": None,
        "boundary_value": 4,
    }
    assert summary["totals"] == {
        "sent": 1,
        "completed": 1,
        "errored": 0,
        "cancelled": None,
    }
    assert summary["throughput"] == 2.0
    assert summary["sla_passed_during_sustain"] is True
    strategy._lifecycle.cancel.assert_not_called()
    strategy._lifecycle.mark_sending_complete.assert_called_once_with(
        timeout_triggered=False
    )
    strategy._progress.freeze_sent_counts.assert_called_once()
    assert strategy._progress.all_credits_sent_event.is_set()


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
    assert summary["status"] == "completed"
    assert summary["boundary_concurrency"] is None
    assert summary["result"]["boundary_value"] is None
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
    summary = orjson.loads((tmp_path / "adaptive_scale_summary.json").read_bytes())
    assert summary["status"] == "completed"
    assert summary["completed_reason"] == "no_sustainable_concurrency_found"
    assert summary["result"] == {
        "last_passing_value": None,
        "first_failing_value": 2,
        "boundary_value": None,
    }


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
    assert events[-1]["adaptive_iteration"] == 0
    assert events[-1]["sla_passed"] is None
    assert "inconclusive" in events[-1]["reason"]
    assert strategy._adaptive_iteration == 1


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


def test_window_stats_total_and_zero_elapsed_throughput() -> None:
    from aiperf.timing.strategies.adaptive_scale import WindowStats

    stats = WindowStats(samples=[1, 2], errors=3, elapsed_sec=0.0)

    assert stats.total == 5
    assert stats.throughput == 0.0


@pytest.mark.parametrize(
    ("stat", "expected"),
    [
        ("avg", 20.0),
        ("min", 10.0),
        ("max", 30.0),
        ("p50", 20.0),
    ],
)
def test_request_latency_value_stats(stat: str, expected: float) -> None:
    samples_ns = [10_000_000, 20_000_000, 30_000_000]

    assert AdaptiveScaleStrategy._request_latency_value(samples_ns, stat) == expected


@pytest.mark.parametrize(
    ("sla", "match"),
    [
        (
            SLAFilter.model_construct(
                metric_tag="request_latency", stat="median", op="le", threshold=1.0
            ),
            "Unsupported request_latency",
        ),
        (
            SLAFilter.model_construct(
                metric_tag="throughput", stat="p95", op="ge", threshold=1.0
            ),
            "Unsupported throughput",
        ),
        (
            SLAFilter.model_construct(
                metric_tag="goodput_ratio", stat="p95", op="ge", threshold=1.0
            ),
            "Unsupported goodput_ratio",
        ),
        (
            SLAFilter.model_construct(
                metric_tag="request_latency", stat="avg", op="eq", threshold=1.0
            ),
            "Unsupported SLA operator",
        ),
    ],
)
def test_invalid_sla_filters_raise_clear_errors(sla: SLAFilter, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AdaptiveScaleStrategy._validate_single_sla_filter(sla)


def test_value_helpers_reject_invalid_inputs(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    throughput_sla = SLAFilter.model_construct(
        metric_tag="throughput", stat="p95", op="ge", threshold=1.0
    )
    unknown_sla = SLAFilter.model_construct(
        metric_tag="time_to_first_token", stat="avg", op="le", threshold=1.0
    )

    with pytest.raises(ValueError, match="completed request samples"):
        AdaptiveScaleStrategy._request_latency_value([], "avg")
    with pytest.raises(ValueError, match="Unsupported request_latency"):
        AdaptiveScaleStrategy._request_latency_value([1], "median")
    with pytest.raises(ValueError, match="Unsupported throughput"):
        strategy._sla_value(throughput_sla, MagicMock(throughput=1.0))
    with pytest.raises(ValueError, match="supports request_latency"):
        strategy._sla_value(unknown_sla, MagicMock())


@pytest.mark.asyncio
async def test_setup_phase_sets_initial_concurrency_and_event(tmp_path) -> None:
    strategy = _strategy(tmp_path)

    await strategy.setup_phase()

    strategy._concurrency_manager.set_session_limit.assert_called_with(
        CreditPhase.PROFILING, 2
    )
    events = [
        orjson.loads(line)
        for line in (tmp_path / "adaptive_scale_events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_phase_started"
    assert events[-1]["active_concurrency"] == 2


@pytest.mark.asyncio
async def test_assessment_loop_failure_completes_and_cancels(
    tmp_path, monkeypatch
) -> None:
    strategy = _strategy(tmp_path)
    strategy._assessment_period = 0

    async def fail_window() -> None:
        raise ValueError("bad window")

    monkeypatch.setattr(strategy, "_assess_window", fail_window)

    await strategy._assessment_loop()

    assert strategy._completed_reason == "assessment_failed: bad window"
    strategy._lifecycle.cancel.assert_called_once()
    events = [
        orjson.loads(line)
        for line in (tmp_path / "adaptive_scale_events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_failed"
    summary = orjson.loads((tmp_path / "adaptive_scale_summary.json").read_bytes())
    assert summary["status"] == "failed"
    assert summary["completed_reason"] == "assessment_failed: bad window"


@pytest.mark.asyncio
async def test_assess_window_evaluates_sustain_phase(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    strategy._controller_phase = "sustain"
    strategy._last_good_concurrency = 4
    strategy._current_concurrency = 4
    strategy._window_latency_ns = [10_000_000, 20_000_000]

    await strategy._assess_window()

    events = [
        orjson.loads(line)
        for line in (tmp_path / "adaptive_scale_events.jsonl").read_text().splitlines()
    ]
    assert {event["adaptive_iteration"] for event in events} == {0}
    assert strategy._adaptive_iteration == 1
    assert strategy._sustain_windows == 1
    assert strategy._sustain_passed_windows == 1


@pytest.mark.asyncio
async def test_assess_window_all_failed_without_boundary_fails(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    strategy._window_errors = 2

    await strategy._assess_window()

    assert strategy._completed_reason == "no_sustainable_concurrency_found"
    strategy._lifecycle.cancel.assert_not_called()
    strategy._lifecycle.mark_sending_complete.assert_called_once_with(
        timeout_triggered=False
    )
    assert strategy._progress.all_credits_sent_event.is_set()
    events = [
        orjson.loads(line)
        for line in (tmp_path / "adaptive_scale_events.jsonl").read_text().splitlines()
    ]
    assert events[-2]["reason"] == "all requests failed in assessment window"
    assert events[-1]["event"] == "adaptive_failed"


def test_all_failed_sustain_window_downshifts_with_reason(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    strategy._controller_phase = "sustain"
    strategy._last_good_concurrency = 4
    strategy._current_concurrency = 6
    stats = MagicMock(samples=[], errors=3, throughput=0.0)

    strategy._assess_failed_window(stats)

    strategy._concurrency_manager.set_session_limit.assert_called_with(
        CreditPhase.PROFILING, 4
    )
    events = [
        orjson.loads(line)
        for line in (tmp_path / "adaptive_scale_events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["reason"] == "all requests failed in assessment window"
    assert events[-1]["step_size"] == 2


def test_sustain_breach_at_minimum_fails_unrecoverably(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    strategy._controller_phase = "sustain"
    strategy._current_concurrency = 2
    strategy._last_good_concurrency = None
    stats = MagicMock(samples=[150_000_000], errors=0, throughput=1.0)

    strategy._assess_sustain(150.0, False, stats)

    assert strategy._completed_reason == "sustain_failed_sla_unrecoverable"
    strategy._lifecycle.cancel.assert_not_called()
    strategy._lifecycle.mark_sending_complete.assert_called_once_with(
        timeout_triggered=False
    )
    assert strategy._progress.all_credits_sent_event.is_set()


def test_sustain_breach_downshift_does_not_promote_unconfirmed_target(
    tmp_path,
) -> None:
    strategy = _strategy(tmp_path)
    strategy._controller_phase = "sustain"
    strategy._current_concurrency = 6
    strategy._last_good_concurrency = 8
    stats = MagicMock(samples=[150_000_000], errors=0, throughput=1.0)

    strategy._assess_sustain(150.0, False, stats)

    assert strategy._current_concurrency < 6
    assert strategy._last_good_concurrency == 8


def test_enter_sustain_requires_last_good_boundary(tmp_path) -> None:
    strategy = _strategy(tmp_path)

    with pytest.raises(RuntimeError, match="passing boundary"):
        strategy._enter_sustain(
            None, MagicMock(samples=[], errors=0, throughput=0.0), "x"
        )


@pytest.mark.parametrize(
    ("op", "observed", "expected"),
    [
        ("lt", 9.0, True),
        ("gt", 11.0, True),
        ("ge", 10.0, True),
        ("lt", 10.0, False),
    ],
)
def test_passes_single_sla_operator_variants(
    op: str, observed: float, expected: bool
) -> None:
    sla = SLAFilter.model_construct(
        metric_tag="request_latency", stat="avg", op=op, threshold=10.0
    )

    assert AdaptiveScaleStrategy._passes_single_sla(sla, observed) is expected


def test_step_size_uses_base_step_without_usable_margins(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    zero_threshold = SLAFilter.model_construct(
        metric_tag="request_latency", stat="avg", op="le", threshold=0.0
    )
    strategy._sla_filters = [zero_threshold]

    assert (
        strategy._sla_margin_step_size(None)
        == strategy._config.adaptive_scale_base_step
    )
    assert (
        strategy._sla_margin_step_size({strategy._sla_key(zero_threshold): 1.0})
        == strategy._config.adaptive_scale_base_step
    )


def test_artifact_disabled_paths_do_not_write(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    strategy._config = strategy._config.model_copy(update={"artifact_dir": None})
    strategy._event_path = None
    strategy._summary_path = None

    strategy._emit_event(
        event="noop",
        reason="artifact disabled",
        sla_value=None,
        throughput=0.0,
        sample_count=0,
        error_count=0,
    )
    strategy._complete_controller(reason="done")
    strategy._complete_controller(reason="ignored")

    assert strategy._completed_reason == "done"
    assert not (tmp_path / "adaptive_scale_events.jsonl").exists()
    assert not (tmp_path / "adaptive_scale_summary.json").exists()


def test_percentile_empty_single_and_exact_rank() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        _percentile([], 50)
    assert _percentile([42], 95) == 42.0
    assert _percentile([10, 20, 30], 50) == 20.0


@pytest.mark.asyncio
async def test_artifact_writer_continues_after_failed_write() -> None:
    writer = AdaptiveScaleArtifactWriter()
    await writer.start()
    completed: list[str] = []

    def fail() -> None:
        raise OSError("disk write failed")

    def succeed() -> None:
        completed.append("ok")

    writer._schedule_write(fail)
    writer._schedule_write(succeed)

    await asyncio.wait_for(writer.flush(), timeout=1.0)
    await writer.close()

    assert completed == ["ok"]
