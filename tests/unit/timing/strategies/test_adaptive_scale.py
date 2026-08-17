# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.credit.messages import CreditReturn, FirstToken
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
from aiperf.timing.strategies.adaptive_scale_sla import AdaptiveScaleSLAEvaluator
from aiperf.timing.strategies.adaptive_scale_types import (
    WindowRequestSample,
    WindowStats,
)


def _strategy(
    tmp_path: Path, *, threshold: float = 100.0, **overrides
) -> AdaptiveScaleStrategy:
    config_kwargs = {
        "phase": CreditPhase.PROFILING,
        "phase_index": 0,
        "profiling_index": 0,
        "phase_name": "profiling",
        "phase_kind": "profiling",
        "timing_mode": TimingMode.ADAPTIVE_SCALE,
        "expected_duration_sec": 60.0,
        "concurrency": 10,
        "arrival_pattern": ArrivalPattern.CONCURRENCY_BURST,
        "adaptive_sustain_duration_sec": 10.0,
        "adaptive_assessment_period_sec": 1.0,
        "adaptive_control_min": 2,
        "adaptive_control_max": 10,
        "adaptive_sla_filters": [
            SLAFilter(
                metric_tag="request_latency",
                stat="p95",
                op="le",
                threshold=threshold,
            )
        ],
        "artifact_dir": tmp_path,
    }
    config_kwargs.update(overrides)
    cfg = CreditPhaseConfig(**config_kwargs)
    lifecycle = MagicMock()
    lifecycle.is_sending_complete = False
    progress = MagicMock()
    progress.all_credits_sent_event = asyncio.Event()
    strategy = AdaptiveScaleStrategy(
        config=cfg,
        conversation_source=MagicMock(),
        scheduler=MagicMock(),
        stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
        credit_issuer=MagicMock(),
        lifecycle=lifecycle,
        concurrency_manager=MagicMock(),
        progress=progress,
    )
    strategy._artifacts._schedule_write = lambda write: write()
    return strategy


def _event_path(tmp_path: Path, phase_name: str = "profiling") -> Path:
    return tmp_path / "phases" / phase_name / "adaptive_scale_events.jsonl"


def _summary_path(tmp_path: Path, phase_name: str = "profiling") -> Path:
    return tmp_path / "phases" / phase_name / "adaptive_scale_summary.json"


def _manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "adaptive_scale_manifest.json"


def _assert_event_clock_fields(event: dict) -> None:
    assert event["schema_version"] == 2
    assert isinstance(event["timestamp"], int)
    assert event["timestamp_ns"] == event["timestamp"]
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
        event["timestamp_utc"],
    )


def test_percentile_interpolates_p95() -> None:
    assert _percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48.0)


@pytest.mark.asyncio
async def test_setup_phase_writes_phase_scoped_manifest(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)

    await strategy.setup_phase()
    await strategy._artifacts.flush()
    await strategy._artifacts.close()

    manifest = orjson.loads(_manifest_path(tmp_path).read_bytes())
    assert manifest["adaptive_phases"] == [
        {
            "events_path": "phases/profiling/adaptive_scale_events.jsonl",
            "phase_index": 0,
            "phase_kind": "profiling",
            "phase_name": "profiling",
            "profiling_index": 0,
            "summary_path": "phases/profiling/adaptive_scale_summary.json",
        }
    ]
    assert _event_path(tmp_path).exists()
    assert not (tmp_path / "adaptive_scale_events.jsonl").exists()
    assert not (tmp_path / "adaptive_scale_summary.json").exists()


@pytest.mark.asyncio
async def test_setup_phase_failure_does_not_write_manifest(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    strategy._user_strategy = MagicMock()
    strategy._user_strategy.setup_phase = AsyncMock(
        side_effect=RuntimeError("setup failed")
    )

    with pytest.raises(RuntimeError, match="setup failed"):
        await strategy.setup_phase()

    assert not _manifest_path(tmp_path).exists()
    assert not _event_path(tmp_path).exists()
    assert not _summary_path(tmp_path).exists()


def test_manifest_entries_are_sorted_and_replaced_by_phase_identity(
    tmp_path: Path,
) -> None:
    writer = AdaptiveScaleArtifactWriter()
    writer._schedule_write = lambda write: write()

    writer.write_manifest_entry(
        tmp_path,
        {
            "events_path": "phases/storm_1/adaptive_scale_events.jsonl",
            "phase_index": 2,
            "phase_kind": "profiling",
            "phase_name": "storm_1",
            "profiling_index": 1,
            "summary_path": "phases/storm_1/adaptive_scale_summary.json",
        },
    )
    writer.write_manifest_entry(
        tmp_path,
        {
            "events_path": "phases/low/adaptive_scale_events.jsonl",
            "phase_index": 1,
            "phase_kind": "profiling",
            "phase_name": "low",
            "profiling_index": 0,
            "summary_path": "phases/low/adaptive_scale_summary.json",
        },
    )
    writer.write_manifest_entry(
        tmp_path,
        {
            "events_path": "phases/storm_1/adaptive_scale_events.v2.jsonl",
            "phase_index": 2,
            "phase_kind": "profiling",
            "phase_name": "storm_1",
            "profiling_index": 1,
            "summary_path": "phases/storm_1/adaptive_scale_summary.v2.json",
        },
    )

    manifest = orjson.loads(_manifest_path(tmp_path).read_bytes())

    assert manifest["schema_version"] == 2
    assert manifest["adaptive_phases"] == [
        {
            "events_path": "phases/low/adaptive_scale_events.jsonl",
            "phase_index": 1,
            "phase_kind": "profiling",
            "phase_name": "low",
            "profiling_index": 0,
            "summary_path": "phases/low/adaptive_scale_summary.json",
        },
        {
            "events_path": "phases/storm_1/adaptive_scale_events.v2.jsonl",
            "phase_index": 2,
            "phase_kind": "profiling",
            "phase_name": "storm_1",
            "profiling_index": 1,
            "summary_path": "phases/storm_1/adaptive_scale_summary.v2.json",
        },
    ]


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
    assert stats.cancelled == 1


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


@pytest.mark.asyncio
@pytest.mark.parametrize("use_user_strategy", [False, True])
async def test_handle_credit_return_accepts_error_kwarg(
    tmp_path, use_user_strategy: bool
) -> None:
    # Regression (PR #1165 review): the callback handler calls
    # handle_credit_return(credit, error=...) on the run's strategy. Every other
    # strategy accepts the error kwarg; AdaptiveScaleStrategy did not, so every
    # credit return under adaptive_scale raised TypeError. Cover both the plain
    # path and the delegated (_user_strategy) path.
    strategy = _strategy(tmp_path)
    if use_user_strategy:
        strategy._user_strategy = MagicMock()
        strategy._user_strategy.handle_credit_return = AsyncMock()
    credit = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="c",
        x_correlation_id="x",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns() - 5_000_000,
    )

    # Must not raise TypeError on the error keyword.
    await strategy.handle_credit_return(credit, error="boom")

    if use_user_strategy:
        strategy._user_strategy.handle_credit_return.assert_awaited_once_with(
            credit, error="boom"
        )


def test_unsupported_sla_metric_fails_during_strategy_construction(tmp_path) -> None:
    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.ADAPTIVE_SCALE,
        expected_duration_sec=60.0,
        concurrency=10,
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        adaptive_sustain_duration_sec=10.0,
        adaptive_assessment_period_sec=1.0,
        adaptive_control_min=2,
        adaptive_control_max=10,
        adaptive_sla_filters=[
            SLAFilter(
                metric_tag="unsupported_metric",
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

    strategy._concurrency_manager.set_session_limit.assert_called_with(0, 10)
    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_decision"
    assert events[-1]["control_value_before"] == 2
    assert events[-1]["control_value_after"] == 10
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

    strategy._concurrency_manager.set_session_limit.assert_called_with(0, 3)


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


def test_success_rate_uses_successes_over_attempts(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    stats = MagicMock(samples=[10_000_000, 20_000_000], errors=1, total=3)
    sla = SLAFilter(
        metric_tag="success_rate",
        stat="avg",
        op="ge",
        threshold=0.95,
    )

    assert strategy._sla_value(sla, stats) == pytest.approx(2 / 3)


def test_output_token_throughput_uses_window_osl_per_second(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    stats = WindowStats(
        samples=[10_000_000, 20_000_000, 30_000_000],
        errors=0,
        elapsed_sec=2.0,
        successful_requests=[
            WindowRequestSample(10_000_000, output_sequence_length=20),
            WindowRequestSample(20_000_000, output_sequence_length=None),
            WindowRequestSample(30_000_000, output_sequence_length=40),
        ],
    )
    sla = SLAFilter(
        metric_tag="output_token_throughput",
        stat="avg",
        op="ge",
        threshold=10.0,
    )

    assert strategy._sla_value(sla, stats) == pytest.approx(30.0)


def test_goodput_ratio_uses_quality_passes_over_attempts(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    latency_sla = SLAFilter(
        metric_tag="request_latency",
        stat="p95",
        op="le",
        threshold=100.0,
    )
    fraction_sla = SLAFilter(
        metric_tag="goodput_ratio",
        stat="avg",
        op="ge",
        threshold=0.5,
    )
    stats = WindowStats(
        samples=[80_000_000, 120_000_000],
        errors=1,
        cancelled=1,
        elapsed_sec=1.0,
        successful_requests=[
            WindowRequestSample(request_latency_ns=80_000_000),
            WindowRequestSample(request_latency_ns=120_000_000),
        ],
    )

    strategy._sla_filters = [latency_sla, fraction_sla]

    assert strategy._sla_value(fraction_sla, stats) == pytest.approx(1 / 4)


def test_goodput_ratio_uses_quality_fraction_semantics(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    latency_sla = SLAFilter(
        metric_tag="request_latency",
        stat="p95",
        op="le",
        threshold=100.0,
    )
    alias_sla = SLAFilter(
        metric_tag="goodput_ratio",
        stat="avg",
        op="ge",
        threshold=0.5,
    )
    stats = WindowStats(
        samples=[80_000_000, 120_000_000],
        errors=1,
        elapsed_sec=1.0,
        successful_requests=[
            WindowRequestSample(request_latency_ns=80_000_000),
            WindowRequestSample(request_latency_ns=120_000_000),
        ],
    )
    strategy._sla_filters = [latency_sla, alias_sla]

    assert strategy._sla_value(alias_sla, stats) == pytest.approx(1 / 3)


def test_success_rate_sla_participates_in_pass_fail(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    latency_sla = strategy._primary_sla
    success_sla = SLAFilter(
        metric_tag="success_rate",
        stat="avg",
        op="ge",
        threshold=0.95,
    )
    strategy._sla_filters = [latency_sla, success_sla]
    observed = {
        strategy._sla_key(latency_sla): 10.0,
        strategy._sla_key(success_sla): 0.90,
    }

    assert strategy._passes_sla(observed) is False


def test_inter_token_latency_aliases_use_itl_samples(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    stats = WindowStats(
        samples=[100_000_000],
        errors=0,
        elapsed_sec=1.0,
        itl_samples=[10_000_000, 30_000_000],
    )

    for metric_tag in ("inter_token_latency", "itl", "tpot"):
        sla = SLAFilter(
            metric_tag=metric_tag,
            stat="p50",
            op="le",
            threshold=25.0,
        )

        assert strategy._sla_value(sla, stats) == pytest.approx(20.0)


def test_missing_inter_token_latency_sample_fails_lower_is_better_sla(
    tmp_path,
) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    stats = WindowStats(samples=[100_000_000], errors=0, elapsed_sec=1.0)
    sla = SLAFilter(
        metric_tag="inter_token_latency",
        stat="p95",
        op="le",
        threshold=25.0,
    )

    observed = {strategy._sla_key(sla): strategy._sla_value(sla, stats)}

    assert strategy._passes_single_sla(sla, observed[strategy._sla_key(sla)]) is False


def test_goodput_counts_requests_passing_quality_filters(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    latency_sla = SLAFilter(
        metric_tag="request_latency", stat="p95", op="le", threshold=100.0
    )
    ttft_sla = SLAFilter(metric_tag="ttft", stat="p95", op="le", threshold=30.0)
    itl_sla = SLAFilter(metric_tag="itl", stat="p95", op="le", threshold=20.0)
    goodput_sla = SLAFilter(metric_tag="goodput", stat="avg", op="ge", threshold=1.0)
    strategy._sla_filters = [latency_sla, ttft_sla, itl_sla, goodput_sla]
    stats = WindowStats(
        samples=[80_000_000, 90_000_000, 120_000_000],
        errors=0,
        elapsed_sec=2.0,
        ttft_samples=[20_000_000, 35_000_000, 20_000_000],
        itl_samples=[10_000_000, 10_000_000, 30_000_000],
        successful_requests=[
            WindowRequestSample(80_000_000, 20_000_000, 10_000_000),
            WindowRequestSample(90_000_000, 35_000_000, 10_000_000),
            WindowRequestSample(120_000_000, 20_000_000, 30_000_000),
        ],
    )

    assert strategy._sla_value(goodput_sla, stats) == pytest.approx(0.5)


def test_goodput_requires_quality_filter(tmp_path) -> None:
    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.ADAPTIVE_SCALE,
        expected_duration_sec=60.0,
        concurrency=10,
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        adaptive_sustain_duration_sec=10.0,
        adaptive_assessment_period_sec=1.0,
        adaptive_control_min=2,
        adaptive_control_max=10,
        adaptive_sla_filters=[
            SLAFilter(metric_tag="goodput", stat="avg", op="ge", threshold=1.0)
        ],
        artifact_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="quality goodput SLA requires"):
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


@pytest.mark.asyncio
async def test_handle_credit_result_records_itl_and_quality_sample(tmp_path) -> None:
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

    await strategy.handle_first_token(
        FirstToken(credit_id=1, phase=CreditPhase.PROFILING, ttft_ns=20_000_000)
    )
    await strategy.handle_credit_result(
        CreditReturn(
            credit=credit,
            request_latency_ns=100_000_000,
            inter_token_latency_ns=10_000_000,
            output_sequence_length=64,
        )
    )
    stats = await strategy._take_window()

    assert stats.itls == [10_000_000]
    assert stats.requests == [
        WindowRequestSample(
            request_latency_ns=100_000_000,
            ttft_ns=20_000_000,
            inter_token_latency_ns=10_000_000,
            output_sequence_length=64,
        )
    ]


@pytest.mark.asyncio
async def test_take_window_excludes_ttft_from_errored_requests(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    successful_credit = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="c1",
        x_correlation_id="x1",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns() - 5_000_000,
    )
    errored_credit = Credit(
        id=2,
        phase=CreditPhase.PROFILING,
        conversation_id="c2",
        x_correlation_id="x2",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns() - 5_000_000,
    )
    cancelled_credit = Credit(
        id=3,
        phase=CreditPhase.PROFILING,
        conversation_id="c3",
        x_correlation_id="x3",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns() - 5_000_000,
    )

    await strategy.handle_first_token(
        FirstToken(credit_id=1, phase=CreditPhase.PROFILING, ttft_ns=180_000_000)
    )
    await strategy.handle_credit_result(
        CreditReturn(credit=successful_credit, request_latency_ns=200_000_000)
    )
    await strategy.handle_first_token(
        FirstToken(credit_id=2, phase=CreditPhase.PROFILING, ttft_ns=20_000_000)
    )
    await strategy.handle_credit_result(
        CreditReturn(credit=errored_credit, error="stream failed")
    )
    await strategy.handle_first_token(
        FirstToken(credit_id=3, phase=CreditPhase.PROFILING, ttft_ns=30_000_000)
    )
    await strategy.handle_credit_result(
        CreditReturn(credit=cancelled_credit, cancelled=True)
    )

    stats = await strategy._take_window()

    assert stats.ttfts == [180_000_000]
    assert stats.errors == 1
    assert stats.cancelled == 1


@pytest.mark.asyncio
async def test_adaptive_window_reports_itl_and_goodput_sla_values(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    latency_sla = strategy._primary_sla
    itl_sla = SLAFilter(metric_tag="itl", stat="avg", op="le", threshold=20.0)
    goodput_sla = SLAFilter(metric_tag="goodput", stat="avg", op="ge", threshold=1.0)
    strategy._sla_filters = [latency_sla, itl_sla, goodput_sla]
    strategy._window_latency_ns = [80_000_000]
    strategy._window_itl_ns = [10_000_000]
    strategy._window_successful_requests = [
        WindowRequestSample(80_000_000, None, 10_000_000)
    ]
    strategy._window_started_at = time.perf_counter() - 1.0

    await strategy._assess_window()

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    window = next(event for event in events if event["event"] == "adaptive_window")
    assert window["sla_values"][strategy._sla_key(itl_sla)] == pytest.approx(10.0)
    assert window["sla_values"][strategy._sla_key(goodput_sla)] >= 0.9


def test_breach_enters_sustain_at_last_good_boundary(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    strategy._last_good_concurrency = 4
    strategy._set_control(5)
    stats = MagicMock(samples=[150_000_000], errors=0, throughput=1.0)

    strategy._assess_discover(150.0, False, stats)

    assert strategy._controller_phase == "sustain"
    assert strategy._boundary_concurrency == 4
    strategy._concurrency_manager.set_session_limit.assert_called_with(0, 4)
    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "sustain_started",
        "boundary_discovered",
    ]
    assert events[-1]["phase"] == "sustain"
    assert events[-1]["control_value_after"] == 4


def test_sustain_completion_writes_complete_event_and_summary(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    strategy._controller_phase = "sustain"
    strategy._boundary_concurrency = 4
    strategy._last_good_concurrency = 4
    strategy._set_control(4)
    strategy._sustain_started_at = time.perf_counter() - 20.0
    strategy._sustain_started_at_ns = 123
    stats = MagicMock(samples=[50_000_000], errors=0, throughput=2.0)

    strategy._assess_sustain(50.0, True, stats)

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_complete"
    summary = orjson.loads(_summary_path(tmp_path).read_bytes())
    assert summary["schema_version"] == 2
    assert summary["status"] == "completed"
    assert summary["boundary_value"] == 4
    assert summary["last_passing_value"] == 4
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
        "cancelled": 0,
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
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_complete"
    assert events[-1]["reason"] == "phase_stopped"
    summary = orjson.loads(_summary_path(tmp_path).read_bytes())
    assert summary["status"] == "completed"
    assert summary["boundary_value"] is None
    assert summary["result"]["boundary_value"] is None
    assert summary["completed_reason"] == "phase_stopped"


def test_all_failed_discover_window_enters_sustain(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    strategy._last_good_concurrency = 4
    strategy._set_control(6)
    stats = MagicMock(samples=[], errors=3, throughput=0.0)

    strategy._assess_failed_window(stats)

    assert strategy._controller_phase == "sustain"
    assert strategy._boundary_concurrency == 4
    strategy._concurrency_manager.set_session_limit.assert_called_with(0, 4)
    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
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
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_failed"
    assert events[-1]["reason"] == "no_sustainable_concurrency_found"
    assert events[-1]["first_failing_value"] == 2
    summary = orjson.loads(_summary_path(tmp_path).read_bytes())
    assert summary["status"] == "failed"
    assert summary["completed_reason"] == "no_sustainable_concurrency_found"
    assert summary["result"] == {
        "last_passing_value": None,
        "first_failing_value": 2,
        "boundary_value": None,
    }


def test_max_concurrency_passing_is_incomplete_not_boundary(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    strategy._set_control(10)
    stats = MagicMock(samples=[50_000_000], errors=0, throughput=1.0)

    strategy._assess_discover(50.0, True, stats)

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_incomplete"
    assert events[-1]["reason"] == "max_control_value_reached_without_saturation"
    assert events[-1]["last_passing_value"] == 10


@pytest.mark.asyncio
async def test_sparse_window_is_inconclusive(tmp_path) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    strategy._min_completed_requests = 2
    strategy._window_latency_ns = [10_000_000]

    await strategy._assess_window()

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
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
            adaptive_control_min=2,
            adaptive_control_max=10,
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
        param("avg", 20.0, id="avg"),
        param("min", 10.0, id="min"),
        param("max", 30.0, id="max"),
        param("p50", 20.0, id="p50"),
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
        metric_tag="unsupported_metric", stat="avg", op="le", threshold=1.0
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

    strategy._concurrency_manager.set_session_limit.assert_called_with(0, 2)
    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_phase_started"
    assert events[-1]["control_value"] == 2


@pytest.mark.asyncio
async def test_execute_phase_failure_writes_failed_terminal_artifacts(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    strategy._user_strategy = MagicMock()
    strategy._user_strategy.execute_phase = AsyncMock(
        side_effect=RuntimeError("execute failed")
    )

    with pytest.raises(RuntimeError, match="execute failed"):
        await strategy.execute_phase()

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_failed"
    assert events[-1]["reason"] == "phase_failed: execute failed"
    summary = orjson.loads(_summary_path(tmp_path).read_bytes())
    assert summary["status"] == "failed"
    assert summary["completed_reason"] == "phase_failed: execute failed"


@pytest.mark.asyncio
async def test_execute_phase_cancellation_writes_failed_terminal_artifacts(
    tmp_path,
) -> None:
    strategy = _strategy(tmp_path)
    strategy._user_strategy = MagicMock()
    strategy._user_strategy.execute_phase = AsyncMock(
        side_effect=asyncio.CancelledError()
    )

    with pytest.raises(asyncio.CancelledError):
        await strategy.execute_phase()

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_failed"
    assert events[-1]["reason"] == "phase_cancelled"
    summary = orjson.loads(_summary_path(tmp_path).read_bytes())
    assert summary["status"] == "failed"
    assert summary["completed_reason"] == "phase_cancelled"


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
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_failed"
    summary = orjson.loads(_summary_path(tmp_path).read_bytes())
    assert summary["status"] == "failed"
    assert summary["completed_reason"] == "assessment_failed: bad window"


@pytest.mark.asyncio
async def test_assess_window_evaluates_sustain_phase(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    strategy._controller_phase = "sustain"
    strategy._last_good_concurrency = 4
    strategy._set_control(4)
    strategy._window_latency_ns = [10_000_000, 20_000_000]

    await strategy._assess_window()

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
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
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-2]["reason"] == "no successful requests in assessment window"
    assert events[-1]["event"] == "adaptive_failed"


@pytest.mark.asyncio
async def test_cancellation_only_window_fails_and_reports_cancelled(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    strategy._window_cancelled = 2

    await strategy._assess_window()

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-2]["event"] == "adaptive_window"
    assert events[-2]["reason"] == "no successful requests in assessment window"
    assert events[-2]["sent"] == 2
    assert events[-2]["cancelled"] == 2
    assert events[-1]["event"] == "adaptive_failed"
    assert events[-1]["cancelled"] == 2
    summary = orjson.loads(_summary_path(tmp_path).read_bytes())
    assert summary["status"] == "failed"
    assert summary["totals"] == {
        "sent": 2,
        "completed": 0,
        "errored": 0,
        "cancelled": 2,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_tag", ["request_error_rate", "error_rate"])
async def test_all_error_rate_sla_window_evaluates_without_successes(
    tmp_path, metric_tag: str
) -> None:
    error_sla = SLAFilter(
        metric_tag=metric_tag,
        stat="avg",
        op="le",
        threshold=100.0,
    )
    strategy = _strategy(tmp_path, adaptive_sla_filters=[error_sla])
    strategy._window_errors = 2
    strategy._window_started_at = time.perf_counter() - 1.0

    await strategy._assess_window()

    assert strategy._completed_reason is None
    strategy._concurrency_manager.set_session_limit.assert_called_with(0, 10)
    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    window = next(event for event in events if event["event"] == "adaptive_window")
    assert window["reason"] == "SLA window evaluated"
    assert window["sla_passed"] is True
    assert window["sla_values"] == {f"{metric_tag}:avg:le:100": 100.0}
    assert events[-1]["event"] == "adaptive_decision"


@pytest.mark.asyncio
async def test_cancelled_window_does_not_pass_request_error_rate_only_sla(tmp_path) -> None:
    error_sla = SLAFilter(
        metric_tag="request_error_rate",
        stat="avg",
        op="le",
        threshold=0.0,
    )
    strategy = _strategy(tmp_path, adaptive_sla_filters=[error_sla])
    strategy._window_cancelled = 2
    strategy._window_started_at = time.perf_counter() - 1.0

    await strategy._assess_window()

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    window = next(event for event in events if event["event"] == "adaptive_window")
    assert window["reason"] == "no successful requests in assessment window"
    assert window["sla_passed"] is False
    assert window["sla_values"] == {}
    assert not any(event["event"] == "adaptive_decision" for event in events)


@pytest.mark.asyncio
async def test_mixed_error_cancel_window_does_not_pass_terminal_rate_slas(
    tmp_path,
) -> None:
    error_sla = SLAFilter(
        metric_tag="request_error_rate",
        stat="avg",
        op="le",
        threshold=50.0,
    )
    cancellation_sla = SLAFilter(
        metric_tag="cancellation_rate",
        stat="avg",
        op="le",
        threshold=0.5,
    )
    strategy = _strategy(
        tmp_path,
        adaptive_sla_filters=[error_sla, cancellation_sla],
        adaptive_min_completed_requests=5,
    )
    strategy._window_errors = 1
    strategy._window_cancelled = 1
    strategy._window_started_at = time.perf_counter() - 1.0

    await strategy._assess_window()

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    window = next(event for event in events if event["event"] == "adaptive_window")
    assert window["reason"] == "no successful requests in assessment window"
    assert window["sla_passed"] is False
    assert window["sla_values"] == {}
    assert not any(event["event"] == "adaptive_decision" for event in events)


@pytest.mark.asyncio
async def test_error_window_does_not_pass_cancellation_rate_only_sla(tmp_path) -> None:
    cancellation_sla = SLAFilter(
        metric_tag="cancellation_rate",
        stat="avg",
        op="le",
        threshold=0.0,
    )
    strategy = _strategy(tmp_path, adaptive_sla_filters=[cancellation_sla])
    strategy._window_errors = 2
    strategy._window_started_at = time.perf_counter() - 1.0

    await strategy._assess_window()

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    window = next(event for event in events if event["event"] == "adaptive_window")
    assert window["reason"] == "no successful requests in assessment window"
    assert window["sla_passed"] is False
    assert window["sla_values"] == {}
    assert not any(event["event"] == "adaptive_decision" for event in events)


@pytest.mark.asyncio
async def test_error_window_does_not_pass_request_error_rate_plus_throughput_cap(
    tmp_path,
) -> None:
    error_sla = SLAFilter(
        metric_tag="request_error_rate",
        stat="avg",
        op="le",
        threshold=100.0,
    )
    throughput_sla = SLAFilter(
        metric_tag="throughput",
        stat="avg",
        op="le",
        threshold=100.0,
    )
    strategy = _strategy(tmp_path, adaptive_sla_filters=[error_sla, throughput_sla])
    strategy._window_errors = 3
    strategy._window_started_at = time.perf_counter() - 1.0

    await strategy._assess_window()

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    window = next(event for event in events if event["event"] == "adaptive_window")
    assert window["reason"] == "no successful requests in assessment window"
    assert window["sla_passed"] is False
    assert window["sla_values"] == {}
    assert not any(event["event"] == "adaptive_decision" for event in events)


@pytest.mark.asyncio
async def test_error_window_does_not_pass_output_token_throughput_cap(tmp_path) -> None:
    throughput_sla = SLAFilter(
        metric_tag="output_token_throughput",
        stat="avg",
        op="le",
        threshold=1.0,
    )
    strategy = _strategy(tmp_path, adaptive_sla_filters=[throughput_sla])
    strategy._window_errors = 2
    strategy._window_started_at = time.perf_counter() - 1.0

    await strategy._assess_window()

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    window = next(event for event in events if event["event"] == "adaptive_window")
    assert window["reason"] == "no successful requests in assessment window"
    assert window["sla_passed"] is False
    assert window["sla_values"] == {}
    assert not any(event["event"] == "adaptive_decision" for event in events)


@pytest.mark.asyncio
async def test_error_window_success_rate_does_not_cover_cancellation_sla(
    tmp_path,
) -> None:
    success_sla = SLAFilter(
        metric_tag="success_rate",
        stat="avg",
        op="ge",
        threshold=0.0,
    )
    cancellation_sla = SLAFilter(
        metric_tag="cancellation_rate",
        stat="avg",
        op="le",
        threshold=0.0,
    )
    strategy = _strategy(
        tmp_path,
        adaptive_sla_filters=[success_sla, cancellation_sla],
        adaptive_min_completed_requests=5,
    )
    strategy._window_errors = 3
    strategy._window_started_at = time.perf_counter() - 1.0

    await strategy._assess_window()

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    window = next(event for event in events if event["event"] == "adaptive_window")
    assert window["reason"] == "no successful requests in assessment window"
    assert window["sla_passed"] is False
    assert window["sla_values"] == {}
    assert not any(event["event"] == "adaptive_decision" for event in events)


def test_all_failed_sustain_window_downshifts_with_reason(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    strategy._controller_phase = "sustain"
    strategy._last_good_concurrency = 4
    strategy._set_control(6)
    stats = MagicMock(samples=[], errors=3, throughput=0.0)

    strategy._assess_failed_window(stats)

    strategy._concurrency_manager.set_session_limit.assert_called_with(0, 4)
    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-1]["reason"] == "all requests failed in assessment window"
    assert events[-1]["step_size"] == 2


def test_second_sustain_breach_after_recovery_fails(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    strategy._controller_phase = "sustain"
    strategy._set_control(6)
    strategy._last_good_concurrency = 4
    stats = MagicMock(samples=[150_000_000], errors=0, cancelled=0, throughput=1.0)

    strategy._assess_sustain(150.0, False, stats)
    assert strategy._completed_reason is None

    strategy._assess_sustain(160.0, False, stats)

    assert strategy._completed_reason == "sustain_failed_after_recovery"
    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-1]["event"] == "adaptive_failed"
    summary = orjson.loads(_summary_path(tmp_path).read_bytes())
    assert summary["status"] == "failed"
    assert summary["completed_reason"] == "sustain_failed_after_recovery"


def test_sustain_breach_at_minimum_fails_unrecoverably(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    strategy._controller_phase = "sustain"
    strategy._set_control(2)
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
    strategy._set_control(6)
    strategy._last_good_concurrency = 8
    stats = MagicMock(samples=[150_000_000], errors=0, throughput=1.0)

    strategy._assess_sustain(150.0, False, stats)

    assert strategy._control.current < 6
    assert strategy._last_good_concurrency == 8


@pytest.mark.asyncio
async def test_pre_sustain_credit_results_do_not_poison_sustain_window(
    tmp_path,
) -> None:
    strategy = _strategy(tmp_path, threshold=100.0)
    strategy._controller_phase = "sustain"
    strategy._last_good_concurrency = strategy._control.minimum
    strategy._set_control(strategy._control.minimum)
    strategy._sustain_started_at_ns = time.time_ns()
    old_credit = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="old",
        x_correlation_id="old",
        turn_index=0,
        num_turns=1,
        issued_at_ns=strategy._sustain_started_at_ns - 1,
    )
    new_credit = Credit(
        id=2,
        phase=CreditPhase.PROFILING,
        conversation_id="new",
        x_correlation_id="new",
        turn_index=0,
        num_turns=1,
        issued_at_ns=strategy._sustain_started_at_ns + 1,
    )

    await strategy.handle_credit_result(
        CreditReturn(credit=old_credit, request_latency_ns=150_000_000)
    )
    await strategy.handle_credit_result(
        CreditReturn(credit=new_credit, request_latency_ns=10_000_000)
    )

    stats = await strategy._take_window()

    assert stats.samples == [10_000_000]


def test_enter_sustain_requires_last_good_boundary(tmp_path) -> None:
    strategy = _strategy(tmp_path)

    with pytest.raises(RuntimeError, match="passing boundary"):
        strategy._enter_sustain(
            None, MagicMock(samples=[], errors=0, throughput=0.0), "x"
        )


@pytest.mark.parametrize(
    ("op", "observed", "expected"),
    [
        param("lt", 9.0, True, id="lt-passing"),
        param("gt", 11.0, True, id="gt-passing"),
        param("ge", 10.0, True, id="ge-passing"),
        param("lt", 10.0, False, id="lt-failing-at-bound"),
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


def test_emit_event_preserves_zero_candidate_value(tmp_path) -> None:
    strategy = _strategy(tmp_path)
    strategy._set_control(5)

    strategy._emit_event(
        event="adaptive_decision",
        reason="zero candidate",
        sla_value=None,
        throughput=0.0,
        sample_count=0,
        error_count=0,
        before=0,
    )

    events = [
        orjson.loads(line) for line in _event_path(tmp_path).read_text().splitlines()
    ]
    assert events[-1]["candidate_value"] == 0
    assert events[-1]["accepted_value"] == 5


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
    assert not _event_path(tmp_path).exists()
    assert not _summary_path(tmp_path).exists()


def test_percentile_empty_single_and_exact_rank() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        _percentile([], 50)
    assert _percentile([42], 95) == 42.0
    assert _percentile([10, 20, 30], 50) == 20.0


def test_percentile_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        _percentile([1, 2, 3], -1)
    with pytest.raises(ValueError, match="between 0 and 100"):
        _percentile([1, 2, 3], 101)


def test_candidate_payload_reports_success_rate_not_goodput_ratio() -> None:
    payload = AdaptiveScaleArtifactWriter.candidate_payload(
        adaptive_iteration=1,
        candidate_value=4,
        stats=WindowStats(
            samples=[10_000_000, 20_000_000],
            errors=1,
            cancelled=1,
            elapsed_sec=2.0,
        ),
        accepted=True,
        rejection_reason="",
    )

    assert payload["success_rate"] == pytest.approx(0.5)
    assert "goodput_ratio" not in payload


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

    with pytest.raises(OSError, match="disk write failed"):
        await asyncio.wait_for(writer.flush(), timeout=1.0)
    with pytest.raises(OSError, match="disk write failed"):
        await writer.close()

    assert completed == ["ok"]


def test_artifact_writer_requires_start_before_write() -> None:
    writer = AdaptiveScaleArtifactWriter()

    with pytest.raises(RuntimeError, match="not started"):
        writer._schedule_write(lambda: None)


def test_prefill_control_backend_sets_prefill_limit(tmp_path) -> None:
    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.ADAPTIVE_SCALE,
        expected_duration_sec=60.0,
        concurrency=10,
        prefill_concurrency=8,
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        adaptive_sustain_duration_sec=10.0,
        adaptive_assessment_period_sec=1.0,
        adaptive_control_variable="prefill_concurrency",
        adaptive_control_min=2,
        adaptive_control_max=8,
        adaptive_sla_filters=[
            SLAFilter(
                metric_tag="request_latency", stat="p95", op="le", threshold=100.0
            )
        ],
        artifact_dir=tmp_path,
    )
    progress = MagicMock()
    progress.all_credits_sent_event = asyncio.Event()
    strategy = AdaptiveScaleStrategy(
        config=cfg,
        conversation_source=MagicMock(),
        scheduler=MagicMock(),
        stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
        credit_issuer=MagicMock(),
        lifecycle=MagicMock(is_sending_complete=False),
        concurrency_manager=MagicMock(),
        progress=progress,
    )

    strategy._set_control(6)

    strategy._concurrency_manager.set_prefill_limit.assert_called_with(
        CreditPhase.PROFILING, 6
    )


def test_request_rate_control_backend_sets_rate(tmp_path) -> None:
    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.ADAPTIVE_SCALE,
        expected_duration_sec=60.0,
        concurrency=10,
        request_rate=20.0,
        arrival_pattern=ArrivalPattern.POISSON,
        adaptive_sustain_duration_sec=10.0,
        adaptive_assessment_period_sec=1.0,
        adaptive_control_variable="request_rate",
        adaptive_control_min=2.0,
        adaptive_control_max=20.0,
        adaptive_sla_filters=[
            SLAFilter(
                metric_tag="request_latency", stat="p95", op="le", threshold=100.0
            )
        ],
        artifact_dir=tmp_path,
    )
    progress = MagicMock()
    progress.all_credits_sent_event = asyncio.Event()
    strategy = AdaptiveScaleStrategy(
        config=cfg,
        conversation_source=MagicMock(),
        scheduler=MagicMock(),
        stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
        credit_issuer=MagicMock(),
        lifecycle=MagicMock(is_sending_complete=False),
        concurrency_manager=MagicMock(),
        progress=progress,
    )

    strategy._set_control(12.5)

    assert strategy._rate_generator.rate == pytest.approx(12.5)


def test_control_backend_clamps_and_snapshots(tmp_path) -> None:
    strategy = _strategy(tmp_path, adaptive_control_min=2, adaptive_control_max=10)

    strategy._set_control(99)

    assert strategy._control.current == 10
    strategy._concurrency_manager.set_session_limit.assert_called_with(0, 10)
    assert strategy._control.snapshot() == {"target_value": 10, "actual_value": 10}


def test_users_control_backend_uses_strategy_snapshot() -> None:
    from aiperf.timing.strategies.adaptive_scale_backends import UsersControlBackend

    strategy = MagicMock()
    strategy.user_control_snapshot.return_value = {
        "actual_value": 3,
        "active_users": 3,
    }
    backend = UsersControlBackend(strategy=strategy, minimum=1, maximum=5)

    backend.set(99)

    strategy.set_target_users.assert_called_once_with(5)
    assert backend.snapshot() == {
        "actual_value": 3,
        "active_users": 3,
        "target_value": 5,
    }


def test_users_control_backend_snapshot_falls_back_without_snapshotter() -> None:
    from aiperf.timing.strategies.adaptive_scale_backends import UsersControlBackend

    strategy = MagicMock()
    del strategy.user_control_snapshot
    backend = UsersControlBackend(strategy=strategy, minimum=1, maximum=5)

    assert backend.snapshot() == {"target_value": 1, "actual_value": 1}


def test_users_control_backend_requires_setter() -> None:
    from aiperf.timing.strategies.adaptive_scale_backends import UsersControlBackend

    backend = UsersControlBackend(strategy=object(), minimum=1, maximum=5)

    with pytest.raises(ValueError, match="adaptive users requires"):
        backend.set(2)


def test_build_backend_rejects_invalid_construction(tmp_path) -> None:
    from aiperf.timing.strategies.adaptive_scale_backends import (
        build_adaptive_control_backend,
    )

    strategy = MagicMock(set_request_rate=MagicMock())
    manager = MagicMock()
    base = dict(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.ADAPTIVE_SCALE,
        expected_duration_sec=60.0,
        concurrency=10,
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        adaptive_sustain_duration_sec=10.0,
        adaptive_assessment_period_sec=1.0,
        adaptive_control_min=2,
        adaptive_sla_filters=[
            SLAFilter(
                metric_tag="request_latency", stat="p95", op="le", threshold=100.0
            )
        ],
        artifact_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="requires a rate-controlled phase"):
        build_adaptive_control_backend(
            strategy=strategy,
            concurrency_manager=manager,
            config=CreditPhaseConfig(
                **base,
                adaptive_control_variable="request_rate",
                adaptive_control_max=10,
                request_rate=10.0,
            ),
        )

    with pytest.raises(ValueError, match="requires prefill_concurrency"):
        build_adaptive_control_backend(
            strategy=strategy,
            concurrency_manager=manager,
            config=CreditPhaseConfig(
                **base,
                adaptive_control_variable="prefill_concurrency",
                adaptive_control_max=None,
            ),
        )

    with pytest.raises(ValueError, match="requires user-centric num_users"):
        build_adaptive_control_backend(
            strategy=strategy,
            concurrency_manager=manager,
            config=CreditPhaseConfig(
                **base, adaptive_control_variable="users", adaptive_control_max=10
            ),
        )


def test_sla_evaluator_rate_metrics_and_failures() -> None:
    evaluator = AdaptiveScaleSLAEvaluator()
    stats = WindowStats(
        samples=[100_000_000, 200_000_000],
        errors=1,
        cancelled=1,
        elapsed_sec=2.0,
    )

    assert evaluator.value(
        SLAFilter(metric_tag="request_throughput", stat="min", op="ge", threshold=1),
        stats,
    ) == pytest.approx(1.0)
    request_error_rate = evaluator.value(
        SLAFilter(metric_tag="request_error_rate", stat="avg", op="le", threshold=34),
        stats,
    )
    assert request_error_rate == pytest.approx(100.0 / 3.0)
    assert evaluator.value(
        SLAFilter(metric_tag="error_rate", stat="avg", op="le", threshold=34),
        stats,
    ) == pytest.approx(request_error_rate)
    assert evaluator.value(
        SLAFilter(
            metric_tag="request_cancellation_rate", stat="max", op="le", threshold=1
        ),
        stats,
    ) == pytest.approx(0.25)
    assert evaluator.value(
        SLAFilter(metric_tag="request_success_rate", stat="avg", op="ge", threshold=0),
        stats,
    ) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="Unsupported throughput SLA stat"):
        evaluator.validate_single_filter(
            SLAFilter(metric_tag="throughput", stat="p95", op="ge", threshold=1)
        )
    with pytest.raises(ValueError, match="supports request_latency"):
        evaluator.validate_single_filter(
            SLAFilter(metric_tag="tokens", stat="avg", op="ge", threshold=1)
        )
    with pytest.raises(ValueError, match="Unsupported SLA operator"):
        evaluator.passes_single(
            SLAFilter.model_construct(
                metric_tag="request_latency", stat="avg", op="eq", threshold=1
            ),
            1.0,
        )
    for metric_tag in ("request_error_rate", "error_rate"):
        for boundary in (0.0, 100.0):
            evaluator.validate_single_filter(
                SLAFilter(
                    metric_tag=metric_tag,
                    stat="avg",
                    op="le",
                    threshold=boundary,
                )
            )

        for threshold in (-0.1, 100.1):
            with pytest.raises(ValueError, match=r"within \[0, 100\]"):
                evaluator.validate_single_filter(
                    SLAFilter(
                        metric_tag=metric_tag,
                        stat="avg",
                        op="le",
                        threshold=threshold,
                    )
                )


def test_sla_evaluator_supports_ttft_request_error_and_cancellation_rate() -> None:
    evaluator = AdaptiveScaleSLAEvaluator()
    stats = WindowStats(
        samples=[100_000_000, 200_000_000],
        errors=1,
        cancelled=1,
        ttft_samples=[10_000_000, 20_000_000],
        elapsed_sec=2.0,
    )

    assert evaluator.value(
        SLAFilter(metric_tag="ttft", stat="p95", op="le", threshold=25.0),
        stats,
    ) == pytest.approx(19.5)
    assert evaluator.value(
        SLAFilter(metric_tag="request_error_rate", stat="avg", op="le", threshold=34.0),
        stats,
    ) == pytest.approx(100.0 / 3.0)
    assert evaluator.value(
        SLAFilter(metric_tag="cancellation_rate", stat="avg", op="le", threshold=0.5),
        stats,
    ) == pytest.approx(0.25)


def test_error_rate_sla_matches_exported_metric_unit_and_denominator() -> None:
    """request_error_rate must be percentage points over completed requests.

    The exported metric is ``100 * errors / (request_count + errors)``; the
    adaptive-scale evaluator previously returned ``errors / total`` (a 0-1
    ratio whose denominator also included cancelled requests), so
    ``request_error_rate:avg:le:1`` allowed a 100% error rate instead of 1%.
    """
    evaluator = AdaptiveScaleSLAEvaluator()
    stats = WindowStats(
        samples=[100_000_000, 200_000_000],
        errors=1,
        cancelled=1,
        elapsed_sec=2.0,
    )

    for metric_tag in ("request_error_rate", "error_rate"):
        sla = SLAFilter(metric_tag=metric_tag, stat="avg", op="le", threshold=1)
        value = evaluator.value(sla, stats)

        # 100 * 1 / (2 successes + 1 error); the cancellation is excluded.
        assert value == pytest.approx(100.0 / 3.0)
        assert not evaluator.passes([sla], {evaluator.key(sla): value})


def test_missing_ttft_sample_fails_lower_is_better_sla() -> None:
    evaluator = AdaptiveScaleSLAEvaluator()
    sla = SLAFilter(
        metric_tag="time_to_first_token", stat="p95", op="le", threshold=25.0
    )
    stats = WindowStats(samples=[100_000_000], errors=0, elapsed_sec=1.0)
    observed = {evaluator.key(sla): evaluator.value(sla, stats)}

    assert observed[evaluator.key(sla)] == float("inf")
    assert not evaluator.passes([sla], observed)


def test_sla_evaluator_zero_total_rates_and_output_token_stat_errors() -> None:
    evaluator = AdaptiveScaleSLAEvaluator()
    empty_stats = WindowStats(samples=[], errors=0, cancelled=0, elapsed_sec=1.0)

    assert (
        evaluator.value(
            SLAFilter(metric_tag="success_rate", stat="avg", op="ge", threshold=0),
            empty_stats,
        )
        == 0.0
    )
    assert (
        evaluator.value(
            SLAFilter(metric_tag="request_error_rate", stat="min", op="le", threshold=1),
            empty_stats,
        )
        == 0.0
    )
    assert (
        evaluator.value(
            SLAFilter(metric_tag="cancellation_rate", stat="max", op="le", threshold=1),
            empty_stats,
        )
        == 0.0
    )
    assert evaluator.value(
        SLAFilter(
            metric_tag="output_token_throughput",
            stat="avg",
            op="ge",
            threshold=0,
        ),
        WindowStats(
            samples=[1],
            errors=0,
            elapsed_sec=2.0,
            successful_requests=[WindowRequestSample(1, output_sequence_length=8)],
        ),
    ) == pytest.approx(4.0)

    with pytest.raises(ValueError, match="Unsupported output_token_throughput"):
        evaluator.value(
            SLAFilter.model_construct(
                metric_tag="output_token_throughput",
                stat="p95",
                op="ge",
                threshold=1,
            ),
            empty_stats,
        )


def test_build_backend_rejects_invalid_bounds_and_unknown_variable(tmp_path) -> None:
    from aiperf.timing.strategies.adaptive_scale_backends import (
        build_adaptive_control_backend,
    )

    base = dict(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.ADAPTIVE_SCALE,
        expected_duration_sec=60.0,
        concurrency=10,
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        adaptive_sustain_duration_sec=10.0,
        adaptive_assessment_period_sec=1.0,
        adaptive_sla_filters=[
            SLAFilter(
                metric_tag="request_latency", stat="p95", op="le", threshold=100.0
            )
        ],
        artifact_dir=tmp_path,
    )

    with pytest.raises(Exception, match="must be > min"):
        build_adaptive_control_backend(
            strategy=MagicMock(),
            concurrency_manager=MagicMock(),
            config=CreditPhaseConfig(
                **base,
                adaptive_control_variable="concurrency",
                adaptive_control_min=10,
                adaptive_control_max=10,
            ),
        )

    config = MagicMock(adaptive_control_variable="tokens")

    with pytest.raises(ValueError, match="unsupported adaptive control variable"):
        build_adaptive_control_backend(
            strategy=MagicMock(),
            concurrency_manager=MagicMock(),
            config=config,
        )


def test_error_rate_sla_threshold_out_of_percentage_range_is_rejected() -> None:
    """error_rate thresholds are percentage points, so [0, 100] is the domain."""
    from aiperf.timing.strategies.adaptive_scale_sla import AdaptiveScaleSLAEvaluator

    evaluator = AdaptiveScaleSLAEvaluator()

    for threshold in (-1.0, 101.0):
        with pytest.raises(ValueError, match="percentage points"):
            evaluator.validate_single_filter(
                SLAFilter(
                    metric_tag="error_rate",
                    stat="avg",
                    op="le",
                    threshold=threshold,
                )
            )

    for threshold in (0.0, 100.0):
        evaluator.validate_single_filter(
            SLAFilter(
                metric_tag="request_error_rate",
                stat="avg",
                op="le",
                threshold=threshold,
            )
        )


def test_error_rate_sla_fraction_style_threshold_warns(caplog) -> None:
    """A pre-change fraction threshold such as 0.05 is accepted but flagged."""
    from aiperf.timing.strategies.adaptive_scale_sla import AdaptiveScaleSLAEvaluator

    evaluator = AdaptiveScaleSLAEvaluator()

    with caplog.at_level(logging.WARNING):
        evaluator.validate_single_filter(
            SLAFilter(metric_tag="error_rate", stat="avg", op="le", threshold=0.05)
        )

    assert "percentage points" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        evaluator.validate_single_filter(
            SLAFilter(metric_tag="error_rate", stat="avg", op="le", threshold=5.0)
        )
    assert caplog.text == ""
