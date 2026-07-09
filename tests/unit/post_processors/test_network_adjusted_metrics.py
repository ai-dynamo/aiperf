# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for network-RTT-adjusted metrics injected by MetricsAccumulator."""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import MetricResult
from aiperf.metrics.accumulator import MetricsAccumulator
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.network_adjusted_metrics import (
    NETWORK_ADJUSTED_SOURCES,
    NetworkRttMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.time_to_first_output_token_metric import (
    TimeToFirstOutputTokenMetric,
)
from aiperf.metrics.types.ttft_metric import TTFTMetric
from tests.unit.post_processors.conftest import create_metric_records_message

_PERCENTILE_ATTRS = ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99")
_SHIFT_ATTRS = ("min", "max", "avg", *_PERCENTILE_ATTRS)

# Known request_latency distribution in nanoseconds (1ms .. 10ms).
_REQUEST_LATENCY_NS = [
    1_000_000.0,
    2_000_000.0,
    3_000_000.0,
    4_000_000.0,
    5_000_000.0,
    6_000_000.0,
    7_000_000.0,
    8_000_000.0,
    9_000_000.0,
    10_000_000.0,
]


async def _seed_records(
    accumulator: MetricsAccumulator,
    series_by_tag: dict[str, list[float]],
) -> None:
    """Seed equal-length per-record metric series into the accumulator."""
    length = len(next(iter(series_by_tag.values())))
    for idx in range(length):
        metrics = {tag: values[idx] for tag, values in series_by_tag.items()}
        await accumulator.process_record(
            create_metric_records_message(
                x_request_id=f"r-{idx}",
                request_start_ns=1_000_000_000 + idx,
                request_end_ns=1_100_000_000 + idx,
                results=[metrics],
            ).to_data()
        )


def _results_by_tag(results: dict[str, MetricResult]) -> dict[str, MetricResult]:
    return results


class TestNetworkAdjustedShift:
    """The adjustment subtracts a constant RTT, shifting every quantile uniformly."""

    @pytest.mark.asyncio
    async def test_adjusted_request_latency_every_stat_shifts_by_rtt(
        self, mock_run
    ) -> None:
        rtt_ns = 500_000.0  # 0.5 ms, below the minimum sample so no clamping
        accumulator = MetricsAccumulator(mock_run)
        await _seed_records(
            accumulator, {RequestLatencyMetric.tag: _REQUEST_LATENCY_NS}
        )
        accumulator.set_network_rtt_ns(rtt_ns)

        results = _results_by_tag((await accumulator.summarize()).results)

        raw = results[RequestLatencyMetric.tag]
        adjusted = results["network_adjusted_request_latency"]
        rtt_ms = rtt_ns / 1e6

        for attr in _SHIFT_ATTRS:
            assert getattr(adjusted, attr) == pytest.approx(
                getattr(raw, attr) - rtt_ms
            ), f"{attr} did not shift by exactly rtt_ms"

    @pytest.mark.asyncio
    async def test_adjusted_request_latency_std_unchanged(self, mock_run) -> None:
        rtt_ns = 500_000.0
        accumulator = MetricsAccumulator(mock_run)
        await _seed_records(
            accumulator, {RequestLatencyMetric.tag: _REQUEST_LATENCY_NS}
        )
        accumulator.set_network_rtt_ns(rtt_ns)

        results = _results_by_tag((await accumulator.summarize()).results)

        assert results["network_adjusted_request_latency"].std == pytest.approx(
            results[RequestLatencyMetric.tag].std
        )

    @pytest.mark.asyncio
    async def test_adjusted_count_matches_source(self, mock_run) -> None:
        accumulator = MetricsAccumulator(mock_run)
        await _seed_records(
            accumulator, {RequestLatencyMetric.tag: _REQUEST_LATENCY_NS}
        )
        accumulator.set_network_rtt_ns(500_000.0)

        results = _results_by_tag((await accumulator.summarize()).results)

        assert (
            results["network_adjusted_request_latency"].count
            == results[RequestLatencyMetric.tag].count
            == len(_REQUEST_LATENCY_NS)
        )


class TestNetworkAdjustedClamp:
    """RTT larger than some samples clamps the adjusted distribution at 0."""

    @pytest.mark.asyncio
    async def test_rtt_exceeds_some_samples_floors_at_zero(self, mock_run) -> None:
        # RTT of 3.5 ms exceeds the three smallest samples (1, 2, 3 ms).
        rtt_ns = 3_500_000.0
        accumulator = MetricsAccumulator(mock_run)
        await _seed_records(
            accumulator, {RequestLatencyMetric.tag: _REQUEST_LATENCY_NS}
        )
        accumulator.set_network_rtt_ns(rtt_ns)

        results = _results_by_tag((await accumulator.summarize()).results)
        adjusted = results["network_adjusted_request_latency"]

        for attr in _SHIFT_ATTRS:
            assert getattr(adjusted, attr) >= 0.0, f"{attr} went negative after clamp"
        assert adjusted.min == pytest.approx(0.0)
        assert adjusted.p1 == pytest.approx(0.0)


class TestNetworkAdjustedInterTokenInvariance:
    """The headline property: ITL is network-invariant and must NOT be adjusted."""

    @pytest.mark.asyncio
    async def test_no_network_adjusted_inter_token_latency_tag_emitted(
        self, mock_run
    ) -> None:
        accumulator = MetricsAccumulator(mock_run)
        await _seed_records(
            accumulator,
            {
                RequestLatencyMetric.tag: _REQUEST_LATENCY_NS,
                TTFTMetric.tag: [v / 2 for v in _REQUEST_LATENCY_NS],
            },
        )
        accumulator.set_network_rtt_ns(500_000.0)

        tags = set((await accumulator.summarize()).results)

        assert "network_adjusted_inter_token_latency" not in tags
        assert "network_adjusted_inter_chunk_latency" not in tags

    @pytest.mark.asyncio
    async def test_itl_algebraically_cancels_rtt(self, mock_run) -> None:
        """ITL = request_latency - ttft, so the subtracted RTT cancels exactly."""
        rtt_ns = 500_000.0
        ttft_ns = [v / 2 for v in _REQUEST_LATENCY_NS]
        accumulator = MetricsAccumulator(mock_run)
        await _seed_records(
            accumulator,
            {
                RequestLatencyMetric.tag: _REQUEST_LATENCY_NS,
                TTFTMetric.tag: ttft_ns,
            },
        )
        accumulator.set_network_rtt_ns(rtt_ns)

        results = _results_by_tag((await accumulator.summarize()).results)

        adj_rl = results["network_adjusted_request_latency"].avg
        adj_ttft = results["network_adjusted_time_to_first_token"].avg
        raw_rl = results[RequestLatencyMetric.tag].avg
        raw_ttft = results[TTFTMetric.tag].avg

        assert (adj_rl - adj_ttft) == pytest.approx(raw_rl - raw_ttft)


class TestNetworkAdjustedNonDestructive:
    """Setting the RTT must never mutate the raw source metric results."""

    @pytest.mark.asyncio
    async def test_raw_metrics_identical_with_and_without_rtt(self, mock_run) -> None:
        ttft_ns = [v / 2 for v in _REQUEST_LATENCY_NS]

        baseline = MetricsAccumulator(mock_run)
        await _seed_records(
            baseline,
            {
                RequestLatencyMetric.tag: _REQUEST_LATENCY_NS,
                TTFTMetric.tag: ttft_ns,
            },
        )
        baseline_results = _results_by_tag((await baseline.summarize()).results)

        adjusted = MetricsAccumulator(mock_run)
        await _seed_records(
            adjusted,
            {
                RequestLatencyMetric.tag: _REQUEST_LATENCY_NS,
                TTFTMetric.tag: ttft_ns,
            },
        )
        adjusted.set_network_rtt_ns(500_000.0)
        adjusted_results = _results_by_tag((await adjusted.summarize()).results)

        for tag in (RequestLatencyMetric.tag, TTFTMetric.tag):
            for attr in (*_SHIFT_ATTRS, "std", "count", "sum"):
                assert getattr(adjusted_results[tag], attr) == pytest.approx(
                    getattr(baseline_results[tag], attr)
                ), f"raw {tag}.{attr} was mutated by RTT injection"

    @pytest.mark.asyncio
    async def test_source_column_not_mutated_in_place(self, mock_run) -> None:
        accumulator = MetricsAccumulator(mock_run)
        await _seed_records(
            accumulator, {RequestLatencyMetric.tag: _REQUEST_LATENCY_NS}
        )
        original = accumulator.column_store.numeric(RequestLatencyMetric.tag).copy()
        accumulator.set_network_rtt_ns(500_000.0)

        await accumulator.summarize()

        np.testing.assert_array_equal(
            accumulator.column_store.numeric(RequestLatencyMetric.tag), original
        )


class TestNetworkAdjustedNoOp:
    """No RTT set means no adjusted metrics are emitted."""

    @pytest.mark.asyncio
    async def test_rtt_never_set_emits_no_adjusted_rows(self, mock_run) -> None:
        accumulator = MetricsAccumulator(mock_run)
        await _seed_records(
            accumulator, {RequestLatencyMetric.tag: _REQUEST_LATENCY_NS}
        )

        tags = set((await accumulator.summarize()).results)

        assert not any(tag.startswith("network_adjusted_") for tag in tags)
        assert NetworkRttMetric.tag not in tags

    @pytest.mark.asyncio
    async def test_set_rtt_none_emits_no_adjusted_rows(self, mock_run) -> None:
        accumulator = MetricsAccumulator(mock_run)
        await _seed_records(
            accumulator, {RequestLatencyMetric.tag: _REQUEST_LATENCY_NS}
        )
        accumulator.set_network_rtt_ns(None)

        tags = set((await accumulator.summarize()).results)

        assert not any(tag.startswith("network_adjusted_") for tag in tags)
        assert NetworkRttMetric.tag not in tags


class TestNetworkRttSummary:
    """The network_rtt summary row reports the subtracted RTT in display units (ms)."""

    @pytest.mark.asyncio
    async def test_network_rtt_row_present_with_avg_in_ms(self, mock_run) -> None:
        rtt_ns = 750_000.0
        accumulator = MetricsAccumulator(mock_run)
        await _seed_records(
            accumulator, {RequestLatencyMetric.tag: _REQUEST_LATENCY_NS}
        )
        accumulator.set_network_rtt_ns(rtt_ns)

        results = _results_by_tag((await accumulator.summarize()).results)

        assert NetworkRttMetric.tag in results
        net_rtt = results[NetworkRttMetric.tag]
        assert net_rtt.unit == "ms"
        assert net_rtt.avg == pytest.approx(rtt_ns / 1e6)


class TestNetworkAdjustedTimeslices:
    """Timeslice summaries get the same RTT adjustment as the overall summary."""

    @pytest.mark.asyncio
    async def test_timeslice_adjusted_request_latency_uses_slice_mask(
        self, mock_run
    ) -> None:
        mock_run.cfg.artifacts.slice_duration = 1.0
        rtt_ns = 500_000.0
        accumulator = MetricsAccumulator(mock_run)
        records = [
            (0, 0.1, 0.2, 2_000_000.0),
            (1, 0.2, 0.3, 4_000_000.0),
            (2, 1.2, 1.3, 8_000_000.0),
        ]
        for session_num, start_s, end_s, request_latency_ns in records:
            await accumulator.process_record(
                create_metric_records_message(
                    x_request_id=f"r-{session_num}",
                    session_num=session_num,
                    request_start_ns=int(start_s * NANOS_PER_SECOND),
                    request_end_ns=int(end_s * NANOS_PER_SECOND),
                    results=[
                        {
                            RequestLatencyMetric.tag: request_latency_ns,
                            TTFTMetric.tag: request_latency_ns / 2,
                            TimeToFirstOutputTokenMetric.tag: request_latency_ns / 2,
                        }
                    ],
                ).to_data()
            )
        accumulator.set_network_rtt_ns(rtt_ns)

        summary = await accumulator.summarize()

        assert summary.timeslices is not None
        assert len(summary.timeslices) == 2
        ts0 = summary.timeslices[0].metric_results
        ts1 = summary.timeslices[1].metric_results
        assert ts0[RequestLatencyMetric.tag].avg == pytest.approx(3.0)
        assert ts0["network_adjusted_request_latency"].count == 2
        assert ts0["network_adjusted_request_latency"].avg == pytest.approx(2.5)
        assert ts0[NetworkRttMetric.tag].avg == pytest.approx(0.5)
        assert ts1[RequestLatencyMetric.tag].avg == pytest.approx(8.0)
        assert ts1["network_adjusted_request_latency"].count == 1
        assert ts1["network_adjusted_request_latency"].avg == pytest.approx(7.5)
        assert ts1[NetworkRttMetric.tag].avg == pytest.approx(0.5)


class TestNetworkAdjustedRegistry:
    """All injected tags must be registered in the real MetricRegistry."""

    @pytest.mark.parametrize(
        "tag",
        [
            "network_adjusted_request_latency",
            "network_adjusted_time_to_first_token",
            "network_adjusted_time_to_first_output_token",
            "network_rtt",
        ],
    )
    def test_tag_resolves_in_registry(self, tag: str) -> None:
        assert tag in MetricRegistry.all_tags()
        assert MetricRegistry.get_class(tag).tag == tag

    def test_time_to_second_token_is_not_adjusted(self) -> None:
        # TTST is an intra-stream gap (second_response - first_response), not
        # request-start-anchored, so it does not carry the network RTT.
        assert "network_adjusted_time_to_second_token" not in MetricRegistry.all_tags()
        assert "network_adjusted_time_to_second_token" not in NETWORK_ADJUSTED_SOURCES

    def test_network_adjusted_sources_map_to_registered_metrics(self) -> None:
        expected = {
            "network_adjusted_request_latency": RequestLatencyMetric.tag,
            "network_adjusted_time_to_first_token": TTFTMetric.tag,
            "network_adjusted_time_to_first_output_token": (
                TimeToFirstOutputTokenMetric.tag
            ),
        }
        assert expected == NETWORK_ADJUSTED_SOURCES
        for adjusted_tag, source_tag in NETWORK_ADJUSTED_SOURCES.items():
            assert adjusted_tag in MetricRegistry.all_tags()
            assert source_tag in MetricRegistry.all_tags()
