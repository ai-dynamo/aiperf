# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the network-RTT-adjusted metric injection in MetricResultsProcessor.

These tests exercise the real MetricRegistry (not the mocked one) so that the
registered ``network_adjusted_*`` / ``network_rtt`` classes resolve and flow
through the normal ``_create_metric_result`` / ``to_display_unit`` path.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.common.models import MetricResult
from aiperf.metrics.metric_dicts import MetricArray
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
from aiperf.post_processors.metric_results_processor import MetricResultsProcessor

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


def _seed_array(processor: MetricResultsProcessor, tag: str, values_ns) -> None:
    """Seed a MetricArray of nanosecond values into the processor results dict."""
    array = MetricArray()
    array.extend(values_ns)
    processor._results[tag] = array


def _results_by_tag(results: list[MetricResult]) -> dict[str, MetricResult]:
    return {r.tag: r for r in results}


class TestNetworkAdjustedShift:
    """The adjustment subtracts a constant RTT, shifting every quantile uniformly."""

    @pytest.mark.asyncio
    async def test_adjusted_request_latency_every_stat_shifts_by_rtt(
        self, mock_run
    ) -> None:
        rtt_ns = 500_000.0  # 0.5 ms, below the minimum sample so no clamping
        processor = MetricResultsProcessor(mock_run)
        _seed_array(processor, RequestLatencyMetric.tag, _REQUEST_LATENCY_NS)
        processor.set_network_rtt_ns(rtt_ns)

        results = _results_by_tag(await processor.summarize())

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
        processor = MetricResultsProcessor(mock_run)
        _seed_array(processor, RequestLatencyMetric.tag, _REQUEST_LATENCY_NS)
        processor.set_network_rtt_ns(rtt_ns)

        results = _results_by_tag(await processor.summarize())

        assert results["network_adjusted_request_latency"].std == pytest.approx(
            results[RequestLatencyMetric.tag].std
        )

    @pytest.mark.asyncio
    async def test_adjusted_count_matches_source(self, mock_run) -> None:
        processor = MetricResultsProcessor(mock_run)
        _seed_array(processor, RequestLatencyMetric.tag, _REQUEST_LATENCY_NS)
        processor.set_network_rtt_ns(500_000.0)

        results = _results_by_tag(await processor.summarize())

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
        processor = MetricResultsProcessor(mock_run)
        _seed_array(processor, RequestLatencyMetric.tag, _REQUEST_LATENCY_NS)
        processor.set_network_rtt_ns(rtt_ns)

        results = _results_by_tag(await processor.summarize())
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
        processor = MetricResultsProcessor(mock_run)
        _seed_array(processor, RequestLatencyMetric.tag, _REQUEST_LATENCY_NS)
        _seed_array(processor, TTFTMetric.tag, [v / 2 for v in _REQUEST_LATENCY_NS])
        processor.set_network_rtt_ns(500_000.0)

        tags = {r.tag for r in await processor.summarize()}

        assert "network_adjusted_inter_token_latency" not in tags
        assert "network_adjusted_inter_chunk_latency" not in tags

    @pytest.mark.asyncio
    async def test_itl_algebraically_cancels_rtt(self, mock_run) -> None:
        """ITL = request_latency - ttft, so the subtracted RTT cancels exactly."""
        rtt_ns = 500_000.0
        ttft_ns = [v / 2 for v in _REQUEST_LATENCY_NS]
        processor = MetricResultsProcessor(mock_run)
        _seed_array(processor, RequestLatencyMetric.tag, _REQUEST_LATENCY_NS)
        _seed_array(processor, TTFTMetric.tag, ttft_ns)
        processor.set_network_rtt_ns(rtt_ns)

        results = _results_by_tag(await processor.summarize())

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

        baseline = MetricResultsProcessor(mock_run)
        _seed_array(baseline, RequestLatencyMetric.tag, _REQUEST_LATENCY_NS)
        _seed_array(baseline, TTFTMetric.tag, ttft_ns)
        baseline_results = _results_by_tag(await baseline.summarize())

        adjusted = MetricResultsProcessor(mock_run)
        _seed_array(adjusted, RequestLatencyMetric.tag, _REQUEST_LATENCY_NS)
        _seed_array(adjusted, TTFTMetric.tag, ttft_ns)
        adjusted.set_network_rtt_ns(500_000.0)
        adjusted_results = _results_by_tag(await adjusted.summarize())

        for tag in (RequestLatencyMetric.tag, TTFTMetric.tag):
            for attr in (*_SHIFT_ATTRS, "std", "count", "sum"):
                assert getattr(adjusted_results[tag], attr) == pytest.approx(
                    getattr(baseline_results[tag], attr)
                ), f"raw {tag}.{attr} was mutated by RTT injection"

    @pytest.mark.asyncio
    async def test_source_array_object_not_mutated_in_place(self, mock_run) -> None:
        processor = MetricResultsProcessor(mock_run)
        _seed_array(processor, RequestLatencyMetric.tag, _REQUEST_LATENCY_NS)
        original = np.array(processor._results[RequestLatencyMetric.tag].data)
        processor.set_network_rtt_ns(500_000.0)

        await processor.summarize()

        np.testing.assert_array_equal(
            processor._results[RequestLatencyMetric.tag].data, original
        )


class TestNetworkAdjustedNoOp:
    """No RTT set means no adjusted metrics are emitted."""

    @pytest.mark.asyncio
    async def test_rtt_never_set_emits_no_adjusted_rows(self, mock_run) -> None:
        processor = MetricResultsProcessor(mock_run)
        _seed_array(processor, RequestLatencyMetric.tag, _REQUEST_LATENCY_NS)

        tags = {r.tag for r in await processor.summarize()}

        assert not any(tag.startswith("network_adjusted_") for tag in tags)
        assert NetworkRttMetric.tag not in tags

    @pytest.mark.asyncio
    async def test_set_rtt_none_emits_no_adjusted_rows(self, mock_run) -> None:
        processor = MetricResultsProcessor(mock_run)
        _seed_array(processor, RequestLatencyMetric.tag, _REQUEST_LATENCY_NS)
        processor.set_network_rtt_ns(None)

        tags = {r.tag for r in await processor.summarize()}

        assert not any(tag.startswith("network_adjusted_") for tag in tags)
        assert NetworkRttMetric.tag not in tags


class TestNetworkRttSummary:
    """The network_rtt summary row reports the subtracted RTT in display units (ms)."""

    @pytest.mark.asyncio
    async def test_network_rtt_row_present_with_avg_in_ms(self, mock_run) -> None:
        rtt_ns = 750_000.0
        processor = MetricResultsProcessor(mock_run)
        _seed_array(processor, RequestLatencyMetric.tag, _REQUEST_LATENCY_NS)
        processor.set_network_rtt_ns(rtt_ns)

        results = _results_by_tag(await processor.summarize())

        assert NetworkRttMetric.tag in results
        net_rtt = results[NetworkRttMetric.tag]
        assert net_rtt.unit == "ms"
        assert net_rtt.avg == pytest.approx(rtt_ns / 1e6)


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
