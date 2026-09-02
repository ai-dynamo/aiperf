# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Field-level tests for ``ServerMetricsAccumulator.realtime_snapshot``.

Covers the per-field extraction helpers (SGLang counter/gauge fallbacks,
external prefix cache, KV cache gauges, HiCache host-tier ratio, queue depth)
and the windowed counter baselines used when ``start_ns`` is provided, plus
the ``AccumulatorProtocol`` aliases (``process_record`` / ``query_time_range``).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from pytest import param

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.accumulator import ServerMetricsAccumulator
from tests.unit.conftest import make_run_from_cli

_COUNTER = PrometheusMetricType.COUNTER
_GAUGE = PrometheusMetricType.GAUGE
_ENDPOINT = "http://node1:8081/metrics"
_NS = 1_000_000_000


def _accumulator() -> ServerMetricsAccumulator:
    return ServerMetricsAccumulator(
        run=make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                endpoint_type=EndpointType.CHAT,
                streaming=False,
            )
        )
    )


async def _feed(
    acc: ServerMetricsAccumulator,
    points: list[tuple[int, dict[str, float]]],
    types: dict[str, PrometheusMetricType],
    endpoint: str = _ENDPOINT,
) -> None:
    """Feed (timestamp_ns, {metric_name: value}) snapshots into one endpoint."""
    for ts_ns, values in points:
        families = {
            name: MetricFamily(
                type=types[name],
                description=name,
                samples=[MetricSample(value=float(v))],
            )
            for name, v in values.items()
        }
        await acc.process_server_metrics_record(
            ServerMetricsRecord(
                endpoint_url=endpoint,
                timestamp_ns=ts_ns,
                metrics=families,
            )
        )


def _run(coro) -> dict[str, float]:
    return asyncio.run(coro)


def test_realtime_snapshot_no_endpoints_returns_empty_dict() -> None:
    acc = _accumulator()
    assert acc.realtime_snapshot() == {}


def test_process_record_alias_and_query_time_range() -> None:
    async def body() -> ServerMetricsAccumulator:
        acc = _accumulator()
        # Empty accumulator -> empty boolean mask.
        empty = acc.query_time_range(0, 10 * _NS)
        assert empty.dtype == bool
        assert len(empty) == 0
        for ts in (0, 1 * _NS, 2 * _NS):
            await acc.process_record(
                ServerMetricsRecord(
                    endpoint_url=_ENDPOINT,
                    timestamp_ns=ts,
                    metrics={
                        "vllm:num_requests_running": MetricFamily(
                            type=_GAUGE,
                            description="running",
                            samples=[MetricSample(value=1.0)],
                        )
                    },
                )
            )
        return acc

    acc = asyncio.run(body())
    mask = acc.query_time_range(1 * _NS, 2 * _NS)
    assert mask.tolist() == [False, True, False]
    assert np.count_nonzero(acc.query_time_range(0, 3 * _NS)) == 3


def test_sglang_counter_pair_hit_rate() -> None:
    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [
                (0, {"sglang:cached_tokens": 0.0, "sglang:prompt_tokens": 0.0}),
                (_NS, {"sglang:cached_tokens": 30.0, "sglang:prompt_tokens": 120.0}),
            ],
            {"sglang:cached_tokens": _COUNTER, "sglang:prompt_tokens": _COUNTER},
        )
        return acc.realtime_snapshot()

    out = _run(body())
    assert out["prefix_cache_hit_rate"] == pytest.approx(25.0)
    assert out["unique_input_tokens_srv"] == pytest.approx(90.0)


def test_sglang_gauge_fallback_hit_rate() -> None:
    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [(0, {"sglang:cache_hit_rate": 0.42})],
            {"sglang:cache_hit_rate": _GAUGE},
        )
        return acc.realtime_snapshot()

    out = _run(body())
    assert out["prefix_cache_hit_rate"] == pytest.approx(42.0)


def test_external_prefix_cache_hit_rate() -> None:
    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [
                (
                    0,
                    {
                        "vllm:external_prefix_cache_hits": 0.0,
                        "vllm:external_prefix_cache_queries": 0.0,
                    },
                ),
                (
                    _NS,
                    {
                        "vllm:external_prefix_cache_hits": 40.0,
                        "vllm:external_prefix_cache_queries": 80.0,
                    },
                ),
            ],
            {
                "vllm:external_prefix_cache_hits": _COUNTER,
                "vllm:external_prefix_cache_queries": _COUNTER,
            },
        )
        return acc.realtime_snapshot()

    out = _run(body())
    assert out["external_prefix_cache_hit_rate"] == pytest.approx(50.0)


@pytest.mark.parametrize(
    "gauge_value,expected_pct",
    [
        param(0.5, 50.0, id="fraction_scaled_to_pct"),
        param(73.5, 73.5, id="already_pct_passthrough"),
    ],
)  # fmt: skip
def test_kv_cache_usage_pct(gauge_value: float, expected_pct: float) -> None:
    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [(0, {"vllm:kv_cache_usage_perc": gauge_value})],
            {"vllm:kv_cache_usage_perc": _GAUGE},
        )
        return acc.realtime_snapshot()

    out = _run(body())
    assert out["kv_cache_usage_pct"] == pytest.approx(expected_pct)


def test_cpu_kv_cache_from_sglang_hicache_ratio() -> None:
    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [
                (
                    0,
                    {
                        "sglang:hicache_host_used_tokens": 25.0,
                        "sglang:hicache_host_total_tokens": 100.0,
                    },
                )
            ],
            {
                "sglang:hicache_host_used_tokens": _GAUGE,
                "sglang:hicache_host_total_tokens": _GAUGE,
            },
        )
        return acc.realtime_snapshot()

    out = _run(body())
    assert out["cpu_kv_cache_usage_pct"] == pytest.approx(25.0)


def test_cpu_kv_cache_ratio_pairs_within_endpoint() -> None:
    """The used/total ratio must pair within each endpoint, taking the max."""

    async def body() -> dict[str, float]:
        acc = _accumulator()
        types = {
            "sglang:hicache_host_used_tokens": _GAUGE,
            "sglang:hicache_host_total_tokens": _GAUGE,
        }
        await _feed(
            acc,
            [
                (
                    0,
                    {
                        "sglang:hicache_host_used_tokens": 10.0,
                        "sglang:hicache_host_total_tokens": 100.0,
                    },
                )
            ],
            types,
            endpoint="http://node1:8081/metrics",
        )
        await _feed(
            acc,
            [
                (
                    0,
                    {
                        "sglang:hicache_host_used_tokens": 90.0,
                        "sglang:hicache_host_total_tokens": 100.0,
                    },
                )
            ],
            types,
            endpoint="http://node2:8081/metrics",
        )
        return acc.realtime_snapshot()

    out = _run(body())
    assert out["cpu_kv_cache_usage_pct"] == pytest.approx(90.0)


def test_queue_depth_running_and_waiting() -> None:
    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [
                (
                    0,
                    {
                        "vllm:num_requests_running": 3.0,
                        "vllm:num_requests_waiting": 7.0,
                    },
                )
            ],
            {
                "vllm:num_requests_running": _GAUGE,
                "vllm:num_requests_waiting": _GAUGE,
            },
        )
        return acc.realtime_snapshot()

    out = _run(body())
    assert out["num_running"] == pytest.approx(3.0)
    assert out["num_waiting"] == pytest.approx(7.0)


def test_counter_delta_with_start_ns_uses_pre_window_baseline() -> None:
    """With start_ns mid-series the delta baselines at the last pre-window sample."""

    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [
                (0, {"vllm:prefix_cache_hits": 0.0, "vllm:prefix_cache_queries": 0.0}),
                (
                    10 * _NS,
                    {
                        "vllm:prefix_cache_hits": 50.0,
                        "vllm:prefix_cache_queries": 100.0,
                    },
                ),
                (
                    20 * _NS,
                    {
                        "vllm:prefix_cache_hits": 75.0,
                        "vllm:prefix_cache_queries": 150.0,
                    },
                ),
            ],
            {
                "vllm:prefix_cache_hits": _COUNTER,
                "vllm:prefix_cache_queries": _COUNTER,
            },
        )
        return acc.realtime_snapshot(start_ns=15 * _NS)

    out = _run(body())
    # Baseline is the 10s sample: hits delta 25, queries delta 50 -> 50%.
    assert out["prefix_cache_hit_rate"] == pytest.approx(50.0)
    assert out["unique_input_tokens_srv"] == pytest.approx(25.0)


def test_counter_delta_start_after_all_samples_suppressed() -> None:
    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [
                (0, {"vllm:prefix_cache_hits": 0.0, "vllm:prefix_cache_queries": 0.0}),
                (
                    _NS,
                    {
                        "vllm:prefix_cache_hits": 50.0,
                        "vllm:prefix_cache_queries": 100.0,
                    },
                ),
            ],
            {
                "vllm:prefix_cache_hits": _COUNTER,
                "vllm:prefix_cache_queries": _COUNTER,
            },
        )
        return acc.realtime_snapshot(start_ns=5 * _NS)

    out = _run(body())
    assert "prefix_cache_hit_rate" not in out


def test_counter_rate_windowed_from_start_ns() -> None:
    """The rate baseline is the first sample AT/AFTER start_ns (not before it)."""

    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [
                (0, {"vllm:prompt_tokens": 0.0}),
                (10 * _NS, {"vllm:prompt_tokens": 100.0}),
                (20 * _NS, {"vllm:prompt_tokens": 300.0}),
            ],
            {"vllm:prompt_tokens": _COUNTER},
        )
        return acc.realtime_snapshot(start_ns=5 * _NS)

    out = _run(body())
    # Window runs 10s -> 20s: delta 200 tokens over 10 seconds.
    assert out["input_token_throughput_srv"] == pytest.approx(20.0)


def test_counter_rate_suppressed_when_no_two_point_window() -> None:
    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [
                (0, {"vllm:generation_tokens": 0.0}),
                (10 * _NS, {"vllm:generation_tokens": 100.0}),
            ],
            {"vllm:generation_tokens": _COUNTER},
        )
        # Baseline lands on the final sample -> no two-point window.
        return acc.realtime_snapshot(start_ns=15 * _NS)

    out = _run(body())
    assert "output_token_throughput_srv" not in out


def test_type_collisions_and_short_series_are_skipped() -> None:
    """Entries with mismatched metric types or too-few samples are ignored."""

    async def body() -> dict[str, float]:
        acc = _accumulator()
        # Counter-named metric stored as GAUGE (and vice versa) must be
        # skipped by the type filters rather than cross-contaminating.
        await _feed(
            acc,
            [
                (
                    0,
                    {
                        "vllm:prompt_tokens": 100.0,
                        "vllm:num_requests_running": 1.0,
                        "vllm:prefix_cache_hits": 10.0,
                    },
                ),
                (
                    _NS,
                    {
                        "vllm:prompt_tokens": 200.0,
                        "vllm:num_requests_running": 2.0,
                        "vllm:prefix_cache_hits": 20.0,
                    },
                ),
            ],
            {
                "vllm:prompt_tokens": _GAUGE,
                "vllm:num_requests_running": _COUNTER,
                "vllm:prefix_cache_hits": _GAUGE,
            },
        )
        # A single-sample counter is skipped by the rate loop.
        await _feed(
            acc,
            [(0, {"vllm:generation_tokens": 50.0})],
            {"vllm:generation_tokens": _COUNTER},
            endpoint="http://node2:8081/metrics",
        )
        return acc.realtime_snapshot()

    out = _run(body())
    assert "input_token_throughput_srv" not in out
    assert "output_token_throughput_srv" not in out
    assert "num_running" not in out
    assert "prefix_cache_hit_rate" not in out


def test_defensive_helpers_short_series_and_empty_gauges() -> None:
    """Static helpers guard single-sample series and empty gauge entries."""
    from aiperf.server_metrics.storage import (
        ScalarTimeSeries,
        ServerMetricEntry,
        ServerMetricKey,
        ServerMetricsTimeSeries,
    )

    single = ScalarTimeSeries()
    single.append(0, MetricSample(value=1.0))
    assert ServerMetricsAccumulator._counter_baseline_idx(single, 5) is None
    assert ServerMetricsAccumulator._counter_rate_baseline_idx(single, 5) is None

    # An endpoint holding an empty gauge series is skipped by the ratio scan.
    ts = ServerMetricsTimeSeries()
    key = ServerMetricKey.from_name_and_labels("sglang:hicache_host_used_tokens", None)
    ts.metrics[key] = ServerMetricEntry(
        metric_type=_GAUGE, description="empty", data=ScalarTimeSeries()
    )
    assert (
        ServerMetricsAccumulator._max_endpoint_gauge_ratio(
            [ts],
            "sglang:hicache_host_used_tokens",
            "sglang:hicache_host_total_tokens",
        )
        is None
    )
