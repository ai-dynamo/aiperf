# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for ``ServerMetricsAccumulator.realtime_snapshot``.

The realtime ``srv`` row reads Prometheus counter pairs via ``_counter_delta``
and ``_counter_rate`` (accumulator.py:606-716). Neither helper clamps negative
deltas, unlike the final-export path which explicitly does
``max(raw_delta, 0.0)`` on counter resets (export_stats.py:354). A server
restart mid-run (counters reset to ~0) therefore makes the realtime row emit:

* a NEGATIVE ``prefix_cache_hit_rate``,
* NEGATIVE ``input_token_throughput_srv`` / ``output_token_throughput_srv``,

and a query counter that lags a batched hits update can make the hit rate
exceed 100%. None of these are physically valid percentages/rates.
"""

from __future__ import annotations

import asyncio

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.accumulator import ServerMetricsAccumulator

_COUNTER = PrometheusMetricType.COUNTER
_ENDPOINT = "http://node1:8081/metrics"


def _accumulator() -> ServerMetricsAccumulator:
    return ServerMetricsAccumulator(
        UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                streaming=False,
            )
        )
    )


async def _feed(
    acc: ServerMetricsAccumulator,
    points: list[tuple[int, dict[str, float]]],
    types: dict[str, PrometheusMetricType],
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
                endpoint_url=_ENDPOINT,
                timestamp_ns=ts_ns,
                metrics=families,
            )
        )


def _run(coro) -> dict[str, float]:
    return asyncio.run(coro)


def test_normal_monotonic_counters_give_valid_hit_rate() -> None:
    """Baseline: monotonic counters produce a sane 0-100% hit rate."""

    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [
                (0, {"vllm:prefix_cache_hits": 0.0, "vllm:prefix_cache_queries": 0.0}),
                (
                    1_000_000_000,
                    {
                        "vllm:prefix_cache_hits": 75.0,
                        "vllm:prefix_cache_queries": 100.0,
                    },
                ),
            ],
            {
                "vllm:prefix_cache_hits": _COUNTER,
                "vllm:prefix_cache_queries": _COUNTER,
            },
        )
        return acc.realtime_snapshot()

    out = _run(body())
    assert out["prefix_cache_hit_rate"] == pytest.approx(75.0)
    assert out["unique_input_tokens_srv"] == pytest.approx(25.0)


def test_hit_rate_negative_on_counter_reset() -> None:
    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [
                (
                    0,
                    {
                        "vllm:prefix_cache_hits": 100.0,
                        "vllm:prefix_cache_queries": 200.0,
                    },
                ),
                # Server restarted: counters reset far below prior values.
                (
                    1_000_000_000,
                    {
                        "vllm:prefix_cache_hits": 5.0,
                        "vllm:prefix_cache_queries": 300.0,
                    },
                ),
            ],
            {
                "vllm:prefix_cache_hits": _COUNTER,
                "vllm:prefix_cache_queries": _COUNTER,
            },
        )
        return acc.realtime_snapshot()

    out = _run(body())
    assert out["prefix_cache_hit_rate"] >= 0.0


def test_input_token_throughput_negative_on_counter_reset() -> None:
    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [
                (0, {"vllm:prompt_tokens": 1_000_000.0}),
                (1_000_000_000, {"vllm:prompt_tokens": 50.0}),  # restart
            ],
            {"vllm:prompt_tokens": _COUNTER},
        )
        return acc.realtime_snapshot()

    out = _run(body())
    assert out["input_token_throughput_srv"] >= 0.0


def test_hit_rate_can_exceed_100_with_lagging_query_counter() -> None:
    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            [
                (0, {"vllm:prefix_cache_hits": 0.0, "vllm:prefix_cache_queries": 0.0}),
                (
                    1_000_000_000,
                    {
                        "vllm:prefix_cache_hits": 120.0,
                        "vllm:prefix_cache_queries": 100.0,
                    },
                ),
            ],
            {
                "vllm:prefix_cache_hits": _COUNTER,
                "vllm:prefix_cache_queries": _COUNTER,
            },
        )
        return acc.realtime_snapshot()

    out = _run(body())
    assert out["prefix_cache_hit_rate"] <= 100.0
