# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pathological / adversarial probes of the server-metrics realtime snapshot,
``_to_pct`` normalization, multi-endpoint gauge mixing, the warmup-baseline
counter-rate window, and the theoretical prefix-cache accumulator.

These deliberately avoid the already-covered counter-reset / lagging-counter
families (negative or >100% realtime rates from a single-endpoint reset, and
the theoretical-cache ``hit_blocks > total_blocks`` clamp). The new targets:

* ``_counter_rate`` with a realtime ``start_ns`` builds the rate window from
  the *warmup* baseline timestamp (the last sample before ``start_ns``), so the
  denominator includes the entire warmup->start gap while the numerator only
  spans the profiling window -> a token throughput that understates the true
  in-window rate (accumulator.py:706-716).
* ``_to_pct`` boundary and the heuristic ``fraction <= 1.0`` misscale: a gauge
  that already reports a fraction *greater than* 1.0 (e.g. 1.5x oversubscribed
  KV usage) is passed through unscaled as ``1.5`` instead of ``150.0``
  (accumulator.py:601-604).
* ``_add_cpu_kv_cache_usage_pct`` reads SGLang ``hicache_host_used_tokens`` and
  ``hicache_host_total_tokens`` via two *independent* ``_gauge_latest_max``
  calls, so on a two-endpoint deployment the numerator can come from one node
  and the denominator from another -> a ratio that maps to no real node
  (accumulator.py:505-516).
* external prefix cache 0/0 suppression, SGLang/vLLM precedence, single-sample
  baselines, and theoretical-cache turn-index / conversation-id guards —
  characterizations of intended behavior.
"""

from __future__ import annotations

import asyncio

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import CreditPhase, PrometheusMetricType
from aiperf.common.messages import MetricRecordsData
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    MetricRecordMetadata,
    TurnMetadata,
)
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.metrics.theoretical_prefix_cache import TheoreticalPrefixCacheAccumulator
from aiperf.plugin.enums import DatasetSamplingStrategy, EndpointType
from aiperf.server_metrics.accumulator import ServerMetricsAccumulator

_COUNTER = PrometheusMetricType.COUNTER
_GAUGE = PrometheusMetricType.GAUGE


# =============================================================================
# Helpers
# =============================================================================


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
    endpoint_url: str,
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
                endpoint_url=endpoint_url,
                timestamp_ns=ts_ns,
                metrics=families,
            )
        )


def _theoretical_accumulator() -> TheoreticalPrefixCacheAccumulator:
    return TheoreticalPrefixCacheAccumulator(
        UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                streaming=False,
            )
        )
    )


def _record(
    *, conversation_id: str | None, turn_index: int | None
) -> MetricRecordsData:
    return MetricRecordsData(
        metadata=MetricRecordMetadata(
            session_num=turn_index or 0,
            request_start_ns=1000,
            request_end_ns=2000,
            conversation_id=conversation_id,
            turn_index=turn_index,
            record_processor_id="rp",
            benchmark_phase=CreditPhase.PROFILING,
            worker_id="worker",
        ),
        metrics={},
        error=None,
    )


# =============================================================================
# _counter_rate warmup-baseline window pathology
# =============================================================================


def test_input_token_throughput_warmup_gap_inflates_rate_window_understates() -> None:
    async def body() -> float:
        acc = _accumulator()
        # Warmup baseline at t=0 (0 tokens), idle for 10s, then the profiling
        # window: at t=10s still 0 tokens, at t=11s 1000 tokens. All 1000
        # tokens accrue inside the 1s profiling window -> true rate 1000 tok/s.
        await _feed(
            acc,
            "http://node1:8081/metrics",
            [
                (0, {"vllm:prompt_tokens": 0.0}),
                (10_000_000_000, {"vllm:prompt_tokens": 0.0}),
                (11_000_000_000, {"vllm:prompt_tokens": 1000.0}),
            ],
            {"vllm:prompt_tokens": _COUNTER},
        )
        return acc.realtime_snapshot(start_ns=10_000_000_000)[
            "input_token_throughput_srv"
        ]

    rate = asyncio.run(body())
    # 1000 tokens over the 1s profiling window = 1000 tok/s. The bug yields
    # ~90.9 tok/s (1000 / 11s) because the warmup gap is in the denominator.
    assert rate == pytest.approx(1000.0)


def test_counter_rate_without_start_uses_full_window_characterization() -> None:
    """Characterization: with no ``start_ns`` the baseline is index 0 and the
    rate spans the whole observed series. Same data as the warmup test gives
    1000 tokens / 11s ~= 90.9 tok/s, which is the *intended* full-run average
    rate when warmup is not being excluded."""

    async def body() -> float:
        acc = _accumulator()
        await _feed(
            acc,
            "http://node1:8081/metrics",
            [
                (0, {"vllm:prompt_tokens": 0.0}),
                (10_000_000_000, {"vllm:prompt_tokens": 0.0}),
                (11_000_000_000, {"vllm:prompt_tokens": 1000.0}),
            ],
            {"vllm:prompt_tokens": _COUNTER},
        )
        return acc.realtime_snapshot()["input_token_throughput_srv"]

    rate = asyncio.run(body())
    assert rate == pytest.approx(1000.0 / 11.0)


# =============================================================================
# _to_pct normalization heuristic
# =============================================================================


def test_kv_cache_usage_pct_fraction_at_one_maps_to_100_characterization() -> None:
    """Characterization: a KV-cache gauge of exactly 1.0 (a saturated fraction)
    is normalized to 100.0 by _to_pct's ``fraction <= 1.0`` branch
    (accumulator.py:604)."""

    async def body() -> float:
        acc = _accumulator()
        await _feed(
            acc,
            "http://node1:8081/metrics",
            [(1_000_000_000, {"vllm:kv_cache_usage_perc": 1.0})],
            {"vllm:kv_cache_usage_perc": _GAUGE},
        )
        return acc.realtime_snapshot()["kv_cache_usage_pct"]

    assert asyncio.run(body()) == pytest.approx(100.0)


def test_kv_cache_usage_pct_value_above_one_passes_through_characterization() -> None:
    """Characterization: _to_pct passes a value > 1.0 through unchanged.

    A KV-cache gauge reading of 1.5 stays 1.5 -- the ``fraction <= 1.0 ?
    x*100 : x`` heuristic treats anything > 1.0 as already a percentage. This
    is the accepted behavior: the usage gauges are 0-1 fractions in practice,
    and treating a > 1.0 reading as already-percent avoids double-scaling a
    backend that emits percentages. The cost is that a genuine oversubscription
    (1.5 meaning 150%) is not rescaled -- a known, accepted limitation of the
    unit-less heuristic. Pinned so a change to _to_pct's boundary is caught.
    """

    async def body() -> float:
        acc = _accumulator()
        await _feed(
            acc,
            "http://node1:8081/metrics",
            [(1_000_000_000, {"vllm:kv_cache_usage_perc": 1.5})],
            {"vllm:kv_cache_usage_perc": _GAUGE},
        )
        return acc.realtime_snapshot()["kv_cache_usage_pct"]

    assert asyncio.run(body()) == pytest.approx(1.5)


def test_sglang_cache_hit_rate_small_fraction_scaled_characterization() -> None:
    """Characterization: a small SGLang ``cache_hit_rate`` fraction of 0.005
    (0.5% hit rate) is correctly scaled to 0.5% by _to_pct. This documents that
    the heuristic IS correct for genuine 0-1 fractions; the ambiguity only bites
    inputs already expressed as percentages."""

    async def body() -> float:
        acc = _accumulator()
        await _feed(
            acc,
            "http://node1:8081/metrics",
            [(1_000_000_000, {"sglang:cache_hit_rate": 0.005})],
            {"sglang:cache_hit_rate": _GAUGE},
        )
        return acc.realtime_snapshot()["prefix_cache_hit_rate"]

    assert asyncio.run(body()) == pytest.approx(0.5)


# =============================================================================
# Multi-endpoint CPU KV cache ratio mixing
# =============================================================================


def test_cpu_kv_cache_ratio_mixes_numerator_and_denominator_across_endpoints() -> None:
    async def body() -> float:
        acc = _accumulator()
        # Node A: 900k/1M host tokens used = 90% (busy, small host cache).
        await _feed(
            acc,
            "http://nodeA:8081/metrics",
            [
                (
                    1_000_000_000,
                    {
                        "sglang:hicache_host_used_tokens": 900_000.0,
                        "sglang:hicache_host_total_tokens": 1_000_000.0,
                    },
                )
            ],
            {
                "sglang:hicache_host_used_tokens": _GAUGE,
                "sglang:hicache_host_total_tokens": _GAUGE,
            },
        )
        # Node B: 200k/5M host tokens used = 4% (idle, large host cache).
        await _feed(
            acc,
            "http://nodeB:8081/metrics",
            [
                (
                    1_000_000_000,
                    {
                        "sglang:hicache_host_used_tokens": 200_000.0,
                        "sglang:hicache_host_total_tokens": 5_000_000.0,
                    },
                )
            ],
            {
                "sglang:hicache_host_used_tokens": _GAUGE,
                "sglang:hicache_host_total_tokens": _GAUGE,
            },
        )
        return acc.realtime_snapshot()["cpu_kv_cache_usage_pct"]

    pct = asyncio.run(body())
    # Real per-node ratios are 90% and 4%. The mixed value (max used 900k from
    # A / max total 5M from B = 18%) is neither; assert it matches one real node.
    assert pct == pytest.approx(90.0) or pct == pytest.approx(4.0)


def test_kv_cache_usage_pct_takes_max_across_endpoints_characterization() -> None:
    """Characterization: ``kv_cache_usage_pct`` uses a single-name
    ``_gauge_latest_max``, so it correctly reports the busiest node's usage
    (no numerator/denominator mixing). Two nodes at 30% and 70% -> 70%."""

    async def body() -> float:
        acc = _accumulator()
        await _feed(
            acc,
            "http://nodeA:8081/metrics",
            [(1_000_000_000, {"vllm:kv_cache_usage_perc": 0.30})],
            {"vllm:kv_cache_usage_perc": _GAUGE},
        )
        await _feed(
            acc,
            "http://nodeB:8081/metrics",
            [(1_000_000_000, {"vllm:kv_cache_usage_perc": 0.70})],
            {"vllm:kv_cache_usage_perc": _GAUGE},
        )
        return acc.realtime_snapshot()["kv_cache_usage_pct"]

    assert asyncio.run(body()) == pytest.approx(70.0)


# =============================================================================
# external prefix cache 0/0 suppression and counter baseline edges
# =============================================================================


def test_external_prefix_cache_zero_queries_suppressed_characterization() -> None:
    """Characterization: external prefix cache hit rate is suppressed when the
    query-counter delta is 0 (offload=none nodes share the family with
    offload=cpu peers), avoiding a misleading ext_cache_hit=0.0% row
    (accumulator.py:486-487)."""

    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            "http://node1:8081/metrics",
            [
                (
                    0,
                    {
                        "vllm:external_prefix_cache_hits": 0.0,
                        "vllm:external_prefix_cache_queries": 0.0,
                    },
                ),
                (
                    1_000_000_000,
                    {
                        "vllm:external_prefix_cache_hits": 0.0,
                        "vllm:external_prefix_cache_queries": 0.0,
                    },
                ),
            ],
            {
                "vllm:external_prefix_cache_hits": _COUNTER,
                "vllm:external_prefix_cache_queries": _COUNTER,
            },
        )
        return acc.realtime_snapshot()

    out = asyncio.run(body())
    assert "external_prefix_cache_hit_rate" not in out


def test_counter_baseline_start_after_all_samples_suppresses_rate() -> None:
    """Characterization: when ``start_ns`` is after every sample,
    _counter_baseline_idx returns None (first_in_window >= len), so the
    throughput row is suppressed rather than emitting a garbage rate
    (accumulator.py:654-655)."""

    async def body() -> dict[str, float]:
        acc = _accumulator()
        await _feed(
            acc,
            "http://node1:8081/metrics",
            [
                (0, {"vllm:prompt_tokens": 0.0}),
                (1_000_000_000, {"vllm:prompt_tokens": 1000.0}),
            ],
            {"vllm:prompt_tokens": _COUNTER},
        )
        return acc.realtime_snapshot(start_ns=5_000_000_000)

    out = asyncio.run(body())
    assert "input_token_throughput_srv" not in out


def test_prefix_cache_prefers_vllm_counters_over_sglang_when_both_present() -> None:
    """Characterization: when both vLLM and SGLang counter pairs exist, the
    vLLM pair wins (checked first in _add_prefix_cache_hit_rate). Documents the
    precedence so a mixed-export deployment cannot silently flip backends."""

    async def body() -> float:
        acc = _accumulator()
        await _feed(
            acc,
            "http://node1:8081/metrics",
            [
                (
                    0,
                    {
                        "vllm:prefix_cache_hits": 0.0,
                        "vllm:prefix_cache_queries": 0.0,
                        "sglang:cached_tokens": 0.0,
                        "sglang:prompt_tokens": 0.0,
                    },
                ),
                (
                    1_000_000_000,
                    {
                        # vLLM: 80/100 = 80% (this should win)
                        "vllm:prefix_cache_hits": 80.0,
                        "vllm:prefix_cache_queries": 100.0,
                        # SGLang: 10/100 = 10% (must be ignored)
                        "sglang:cached_tokens": 10.0,
                        "sglang:prompt_tokens": 100.0,
                    },
                ),
            ],
            {
                "vllm:prefix_cache_hits": _COUNTER,
                "vllm:prefix_cache_queries": _COUNTER,
                "sglang:cached_tokens": _COUNTER,
                "sglang:prompt_tokens": _COUNTER,
            },
        )
        return acc.realtime_snapshot()["prefix_cache_hit_rate"]

    assert asyncio.run(body()) == pytest.approx(80.0)


# =============================================================================
# Theoretical prefix cache accumulator: turn-index / conversation-id guards
# =============================================================================


def test_theoretical_cache_out_of_range_turn_index_ignored_characterization() -> None:
    """Characterization: a record whose turn_index is beyond the loader's
    per-turn tuple is silently dropped (theoretical_prefix_cache.py:72). A trace
    that replays more turns than the loader stamped contributes nothing rather
    than raising IndexError."""

    async def body() -> list:
        acc = _theoretical_accumulator()
        acc.on_dataset_configured(
            DatasetMetadata(
                conversations=[
                    ConversationMetadata(
                        conversation_id="trace-a",
                        turns=[
                            TurnMetadata(
                                theoretical_prefix_cache_hit_blocks=2,
                                theoretical_prefix_cache_total_blocks=4,
                            )
                        ],
                    )
                ],
                sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            )
        )
        # Only turn 0 exists; turn 5 is out of range and must be ignored.
        await acc.process_record(_record(conversation_id="trace-a", turn_index=5))
        return await acc.summarize()

    assert asyncio.run(body()) == []


def test_theoretical_cache_negative_turn_index_ignored_characterization() -> None:
    """Characterization: a negative turn_index is guarded explicitly
    (``turn_index < 0``) so it cannot index from the end of the per-turn tuple
    (theoretical_prefix_cache.py:72)."""

    async def body() -> list:
        acc = _theoretical_accumulator()
        acc.on_dataset_configured(
            DatasetMetadata(
                conversations=[
                    ConversationMetadata(
                        conversation_id="trace-a",
                        turns=[
                            TurnMetadata(
                                theoretical_prefix_cache_hit_blocks=2,
                                theoretical_prefix_cache_total_blocks=4,
                            )
                        ],
                    )
                ],
                sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            )
        )
        await acc.process_record(_record(conversation_id="trace-a", turn_index=-1))
        return await acc.summarize()

    assert asyncio.run(body()) == []


def test_theoretical_cache_none_conversation_id_ignored_characterization() -> None:
    """Characterization: a record with conversation_id=None is dropped before
    any lookup (theoretical_prefix_cache.py:69), so single-turn synthetic
    records without conversation provenance never corrupt the hit accounting."""

    async def body() -> list:
        acc = _theoretical_accumulator()
        acc.on_dataset_configured(
            DatasetMetadata(
                conversations=[
                    ConversationMetadata(
                        conversation_id="trace-a",
                        turns=[
                            TurnMetadata(
                                theoretical_prefix_cache_hit_blocks=2,
                                theoretical_prefix_cache_total_blocks=4,
                            )
                        ],
                    )
                ],
                sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            )
        )
        await acc.process_record(_record(conversation_id=None, turn_index=0))
        return await acc.summarize()

    assert asyncio.run(body()) == []
