# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for ``TheoreticalPrefixCacheAccumulator``.

The accumulator sums loader-provided ``(hit_blocks, total_blocks)`` per turn
and reports ``100 * hit_blocks / total_blocks`` (theoretical_prefix_cache.py:
77-87). It guards ``total_blocks <= 0`` but never enforces the
``hit_blocks <= total_blocks`` invariant, so a loader that violates it (a
miscount in WEKA prefix accounting) produces a hit rate > 100% with no clamp
and no warning -- a metric a reader will treat as a percentage.
"""

from __future__ import annotations

import asyncio

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.messages import MetricRecordsData
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    MetricRecordMetadata,
    TurnMetadata,
)
from aiperf.metrics.theoretical_prefix_cache import TheoreticalPrefixCacheAccumulator
from aiperf.plugin.enums import DatasetSamplingStrategy, EndpointType


def _accumulator() -> TheoreticalPrefixCacheAccumulator:
    return TheoreticalPrefixCacheAccumulator(
        UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                streaming=False,
            )
        )
    )


def _record(*, conversation_id: str, turn_index: int) -> MetricRecordsData:
    return MetricRecordsData(
        metadata=MetricRecordMetadata(
            session_num=turn_index,
            request_start_ns=1000 + turn_index,
            request_end_ns=2000 + turn_index,
            conversation_id=conversation_id,
            turn_index=turn_index,
            record_processor_id="rp",
            benchmark_phase=CreditPhase.PROFILING,
            worker_id="worker",
        ),
        metrics={},
        error=None,
    )


def test_hit_rate_clamped_when_hit_blocks_exceeds_total() -> None:
    async def body() -> float:
        acc = _accumulator()
        acc.on_dataset_configured(
            DatasetMetadata(
                conversations=[
                    ConversationMetadata(
                        conversation_id="trace-a",
                        turns=[
                            TurnMetadata(
                                theoretical_prefix_cache_hit_blocks=10,
                                theoretical_prefix_cache_total_blocks=8,
                            )
                        ],
                    )
                ],
                sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            )
        )
        await acc.process_record(_record(conversation_id="trace-a", turn_index=0))
        [result] = await acc.summarize()
        return result.current

    current = asyncio.run(body())
    assert current <= 100.0


def test_rate_preserved_across_repeated_replays() -> None:
    """Characterization: replaying the same (conversation, turn) N times keeps
    the *rate* stable (numerator and denominator both scale), even though the
    absolute block counts inflate. Confirms the metric is replay-count
    invariant, the property agentic recycle relies on.
    """

    async def body() -> tuple[float, int, int]:
        acc = _accumulator()
        acc.on_dataset_configured(
            DatasetMetadata(
                conversations=[
                    ConversationMetadata(
                        conversation_id="trace-a",
                        turns=[
                            TurnMetadata(
                                theoretical_prefix_cache_hit_blocks=3,
                                theoretical_prefix_cache_total_blocks=4,
                            )
                        ],
                    )
                ],
                sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            )
        )
        for _ in range(10):
            await acc.process_record(_record(conversation_id="trace-a", turn_index=0))
        [result] = await acc.summarize()
        return result.current, int(result.sum), int(result.count)

    current, total_hits, total_blocks = asyncio.run(body())
    assert current == pytest.approx(75.0)
    assert total_hits == 30  # 3 * 10 replays
    assert total_blocks == 40  # 4 * 10 replays
