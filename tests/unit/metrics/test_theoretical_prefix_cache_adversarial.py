# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for ``TheoreticalPrefixCacheAccumulator``."""

from __future__ import annotations

import asyncio

import pytest

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
from tests.unit.conftest import make_benchmark_run


def _accumulator() -> TheoreticalPrefixCacheAccumulator:
    return TheoreticalPrefixCacheAccumulator(
        make_benchmark_run(endpoint_type=EndpointType.CHAT)
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
    """Characterization: replaying the same (conversation, turn) N times keeps"""

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
    assert total_hits == 30
    assert total_blocks == 40
