# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.messages import MetricRecordsData
from aiperf.common.models import (
    Conversation,
    ConversationMetadata,
    DatasetMetadata,
    ErrorDetails,
    MetricRecordMetadata,
    Turn,
    TurnMetadata,
)
from aiperf.metrics.theoretical_prefix_cache import (
    THEORETICAL_PREFIX_CACHE_HIT_TAG,
    TheoreticalPrefixCacheAccumulator,
)
from aiperf.plugin.enums import DatasetSamplingStrategy, EndpointType
from tests.unit.conftest import make_benchmark_run


def _record(
    *,
    conversation_id: str,
    turn_index: int,
    error: ErrorDetails | None = None,
) -> MetricRecordsData:
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
        error=error,
    )


def test_turn_metadata_carries_theoretical_prefix_counts() -> None:
    turn = Turn(
        theoretical_prefix_cache_hit_blocks=3,
        theoretical_prefix_cache_total_blocks=5,
    )

    metadata = turn.metadata()

    assert metadata.theoretical_prefix_cache_hit_blocks == 3
    assert metadata.theoretical_prefix_cache_total_blocks == 5


def test_conversation_metadata_carries_theoretical_prefix_counts() -> None:
    conversation = Conversation(
        session_id="trace-a",
        turns=[
            Turn(
                theoretical_prefix_cache_hit_blocks=3,
                theoretical_prefix_cache_total_blocks=5,
            )
        ],
    )

    [metadata] = conversation.metadata().turns

    assert metadata.theoretical_prefix_cache_hit_blocks == 3
    assert metadata.theoretical_prefix_cache_total_blocks == 5


def test_accumulator_reports_cumulative_theoretical_prefix_hit_rate() -> None:
    asyncio.run(_run_accumulator_reports_cumulative_theoretical_prefix_hit_rate())


async def _run_accumulator_reports_cumulative_theoretical_prefix_hit_rate() -> None:
    acc = TheoreticalPrefixCacheAccumulator(
        make_benchmark_run(endpoint_type=EndpointType.CHAT)
    )
    acc.on_dataset_configured(
        DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="trace-a",
                    turns=[
                        TurnMetadata(
                            theoretical_prefix_cache_hit_blocks=0,
                            theoretical_prefix_cache_total_blocks=3,
                        ),
                        TurnMetadata(
                            theoretical_prefix_cache_hit_blocks=3,
                            theoretical_prefix_cache_total_blocks=4,
                        ),
                    ],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
    )

    await acc.process_record(_record(conversation_id="trace-a", turn_index=0))
    await acc.process_record(_record(conversation_id="trace-a", turn_index=1))

    [result] = await acc.summarize()
    assert result.tag == THEORETICAL_PREFIX_CACHE_HIT_TAG
    assert result.current == pytest.approx(100.0 * 3 / 7)
    assert result.avg == pytest.approx(100.0 * 3 / 7)
    assert result.count == 7
    assert result.sum == 3


def test_accumulator_skips_missing_metadata_and_errors() -> None:
    asyncio.run(_run_accumulator_skips_missing_metadata_and_errors())


async def _run_accumulator_skips_missing_metadata_and_errors() -> None:
    acc = TheoreticalPrefixCacheAccumulator(
        make_benchmark_run(endpoint_type=EndpointType.CHAT)
    )
    acc.on_dataset_configured(
        DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="trace-a",
                    turns=[
                        TurnMetadata(
                            theoretical_prefix_cache_hit_blocks=1,
                            theoretical_prefix_cache_total_blocks=2,
                        )
                    ],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
    )

    await acc.process_record(_record(conversation_id="missing", turn_index=0))
    await acc.process_record(
        _record(
            conversation_id="trace-a",
            turn_index=0,
            error=ErrorDetails(message="bad request"),
        )
    )

    assert await acc.summarize() == []
