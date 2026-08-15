# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TheoreticalPrefixCacheAccumulator joining loader-stamped per-node/per-turn block counts into a cumulative hit percent."""

from __future__ import annotations

import pytest

from aiperf.common.enums import CreditPhase, GenericMetricUnit
from aiperf.common.models.dataset_models import (
    ConversationMetadata,
    DatasetMetadata,
    GraphDatasetMetadata,
    TurnMetadata,
)
from aiperf.metrics.theoretical_prefix_cache import (
    THEORETICAL_PREFIX_CACHE_HIT_TAG,
    TheoreticalPrefixCacheAccumulator,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from tests.unit.post_processors.conftest import (
    create_metric_metadata,
    create_metric_records_data,
)

# A 3-turn linear chain, hash_ids [1,2] / [1,2,3] / [1,2,3,4], block_size 64,
# local hash_id scope. Over ONE shared per-trace seen-set in time order:
#   turn 0: hit 0 / total 2   (seen={} -> {1,2})
#   turn 1: hit 2 / total 3   ([1,2] hit, 3 miss -> {1,2,3})
#   turn 2: hit 3 / total 4   ([1,2,3] hit, 4 miss -> {1,2,3,4})
# cumulative: hit 5 / total 9 -> 100 * 5/9 = 55.5555...%
_EXPECTED_PER_TURN = [(0, 2), (2, 3), (3, 4)]
_EXPECTED_HIT_PCT = 100.0 * 5 / 9

_BASE_TRACE = "trace_03_n3"


def _dataset_metadata(
    *,
    conversations: list[ConversationMetadata] | None = None,
    graph: GraphDatasetMetadata | None = None,
) -> DatasetMetadata:
    """Sequential-sampling DatasetMetadata carrying either per-turn or graph facets."""
    return DatasetMetadata(
        conversations=conversations or [],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        graph=graph,
    )


def _conversation_with_turns(
    conversation_id: str, per_turn: list[tuple[int, int]]
) -> ConversationMetadata:
    """ConversationMetadata whose turns carry (hit_blocks, total_blocks) stamps."""
    conv_meta = ConversationMetadata(conversation_id=conversation_id)
    conv_meta.turns = [
        TurnMetadata(
            theoretical_prefix_cache_hit_blocks=hit,
            theoretical_prefix_cache_total_blocks=total,
        )
        for hit, total in per_turn
    ]
    return conv_meta


def _turn_record(conversation_id: str, turn_index: int, *, overflow_skip: bool = False):
    """A PROFILING-phase metric record addressing one (conversation, turn) pair."""
    # conversation_id is the root-scope TEMPLATE id (instance identity rides
    # x_correlation_id, which this join never reads).
    metadata = create_metric_metadata(
        conversation_id=conversation_id,
        turn_index=turn_index,
        benchmark_phase=CreditPhase.PROFILING,
    )
    metadata.context_overflow_skip = overflow_skip
    return create_metric_records_data(metadata=metadata)


def _accumulator(mock_run) -> TheoreticalPrefixCacheAccumulator:
    """Accumulator under test, bound to the shared mock run."""
    return TheoreticalPrefixCacheAccumulator(run=mock_run, service_id="proc-1")


@pytest.mark.asyncio
async def test_accumulator_emits_cumulative_hit_rate(mock_run) -> None:
    """on_dataset_configured + process_record + summarize yields the cumulative hit percent."""
    acc = _accumulator(mock_run)
    acc.on_dataset_configured(
        _dataset_metadata(
            conversations=[_conversation_with_turns(_BASE_TRACE, _EXPECTED_PER_TURN)]
        )
    )

    for turn_index in range(len(_EXPECTED_PER_TURN)):
        await acc.process_record(_turn_record(_BASE_TRACE, turn_index))

    results = await acc.summarize()
    assert len(results) == 1
    result = results[0]
    assert result.tag == THEORETICAL_PREFIX_CACHE_HIT_TAG
    assert result.unit == str(GenericMetricUnit.PERCENT)
    assert result.avg == pytest.approx(_EXPECTED_HIT_PCT)
    assert result.sum == 5
    assert result.count == 9
    # Finite-discipline: the emitted value must be a finite percent in [0, 100].
    assert 0.0 <= result.avg <= 100.0


@pytest.mark.asyncio
async def test_accumulator_node_map_path_matches_graph_dispatch(mock_run) -> None:
    """Graph facet path: the per-node map is keyed by node id recovered from (conversation_id, turn_index)."""
    acc = _accumulator(mock_run)
    acc.on_dataset_configured(
        _dataset_metadata(
            graph=GraphDatasetMetadata(
                trace_ids=[_BASE_TRACE],
                prefix_cache_by_trace={
                    _BASE_TRACE: {
                        f"{_BASE_TRACE}:{i}": list(counts)
                        for i, counts in enumerate(_EXPECTED_PER_TURN)
                    }
                },
            ),
        )
    )

    for turn_index in range(len(_EXPECTED_PER_TURN)):
        await acc.process_record(_turn_record(_BASE_TRACE, turn_index))

    results = await acc.summarize()
    assert len(results) == 1
    assert results[0].avg == pytest.approx(_EXPECTED_HIT_PCT)
    assert results[0].sum == 5
    assert results[0].count == 9


@pytest.mark.asyncio
async def test_accumulator_no_metadata_emits_nothing(mock_run) -> None:
    """A plain dataset with no prefix-cache metadata summarizes to nothing, not NaN."""
    acc = _accumulator(mock_run)
    acc.on_dataset_configured(
        _dataset_metadata(conversations=[ConversationMetadata(conversation_id="c0")])
    )
    await acc.process_record(_turn_record("c0", 0))
    assert await acc.summarize() == []


@pytest.mark.asyncio
async def test_accumulator_skips_context_overflow_records(mock_run) -> None:
    """An overflow-skip record must not fold its planned blocks into the hit rate."""
    # The RecordsManager trims the error off an overflow-skip record, so it
    # arrives valid=True and only context_overflow_skip distinguishes it.
    acc = _accumulator(mock_run)
    acc.on_dataset_configured(
        _dataset_metadata(
            graph=GraphDatasetMetadata(
                trace_ids=[_BASE_TRACE],
                prefix_cache_by_trace={_BASE_TRACE: {f"{_BASE_TRACE}:0": [3, 4]}},
            ),
        )
    )

    overflow_record = _turn_record(_BASE_TRACE, 0, overflow_skip=True)
    assert overflow_record.valid
    await acc.process_record(overflow_record)
    assert await acc.summarize() == []

    await acc.process_record(_turn_record(_BASE_TRACE, 0, overflow_skip=False))
    results = await acc.summarize()
    assert results[0].sum == 3
    assert results[0].count == 4


@pytest.mark.asyncio
async def test_accumulator_clamps_overcounted_hits(mock_run) -> None:
    """A loader miscount (hit > total) is clamped so the rate stays at most 100%."""
    acc = _accumulator(mock_run)
    acc.on_dataset_configured(
        _dataset_metadata(conversations=[_conversation_with_turns("c0", [(10, 4)])])
    )
    await acc.process_record(_turn_record("c0", 0))
    results = await acc.summarize()
    assert results[0].avg == pytest.approx(100.0)
