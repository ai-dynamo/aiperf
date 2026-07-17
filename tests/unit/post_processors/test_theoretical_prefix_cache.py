# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for theoretical prefix-cache accounting.

Covers the WEKA graph-IR pre-pass that stamps per-turn hit/total block counts
and the ``TheoreticalPrefixCacheAccumulator`` results processor that emits the
cumulative ``theoretical_prefix_cache_hit`` percent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.enums import CreditPhase, GenericMetricUnit
from aiperf.common.models.dataset_models import (
    ConversationMetadata,
    DatasetMetadata,
    GraphDatasetMetadata,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.post_processors.theoretical_prefix_cache import (
    THEORETICAL_PREFIX_CACHE_HIT_TAG,
    TheoreticalPrefixCacheAccumulator,
)
from tests.unit.post_processors.conftest import (
    create_metric_metadata,
    create_metric_records_data,
)

# weka_min.json: a 3-turn linear chain, hash_ids [1,2] / [1,2,3] / [1,2,3,4],
# times 0.0 / 1.5 / 3.0, block_size 64, hash_id_scope local. Over ONE shared
# per-trace seen-set consumed in time order:
#   turn 0: hit 0 / total 2   (seen={} -> {1,2})
#   turn 1: hit 2 / total 3   ([1,2] hit, 3 miss -> {1,2,3})
#   turn 2: hit 3 / total 4   ([1,2,3] hit, 4 miss -> {1,2,3,4})
# cumulative: hit 5 / total 9 -> 100 * 5/9 = 55.5555...%
_WEKA_MIN = Path(__file__).parents[1] / "graph" / "fixtures" / "weka_min.json"
_EXPECTED_PER_TURN = [(0, 2), (2, 3), (3, 4)]
_EXPECTED_HIT_PCT = 100.0 * 5 / 9


@pytest.mark.asyncio
async def test_accumulator_emits_cumulative_hit_rate(mock_run) -> None:
    """on_dataset_configured + process_record + summarize -> hit-rate percent."""
    acc = TheoreticalPrefixCacheAccumulator(run=mock_run, service_id="proc-1")

    conv_id = "trace_03_n3"
    from aiperf.common.models.dataset_models import TurnMetadata

    conv_meta = ConversationMetadata(conversation_id=conv_id)
    conv_meta.turns = [
        TurnMetadata(
            theoretical_prefix_cache_hit_blocks=h,
            theoretical_prefix_cache_total_blocks=t,
        )
        for h, t in _EXPECTED_PER_TURN
    ]
    metadata = DatasetMetadata(
        conversations=[conv_meta],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    acc.on_dataset_configured(metadata)

    for turn_index in range(len(_EXPECTED_PER_TURN)):
        record = create_metric_records_data(
            metadata=create_metric_metadata(
                conversation_id=conv_id,
                turn_index=turn_index,
                benchmark_phase=CreditPhase.PROFILING,
            )
        )
        await acc.process_record(record)

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
    """Graph path: per-node map + node_id recovered from (conversation_id, turn_index).

    Mirrors the real WEKA graph dispatch where the record carries LEGACY-shaped
    identity: ``conversation_id`` is the trajectory TEMPLATE id (root scope ==
    the trace id) and ``turn_index`` is the node's 0-based turn, so the
    ``{scope}:{turn}`` node id -- and thus the join into the per-node map -- is a
    pure function of those two record fields (no correlation-id parsing).
    """
    acc = TheoreticalPrefixCacheAccumulator(run=mock_run, service_id="proc-1")
    base_trace = "trace_03_n3"
    acc.on_dataset_configured(
        DatasetMetadata(
            conversations=[],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            graph=GraphDatasetMetadata(
                trace_ids=[base_trace],
                prefix_cache_by_trace={
                    base_trace: {
                        f"{base_trace}:0": [0, 2],
                        f"{base_trace}:1": [2, 3],
                        f"{base_trace}:2": [3, 4],
                    }
                },
            ),
        )
    )
    for turn_index in range(3):
        record = create_metric_records_data(
            metadata=create_metric_metadata(
                # conversation_id is the root-scope TEMPLATE id (instance
                # identity rides x_correlation_id, which this join never reads).
                conversation_id=base_trace,
                turn_index=turn_index,
                benchmark_phase=CreditPhase.PROFILING,
            )
        )
        await acc.process_record(record)
    results = await acc.summarize()
    assert len(results) == 1
    assert results[0].avg == pytest.approx(_EXPECTED_HIT_PCT)
    assert results[0].sum == 5
    assert results[0].count == 9


@pytest.mark.asyncio
async def test_accumulator_no_metadata_emits_nothing(mock_run) -> None:
    """No prefix-cache metadata (non-weka dataset) -> empty summary, no NaN."""
    acc = TheoreticalPrefixCacheAccumulator(run=mock_run, service_id="proc-1")
    metadata = DatasetMetadata(
        conversations=[ConversationMetadata(conversation_id="c0")],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    acc.on_dataset_configured(metadata)
    record = create_metric_records_data(
        metadata=create_metric_metadata(
            conversation_id="c0", turn_index=0, benchmark_phase=CreditPhase.PROFILING
        )
    )
    await acc.process_record(record)
    assert await acc.summarize() == []


@pytest.mark.asyncio
async def test_accumulator_skips_context_overflow_records(mock_run) -> None:
    """WK4: an overflow-skip record (valid=True, error trimmed by the
    RecordsManager) must NOT fold its planned blocks into the hit rate.

    The trimmed carrier keeps ``metadata.context_overflow_skip=True``; the
    accumulator keys off that flag. An identical record without the flag still
    accumulates (control).
    """
    acc = TheoreticalPrefixCacheAccumulator(run=mock_run, service_id="proc-1")
    base_trace = "trace_03_n3"
    acc.on_dataset_configured(
        DatasetMetadata(
            conversations=[],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            graph=GraphDatasetMetadata(
                trace_ids=[base_trace],
                prefix_cache_by_trace={base_trace: {f"{base_trace}:0": [3, 4]}},
            ),
        )
    )

    def _record(overflow_skip: bool):
        metadata = create_metric_metadata(
            # conversation_id is the root-scope TEMPLATE id; turn_index 0 recovers
            # node id ``{base_trace}:0``.
            conversation_id=base_trace,
            turn_index=0,
            benchmark_phase=CreditPhase.PROFILING,
        )
        metadata.context_overflow_skip = overflow_skip
        return create_metric_records_data(metadata=metadata)

    overflow_record = _record(overflow_skip=True)
    assert overflow_record.valid  # trimmed carrier arrives error-free
    await acc.process_record(overflow_record)
    assert await acc.summarize() == []

    await acc.process_record(_record(overflow_skip=False))
    results = await acc.summarize()
    assert results[0].sum == 3
    assert results[0].count == 4


@pytest.mark.asyncio
async def test_accumulator_clamps_overcounted_hits(mock_run) -> None:
    """A loader miscount (hit > total) is clamped so the rate stays <= 100%."""
    acc = TheoreticalPrefixCacheAccumulator(run=mock_run, service_id="proc-1")
    from aiperf.common.models.dataset_models import TurnMetadata

    conv_meta = ConversationMetadata(conversation_id="c0")
    conv_meta.turns = [
        TurnMetadata(
            theoretical_prefix_cache_hit_blocks=10,
            theoretical_prefix_cache_total_blocks=4,
        )
    ]
    acc.on_dataset_configured(
        DatasetMetadata(
            conversations=[conv_meta],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
    )
    record = create_metric_records_data(
        metadata=create_metric_metadata(
            conversation_id="c0", turn_index=0, benchmark_phase=CreditPhase.PROFILING
        )
    )
    await acc.process_record(record)
    results = await acc.summarize()
    assert results[0].avg == pytest.approx(100.0)
