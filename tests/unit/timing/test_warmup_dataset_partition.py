# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase pools remain disjoint across sampler wraparound and partial warmup."""

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.dataset.dataset_samplers import ShuffleSampler
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import CreditPhaseConfig, TimingConfig
from aiperf.timing.conversation_source import (
    ConversationSource,
    partition_warmup_dataset,
)
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.phase_orchestrator import PhaseOrchestrator


def _metadata(strategy: DatasetSamplingStrategy) -> DatasetMetadata:
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id=f"c{i}", turns=[TurnMetadata()] * (i + 1)
            )
            for i in range(6)
        ],
        sampling_strategy=strategy,
    )


@pytest.mark.parametrize("strategy", list(DatasetSamplingStrategy))
def test_phase_sources_are_disjoint_and_warmup_independent(
    strategy: DatasetSamplingStrategy,
) -> None:
    metadata = _metadata(strategy)
    # Exercise metadata cached before partitioning: counts must be recomputed.
    assert metadata.total_turn_count == 21
    assert len(metadata.sampleable_conversation_ids) == 6
    config = TimingConfig(phase_configs=[], warmup_dataset_entries=2)

    def create() -> PhaseOrchestrator:
        return PhaseOrchestrator(
            config=config,
            dataset_metadata=metadata,
            phase_publisher=MagicMock(),
            credit_router=MagicMock(),
            service_id="test-orchestrator",
        )

    orchestrator = create()
    warmup = orchestrator._warmup_conversation_source
    profiling = orchestrator.conversation_source
    warmup_ids = {warmup.next().conversation_id for _ in range(40)}
    profiling_ids = [profiling.next().conversation_id for _ in range(40)]
    assert warmup_ids <= {"c0", "c1"}
    assert set(profiling_ids) <= {"c2", "c3", "c4", "c5"}
    assert warmup_ids.isdisjoint(profiling_ids)
    assert profiling.dataset_metadata.total_turn_count == 18
    other = create()
    # Aborted or duration-limited warmup cannot advance profiling's RNG.
    other._warmup_conversation_source.next()
    assert [
        other.conversation_source.next().conversation_id for _ in range(40)
    ] == profiling_ids
    assert metadata.total_turn_count == 21
    assert len(metadata.conversations) == 6


@pytest.mark.asyncio
async def test_sessions_all_uses_profiling_partition_size() -> None:
    _, metadata = partition_warmup_dataset(
        _metadata(DatasetSamplingStrategy.SHUFFLE), 2
    )
    source = ConversationSource(
        metadata, ShuffleSampler(metadata.sampleable_conversation_ids)
    )
    runner = PhaseRunner(
        config=CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            sessions_from_dataset=True,
        ),
        conversation_source=source,
        phase_publisher=MagicMock(),
        credit_router=MagicMock(),
        concurrency_manager=MagicMock(),
        callback_handler=MagicMock(),
    )
    assert runner._config.expected_num_sessions == 4


@pytest.mark.parametrize("count", [0, 6, 7])
def test_loaded_partition_cannot_be_empty(count: int) -> None:
    with pytest.raises(ValueError, match="loaded profiling"):
        partition_warmup_dataset(_metadata(DatasetSamplingStrategy.SEQUENTIAL), count)
