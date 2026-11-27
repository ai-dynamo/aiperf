# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``sessions: all`` (one full pass) requires sampling without replacement."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase_orchestrator import PhaseOrchestrator


def _timing_config(*, sessions_from_dataset: bool) -> SimpleNamespace:
    phase = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        phase_name="profiling",
        timing_mode=TimingMode.REQUEST_RATE,
        concurrency=4,
        sessions_from_dataset=sessions_from_dataset,
    )
    return SimpleNamespace(phase_configs=[phase])


def _dataset(strategy: DatasetSamplingStrategy) -> DatasetMetadata:
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(conversation_id=f"c{i}", turns=[TurnMetadata()])
            for i in range(3)
        ],
        sampling_strategy=strategy,
    )


def test_phase_orchestrator_full_pass_with_random_sampling_raises() -> None:
    with pytest.raises(ValueError, match="random sampling"):
        PhaseOrchestrator(
            config=_timing_config(sessions_from_dataset=True),
            phase_publisher=MagicMock(),
            credit_router=MagicMock(),
            dataset_metadata=_dataset(DatasetSamplingStrategy.RANDOM),
            service_id="test-orchestrator",
        )


@pytest.mark.parametrize(
    "strategy,sessions_from_dataset",
    [
        param(DatasetSamplingStrategy.SEQUENTIAL, True, id="sequential_full_pass"),
        param(DatasetSamplingStrategy.SHUFFLE, True, id="shuffle_full_pass"),
        param(DatasetSamplingStrategy.RANDOM, False, id="random_bounded"),
    ],
)  # fmt: skip
def test_reject_full_pass_with_random_sampling_allows_other_combinations(
    strategy: DatasetSamplingStrategy, sessions_from_dataset: bool
) -> None:
    orchestrator = SimpleNamespace(
        _config=_timing_config(sessions_from_dataset=sessions_from_dataset),
        _dataset_metadata=_dataset(strategy),
    )

    PhaseOrchestrator._reject_full_pass_with_random_sampling(orchestrator)
