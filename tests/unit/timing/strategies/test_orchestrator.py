# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PhaseOrchestrator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase, DatasetSamplingStrategy, TimingMode
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.timing.config import TimingConfig
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
from tests.unit.timing.conftest import make_phase_config, make_timing_config


def make_dataset_metadata(
    num_conversations: int = 3,
    turns_per_conversation: int = 1,
) -> DatasetMetadata:
    """Create DatasetMetadata for testing."""
    conversations = [
        ConversationMetadata(
            conversation_id=f"conv-{i}",
            turns=[TurnMetadata() for _ in range(turns_per_conversation)],
        )
        for i in range(num_conversations)
    ]
    return DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


@pytest.fixture
def mock_credit_router():
    router = MagicMock()
    router.send_credit = AsyncMock()
    router.cancel_all_credits = AsyncMock()
    router.mark_credits_complete = MagicMock()
    router.set_return_callback = MagicMock()
    router.set_first_token_callback = MagicMock()
    router.reset = MagicMock()
    return router


@pytest.fixture
def mock_phase_publisher():
    publisher = MagicMock()
    publisher.publish_phase_start = AsyncMock()
    publisher.publish_phase_complete = AsyncMock()
    publisher.publish_phase_sending_complete = AsyncMock()
    publisher.publish_progress = AsyncMock()
    publisher.publish_credits_complete = AsyncMock()
    return publisher


@pytest.fixture
def timing_config():
    return make_timing_config(
        timing_mode=TimingMode.REQUEST_RATE,
        request_count=5,
        request_rate=10.0,
    )


@pytest.fixture
def dataset_metadata():
    return make_dataset_metadata(num_conversations=3, turns_per_conversation=2)


@pytest.fixture
async def orchestrator(
    timing_config, mock_phase_publisher, mock_credit_router, dataset_metadata
):
    orch = PhaseOrchestrator(
        config=timing_config,
        phase_publisher=mock_phase_publisher,
        credit_router=mock_credit_router,
        dataset_metadata=dataset_metadata,
    )
    await orch.initialize()
    return orch


class TestOrchestratorInitialization:
    @pytest.mark.asyncio
    async def test_registers_return_callback_on_init(
        self, timing_config, mock_phase_publisher, mock_credit_router, dataset_metadata
    ) -> None:
        PhaseOrchestrator(
            config=timing_config,
            phase_publisher=mock_phase_publisher,
            credit_router=mock_credit_router,
            dataset_metadata=dataset_metadata,
        )
        mock_credit_router.set_return_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_registers_first_token_callback_on_init(
        self, timing_config, mock_phase_publisher, mock_credit_router, dataset_metadata
    ) -> None:
        PhaseOrchestrator(
            config=timing_config,
            phase_publisher=mock_phase_publisher,
            credit_router=mock_credit_router,
            dataset_metadata=dataset_metadata,
        )
        mock_credit_router.set_first_token_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_callback_handler(self, orchestrator) -> None:
        assert orchestrator._callback_handler is not None

    @pytest.mark.asyncio
    async def test_creates_concurrency_manager(self, orchestrator) -> None:
        assert orchestrator._concurrency_manager is not None

    @pytest.mark.asyncio
    async def test_creates_cancellation_policy(self, orchestrator) -> None:
        assert orchestrator._cancellation_policy is not None

    @pytest.mark.asyncio
    async def test_creates_conversation_source(self, orchestrator) -> None:
        assert orchestrator.conversation_source is not None

    @pytest.mark.asyncio
    async def test_active_runners_initially_empty(self, orchestrator) -> None:
        assert orchestrator._active_runners == []


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancels_all_credits_via_router(
        self, orchestrator, mock_credit_router
    ) -> None:
        await orchestrator.cancel()
        mock_credit_router.cancel_all_credits.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_is_idempotent(self, orchestrator, mock_credit_router) -> None:
        await orchestrator.cancel()
        await orchestrator.cancel()
        assert mock_credit_router.cancel_all_credits.call_count == 2

    @pytest.mark.asyncio
    async def test_cancel_without_active_runners(
        self, orchestrator, mock_credit_router
    ) -> None:
        assert orchestrator._active_runners == []
        await orchestrator.cancel()
        mock_credit_router.cancel_all_credits.assert_called_once()


class TestPhaseConfiguration:
    @pytest.mark.asyncio
    async def test_warmup_and_profiling_phases_configured(
        self, mock_phase_publisher, mock_credit_router, dataset_metadata
    ) -> None:
        warmup = make_phase_config(
            phase=CreditPhase.WARMUP,
            timing_mode=TimingMode.REQUEST_RATE,
            request_count=5,
            request_rate=10.0,
        )
        profiling = make_phase_config(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            request_count=10,
            request_rate=10.0,
        )
        config = TimingConfig(phase_configs=[warmup, profiling])

        orch = PhaseOrchestrator(
            config=config,
            phase_publisher=mock_phase_publisher,
            credit_router=mock_credit_router,
            dataset_metadata=dataset_metadata,
        )
        await orch.initialize()

        phases = [pc.phase for pc in orch._ordered_phase_configs]
        assert CreditPhase.WARMUP in phases
        assert CreditPhase.PROFILING in phases

    @pytest.mark.asyncio
    async def test_only_profiling_phase_when_no_warmup(self, orchestrator) -> None:
        phases = [pc.phase for pc in orchestrator._ordered_phase_configs]
        assert CreditPhase.PROFILING in phases
        assert CreditPhase.WARMUP not in phases

    @pytest.mark.asyncio
    async def test_phase_order_warmup_before_profiling(
        self, mock_phase_publisher, mock_credit_router, dataset_metadata
    ) -> None:
        warmup = make_phase_config(
            phase=CreditPhase.WARMUP,
            timing_mode=TimingMode.REQUEST_RATE,
            request_count=5,
            request_rate=10.0,
        )
        profiling = make_phase_config(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            request_count=10,
            request_rate=10.0,
        )
        config = TimingConfig(phase_configs=[warmup, profiling])

        orch = PhaseOrchestrator(
            config=config,
            phase_publisher=mock_phase_publisher,
            credit_router=mock_credit_router,
            dataset_metadata=dataset_metadata,
        )
        await orch.initialize()

        phases = [pc.phase for pc in orch._ordered_phase_configs]
        assert phases == [CreditPhase.WARMUP, CreditPhase.PROFILING]


class TestComponentWiring:
    @pytest.mark.asyncio
    async def test_callback_handler_uses_concurrency_manager(
        self, orchestrator
    ) -> None:
        assert (
            orchestrator._callback_handler._concurrency_manager
            is orchestrator._concurrency_manager
        )

    @pytest.mark.asyncio
    async def test_callback_handler_registered_with_router(
        self, timing_config, mock_phase_publisher, mock_credit_router, dataset_metadata
    ) -> None:
        orch = PhaseOrchestrator(
            config=timing_config,
            phase_publisher=mock_phase_publisher,
            credit_router=mock_credit_router,
            dataset_metadata=dataset_metadata,
        )

        mock_credit_router.set_return_callback.assert_called_once_with(
            orch._callback_handler.on_credit_return
        )
        mock_credit_router.set_first_token_callback.assert_called_once_with(
            orch._callback_handler.on_first_token
        )

    @pytest.mark.asyncio
    async def test_conversation_source_is_configured(self, orchestrator) -> None:
        sampled = orchestrator.conversation_source.next()
        assert sampled is not None
        assert sampled.conversation_id.startswith("conv-")
