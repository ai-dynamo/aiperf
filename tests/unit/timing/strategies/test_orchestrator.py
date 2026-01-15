# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PhaseOrchestrator

Tests orchestration concerns:
- Initialization and setup
- Callback registration
- Cancellation
- Phase configuration

Architecture:
    Orchestrator owns long-lived components (ConcurrencyManager, CancellationPolicy,
    ConversationSource, CreditCallbackHandler). Per-phase components (PhaseRunner,
    TimingStrategy) are created during _execute_phases().
"""

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
    """Mock credit router."""
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
    """Mock phase publisher."""
    publisher = MagicMock()
    publisher.publish_phase_start = AsyncMock()
    publisher.publish_phase_complete = AsyncMock()
    publisher.publish_phase_sending_complete = AsyncMock()
    publisher.publish_progress = AsyncMock()
    publisher.publish_credits_complete = AsyncMock()
    return publisher


@pytest.fixture
def timing_config():
    """Basic timing config for testing."""
    return make_timing_config(
        timing_mode=TimingMode.REQUEST_RATE,
        request_count=5,
        request_rate=10.0,
    )


@pytest.fixture
def dataset_metadata():
    """Dataset with multi-turn conversations."""
    return make_dataset_metadata(num_conversations=3, turns_per_conversation=2)


@pytest.fixture
async def orchestrator(
    timing_config, mock_phase_publisher, mock_credit_router, dataset_metadata
):
    """Create and initialize PhaseOrchestrator."""
    orch = PhaseOrchestrator(
        config=timing_config,
        phase_publisher=mock_phase_publisher,
        credit_router=mock_credit_router,
        dataset_metadata=dataset_metadata,
    )
    await orch.initialize()
    return orch


class TestOrchestratorInitialization:
    """Tests for PhaseOrchestrator initialization."""

    @pytest.mark.asyncio
    async def test_registers_return_callback_on_init(
        self, timing_config, mock_phase_publisher, mock_credit_router, dataset_metadata
    ):
        """Orchestrator registers credit return callback."""
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
    ):
        """Orchestrator registers first token callback."""
        PhaseOrchestrator(
            config=timing_config,
            phase_publisher=mock_phase_publisher,
            credit_router=mock_credit_router,
            dataset_metadata=dataset_metadata,
        )

        mock_credit_router.set_first_token_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_callback_handler(self, orchestrator):
        """Orchestrator creates CreditCallbackHandler."""
        assert orchestrator._callback_handler is not None

    @pytest.mark.asyncio
    async def test_creates_concurrency_manager(self, orchestrator):
        """Orchestrator creates ConcurrencyManager."""
        assert orchestrator._concurrency_manager is not None

    @pytest.mark.asyncio
    async def test_creates_cancellation_policy(self, orchestrator):
        """Orchestrator creates RequestCancellationPolicy."""
        assert orchestrator._cancellation_policy is not None

    @pytest.mark.asyncio
    async def test_creates_conversation_source(self, orchestrator):
        """Orchestrator creates conversation source."""
        assert orchestrator.conversation_source is not None

    @pytest.mark.asyncio
    async def test_active_runners_initially_empty(self, orchestrator):
        """Active runners list is empty before execution."""
        assert orchestrator._active_runners == []


class TestCancellation:
    """Tests for PhaseOrchestrator.cancel."""

    @pytest.mark.asyncio
    async def test_cancels_all_credits_via_router(
        self, orchestrator, mock_credit_router
    ):
        """Cancel sends cancel_all_credits to router."""
        await orchestrator.cancel()

        mock_credit_router.cancel_all_credits.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_is_idempotent(self, orchestrator, mock_credit_router):
        """Cancel can be called multiple times safely."""
        await orchestrator.cancel()
        await orchestrator.cancel()

        # Should be called twice
        assert mock_credit_router.cancel_all_credits.call_count == 2

    @pytest.mark.asyncio
    async def test_cancel_without_active_runners(
        self, orchestrator, mock_credit_router
    ):
        """Cancel works when no runners are active."""
        assert orchestrator._active_runners == []
        await orchestrator.cancel()
        # Should complete without error
        mock_credit_router.cancel_all_credits.assert_called_once()


class TestPhaseConfiguration:
    """Tests for phase configuration during orchestrator setup."""

    @pytest.mark.asyncio
    async def test_warmup_and_profiling_phases_configured(
        self, mock_phase_publisher, mock_credit_router, dataset_metadata
    ):
        """Both warmup and profiling phases are tracked in config."""
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
    async def test_only_profiling_phase_when_no_warmup(self, orchestrator):
        """Only profiling phase when warmup not configured."""
        phases = [pc.phase for pc in orchestrator._ordered_phase_configs]
        assert CreditPhase.PROFILING in phases
        assert CreditPhase.WARMUP not in phases

    @pytest.mark.asyncio
    async def test_phase_order_warmup_before_profiling(
        self, mock_phase_publisher, mock_credit_router, dataset_metadata
    ):
        """Warmup phase comes before profiling in execution order."""
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
    """Tests for correct component wiring."""

    @pytest.mark.asyncio
    async def test_callback_handler_uses_concurrency_manager(self, orchestrator):
        """CreditCallbackHandler is wired to ConcurrencyManager."""
        assert (
            orchestrator._callback_handler._concurrency_manager
            is orchestrator._concurrency_manager
        )

    @pytest.mark.asyncio
    async def test_callback_handler_registered_with_router(
        self, timing_config, mock_phase_publisher, mock_credit_router, dataset_metadata
    ):
        """CreditCallbackHandler callbacks are registered with router."""
        orch = PhaseOrchestrator(
            config=timing_config,
            phase_publisher=mock_phase_publisher,
            credit_router=mock_credit_router,
            dataset_metadata=dataset_metadata,
        )

        # Verify callbacks were registered
        mock_credit_router.set_return_callback.assert_called_once_with(
            orch._callback_handler.on_credit_return
        )
        mock_credit_router.set_first_token_callback.assert_called_once_with(
            orch._callback_handler.on_first_token
        )

    @pytest.mark.asyncio
    async def test_conversation_source_is_configured(self, orchestrator):
        """ConversationSource is properly configured."""
        # Verify conversation source can sample conversations
        sampled = orchestrator.conversation_source.next()
        assert sampled is not None
        assert sampled.conversation_id.startswith("conv-")
