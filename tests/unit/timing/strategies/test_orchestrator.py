# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PhaseOrchestrator initialization, cancellation, and phase configuration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import TimingConfig
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
from aiperf.timing.trajectory_source import TrajectorySource
from tests.unit.timing.conftest import make_phase_config, make_timing_config


def make_dataset(num_convs: int = 3, turns: int = 1) -> DatasetMetadata:
    convs = [
        ConversationMetadata(
            conversation_id=f"conv-{i}", turns=[TurnMetadata() for _ in range(turns)]
        )
        for i in range(num_convs)
    ]
    return DatasetMetadata(
        conversations=convs, sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )


def make_router() -> MagicMock:
    r = MagicMock()
    r.send_credit = AsyncMock()
    r.cancel_all_credits = AsyncMock()
    r.mark_credits_complete = MagicMock()
    r.set_return_callback = MagicMock()
    r.set_first_token_callback = MagicMock()
    r.reset = MagicMock()
    return r


def make_publisher() -> MagicMock:
    p = MagicMock()
    p.publish_phase_start = AsyncMock()
    p.publish_phase_complete = AsyncMock()
    p.publish_phase_sending_complete = AsyncMock()
    p.publish_progress = AsyncMock()
    p.publish_credits_complete = AsyncMock()
    return p


@pytest.mark.asyncio
class TestOrchestratorInit:
    """Tests for PhaseOrchestrator initialization behavior."""

    async def test_registers_callbacks_with_router(self) -> None:
        """Orchestrator registers credit return and first token callbacks during init."""
        router = make_router()
        orch = PhaseOrchestrator(
            config=make_timing_config(
                TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
            ),
            phase_publisher=make_publisher(),
            credit_router=router,
            dataset_metadata=make_dataset(3, 2),
        )
        router.set_return_callback.assert_called_once_with(
            orch._callback_handler.on_credit_return
        )
        router.set_first_token_callback.assert_called_once_with(
            orch._callback_handler.on_first_token
        )

    async def test_wires_warmup_abort_to_publish_profile_cancel(self) -> None:
        """The callback handler's live warmup-abort is wired to the publisher's
        publish_profile_cancel. A regression here silently reverts the agentic
        warmup-failure path to the original hang, while leaf unit tests (which
        inject the callback by hand) keep passing -- so pin it explicitly."""
        publisher = make_publisher()
        orch = PhaseOrchestrator(
            config=make_timing_config(
                TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
            ),
            phase_publisher=publisher,
            credit_router=make_router(),
            dataset_metadata=make_dataset(3, 2),
        )
        assert (
            orch._callback_handler.on_warmup_abort is publisher.publish_profile_cancel
        )

    async def test_conversation_source_samples_from_dataset(self) -> None:
        """Conversation source is initialized and can sample conversations."""
        cfg = make_timing_config(
            TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
        )
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=make_publisher(),
            credit_router=make_router(),
            dataset_metadata=make_dataset(3, 2),
        )
        await orch.initialize()
        sampled = orch.conversation_source.next()
        assert sampled is not None
        assert sampled.conversation_id.startswith("conv-")


@pytest.mark.asyncio
class TestCancellation:
    """Tests for PhaseOrchestrator cancellation behavior."""

    async def test_cancel_cancels_router_credits(self) -> None:
        """Calling cancel() triggers cancellation of all in-flight credits."""
        router = make_router()
        cfg = make_timing_config(
            TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
        )
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=make_publisher(),
            credit_router=router,
            dataset_metadata=make_dataset(3, 2),
        )
        await orch.initialize()
        await orch.cancel()
        router.cancel_all_credits.assert_called_once()

    async def test_cancel_can_be_called_multiple_times(self) -> None:
        """Calling cancel() multiple times does not raise errors."""
        router = make_router()
        cfg = make_timing_config(
            TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
        )
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=make_publisher(),
            credit_router=router,
            dataset_metadata=make_dataset(3, 2),
        )
        await orch.initialize()
        await orch.cancel()
        await orch.cancel()
        assert router.cancel_all_credits.call_count == 2


@pytest.mark.asyncio
class TestPhaseConfig:
    """Tests for phase configuration handling."""

    async def test_warmup_and_profiling_phases_in_order(self) -> None:
        """When both warmup and profiling are configured, they execute in order."""
        warmup = make_phase_config(
            CreditPhase.WARMUP,
            TimingMode.REQUEST_RATE,
            request_count=5,
            request_rate=10.0,
        )
        profiling = make_phase_config(
            CreditPhase.PROFILING,
            TimingMode.REQUEST_RATE,
            request_count=10,
            request_rate=10.0,
        )
        cfg = TimingConfig(phase_configs=[warmup, profiling])
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=make_publisher(),
            credit_router=make_router(),
            dataset_metadata=make_dataset(3, 2),
        )
        await orch.initialize()
        phases = [pc.phase for pc in orch._ordered_phase_configs]
        assert phases == [CreditPhase.WARMUP, CreditPhase.PROFILING]

    async def test_profiling_only_excludes_warmup(self) -> None:
        """When only profiling is configured, warmup phase is not present."""
        cfg = make_timing_config(
            TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
        )
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=make_publisher(),
            credit_router=make_router(),
            dataset_metadata=make_dataset(3, 2),
        )
        await orch.initialize()
        phases = [pc.phase for pc in orch._ordered_phase_configs]
        assert CreditPhase.PROFILING in phases
        assert CreditPhase.WARMUP not in phases


@pytest.mark.asyncio
class TestConversationSourceSelection:
    """PhaseOrchestrator picks the right ConversationSource subclass per timing mode."""

    async def test_phase_orchestrator_uses_trajectory_source_for_agentic_replay(
        self,
    ) -> None:
        """AGENTIC_REPLAY phase configs trigger TrajectorySource construction."""
        warmup = make_phase_config(
            CreditPhase.WARMUP,
            TimingMode.AGENTIC_REPLAY,
            concurrency=3,
        )
        profiling = make_phase_config(
            CreditPhase.PROFILING,
            TimingMode.AGENTIC_REPLAY,
            concurrency=3,
        )
        cfg = TimingConfig(
            phase_configs=[warmup, profiling],
            concurrency=3,
            random_seed=12345,
        )
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=make_publisher(),
            credit_router=make_router(),
            dataset_metadata=make_dataset(num_convs=5, turns=4),
        )
        assert isinstance(orch.conversation_source, TrajectorySource)
        assert orch.conversation_source._random_seed == 12345
        assert orch.conversation_source._target_size == 3

    async def test_phase_orchestrator_uses_plain_source_for_non_agentic(self) -> None:
        """Non-AGENTIC_REPLAY modes preserve plain ConversationSource (not the subclass)."""
        for mode in (TimingMode.REQUEST_RATE, TimingMode.FIXED_SCHEDULE):
            cfg = make_timing_config(mode, request_count=5, request_rate=10.0)
            orch = PhaseOrchestrator(
                config=cfg,
                phase_publisher=make_publisher(),
                credit_router=make_router(),
                dataset_metadata=make_dataset(3, 2),
            )
            assert type(orch.conversation_source) is ConversationSource, (
                f"Expected plain ConversationSource for {mode}, got "
                f"{type(orch.conversation_source).__name__}"
            )
