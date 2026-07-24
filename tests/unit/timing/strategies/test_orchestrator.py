# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PhaseOrchestrator initialization, cancellation, and phase configuration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.plugin.enums import ArrivalPattern, DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import TimingConfig
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
from tests.unit.timing.conftest import (
    MockCreditRouter,
    make_phase_config,
    make_timing_config,
)


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

    async def test_seamless_runtime_handoff_is_prepared_on_outgoing_phase(
        self,
    ) -> None:
        source = make_phase_config(
            CreditPhase.PROFILING,
            TimingMode.REQUEST_RATE,
            request_count=5,
            request_rate=10.0,
            seamless=True,
        ).model_copy(update={"phase_index": 0, "phase_name": "source"})
        target = make_phase_config(
            CreditPhase.PROFILING,
            TimingMode.REQUEST_RATE,
            request_count=5,
            request_rate=10.0,
        ).model_copy(update={"phase_index": 1, "phase_name": "target"})
        orch = PhaseOrchestrator(
            config=TimingConfig(phase_configs=[source, target]),
            phase_publisher=make_publisher(),
            credit_router=make_router(),
            dataset_metadata=make_dataset(3, 2),
        )
        orch._callback_handler.start_phase_handoff = MagicMock()
        runners = []

        def make_runner_mock(**kwargs):
            runner = MagicMock()
            runner.phase = kwargs["config"].phase
            runner.run = AsyncMock()
            runner.wait_for_completion = AsyncMock()
            runners.append(runner)
            return runner

        with patch(
            "aiperf.timing.phase_orchestrator.PhaseRunner",
            side_effect=make_runner_mock,
        ):
            await orch._execute_phases()

        orch._callback_handler.start_phase_handoff.assert_called_once_with(0, 1)
        assert runners[0].run.await_args_list == [call(is_final_phase=False)]
        assert runners[1].run.await_args_list == [call(is_final_phase=True)]
        runners[0].wait_for_completion.assert_awaited_once()
        runners[1].wait_for_completion.assert_not_called()

    async def test_live_root_session_continues_with_target_phase_accounting(
        self,
    ) -> None:
        source = make_phase_config(
            CreditPhase.WARMUP,
            TimingMode.REQUEST_RATE,
            request_count=1,
            concurrency=1,
            request_rate=1000.0,
            arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
            seamless=True,
        ).model_copy(update={"phase_index": 0, "phase_name": "warmup"})
        target = make_phase_config(
            CreditPhase.PROFILING,
            TimingMode.REQUEST_RATE,
            request_count=2,
            concurrency=1,
            request_rate=1000.0,
            arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        ).model_copy(update={"phase_index": 1, "phase_name": "profiling"})
        router = MockCreditRouter(auto_return=False)
        orch = PhaseOrchestrator(
            config=TimingConfig(phase_configs=[source, target]),
            phase_publisher=make_publisher(),
            credit_router=router,
            dataset_metadata=make_dataset(1, 5),
        )
        await orch.initialize()

        run_task = asyncio.create_task(orch.start())
        while (
            not router.sent_credits
            or not orch._callback_handler._phase_handlers[
                0
            ].lifecycle.is_sending_complete
        ):
            await asyncio.sleep(0)
        await router.return_credit(router.sent_credits[0])

        while len([c for c in router.sent_credits if c.phase_index == 1]) < 1:
            await asyncio.sleep(0)
        first_target = next(c for c in router.sent_credits if c.phase_index == 1)
        await router.return_credit(first_target)

        while len([c for c in router.sent_credits if c.phase_index == 1]) < 2:
            await asyncio.sleep(0)
        second_target = [c for c in router.sent_credits if c.phase_index == 1][1]
        await router.return_credit(second_target)
        await run_task

        source_credits = [c for c in router.sent_credits if c.phase_index == 0]
        target_credits = [c for c in router.sent_credits if c.phase_index == 1]
        assert len(source_credits) == 1
        assert len(target_credits) == 2
        assert target_credits[0].x_correlation_id == source_credits[0].x_correlation_id
        assert target_credits[0].turn_index == 1
        assert target_credits[0].start_turn_index == 1
        assert target_credits[1].turn_index == 2
