# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase, DatasetSamplingStrategy, TimingMode
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.timing.config import TimingConfig
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
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
    @pytest.mark.parametrize(
        "callback_method",
        ["set_return_callback", "set_first_token_callback"],
    )  # fmt: skip
    async def test_registers_callbacks(self, callback_method) -> None:
        router = make_router()
        PhaseOrchestrator(
            config=make_timing_config(
                TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
            ),
            phase_publisher=make_publisher(),
            credit_router=router,
            dataset_metadata=make_dataset(3, 2),
        )
        getattr(router, callback_method).assert_called_once()

    @pytest.mark.parametrize(
        "attr",
        ["_callback_handler", "_concurrency_manager", "_cancellation_policy", "conversation_source"],
    )  # fmt: skip
    async def test_creates_components(self, attr) -> None:
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
        assert getattr(orch, attr) is not None

    async def test_active_runners_initially_empty(self) -> None:
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
        assert orch._active_runners == []


@pytest.mark.asyncio
class TestCancellation:
    async def test_cancels_credits(self) -> None:
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

    async def test_cancel_is_idempotent(self) -> None:
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
    async def test_warmup_and_profiling(self) -> None:
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

    async def test_profiling_only(self) -> None:
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
class TestComponentWiring:
    async def test_callback_handler_uses_concurrency_manager(self) -> None:
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
        assert orch._callback_handler._concurrency_manager is orch._concurrency_manager

    async def test_callback_handler_registered(self) -> None:
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
        router.set_return_callback.assert_called_once_with(
            orch._callback_handler.on_credit_return
        )
        router.set_first_token_callback.assert_called_once_with(
            orch._callback_handler.on_first_token
        )

    async def test_conversation_source_configured(self) -> None:
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
