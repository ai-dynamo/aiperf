# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import CommAddress, CreditPhase
from aiperf.common.models import (
    ConversationMetadata,
    CreditPhaseStats,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.utils import yield_to_event_loop
from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit, CreditContext, TurnToSend
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    ArrivalPattern,
    DatasetSamplingStrategy,
    PluginType,
    TimingMode,
)
from aiperf.timing.concurrency import ConcurrencyStats
from aiperf.timing.config import (
    CreditPhaseConfig,
    RequestCancellationConfig,
    TimingConfig,
)
from aiperf.timing.phase.publisher import PhasePublisher
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
from aiperf.workers.worker import Worker
from tests.harness.fake_communication import FakeCommunication, FakeCommunicationBus


async def _async_noop(*args, **kwargs) -> None:
    return None


async def _async_true(*args, **kwargs) -> bool:
    return True


@dataclass
class MockCreditRouter:
    sent_credits: list[Credit] = field(default_factory=list)
    auto_return: bool = False
    _return_cb: Callable[[str, CreditReturn], Awaitable[None]] | None = None
    _first_token_cb: Callable[[FirstToken], Awaitable[None]] | None = None
    _pending: list[asyncio.Task] = field(default_factory=list)

    async def send_credit(self, credit: Credit) -> None:
        self.sent_credits.append(credit)
        if self.auto_return and self._return_cb:
            self._pending.append(asyncio.create_task(self._do_return(credit)))
            await yield_to_event_loop()

    async def _do_return(self, credit: Credit) -> None:
        await asyncio.sleep(0.001)
        if self._return_cb:
            await self._return_cb(
                "worker-1",
                CreditReturn(credit=credit, cancelled=False, first_token_sent=True),
            )

    async def cancel_all_credits(self) -> None:
        pass

    def mark_credits_complete(self) -> None:
        pass

    def set_return_callback(
        self, cb: Callable[[str, CreditReturn], Awaitable[None]]
    ) -> None:
        self._return_cb = cb

    def set_first_token_callback(
        self, cb: Callable[[FirstToken], Awaitable[None]]
    ) -> None:
        self._first_token_cb = cb

    async def return_credit(
        self, credit: Credit, cancelled: bool = False, first_token_sent: bool = True
    ) -> None:
        if self._return_cb:
            await self._return_cb(
                "worker-1",
                CreditReturn(
                    credit=credit,
                    cancelled=cancelled,
                    first_token_sent=first_token_sent,
                ),
            )


@dataclass
class OrchestratorHarness:
    orchestrator: PhaseOrchestrator
    router: MockCreditRouter

    @property
    def sent_credits(self) -> list[Credit]:
        return self.router.sent_credits

    async def run_with_auto_return(self) -> None:
        self.router.auto_return = True
        await self.orchestrator.initialize()
        with contextlib.suppress(asyncio.CancelledError):
            await self.orchestrator.start()
        with contextlib.suppress(Exception):
            await self.orchestrator.stop()
        if self.router._pending:
            await asyncio.gather(*self.router._pending, return_exceptions=True)
            self.router._pending.clear()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def return_credit(self, credit: Credit) -> None:
        await self.router.return_credit(credit)


@pytest.fixture
def create_orchestrator_harness(mock_zmq, time_traveler):
    def create(
        conversations: list[tuple[str, int]] | None = None,
        *,
        schedule: list[tuple[int | float, str]] | None = None,
        request_count: int | None = None,
        num_sessions: int | None = None,
        num_users: int | None = None,
        concurrency: int | None = None,
        request_rate: float | None = None,
        user_centric_rate: float | None = None,
        arrival_pattern: ArrivalPattern = ArrivalPattern.POISSON,
        random_seed: int = 42,
        sampling_strategy: DatasetSamplingStrategy = DatasetSamplingStrategy.SHUFFLE,
        timing_mode: TimingMode | None = None,
        auto_offset_timestamps: bool = False,
        fixed_schedule_start_offset: int | None = None,
    ) -> OrchestratorHarness:
        if schedule is not None:
            dataset = make_dataset_with_schedule(schedule, sampling_strategy)
            mode = timing_mode or TimingMode.FIXED_SCHEDULE
            req_count = request_count or len(schedule)
        elif conversations is not None:
            convs = [
                ConversationMetadata(
                    conversation_id=cid, turns=[TurnMetadata() for _ in range(n)]
                )
                for cid, n in conversations
            ]
            dataset = DatasetMetadata(
                conversations=convs, sampling_strategy=sampling_strategy
            )
            mode = timing_mode or (
                TimingMode.USER_CENTRIC_RATE
                if user_centric_rate
                else TimingMode.REQUEST_RATE
            )
            req_count = request_count
        else:
            raise ValueError("conversations or schedule required")

        rate = user_centric_rate if user_centric_rate is not None else request_rate
        if concurrency is not None and rate is None:
            arrival_pattern = ArrivalPattern.CONCURRENCY_BURST

        cfg = make_timing_config(
            timing_mode=mode,
            arrival_pattern=arrival_pattern,
            concurrency=concurrency,
            request_rate=rate,
            request_count=req_count,
            num_sessions=num_sessions,
            num_users=num_users,
            random_seed=random_seed,
            auto_offset_timestamps=auto_offset_timestamps,
            fixed_schedule_start_offset=fixed_schedule_start_offset,
        )
        router = MockCreditRouter()
        pub = MagicMock()
        pub.publish = _async_noop
        publisher = PhasePublisher(pub_client=pub, service_id="test")
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=publisher,
            credit_router=router,
            dataset_metadata=dataset,
            service_id="test-orchestrator",
        )
        return OrchestratorHarness(orchestrator=orch, router=router)

    return create


def make_credit(
    id: int = 1,
    conv_id: str = "conv1",
    turn: int = 0,
    num_turns: int | None = None,
    is_final: bool | None = None,
    phase: CreditPhase = CreditPhase.PROFILING,
    corr_id: str | None = None,
) -> Credit:
    if num_turns is not None:
        n = num_turns
    elif is_final is not None:
        n = turn + 1 if is_final else turn + 2
    else:
        n = turn + 1
    return Credit(
        id=id,
        phase=phase,
        conversation_id=conv_id,
        x_correlation_id=corr_id or f"corr-{conv_id}",
        turn_index=turn,
        num_turns=n,
        issued_at_ns=time.time_ns(),
    )


def make_turn(
    conv_id: str = "conv1",
    turn: int = 0,
    num_turns: int = 1,
    corr_id: str | None = None,
) -> TurnToSend:
    return TurnToSend(
        conversation_id=conv_id,
        x_correlation_id=corr_id or f"corr-{conv_id}",
        turn_index=turn,
        num_turns=num_turns,
    )


def make_credit_return(
    credit: Credit,
    cancelled: bool = False,
    first_token_sent: bool = True,
    error: str | None = None,
) -> CreditReturn:
    return CreditReturn(
        credit=credit,
        cancelled=cancelled,
        first_token_sent=first_token_sent,
        error=error,
    )


def make_dataset(
    conv_ids: list[str],
    timestamps: list[int | float | None] | None = None,
    delays: list[list[int | float] | None] | None = None,
    turn_counts: list[int] | None = None,
    strategy: DatasetSamplingStrategy = DatasetSamplingStrategy.SEQUENTIAL,
) -> DatasetMetadata:
    convs = []
    for i, cid in enumerate(conv_ids):
        turns = []
        n = turn_counts[i] if turn_counts else 1
        if timestamps:
            ts = timestamps[i]
            turns.append(TurnMetadata(timestamp_ms=ts, delay_ms=None))
            if n > 1:
                d = delays[i] if delays and i < len(delays) else None
                if d:
                    for j, delay in enumerate(d):
                        t = ts + sum(d[: j + 1]) if ts is not None else None
                        turns.append(TurnMetadata(timestamp_ms=t, delay_ms=delay))
                else:
                    turns.extend(
                        [
                            TurnMetadata(timestamp_ms=None, delay_ms=None)
                            for _ in range(n - 1)
                        ]
                    )
        else:
            turns = [TurnMetadata(timestamp_ms=None, delay_ms=None) for _ in range(n)]
        convs.append(ConversationMetadata(conversation_id=cid, turns=turns))
    return DatasetMetadata(conversations=convs, sampling_strategy=strategy)


def make_dataset_with_schedule(
    schedule: list[tuple[int, str]],
    strategy: DatasetSamplingStrategy = DatasetSamplingStrategy.SEQUENTIAL,
) -> DatasetMetadata:
    by_conv: dict[str, list[int]] = {}
    for ts, cid in schedule:
        by_conv.setdefault(cid, []).append(ts)
    convs = []
    for cid, tss in by_conv.items():
        turns = [TurnMetadata(timestamp_ms=tss[0], delay_ms=None)]
        turns.extend(
            TurnMetadata(timestamp_ms=tss[i], delay_ms=tss[i] - tss[i - 1])
            for i in range(1, len(tss))
        )
        convs.append(ConversationMetadata(conversation_id=cid, turns=turns))
    return DatasetMetadata(conversations=convs, sampling_strategy=strategy)


def make_phase_config(
    phase: CreditPhase = CreditPhase.PROFILING,
    timing_mode: TimingMode = TimingMode.REQUEST_RATE,
    request_count: int | None = None,
    num_sessions: int | None = None,
    duration_sec: float | None = None,
    concurrency: int | None = None,
    prefill_concurrency: int | None = None,
    request_rate: float | None = None,
    arrival_pattern: ArrivalPattern = ArrivalPattern.POISSON,
    num_users: int | None = None,
    grace_period_sec: float | None = None,
    seamless: bool = False,
    auto_offset_timestamps: bool = False,
    fixed_schedule_start_offset: int | None = None,
    fixed_schedule_end_offset: int | None = None,
    concurrency_ramp_duration_sec: float | None = None,
    prefill_concurrency_ramp_duration_sec: float | None = None,
    request_rate_ramp_duration_sec: float | None = None,
) -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=phase,
        timing_mode=timing_mode,
        total_expected_requests=request_count,
        expected_num_sessions=num_sessions,
        expected_duration_sec=duration_sec,
        concurrency=concurrency,
        prefill_concurrency=prefill_concurrency,
        request_rate=request_rate,
        arrival_pattern=arrival_pattern,
        num_users=num_users,
        grace_period_sec=grace_period_sec,
        seamless=seamless,
        auto_offset_timestamps=auto_offset_timestamps,
        fixed_schedule_start_offset=fixed_schedule_start_offset,
        fixed_schedule_end_offset=fixed_schedule_end_offset,
        concurrency_ramp_duration_sec=concurrency_ramp_duration_sec,
        prefill_concurrency_ramp_duration_sec=prefill_concurrency_ramp_duration_sec,
        request_rate_ramp_duration_sec=request_rate_ramp_duration_sec,
    )


def make_timing_config(
    timing_mode: TimingMode = TimingMode.REQUEST_RATE,
    phase: CreditPhase = CreditPhase.PROFILING,
    request_count: int | None = None,
    num_sessions: int | None = None,
    duration_sec: float | None = None,
    concurrency: int | None = None,
    prefill_concurrency: int | None = None,
    request_rate: float | None = None,
    arrival_pattern: ArrivalPattern = ArrivalPattern.POISSON,
    num_users: int | None = None,
    grace_period_sec: float | None = None,
    random_seed: int | None = None,
    request_cancellation_rate: float | None = None,
    request_cancellation_delay: float = 0.0,
    auto_offset_timestamps: bool = False,
    fixed_schedule_start_offset: int | None = None,
    fixed_schedule_end_offset: int | None = None,
    phase_configs: list[CreditPhaseConfig] | None = None,
    concurrency_ramp_duration_sec: float | None = None,
    prefill_concurrency_ramp_duration_sec: float | None = None,
    request_rate_ramp_duration_sec: float | None = None,
) -> TimingConfig:
    if phase_configs is None:
        phase_configs = [
            make_phase_config(
                phase=phase,
                timing_mode=timing_mode,
                request_count=request_count,
                num_sessions=num_sessions,
                duration_sec=duration_sec,
                concurrency=concurrency,
                prefill_concurrency=prefill_concurrency,
                request_rate=request_rate,
                arrival_pattern=arrival_pattern,
                num_users=num_users,
                grace_period_sec=grace_period_sec,
                auto_offset_timestamps=auto_offset_timestamps,
                fixed_schedule_start_offset=fixed_schedule_start_offset,
                fixed_schedule_end_offset=fixed_schedule_end_offset,
                concurrency_ramp_duration_sec=concurrency_ramp_duration_sec,
                prefill_concurrency_ramp_duration_sec=prefill_concurrency_ramp_duration_sec,
                request_rate_ramp_duration_sec=request_rate_ramp_duration_sec,
            )
        ]
    return TimingConfig(
        phase_configs=phase_configs,
        random_seed=random_seed,
        request_cancellation=RequestCancellationConfig(
            rate=request_cancellation_rate, delay=request_cancellation_delay
        ),
    )


def profiling_stats_from_config(cfg: TimingConfig) -> CreditPhaseStats:
    pc = next(
        (p for p in cfg.phase_configs if p.phase == CreditPhase.PROFILING),
        cfg.phase_configs[0] if cfg.phase_configs else None,
    )
    return CreditPhaseStats(
        phase=CreditPhase.PROFILING,
        start_ns=time.time_ns(),
        total_expected_requests=pc.total_expected_requests if pc else None,
    )


def get_session_stats(
    orch: PhaseOrchestrator, phase: CreditPhase | None = None
) -> ConcurrencyStats | None:
    return orch._concurrency_manager.get_session_stats(phase)


def make_sampler(
    conv_ids: list[str] | None = None,
    strategy: DatasetSamplingStrategy = DatasetSamplingStrategy.SEQUENTIAL,
):
    SamplerClass = plugins.get_class(PluginType.DATASET_SAMPLER, strategy)
    return SamplerClass(conversation_ids=conv_ids or ["conv1", "conv2", "conv3"])


class InstantWorker(Worker):
    received_credits: list[Credit]
    received_timestamps: list[int]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.received_credits, self.received_timestamps = [], []

    async def _process_credit(self, ctx: CreditContext) -> None:
        self.received_credits.append(ctx.credit)
        self.received_timestamps.append(time.time_ns())
        ctx.first_token_sent = False


class TimingHarness:
    def __init__(self, service_config: ServiceConfig, user_config) -> None:
        from aiperf.credit.sticky_router import StickyCreditRouter

        self.bus = FakeCommunicationBus()
        FakeCommunication.set_shared_bus(self.bus)
        self.router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        self.publisher = PhasePublisher(
            pub_client=self.router.comms.create_pub_client(
                CommAddress.EVENT_BUS_PROXY_FRONTEND
            ),
            service_id="test-service",
        )
        self._worker = InstantWorker(
            service_config=service_config,
            user_config=user_config,
            service_id="instant-worker-1",
        )

    @property
    def dropped_credits(self) -> list[Credit]:
        return self._worker.received_credits

    @property
    def dropped_timestamps(self) -> list[int]:
        return self._worker.received_timestamps

    async def create_orchestrator(
        self, cfg: TimingConfig, dataset: DatasetMetadata | None = None, **kwargs
    ) -> PhaseOrchestrator:
        if dataset is None:
            dataset = make_dataset(conv_ids=["conv1", "conv2", "conv3"])
        orch = PhaseOrchestrator(
            config=cfg,
            phase_publisher=self.publisher,
            credit_router=self.router,
            dataset_metadata=dataset,
        )
        await orch.initialize()
        return orch

    async def run_orchestrator(self, orch: PhaseOrchestrator) -> None:
        await self.router.initialize()
        await self.router.start()
        await self._worker.initialize()
        await self._worker.start()
        task = asyncio.create_task(orch.start())
        while not task.done():
            await asyncio.sleep(0.01)
        await task


class MockCreditSender:
    def __init__(self) -> None:
        self.sent_credits: list[Credit] = []
        self.cancelled = False
        self._cb: Callable[[str, CreditReturn], Awaitable[None]] | None = None

    async def send_credit(self, credit: Credit) -> None:
        self.sent_credits.append(credit)

    async def cancel_all_credits(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.sent_credits.clear()
        self.cancelled = False

    def set_return_callback(
        self, cb: Callable[[str, CreditReturn], Awaitable[None]]
    ) -> None:
        self._cb = cb


@pytest.fixture
def timing_harness(
    service_config, user_config, skip_service_registration
) -> TimingHarness:
    return TimingHarness(service_config=service_config, user_config=user_config)


@pytest.fixture
def mock_credit_sender() -> MockCreditSender:
    return MockCreditSender()


@pytest.fixture
def router_with_worker(service_config):
    from aiperf.credit.sticky_router import StickyCreditRouter, WorkerLoad

    router = StickyCreditRouter(service_config=service_config, service_id="test-router")
    router._workers = {
        "worker-1": WorkerLoad(worker_id="worker-1", in_flight_credits=0)
    }
    return router


@pytest.fixture
def mock_phase_publisher() -> MagicMock:
    m = MagicMock()
    m.publish_phase_start = m.publish_phase_sending_complete = (
        m.publish_phase_complete
    ) = _async_noop
    m.publish_progress = m.publish_credits_complete = _async_noop
    return m


@pytest.fixture
def mock_credit_router() -> MagicMock:
    m = MagicMock()
    m.send_credit = m.cancel_all_credits = _async_noop
    m.mark_credits_complete = m.reset = m.set_return_callback = (
        m.set_first_token_callback
    ) = MagicMock()
    return m


@pytest.fixture
def mock_concurrency_manager() -> MagicMock:
    m = MagicMock()
    m.configure_for_phase = m.release_session_slot = m.release_prefill_slot = (
        MagicMock()
    )
    m.set_session_limit = m.set_prefill_limit = MagicMock()
    m.acquire_session_slot = m.acquire_prefill_slot = _async_true
    m.release_stuck_slots = MagicMock(return_value=(0, 0))
    m.get_session_stats = m.get_prefill_stats = MagicMock(return_value=None)
    return m


@pytest.fixture
def mock_stop_checker() -> MagicMock:
    m = MagicMock()
    m.can_send_any_turn = m.can_start_new_session = MagicMock(return_value=True)
    return m


@pytest.fixture
def mock_progress_tracker() -> MagicMock:
    m = MagicMock()
    m.increment_sent = MagicMock(return_value=(1, False))
    m.increment_returned = MagicMock(return_value=False)
    m.increment_prefill_released = m.freeze_sent_counts = m.freeze_completed_counts = (
        m.create_stats
    ) = MagicMock()
    m.check_all_returned_or_cancelled = MagicMock(return_value=False)
    m.all_credits_sent_event = asyncio.Event()
    m.all_credits_returned_event = asyncio.Event()
    m.in_flight_sessions = 0
    m.counter = MagicMock()
    return m


@pytest.fixture
def mock_lifecycle() -> MagicMock:
    m = MagicMock()
    m.is_complete = m.is_sending_complete = False
    m.started_at_perf_ns = 1_000_000_000
    m.start = m.mark_sending_complete = m.mark_complete = m.cancel = MagicMock()
    m.time_left_in_seconds = MagicMock(return_value=60.0)
    return m


@pytest.fixture
def mock_callback_handler() -> MagicMock:
    m = MagicMock()
    m.register_phase = m.unregister_phase = MagicMock()
    m.on_credit_return = m.on_first_token = _async_noop
    return m


@pytest.fixture
def mock_cancellation_policy() -> MagicMock:
    m = MagicMock()
    m.next_cancellation_delay_ns = MagicMock(return_value=None)
    return m


@pytest.fixture
def mock_orchestrator(create_orchestrator_harness):
    def create(
        conversations: list[tuple[str, int]],
        *,
        num_sessions: int | None = None,
        request_count: int | None = None,
        concurrency: int | None = None,
        request_rate: float | None = None,
    ) -> OrchestratorHarness:
        return create_orchestrator_harness(
            conversations=conversations,
            num_sessions=num_sessions,
            request_count=request_count,
            concurrency=concurrency,
            request_rate=request_rate,
        )

    return create
