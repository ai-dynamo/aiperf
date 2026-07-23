# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import BaselineKind, CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.models import (
    ConversationMetadata,
    CreditPhaseStats,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.config.rate_series import RateSeriesConfig
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import ArrivalPattern, DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.runner import PhaseRunner

pytestmark = pytest.mark.looptime


def make_dataset_metadata(conversations: list[tuple[str, int]]) -> DatasetMetadata:
    """Create dataset metadata with specified conversations.

    Args:
        conversations: List of (conversation_id, num_turns) tuples.
    """
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id=conv_id,
                turns=[TurnMetadata() for _ in range(num_turns)],
            )
            for conv_id, num_turns in conversations
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


@dataclass
class MockStrategy:
    setup_called: bool = False
    execute_called: bool = False
    handle_credit_return_calls: list[Credit] = field(default_factory=list)
    execute_delay: float = 0.0
    _execute_event: asyncio.Event = field(default_factory=asyncio.Event)

    async def setup_phase(self) -> None:
        self.setup_called = True

    async def execute_phase(self) -> None:
        self.execute_called = True
        if self.execute_delay > 0:
            await asyncio.sleep(self.execute_delay)
        self._execute_event.set()

    async def handle_credit_return(self, credit: Credit) -> None:
        self.handle_credit_return_calls.append(credit)


@dataclass
class MockRateSettableStrategy(MockStrategy):
    rate_updates: list[float] = field(default_factory=list)

    def set_request_rate(self, rate: float) -> None:
        self.rate_updates.append(rate)


def mock_conc_mgr() -> MagicMock:
    m = MagicMock()
    m.configure_for_phase = MagicMock()
    m.acquire_session_slot = AsyncMock(return_value=True)
    m.acquire_prefill_slot = AsyncMock(return_value=True)
    m.release_session_slot = m.release_prefill_slot = MagicMock()
    m.set_session_limit = m.set_prefill_limit = MagicMock()
    m.release_stuck_slots = MagicMock(return_value=(0, 0))
    return m


def mock_cancel_policy() -> MagicMock:
    m = MagicMock()
    m.next_cancellation_delay_ns = MagicMock(return_value=None)
    return m


def mock_callback() -> MagicMock:
    m = MagicMock()
    m.register_phase = m.unregister_phase = MagicMock()
    m.on_credit_return = m.on_first_token = AsyncMock()
    return m


def cfg(
    phase: CreditPhase = CreditPhase.PROFILING,
    mode: TimingMode = TimingMode.REQUEST_RATE,
    reqs: int | None = 10,
    dur: float | None = None,
    conc: int | None = None,
    prefill_conc: int | None = None,
    rate: float | None = 10.0,
    grace: float | None = 1.0,
    seamless: bool = False,
    conc_ramp: float | None = None,
    prefill_ramp: float | None = None,
    rate_ramp: float | None = None,
    rate_series: RateSeriesConfig | None = None,
) -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=phase,
        timing_mode=mode,
        total_expected_requests=reqs,
        expected_duration_sec=dur,
        concurrency=conc,
        prefill_concurrency=prefill_conc,
        request_rate=rate,
        arrival_pattern=ArrivalPattern.POISSON,
        grace_period_sec=grace,
        seamless=seamless,
        concurrency_ramp_duration_sec=conc_ramp,
        prefill_concurrency_ramp_duration_sec=prefill_ramp,
        request_rate_ramp_duration_sec=rate_ramp,
        request_rate_series=rate_series,
    )


def make_runner(
    config: CreditPhaseConfig,
    conv_src: MagicMock,
    pub: MagicMock,
    router: MagicMock,
    conc: MagicMock,
    cancel: MagicMock,
    cb: MagicMock,
) -> PhaseRunner:
    return PhaseRunner(
        config=config,
        conversation_source=conv_src,
        phase_publisher=pub,
        credit_router=router,
        concurrency_manager=conc,
        cancellation_policy=cancel,
        callback_handler=cb,
    )


@pytest.fixture
def conv_src() -> MagicMock:
    m = MagicMock()
    m.next = MagicMock()
    return m


@pytest.fixture
def pub() -> MagicMock:
    m = MagicMock()
    m.publish_phase_start = AsyncMock()
    m.publish_phase_baseline_request = AsyncMock()
    m.publish_phase_sending_complete = AsyncMock()
    m.publish_phase_complete = AsyncMock()
    m.publish_progress = AsyncMock()
    m.publish_credits_complete = AsyncMock()
    return m


@pytest.fixture
def router() -> MagicMock:
    m = MagicMock()
    m.send_credit = m.cancel_all_credits = AsyncMock()
    m.wait_for_workers = AsyncMock()
    m.mark_credits_complete = MagicMock()
    return m


@pytest.fixture
def conc() -> MagicMock:
    return mock_conc_mgr()


@pytest.fixture
def cancel() -> MagicMock:
    return mock_cancel_policy()


@pytest.fixture
def cb() -> MagicMock:
    return mock_callback()


@pytest.fixture
async def runner(
    conv_src: MagicMock,
    pub: MagicMock,
    router: MagicMock,
    conc: MagicMock,
    cancel: MagicMock,
    cb: MagicMock,
) -> PhaseRunner:
    return make_runner(cfg(), conv_src, pub, router, conc, cancel, cb)


class TestPhaseRunnerLifecycle:
    async def test_baseline_boundary_capture_is_fire_and_forget(
        self, runner: PhaseRunner, pub: MagicMock
    ) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_publish(*args, **kwargs):
            started.set()
            await release.wait()

        pub.publish_phase_baseline_request = AsyncMock(side_effect=slow_publish)

        boundary_ns = runner._capture_baseline_boundary("phase-1", BaselineKind.START)

        assert isinstance(boundary_ns, int)
        await asyncio.wait_for(started.wait(), timeout=0.1)
        pub.publish_phase_baseline_request.assert_called_once_with(
            runner._config, "phase-1", BaselineKind.START
        )
        release.set()
        await asyncio.sleep(0)

    async def test_return_complete_waits_for_end_baseline_publish_before_phase_complete(
        self, runner: PhaseRunner, pub: MagicMock
    ) -> None:
        events: list[str] = []
        release_baseline = asyncio.Event()

        async def slow_baseline_publish(*args, **kwargs):
            events.append("baseline_start")
            await release_baseline.wait()
            events.append("baseline_done")

        async def record_phase_complete(*args, **kwargs):
            events.append("phase_complete")

        pub.publish_phase_baseline_request = AsyncMock(
            side_effect=slow_baseline_publish
        )
        pub.publish_phase_complete = AsyncMock(side_effect=record_phase_complete)
        runner._baseline_start_ns = None
        runner._baseline_end_ns = None
        runner._lifecycle.start()
        runner._lifecycle.mark_sending_complete(timeout_triggered=False)
        runner._progress.all_credits_returned_event.set()

        task = asyncio.create_task(
            runner._wait_for_returning_complete(phase_id="phase-1")
        )
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)

        assert events == ["baseline_start"]
        assert not task.done()
        pub.publish_phase_complete.assert_not_called()

        release_baseline.set()
        await asyncio.wait_for(task, timeout=0.1)

        assert events == ["baseline_start", "baseline_done", "phase_complete"]
        pub.publish_phase_baseline_request.assert_called_once_with(
            runner._config, "phase-1", BaselineKind.END
        )

    async def test_run_creates_strategy_via_factory(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(cfg(), conv_src, pub, router, conc, cancel, cb)
        strategy = MockStrategy()
        captured_kwargs = {}

        def mock_class(**kwargs):
            captured_kwargs.update(kwargs)
            return strategy

        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=mock_class,
        ) as f:
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            await r.run(is_final_phase=True)
            f.assert_called_once()
            assert (
                "scheduler" in captured_kwargs
                and "stop_checker" in captured_kwargs
                and "credit_issuer" in captured_kwargs
                and "lifecycle" in captured_kwargs
            )

    async def test_run_registers_phase_with_callback_handler(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(cfg(), conv_src, pub, router, conc, cancel, cb)
        strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: strategy,
        ):
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            await r.run(is_final_phase=True)
            cb.register_phase.assert_called_once()
            assert cb.register_phase.call_args.kwargs["phase"] == CreditPhase.PROFILING

    async def test_run_configures_concurrency_manager(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        c = cfg(conc=10)
        r = make_runner(c, conv_src, pub, router, conc, cancel, cb)
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: MockStrategy(),
        ):
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            await r.run(is_final_phase=True)
            conc.configure_for_phase.assert_called_once_with(
                c.phase, c.concurrency, c.prefill_concurrency
            )

    async def test_run_publishes_start_and_complete(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(cfg(), conv_src, pub, router, conc, cancel, cb)
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: MockStrategy(),
        ):
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            await r.run(is_final_phase=True)
            pub.publish_phase_start.assert_called_once()
            pub.publish_phase_complete.assert_called_once()

    async def test_run_returns_stats(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(cfg(), conv_src, pub, router, conc, cancel, cb)
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: MockStrategy(),
        ):
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            result = await r.run(is_final_phase=True)
            assert (
                isinstance(result, CreditPhaseStats)
                and result.phase == CreditPhase.PROFILING
            )


class TestRamperCreation:
    async def test_no_rampers_without_ramp_duration(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(
            cfg(conc=10, rate=100.0), conv_src, pub, router, conc, cancel, cb
        )
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: MockStrategy(),
        ):
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            await r.run(is_final_phase=True)
            assert len(r._rampers) == 0

    async def test_session_concurrency_ramper_created(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(
            cfg(conc=10, conc_ramp=5.0), conv_src, pub, router, conc, cancel, cb
        )
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: MockStrategy(),
        ):
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            await r.run(is_final_phase=True)
            assert len(r._rampers) >= 1

    async def test_prefill_concurrency_ramper_created(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(
            cfg(prefill_conc=5, prefill_ramp=3.0),
            conv_src,
            pub,
            router,
            conc,
            cancel,
            cb,
        )
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: MockStrategy(),
        ):
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            await r.run(is_final_phase=True)
            assert len(r._rampers) >= 1

    async def test_rate_ramper_requires_rate_settable_strategy(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(
            cfg(rate=100.0, rate_ramp=10.0), conv_src, pub, router, conc, cancel, cb
        )
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: MockStrategy(),
        ):
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            await r.run(is_final_phase=True)
            assert len(r._rampers) == 0

    async def test_rate_series_controller_created_for_rate_settable_strategy(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        rate_series = RateSeriesConfig(
            points=[{"time_s": 0, "qps": 5}, {"time_s": 10, "qps": 15}]
        )
        r = make_runner(
            cfg(rate=5.0, rate_series=rate_series),
            conv_src,
            pub,
            router,
            conc,
            cancel,
            cb,
        )
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: MockRateSettableStrategy(),
        ):
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            await r.run(is_final_phase=True)
            assert len(r._rampers) == 1

    async def test_rate_series_requires_rate_settable_strategy(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        rate_series = RateSeriesConfig(
            points=[{"time_s": 0, "qps": 5}, {"time_s": 10, "qps": 15}]
        )
        r = make_runner(
            cfg(rate=5.0, rate_series=rate_series),
            conv_src,
            pub,
            router,
            conc,
            cancel,
            cb,
        )
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: MockStrategy(),
        ):
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            await r.run(is_final_phase=True)
            assert len(r._rampers) == 0


class TestPhaseRunnerCancellation:
    async def test_cancel_sets_flag(self, runner: PhaseRunner) -> None:
        assert runner._was_cancelled is False
        runner.cancel()
        assert runner._was_cancelled is True

    async def test_cancel_cancels_lifecycle(self, runner: PhaseRunner) -> None:
        runner.cancel()
        assert runner._lifecycle.was_cancelled is True

    async def test_cancel_stops_rampers(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(
            cfg(conc=10, conc_ramp=5.0), conv_src, pub, router, conc, cancel, cb
        )
        mock_ramper = MagicMock()
        r._rampers = [mock_ramper]
        r.cancel()
        mock_ramper.stop.assert_called_once()

    async def test_cancel_cancels_scheduler(self, runner: PhaseRunner) -> None:
        async def dummy() -> None:
            await asyncio.sleep(10)

        runner._scheduler.schedule_later(10.0, dummy())
        assert runner._scheduler.pending_count > 0
        runner.cancel()
        assert runner._scheduler.pending_count == 0


class TestTimeoutHandling:
    async def test_returns_false_when_event_set(
        self, runner: PhaseRunner, time_traveler: MagicMock
    ) -> None:
        event = asyncio.Event()
        event.set()
        result = await runner._wait_for_event_with_timeout(
            name="t", event=event, timeout=10.0, task_to_cancel=None
        )
        assert result is False

    async def test_returns_true_on_timeout(
        self, runner: PhaseRunner, time_traveler: MagicMock
    ) -> None:
        event = asyncio.Event()
        result = await runner._wait_for_event_with_timeout(
            name="t", event=event, timeout=0.001, task_to_cancel=None
        )
        assert result is True

    async def test_cancels_task_on_timeout(
        self, runner: PhaseRunner, time_traveler: MagicMock
    ) -> None:
        event, never = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(never.wait())
        await runner._wait_for_event_with_timeout(
            name="t", event=event, timeout=0.001, task_to_cancel=task
        )
        await asyncio.sleep(0)
        assert task.cancelled()

    async def test_sets_event_on_timeout_when_configured(
        self, runner: PhaseRunner, time_traveler: MagicMock
    ) -> None:
        event = asyncio.Event()
        await runner._wait_for_event_with_timeout(
            name="t",
            event=event,
            timeout=0.001,
            task_to_cancel=None,
            set_event_on_timeout=True,
        )
        assert event.is_set()

    async def test_returns_true_when_timeout_zero(
        self, runner: PhaseRunner, time_traveler: MagicMock
    ) -> None:
        event = asyncio.Event()
        result = await runner._wait_for_event_with_timeout(
            name="t", event=event, timeout=0, task_to_cancel=None
        )
        assert result is True

    async def test_waits_indefinitely_when_timeout_none(
        self, runner: PhaseRunner, time_traveler: MagicMock
    ) -> None:
        event = asyncio.Event()

        async def set_later() -> None:
            await asyncio.sleep(0.01)
            event.set()

        asyncio.create_task(set_later())
        result = await runner._wait_for_event_with_timeout(
            name="t", event=event, timeout=None, task_to_cancel=None
        )
        assert result is False


class TestStuckSlotsRelease:
    async def test_release_calls_concurrency_manager(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(cfg(), conv_src, pub, router, conc, cancel, cb)
        r._release_stuck_slots()
        conc.release_stuck_slots.assert_called_once_with(CreditPhase.PROFILING)


class TestProgressReporting:
    async def test_progress_loop_publishes_stats(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
        time_traveler: MagicMock,
    ) -> None:
        r = make_runner(cfg(), conv_src, pub, router, conc, cancel, cb)
        task = asyncio.create_task(r._progress_report_loop())
        # Use longer sleep to ensure at least one progress publish in CI environments
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pub.publish_progress.call_count >= 1


class TestSeamlessMode:
    async def test_returns_early_for_non_final_phase(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(cfg(seamless=True), conv_src, pub, router, conc, cancel, cb)
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: MockStrategy(),
        ):
            r._progress.all_credits_sent_event.set()
            result = await asyncio.wait_for(r.run(is_final_phase=False), timeout=1.0)
            assert isinstance(result, CreditPhaseStats)

    async def test_waits_for_returns_on_final_phase(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(cfg(seamless=True), conv_src, pub, router, conc, cancel, cb)
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: MockStrategy(),
        ):
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            await r.run(is_final_phase=True)
            pub.publish_phase_complete.assert_called_once()


class TestComponentOwnership:
    def test_phase_property_returns_configured_phase(self, runner: PhaseRunner) -> None:
        assert runner.phase == CreditPhase.PROFILING


class TestPhaseTypes:
    @pytest.mark.parametrize("phase", [CreditPhase.WARMUP, CreditPhase.PROFILING])
    async def test_runner_works_with_both_phases(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
        phase: CreditPhase,
    ) -> None:
        r = make_runner(cfg(phase=phase), conv_src, pub, router, conc, cancel, cb)
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: MockStrategy(),
        ):
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            result = await r.run(is_final_phase=True)
            assert result.phase == phase


class TestEdgeCases:
    async def test_already_complete_returns_immediately(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(cfg(), conv_src, pub, router, conc, cancel, cb)
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: MockStrategy(),
        ):
            r._progress.all_credits_sent_event.set()
            r._progress.all_credits_returned_event.set()
            r._progress._counter._final_requests_sent = 0
            result = await r.run(is_final_phase=True)
            assert isinstance(result, CreditPhaseStats)

    async def test_cleanup_runs_on_exception(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(cfg(), conv_src, pub, router, conc, cancel, cb)
        mock_ramper = MagicMock()
        r._rampers = [mock_ramper]
        strategy = MagicMock()
        strategy.setup_phase = AsyncMock(side_effect=RuntimeError("Test error"))
        with patch(
            "aiperf.timing.phase.runner.plugins.get_class",
            return_value=lambda **kw: strategy,
        ):
            with pytest.raises(RuntimeError):
                await r.run(is_final_phase=True)
            mock_ramper.stop.assert_called_once()

    async def test_timeout_skips_cancel_drain_when_all_credits_returned(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(cfg(dur=1.0), conv_src, pub, router, conc, cancel, cb)
        stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            start_ns=1,
            sent_end_ns=2,
            requests_end_ns=3,
            final_requests_sent=94,
            requests_sent=94,
            requests_completed=94,
            requests_cancelled=0,
            final_requests_completed=94,
            final_requests_cancelled=0,
            final_request_errors=0,
            final_sent_sessions=1,
            final_completed_sessions=1,
            final_cancelled_sessions=0,
        )
        r._progress.create_stats = MagicMock(return_value=stats)
        # Force the post-timeout cancel branch: pretend credits are still
        # outstanding at the initial check, time out the wait, then recompute
        # need == 0 so the empty-drain guard sets the event without waiting.
        r._progress.check_all_returned_or_cancelled = MagicMock(return_value=False)
        r._lifecycle.start()
        r._lifecycle.mark_sending_complete()

        with patch.object(
            r,
            "_wait_for_event_with_timeout",
            new=AsyncMock(return_value=True),
        ):
            await r._wait_for_returning_complete()

        router.cancel_all_credits.assert_awaited_once()
        assert r._progress.all_credits_returned_event.is_set()
        r._concurrency_manager.release_stuck_slots.assert_not_called()


class TestFixedScheduleConfigCorrection:
    """Tests for FIXED_SCHEDULE mode config correction using actual dataset size."""

    async def test_fixed_schedule_updates_config_from_dataset_metadata(
        self,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        """FIXED_SCHEDULE should use dataset metadata size, not config values."""
        # Config says 100 requests/sessions, but dataset only has 2 conversations
        config = cfg(mode=TimingMode.FIXED_SCHEDULE, reqs=100)
        config = config.model_copy(update={"expected_num_sessions": 100})

        conv_src = MagicMock()
        conv_src.dataset_metadata = make_dataset_metadata([("c1", 3), ("c2", 2)])

        r = make_runner(config, conv_src, pub, router, conc, cancel, cb)

        # Config should be updated to reflect actual dataset size
        assert r._config.total_expected_requests == 5  # 3 + 2 turns
        assert r._config.expected_num_sessions == 2  # 2 conversations

    async def test_fixed_schedule_without_metadata_uses_config(
        self,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        """FIXED_SCHEDULE without metadata falls back to config values."""
        config = cfg(mode=TimingMode.FIXED_SCHEDULE, reqs=100)

        conv_src = MagicMock()
        conv_src.dataset_metadata = None

        r = make_runner(config, conv_src, pub, router, conc, cancel, cb)

        # Config should remain unchanged
        assert r._config.total_expected_requests == 100

    async def test_request_rate_mode_ignores_dataset_metadata(
        self,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        """REQUEST_RATE mode should use config values even with metadata."""
        config = cfg(mode=TimingMode.REQUEST_RATE, reqs=100)

        conv_src = MagicMock()
        conv_src.dataset_metadata = make_dataset_metadata([("c1", 1)])

        r = make_runner(config, conv_src, pub, router, conc, cancel, cb)

        # Config should remain unchanged for REQUEST_RATE
        assert r._config.total_expected_requests == 100

    async def test_user_centric_rate_mode_ignores_dataset_metadata(
        self,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        """USER_CENTRIC_RATE mode should use config values even with metadata."""
        config = cfg(mode=TimingMode.USER_CENTRIC_RATE, reqs=100)

        conv_src = MagicMock()
        conv_src.dataset_metadata = make_dataset_metadata([("c1", 1)])

        r = make_runner(config, conv_src, pub, router, conc, cancel, cb)

        # Config should remain unchanged for USER_CENTRIC_RATE
        assert r._config.total_expected_requests == 100

    async def test_fixed_schedule_filtered_dataset_scenario(
        self,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        """Simulates start/end offset filtering that reduces dataset size."""
        # Original file had 1000 conversations, config reflects that
        config = cfg(mode=TimingMode.FIXED_SCHEDULE, reqs=1000)
        config = config.model_copy(update={"expected_num_sessions": 1000})

        # But filtering reduced to just 2 conversations
        conv_src = MagicMock()
        conv_src.dataset_metadata = make_dataset_metadata(
            [("filtered_1", 1), ("filtered_2", 1)]
        )

        r = make_runner(config, conv_src, pub, router, conc, cancel, cb)

        # Config should reflect the filtered dataset, not the original
        assert r._config.total_expected_requests == 2
        assert r._config.expected_num_sessions == 2


class TestPhaseRunnerWorkerReadiness:
    """The phase must gate credit issuance on worker readiness. Regression
    coverage for the startup-race deadlock (see PhaseRunner._run_strategy)."""

    async def test_run_strategy_awaits_wait_for_workers_with_start_timeout(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        # Pins the barrier's timeout to START_TIMEOUT. Gating is proven by the
        # real-router test below; the mock barrier here never blocks.
        r = make_runner(cfg(), conv_src, pub, router, conc, cancel, cb)
        r._progress.all_credits_sent_event.set()
        r._progress.all_credits_returned_event.set()

        await r._run_strategy(MockStrategy(), is_final_phase=True)

        router.wait_for_workers.assert_awaited_once_with(
            timeout=Environment.SERVICE.START_TIMEOUT
        )

    async def test_run_strategy_blocks_execution_until_worker_registers(
        self,
        benchmark_run,
        conv_src: MagicMock,
        pub: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        # Regression guard for the startup-race deadlock. Uses the REAL
        # StickyCreditRouter barrier (not a mock), so it actually gates
        # execution: with no workers, _run_strategy must block before running
        # execute_phase. A barrier placed after execute_async (the original
        # bug) would let execute_phase run here and fail the first assert.
        real_router = StickyCreditRouter(run=benchmark_run, service_id="test-router")
        r = make_runner(cfg(), conv_src, pub, real_router, conc, cancel, cb)

        class GatedStrategy(MockStrategy):
            async def execute_phase(self) -> None:
                self.execute_called = True
                r._progress.all_credits_sent_event.set()
                r._progress.all_credits_returned_event.set()

        strategy = GatedStrategy()
        run_task = asyncio.create_task(r._run_strategy(strategy, is_final_phase=True))

        # Advance to the barrier. With no worker registered the phase must block
        # there, before kicking off the issuance task. ``_execution_task`` stays
        # None iff the barrier precedes issuance; if it has been created here the
        # barrier was placed after ``execute_async`` (the original deadlock).
        await asyncio.sleep(0.1)
        assert r._execution_task is None, (
            "issuance task was created before the worker-readiness barrier "
            "released: the barrier is not gating execution (startup-race "
            "regression)"
        )
        assert not run_task.done()

        # First worker registers -> barrier releases -> phase runs to completion.
        real_router._register_worker("worker-1")
        await asyncio.wait_for(run_task, timeout=5.0)
        assert strategy.execute_called


class TestWarmupProgressHeartbeat:
    """Warmup emits a throttled INFO heartbeat in _progress_report_loop
    (AIPERF_SERVICE_WARMUP_PROGRESS_LOG_INTERVAL, default 30s, 0 disables) so
    headless warmup stays observable when no interactive UI consumes progress.
    """

    def test_format_warmup_progress_includes_returned_target_and_elapsed(self):
        stats = CreditPhaseStats(
            phase=CreditPhase.WARMUP,
            requests_sent=40,
            requests_completed=30,
            requests_cancelled=2,
            request_errors=1,
            total_expected_requests=50,
        )
        msg = PhaseRunner._format_warmup_progress(stats)
        assert "Phase warmup progress" in msg
        assert "returned=32/50" in msg  # completed + cancelled / target
        assert "sent=40" in msg
        assert "errors=1" in msg
        assert "elapsed=" in msg

    def test_format_warmup_progress_prefers_final_requests_sent(self):
        # When both are populated, final_requests_sent is the primary target;
        # total_expected_requests is only the fallback. Guards a regression
        # that would silently prefer the fallback.
        stats = CreditPhaseStats(
            phase=CreditPhase.WARMUP,
            requests_sent=40,
            requests_completed=30,
            requests_cancelled=2,
            final_requests_sent=42,
            total_expected_requests=50,
        )
        msg = PhaseRunner._format_warmup_progress(stats)
        assert "returned=32/42" in msg  # completed + cancelled / final_requests_sent
        assert "/50" not in msg

    def test_format_warmup_progress_without_target(self):
        stats = CreditPhaseStats(
            phase=CreditPhase.WARMUP,
            requests_sent=5,
            requests_completed=3,
        )
        msg = PhaseRunner._format_warmup_progress(stats)
        assert "returned=3" in msg and "returned=3/" not in msg  # no target

    def test_warmup_log_interval_env_field_exists(self):
        # 0 disables; default is a positive heartbeat cadence.
        assert Environment.SERVICE.WARMUP_PROGRESS_LOG_INTERVAL >= 0.0

    async def _drive_progress_loop(
        self,
        runner: PhaseRunner,
        pub: MagicMock,
        monotonic_values: list[float],
        stop_after_publishes: int,
    ) -> MagicMock:
        """Run _progress_report_loop with a controlled clock and stop it after
        a fixed number of publish_progress calls. Returns the patched info mock.
        """
        stats = CreditPhaseStats(phase=runner._config.phase, requests_sent=1)
        runner._progress.create_stats = MagicMock(return_value=stats)

        publishes = 0

        async def _publish(_stats: CreditPhaseStats) -> None:
            nonlocal publishes
            publishes += 1
            if publishes >= stop_after_publishes:
                raise asyncio.CancelledError

        pub.publish_progress = AsyncMock(side_effect=_publish)

        clock = iter(monotonic_values)
        last = [monotonic_values[-1]]

        def _monotonic() -> float:
            try:
                return next(clock)
            except StopIteration:
                return last[0]

        info_mock = MagicMock()
        with (
            patch("aiperf.timing.phase.runner.time.monotonic", new=_monotonic),
            patch.object(runner, "info", new=info_mock),
            pytest.raises(asyncio.CancelledError),
        ):
            await runner._progress_report_loop()
        return info_mock

    async def test_warmup_heartbeat_throttled_to_once_per_interval(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(
            cfg(phase=CreditPhase.WARMUP), conv_src, pub, router, conc, cancel, cb
        )
        # interval=10; setup reads 0 (next_at=10). Loop reads now at 1,5,11,15,22:
        # fires at 11 (next_at=21) and 22 (next_at=32) -> exactly 2 heartbeats.
        with patch.object(Environment.SERVICE, "WARMUP_PROGRESS_LOG_INTERVAL", 10.0):
            info_mock = await self._drive_progress_loop(
                r, pub, [0, 1, 5, 11, 15, 22], stop_after_publishes=6
            )
        assert info_mock.call_count == 2

    async def test_warmup_heartbeat_disabled_when_interval_zero(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(
            cfg(phase=CreditPhase.WARMUP), conv_src, pub, router, conc, cancel, cb
        )
        with patch.object(Environment.SERVICE, "WARMUP_PROGRESS_LOG_INTERVAL", 0.0):
            info_mock = await self._drive_progress_loop(
                r, pub, [0, 100, 200, 300], stop_after_publishes=3
            )
        info_mock.assert_not_called()

    async def test_no_heartbeat_when_phase_not_warmup(
        self,
        conv_src: MagicMock,
        pub: MagicMock,
        router: MagicMock,
        conc: MagicMock,
        cancel: MagicMock,
        cb: MagicMock,
    ) -> None:
        r = make_runner(
            cfg(phase=CreditPhase.PROFILING), conv_src, pub, router, conc, cancel, cb
        )
        with patch.object(Environment.SERVICE, "WARMUP_PROGRESS_LOG_INTERVAL", 10.0):
            info_mock = await self._drive_progress_loop(
                r, pub, [0, 11, 22, 33], stop_after_publishes=3
            )
        info_mock.assert_not_called()
