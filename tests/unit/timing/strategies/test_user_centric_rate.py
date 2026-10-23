# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.strategies.user_centric_rate import User, UserCentricStrategy
from tests.unit.timing.conftest import OrchestratorHarness, make_sampler

TWO_TURN = [("c1", 2), ("c2", 2), ("c3", 2), ("c4", 2), ("c5", 2)]
MULTI_TURN = [("c1", 3), ("c2", 3), ("c3", 3), ("c4", 3)]


class TestUserCentricInit:
    @pytest.mark.parametrize(
        "num_users,rate,match",
        [(None, 10.0, "num_users must be set"), (5, None, "request_rate must be set")],
    )  # fmt: skip
    def test_missing_params_raises(self, num_users, rate, match) -> None:
        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            request_rate=rate,
            num_users=num_users,
            total_expected_requests=10,
        )
        with pytest.raises(ValueError, match=match):
            UserCentricStrategy(
                config=cfg,
                conversation_source=MagicMock(),
                scheduler=MagicMock(),
                stop_checker=MagicMock(),
                credit_issuer=MagicMock(),
                lifecycle=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_valid_config(self, create_orchestrator_harness) -> None:
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN,
            user_centric_rate=10.0,
            num_users=5,
            request_count=10,
        )
        await h.orchestrator.initialize()
        assert len(h.orchestrator._ordered_phase_configs) == 1
        assert h.orchestrator._ordered_phase_configs[0].phase == CreditPhase.PROFILING


@pytest.mark.asyncio
class TestUserCentricExecution:
    @pytest.mark.parametrize(
        "convs,rate,users,count,expected",
        [
            (TWO_TURN * 10, 20.0, 5, 10, 10),
            (TWO_TURN * 4, 10.0, 10, 10, 10),
            (MULTI_TURN * 2, 20.0, 4, 4, 4),
            (TWO_TURN * 4, 50.0, 10, 25, 25),
            (MULTI_TURN * 3, 40.0, 8, 30, 30),
            (MULTI_TURN, 10.0, 1, 10, 10),
            (TWO_TURN * 20, 100.0, 50, 100, 100),
            (TWO_TURN * 2, 1.0, 2, 2, 2),
            (TWO_TURN * 100, 500.0, 100, 500, 500),
            (TWO_TURN, 10.0, 5, 5, 5),
            (TWO_TURN * 4, 20.0, 20, 10, 10),
            (TWO_TURN * 4, 40.0, 10, 10, 10),
        ],
    )  # fmt: skip
    async def test_issues_expected_credits(
        self, create_orchestrator_harness, convs, rate, users, count, expected
    ) -> None:
        """Verify strategy issues correct number of credits across various configurations."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=convs,
            user_centric_rate=rate,
            num_users=users,
            request_count=count,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) >= expected


@pytest.mark.asyncio
class TestSessionTracking:
    async def test_unique_correlation_ids(self, create_orchestrator_harness) -> None:
        """Each user session should have a unique correlation ID."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN * 4,
            user_centric_rate=25.0,
            num_users=5,
            request_count=15,
        )
        await h.run_with_auto_return()
        assert len({c.x_correlation_id for c in h.sent_credits}) >= 5

    async def test_multi_turn_shares_correlation(
        self, create_orchestrator_harness
    ) -> None:
        """Multiple turns in a session should share the same correlation ID with sequential turn indices."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=MULTI_TURN,
            user_centric_rate=20.0,
            num_users=4,
            request_count=20,
        )
        await h.run_with_auto_return()
        sessions: dict[str, list] = {}
        for c in h.sent_credits:
            sessions.setdefault(c.x_correlation_id, []).append(c)
        multi = [s for s in sessions.values() if len(s) > 1]
        assert len(multi) > 0
        for credits in multi:
            indices = [c.turn_index for c in credits]
            assert indices == sorted(indices)
            assert indices[0] == 0


@pytest.mark.asyncio
class TestStopConditions:
    async def test_stops_at_request_count(self, create_orchestrator_harness) -> None:
        """Strategy should stop issuing credits when request count is reached."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN * 10,
            user_centric_rate=50.0,
            num_users=10,
            request_count=25,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == 25

    async def test_stops_at_session_count(self, create_orchestrator_harness) -> None:
        """Strategy should stop starting new sessions when session count is reached."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=MULTI_TURN * 5,
            user_centric_rate=40.0,
            num_users=8,
            num_sessions=10,
        )
        await h.run_with_auto_return()
        assert len([c for c in h.sent_credits if c.turn_index == 0]) == 10


def make_strategy(num_turns: int = 2) -> UserCentricStrategy:
    """Build a UserCentricStrategy over a real ConversationSource with one
    num_turns-turn conversation and mocked scheduler/issuer/lifecycle."""
    scheduler = MagicMock()
    stop_checker = MagicMock()
    issuer = MagicMock()
    issuer.issue_credit = lambda *a, **k: True
    lifecycle = MagicMock()
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="c1",
                turns=[TurnMetadata(delay_ms=None) for _ in range(num_turns)],
            )
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    sampler = make_sampler(["c1"], DatasetSamplingStrategy.SEQUENTIAL)
    src = ConversationSource(ds, sampler)
    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.USER_CENTRIC_RATE,
        request_rate=10.0,
        num_users=1,
        total_expected_requests=num_turns,
    )
    return UserCentricStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=stop_checker,
        credit_issuer=issuer,
        lifecycle=lifecycle,
    )


@pytest.mark.asyncio
class TestUserCentricCreditReturn:
    async def test_continuation_turn_carries_has_forks(self) -> None:
        """has_forks from the NEXT turn's metadata must ride onto the
        continuation turn (the sticky router defers parent-entry eviction until
        DAG children drain). Regression: from_previous_credit(credit) was called
        without next-turn metadata, so every continuation turn got
        has_forks=False."""
        strategy = make_strategy()
        await strategy.setup_phase()
        # Mark the next turn (index 1) as fork-bearing.
        meta = strategy._conversation_source._metadata_lookup["c1"]
        meta.turns[1].has_forks = True

        captured: list = []
        strategy._credit_issuer.issue_credit = (
            lambda turn: captured.append(turn) or True
        )
        # setup_phase registered the t=0 replacement user; use its session id.
        x_correlation_id = next(iter(strategy._session_to_user))
        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="c1",
            x_correlation_id=x_correlation_id,
            turn_index=0,
            num_turns=2,
            issued_at_ns=1000,
        )
        await strategy.handle_credit_return(credit)
        assert captured and captured[0].has_forks is True


@pytest.mark.asyncio
class TestRealisticScenarios:
    async def test_chat_benchmark(self, create_orchestrator_harness) -> None:
        """Simulate high-concurrency chat benchmark with many users and sessions."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", 3) for i in range(100)],
            user_centric_rate=100.0,
            num_users=50,
            request_count=200,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) >= 200
        assert 0 in {c.turn_index for c in h.sent_credits}

    async def test_kv_cache_benchmark(self, create_orchestrator_harness) -> None:
        """Simulate KV cache benchmark with longer multi-turn conversations."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", 5) for i in range(30)],
            user_centric_rate=40.0,
            num_users=20,
            request_count=100,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) >= 100


class TestAdaptiveTargetUsers:
    def _strategy(self) -> UserCentricStrategy:
        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            request_rate=10.0,
            num_users=4,
            total_expected_requests=10,
        )
        return UserCentricStrategy(
            config=cfg,
            conversation_source=MagicMock(),
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=MagicMock(),
            lifecycle=MagicMock(),
        )

    def test_set_target_users_recomputes_gap_and_staggers_scale_up(self) -> None:
        strategy = self._strategy()
        strategy._spawn_queue = []
        strategy._session_to_user = {"1": MagicMock(), "2": MagicMock()}

        strategy.set_target_users(6)

        assert strategy.target_users == 6
        assert strategy._adaptive_target_enabled is True
        assert strategy._turn_gap == pytest.approx(0.6)
        assert len(strategy._spawn_queue) == 2
        snapshot = strategy.user_control_snapshot()
        assert snapshot["target_value"] == 6
        assert snapshot["actual_value"] == 2
        assert snapshot["retiring_users"] == 0

    def test_set_target_users_rejects_non_positive_and_reports_retiring(self) -> None:
        strategy = self._strategy()
        strategy._session_to_user = {str(i): MagicMock() for i in range(6)}

        with pytest.raises(ValueError, match="target users must be positive"):
            strategy.set_target_users(0)

        strategy.set_target_users(3)

        snapshot = strategy.user_control_snapshot()
        assert snapshot["target_value"] == 3
        assert snapshot["actual_value"] == 6
        assert snapshot["retiring_users"] == 3
        assert snapshot["cancelled"] == 0

    def test_scale_down_suppresses_new_and_replacement_users(self) -> None:
        strategy = self._strategy()
        strategy._session_to_user = {str(i): MagicMock() for i in range(6)}

        strategy.set_target_users(3)

        assert not strategy._should_spawn_user()
        spawn_queue: list[float] = []
        strategy._schedule_replacement_user(
            spawn_queue, spawn_sec=10.0, user=MagicMock(max_turns=2)
        )
        assert spawn_queue == []


class TestUserClass:
    def test_x_correlation_id_delegates_to_sampled(self) -> None:
        """User.x_correlation_id should delegate to sampled session."""
        m = MagicMock()
        m.x_correlation_id = "test-id"
        assert User(user_id=1, sampled=m).x_correlation_id == "test-id"

    def test_build_first_turn_passes_max_turns(self) -> None:
        """User.build_first_turn should pass max_turns to sampled session."""
        m = MagicMock()
        m.build_first_turn.return_value = "turn"
        u = User(user_id=1, sampled=m, max_turns=5)
        assert u.build_first_turn() == "turn"
        m.build_first_turn.assert_called_once_with(max_turns=5)

    def test_dataclass_fields(self) -> None:
        """Verify User dataclass stores fields correctly."""
        m = MagicMock()
        m.x_correlation_id = "c-42"
        u = User(user_id=42, sampled=m, next_send_time=1000, max_turns=3, order=5)
        assert u.user_id == 42
        assert u.sampled == m
        assert u.next_send_time == 1000
        assert u.max_turns == 3
        assert u.order == 5


@pytest.mark.asyncio
class TestArrivalQueueClamping:
    """Deterministic coverage for the clamped arrival queue in execute_phase.

    Contract under test: every arrival slot fires at
    max(planned, previous_arrival_fire + stagger) -- stagger is the
    documented SMALLEST gap between any two users' first turns, so the clamp
    applies to on-plan and overdue slots alike. Issuance is fire-and-track
    (one scheduler task per arrival): a blocked or excepting issuance must
    neither stall nor abort the remaining grid, and replacement slots are
    derived at fire time so a failed issuance cannot kill its user's
    replacement stream.

    All tests run under time_traveler_no_patch_sleep: looptime virtualizes
    the loop clock (real asyncio.sleep calls become virtual loop timers that
    the loop reaches instantly when idle, and pending issuance tasks always
    run BEFORE the idle clock jump), while time.perf_counter reads share the
    same virtual clock -- so fire times are exact. (time_traveler's patched
    sleep advances its offset BEFORE yielding, which would let a later
    queue sleep shift an issuance task's recorded timestamp.) Every wait is
    bounded by asyncio.wait_for with a finite virtual timeout, and cleanup
    always cancels the execution task and drains scheduler-tracked tasks.
    """

    @staticmethod
    def _strategy(num_users: int = 3, rate: float = 20.0) -> UserCentricStrategy:
        """Real ConversationSource (one 2-turn conversation, wrapping sampler)
        and a REAL LoopScheduler (fire-and-track issuance must actually run);
        mocked stop_checker/lifecycle; issuer injected per test.
        rate=20.0 -> stagger=50ms, turn_gap=num_users/rate=150ms."""
        scheduler = LoopScheduler()
        ds = DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="c1",
                    turns=[TurnMetadata(delay_ms=None), TurnMetadata(delay_ms=None)],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = make_sampler(["c1"], DatasetSamplingStrategy.SEQUENTIAL)
        src = ConversationSource(ds, sampler)
        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            request_rate=rate,
            num_users=num_users,
            total_expected_requests=num_users * 2,
        )
        return UserCentricStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=scheduler,
            stop_checker=MagicMock(),
            credit_issuer=MagicMock(),
            lifecycle=MagicMock(),
        )

    @staticmethod
    async def _run(strategy: UserCentricStrategy, timeout: float = 30.0) -> None:
        """Run execute_phase bounded by a finite (virtual) timeout, always
        cancelling the task and draining scheduler-tracked tasks after."""
        task = asyncio.create_task(strategy.execute_phase())
        try:
            await asyncio.wait_for(task, timeout=timeout)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.gather(
                *strategy._scheduler.cancel_all(), return_exceptions=True
            )

    @staticmethod
    def _set_phase_start(
        strategy: UserCentricStrategy, time_traveler, offset_staggers: float = 0.0
    ) -> float:
        """Anchor the phase start (both lifecycle clock fields) at virtual-now
        minus offset_staggers worth of stagger; returns start in seconds."""
        stagger = strategy._stagger
        start_ns = time_traveler.perf_counter_ns() - int(
            offset_staggers * stagger * NANOS_PER_SECOND
        )
        strategy._lifecycle.started_at_perf_ns = start_ns
        strategy._lifecycle.started_at_perf_sec = start_ns / NANOS_PER_SECOND
        return start_ns / NANOS_PER_SECOND

    async def test_healthy_grid_lands_on_planned_slots(
        self, time_traveler_no_patch_sleep
    ) -> None:
        """With no stall the clamp never engages: initial users issue exactly
        on the planned stagger grid (base + order * stagger)."""
        strategy = self._strategy()
        await strategy.setup_phase()
        stagger = strategy._stagger
        start = self._set_phase_start(strategy, time_traveler_no_patch_sleep)

        call_times: list[float] = []

        async def _issue(_turn) -> bool:
            call_times.append(time.perf_counter())
            return len(call_times) < 3  # stop after the 3 initial users

        strategy._credit_issuer.issue_credit = AsyncMock(side_effect=_issue)
        await self._run(strategy)

        planned = [start + i * stagger for i in range(3)]
        assert call_times == pytest.approx(planned, abs=stagger / 10)

    async def test_stalled_grid_respaces_initial_and_replacement_arrivals(
        self, time_traveler_no_patch_sleep
    ) -> None:
        """Started 10 staggers in the past, every planned slot -- initial grid
        and derived replacements alike -- is overdue. The clamp must re-space
        all of them at one stagger per slot instead of bursting them at one
        instant (the pre-fix CI failure: mean stagger 0.0000s)."""
        strategy = self._strategy()
        await strategy.setup_phase()
        stagger = strategy._stagger
        self._set_phase_start(
            strategy, time_traveler_no_patch_sleep, offset_staggers=10.0
        )

        call_times: list[float] = []

        async def _issue(_turn) -> bool:
            call_times.append(time.perf_counter())
            return len(call_times) < 6  # 3 initial users + 3 replacements

        strategy._credit_issuer.issue_credit = AsyncMock(side_effect=_issue)
        await self._run(strategy)

        assert len(call_times) == 6
        gaps = [b - a for a, b in zip(call_times, call_times[1:], strict=False)]
        assert all(gap == pytest.approx(stagger, abs=stagger / 10) for gap in gaps)

    async def test_sub_stagger_planned_gap_is_clamped(
        self, time_traveler_no_patch_sleep
    ) -> None:
        """The min-gap contract applies to on-plan slots too: two arrivals
        planned 0.4 staggers apart must still fire one full stagger apart."""
        strategy = self._strategy()
        await strategy.setup_phase()
        stagger = strategy._stagger
        strategy._initial_users = [
            User(user_id=1, sampled=MagicMock(), max_turns=1, order=0),
            User(user_id=2, sampled=MagicMock(), max_turns=1, order=0.4),
        ]
        self._set_phase_start(strategy, time_traveler_no_patch_sleep)

        call_times: list[float] = []

        async def _issue(_turn) -> bool:
            call_times.append(time.perf_counter())
            return len(call_times) < 2

        strategy._credit_issuer.issue_credit = AsyncMock(side_effect=_issue)
        await self._run(strategy)

        assert len(call_times) == 2
        assert call_times[1] - call_times[0] == pytest.approx(stagger, abs=stagger / 10)

    async def test_issuer_exception_isolated_and_stream_survives(
        self, time_traveler_no_patch_sleep
    ) -> None:
        """One excepting issuance must neither abort nor stall the remaining
        grid, and its user's replacement stream survives: replacement slots
        are derived at fire time, before the issuance outcome is known.
        The failure is captured by the scheduler's exception handler."""
        strategy = self._strategy()
        await strategy.setup_phase()
        stagger = strategy._stagger
        self._set_phase_start(strategy, time_traveler_no_patch_sleep)

        errors: list[BaseException] = []
        strategy._scheduler.set_exception_handler(
            lambda task: errors.append(task.exception())
        )
        call_times: list[float] = []

        async def _issue(_turn) -> bool:
            call_times.append(time.perf_counter())
            if len(call_times) == 1:
                raise RuntimeError("boom")
            return len(call_times) < 6  # 3 initials + 3 replacements

        strategy._credit_issuer.issue_credit = AsyncMock(side_effect=_issue)
        await self._run(strategy)

        # All three replacements fire -- including the failed user's --
        # proving its stream was not killed by the exception.
        assert len(call_times) == 6
        gaps = [b - a for a, b in zip(call_times, call_times[1:], strict=False)]
        # The initial grid keeps exact stagger spacing despite the exception;
        # the clamp's contract is a MINIMUM gap, so the planned initial ->
        # replacement boundary gap (turn_gap >= 2 * stagger here) is allowed
        # to exceed stagger, but no gap may close below it.
        assert gaps[0] == pytest.approx(stagger, abs=stagger / 10)
        assert gaps[1] == pytest.approx(stagger, abs=stagger / 10)
        assert all(gap >= stagger * 0.9 for gap in gaps)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    async def test_blocked_issuance_does_not_stall_grid(
        self, time_traveler_no_patch_sleep
    ) -> None:
        """Fire-and-track: an issuance blocked on slot acquisition must not
        delay later slots -- they fire on the clamped schedule while the
        first issuance is still blocked."""
        strategy = self._strategy()
        await strategy.setup_phase()
        stagger = strategy._stagger
        self._set_phase_start(strategy, time_traveler_no_patch_sleep)

        gate = asyncio.Event()
        third_call = asyncio.Event()
        call_times: list[float] = []

        async def _issue(_turn) -> bool:
            call_times.append(time.perf_counter())
            if len(call_times) == 1:
                await gate.wait()  # first issuance blocks until cleanup
            if len(call_times) == 3:
                third_call.set()
            return len(call_times) < 3

        strategy._credit_issuer.issue_credit = AsyncMock(side_effect=_issue)
        task = asyncio.create_task(strategy.execute_phase())
        try:
            # The third arrival fires (bounded wait) while the first
            # issuance is still blocked.
            await asyncio.wait_for(third_call.wait(), timeout=5.0)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            gate.set()
            await asyncio.gather(
                *strategy._scheduler.cancel_all(), return_exceptions=True
            )

        assert len(call_times) == 3
        gaps = [b - a for a, b in zip(call_times, call_times[1:], strict=False)]
        assert all(gap == pytest.approx(stagger, abs=stagger / 10) for gap in gaps)

    async def test_refusal_stops_arrival_loop(
        self, time_traveler_no_patch_sleep
    ) -> None:
        """A refused issuance (stop condition) flips the loop's stop flag,
        which is re-checked after each clamp sleep, so no further arrival
        fires once the refusal has landed."""
        strategy = self._strategy()
        await strategy.setup_phase()
        self._set_phase_start(strategy, time_traveler_no_patch_sleep)

        call_times: list[float] = []

        async def _issue(_turn) -> bool:
            call_times.append(time.perf_counter())
            return len(call_times) < 2  # refuse from the 2nd arrival on

        strategy._credit_issuer.issue_credit = AsyncMock(side_effect=_issue)
        # Returns (not times out): the loop exits on the refusal flag.
        await self._run(strategy, timeout=5.0)

        assert len(call_times) == 2

    async def test_cancelled_execution_task_strands_no_timers_or_tasks(
        self, time_traveler_no_patch_sleep
    ) -> None:
        """Cancelling the execution task mid-phase leaves no pending timers
        and no running issuance tasks after scheduler cleanup; a timeout
        waiting for the first arrival still runs full cleanup."""
        strategy = self._strategy()
        await strategy.setup_phase()
        self._set_phase_start(strategy, time_traveler_no_patch_sleep)

        gate = asyncio.Event()
        first_call = asyncio.Event()

        async def _issue(_turn) -> bool:
            first_call.set()
            await gate.wait()  # every issuance blocks until cleanup
            return True

        strategy._credit_issuer.issue_credit = AsyncMock(side_effect=_issue)
        task = asyncio.create_task(strategy.execute_phase())
        try:
            await asyncio.wait_for(first_call.wait(), timeout=5.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            gate.set()
            await asyncio.gather(
                *strategy._scheduler.cancel_all(), return_exceptions=True
            )
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert strategy._scheduler.pending_count == 0
        assert strategy._scheduler.running_count == 0
