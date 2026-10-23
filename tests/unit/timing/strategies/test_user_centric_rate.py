# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

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
class TestInitialUserChaining:
    """Deterministic coverage for the chained initial first-turn timers.

    UserCentricStrategy._schedule_initial_user chains each initial user's
    timer to the next so an event-loop stall longer than the stagger window
    re-spaces the remaining arrivals instead of bursting them in one batch
    (the Windows CI flake in TestUserCentricRateStaggerTiming). These tests
    drive the chain helper directly on the real loop with a real
    LoopScheduler; no_sleep only patches asyncio.sleep, which none of these
    tests use -- waits are event/wait_for based and timers run in real time.
    """

    @staticmethod
    def _strategy(
        scheduler: LoopScheduler,
        issuer: MagicMock,
        *,
        sending_complete: bool = False,
    ) -> UserCentricStrategy:
        lifecycle = MagicMock()
        lifecycle.is_sending_complete = sending_complete
        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            request_rate=20.0,  # stagger = 50ms
            num_users=3,
            total_expected_requests=6,
        )
        return UserCentricStrategy(
            config=cfg,
            conversation_source=MagicMock(),
            scheduler=scheduler,
            stop_checker=MagicMock(),
            credit_issuer=issuer,
            lifecycle=lifecycle,
        )

    @staticmethod
    def _users(n: int) -> list[User]:
        return [User(user_id=i, sampled=MagicMock(), order=i) for i in range(n)]

    @staticmethod
    def _recording_issuer(call_times: list[float]) -> MagicMock:
        issuer = MagicMock()

        async def _issue(_turn) -> bool:
            call_times.append(time.perf_counter())
            return True

        issuer.issue_credit = AsyncMock(side_effect=_issue)
        return issuer

    @staticmethod
    async def _wait_for_calls(call_times: list[float], n: int) -> None:
        """Pump the loop until n issuance calls land; each wait_for timeout
        lets the real timers fire. No asyncio.sleep (no_sleep collapses it)."""
        while len(call_times) < n:
            try:
                await asyncio.wait_for(asyncio.Event().wait(), timeout=0.2)
            except asyncio.TimeoutError:
                pass

    @staticmethod
    async def _cleanup(scheduler: LoopScheduler) -> None:
        await asyncio.gather(*scheduler.cancel_all(), return_exceptions=True)

    async def test_chain_respaces_successor_when_timer_fires_late(self) -> None:
        """A successor whose planned slot is already due (collapsed grid after
        a stall) must be re-armed at fire_time + stagger, not fired in the
        same batch."""
        scheduler = LoopScheduler()
        call_times: list[float] = []
        issuer = self._recording_issuer(call_times)
        strategy = self._strategy(scheduler, issuer)
        users = self._users(3)

        now = time.perf_counter()
        users[0].next_send_time = now  # fires ~immediately
        # Planned successor slots are nearly immediate too: without
        # re-spacing, all three would fire in the same loop batch.
        users[1].next_send_time = now + 0.005
        users[2].next_send_time = now + 0.010

        strategy._schedule_initial_user(users, 0)
        await self._wait_for_calls(call_times, 3)
        await self._cleanup(scheduler)

        stagger = strategy._stagger
        assert len(call_times) == 3
        # Each hop is spaced by ~stagger from the previous ACTUAL fire, not
        # by the collapsed planned grid. Late firing only inflates the gap,
        # and the loop clock fires at most one resolution early, so the
        # half-stagger lower bound is robust.
        assert call_times[1] - call_times[0] >= stagger / 2
        assert call_times[2] - call_times[1] >= stagger / 2

    async def test_chain_keeps_planned_grid_when_successor_slot_is_future(self) -> None:
        """On a healthy loop the max() anchor keeps the planned slot: a
        successor planned beyond fire_time + stagger still fires on plan."""
        scheduler = LoopScheduler()
        call_times: list[float] = []
        issuer = self._recording_issuer(call_times)
        strategy = self._strategy(scheduler, issuer)
        users = self._users(2)

        now = time.perf_counter()
        users[0].next_send_time = now
        users[1].next_send_time = now + 0.25  # well past now + stagger (0.05)

        strategy._schedule_initial_user(users, 0)
        await self._wait_for_calls(call_times, 2)
        await self._cleanup(scheduler)

        assert len(call_times) == 2
        assert call_times[1] - call_times[0] >= 0.20

    async def test_chain_advances_while_first_issuance_is_blocked(self) -> None:
        """The successor is armed when the timer fires, before issuance: a
        blocked first issue_credit must not delay or suppress the chain."""
        scheduler = LoopScheduler()
        issuer = MagicMock()
        gate = asyncio.Event()
        second_call = asyncio.Event()
        calls = 0

        async def _issue(_turn) -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                await gate.wait()  # first issuance blocks until released
            else:
                second_call.set()
            return True

        issuer.issue_credit = AsyncMock(side_effect=_issue)
        strategy = self._strategy(scheduler, issuer)
        users = self._users(2)

        now = time.perf_counter()
        users[0].next_send_time = now
        users[1].next_send_time = now  # collapsed planned grid

        strategy._schedule_initial_user(users, 0)
        # Chain must advance to the second user's issuance even though the
        # first issuance is still blocked.
        await asyncio.wait_for(second_call.wait(), timeout=2.0)
        gate.set()
        await self._cleanup(scheduler)

        assert calls == 2

    async def test_chain_not_rearmed_after_sending_cutoff(self) -> None:
        """Once the phase marks sending complete, PhaseRunner's cutoff has
        cancelled all pending timers; a timer firing past the cutoff must not
        recreate new ones."""
        scheduler = LoopScheduler()
        call_times: list[float] = []
        issuer = self._recording_issuer(call_times)
        strategy = self._strategy(scheduler, issuer, sending_complete=True)
        users = self._users(3)

        now = time.perf_counter()
        for i, user in enumerate(users):
            user.next_send_time = now + i * 0.01

        strategy._schedule_initial_user(users, 0)
        await self._wait_for_calls(call_times, 1)

        # The fired timer issued its own credit but armed no successor.
        assert scheduler.pending_count() == 0
        await self._cleanup(scheduler)
        assert len(call_times) == 1

    async def test_pending_chain_timer_cancelled_before_fire(self) -> None:
        """Cancelling the pending chain timer before it fires issues nothing
        and leaves no timers behind."""
        scheduler = LoopScheduler()
        call_times: list[float] = []
        issuer = self._recording_issuer(call_times)
        strategy = self._strategy(scheduler, issuer)
        users = self._users(3)

        now = time.perf_counter()
        for i, user in enumerate(users):
            user.next_send_time = now + 0.05 + i * 0.01

        strategy._schedule_initial_user(users, 0)
        scheduler.cancel_all_pending()
        assert scheduler.pending_count() == 0

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.Event().wait(), timeout=0.1)
        assert not call_times
