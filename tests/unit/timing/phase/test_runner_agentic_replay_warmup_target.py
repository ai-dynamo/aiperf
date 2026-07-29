# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for AGENTIC_REPLAY warmup ``total_expected_requests``."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.trajectory_source import TrajectorySource
from tests.unit.timing._shared_helpers import _make_dataset_metadata

pytestmark = pytest.mark.asyncio


def _warmup_config(concurrency: int) -> CreditPhaseConfig:
    """Mirror the placeholder shape produced by ``_build_warmup_config`` for AGENTIC_REPLAY."""
    return CreditPhaseConfig(
        phase=CreditPhase.WARMUP,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        total_expected_requests=concurrency,
        concurrency=concurrency,
        prefill_concurrency=None,
        request_rate=None,
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        seamless=False,
        grace_period_sec=float("inf"),
    )


def _make_runner(
    config: CreditPhaseConfig,
    conversation_source,
) -> PhaseRunner:
    pub = MagicMock()
    pub.publish_phase_start = AsyncMock()
    pub.publish_phase_sending_complete = AsyncMock()
    pub.publish_phase_complete = AsyncMock()
    pub.publish_progress = AsyncMock()
    router = MagicMock()
    router.send_credit = router.cancel_all_credits = AsyncMock()
    router.mark_credits_complete = MagicMock()
    router.set_return_callback = router.set_first_token_callback = MagicMock()
    conc = MagicMock()
    conc.configure_for_phase = MagicMock()
    conc.acquire_session_slot = AsyncMock(return_value=True)
    conc.acquire_prefill_slot = AsyncMock(return_value=True)
    conc.release_session_slot = conc.release_prefill_slot = MagicMock()
    conc.set_session_limit = conc.set_prefill_limit = MagicMock()
    conc.release_stuck_slots = MagicMock(return_value=(0, 0))
    cancel = MagicMock()
    cancel.next_cancellation_delay_ns = MagicMock(return_value=None)
    cb = MagicMock()
    cb.register_phase = cb.unregister_phase = MagicMock()
    cb.on_credit_return = cb.on_first_token = AsyncMock()
    return PhaseRunner(
        config=config,
        conversation_source=conversation_source,
        phase_publisher=pub,
        credit_router=router,
        concurrency_manager=conc,
        cancellation_policy=cancel,
        callback_handler=cb,
        user_config=None,
    )


class TestAgenticReplayWarmupTarget:
    """``PhaseRunner`` warmup-target behavior under AGENTIC_REPLAY."""

    async def test_concurrency_above_pool_size_wrap_fills_to_concurrency(self) -> None:
        """Pool of 6, concurrency=8 -> 8 lanes via wrap-fill, honouring ``--concurrency`` by reusing trajectories rather than capping load."""
        import itertools

        md = _make_dataset_metadata({f"t{i}": 5 for i in range(6)})
        sampler = MagicMock()
        cycle = itertools.cycle([c.conversation_id for c in md.conversations])
        sampler.next_conversation_id.side_effect = lambda: next(cycle)
        src = TrajectorySource(
            dataset_metadata=md,
            dataset_sampler=sampler,
            concurrency=8,
            random_seed=42,
        )
        assert len(src.trajectories) == 8
        distinct = {t.conversation_id for t in src.trajectories}
        assert len(distinct) == 6  # 6 distinct sources, repeated across 8 lanes

    async def test_concurrency_below_pool_size_uses_concurrency(self) -> None:
        """Pool of 10, concurrency=4 -> 4 trajectories -> target = 4 (unchanged)."""
        md = _make_dataset_metadata({f"t{i}": 5 for i in range(10)})
        sampler = MagicMock()
        sampler.next_conversation_id.side_effect = [
            c.conversation_id for c in md.conversations
        ]
        src = TrajectorySource(
            dataset_metadata=md, dataset_sampler=sampler, concurrency=4, random_seed=42
        )
        assert len(src.trajectories) == 4

        config = _warmup_config(concurrency=4)
        runner = _make_runner(config, src)
        assert runner._config.total_expected_requests == 4

    async def test_short_traces_skipped_below_concurrency_wrap_fills(self) -> None:
        """Pool of 6 with one skipped 1-turn trace, concurrency=8: wrap-fill cycles the 5 usable trajectories to 8 lanes with fresh ``start_turn_index`` salts."""
        import itertools

        md = _make_dataset_metadata({"a": 5, "b": 5, "c": 5, "d": 5, "e": 5, "tiny": 1})
        sampler = MagicMock()
        cycle = itertools.cycle([c.conversation_id for c in md.conversations])
        sampler.next_conversation_id.side_effect = lambda: next(cycle)
        src = TrajectorySource(
            dataset_metadata=md,
            dataset_sampler=sampler,
            concurrency=8,
            random_seed=42,
        )
        assert len(src.trajectories) == 8
        distinct = {t.conversation_id for t in src.trajectories}
        # 5 usable (tiny is skipped), repeated across 8 lanes.
        assert distinct == {"a", "b", "c", "d", "e"}

    async def test_profiling_phase_target_unchanged(self) -> None:
        """The re-anchor only applies to WARMUP, not PROFILING (in-budget run)."""
        md = _make_dataset_metadata({f"t{i}": 5 for i in range(8)})
        sampler = MagicMock()
        sampler.next_conversation_id.side_effect = [
            c.conversation_id for c in md.conversations
        ]
        src = TrajectorySource(
            dataset_metadata=md, dataset_sampler=sampler, concurrency=8, random_seed=42
        )

        profiling = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENTIC_REPLAY,
            total_expected_requests=100,
            expected_duration_sec=900,
            concurrency=8,
            request_rate=None,
            arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        )
        runner = _make_runner(profiling, src)
        # PROFILING target untouched.
        assert runner._config.total_expected_requests == 100

    async def test_warmup_target_reanchored_to_warmup_credit_count(self) -> None:
        """Multi-stream lanes re-anchor the barrier to warmup_credit_count (> concurrency), else the concurrency-sized barrier cancels the closest-to-t* primers."""
        src = MagicMock()
        src.dataset_metadata = None  # skip the FIXED_SCHEDULE re-anchor
        src.warmup_credit_count = 6
        runner = _make_runner(_warmup_config(concurrency=2), src)
        assert runner._config.total_expected_requests == 6

    async def test_cache_warmup_target_uses_actual_lane_count(self) -> None:
        """The placeholder is re-anchored to the actual wrap-filled trajectory lanes."""
        src = MagicMock()
        src.dataset_metadata = None
        src.trajectories = [MagicMock(), MagicMock(), MagicMock()]
        config = _warmup_config(concurrency=4).model_copy(
            update={
                "warmup_requests_per_lane": 10,
                "total_expected_requests": 40,
            }
        )

        runner = _make_runner(config, src)

        assert runner._config.total_expected_requests == 30

    async def test_profiling_not_reanchored_to_warmup_count(self) -> None:
        """PROFILING must NOT be re-anchored to warmup_credit_count."""
        src = MagicMock()
        src.dataset_metadata = None
        src.warmup_credit_count = 6
        profiling = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENTIC_REPLAY,
            total_expected_requests=100,
            expected_duration_sec=900,
            concurrency=2,
            request_rate=None,
            arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        )
        runner = _make_runner(profiling, src)
        assert runner._config.total_expected_requests == 100

    async def test_non_agentic_replay_warmup_target_unchanged(self) -> None:
        """The re-anchor must not touch REQUEST_RATE warmups (in-budget run)."""
        md = _make_dataset_metadata({f"t{i}": 5 for i in range(8)})
        sampler = MagicMock()
        sampler.next_conversation_id.side_effect = [
            c.conversation_id for c in md.conversations
        ]
        src = TrajectorySource(
            dataset_metadata=md, dataset_sampler=sampler, concurrency=8, random_seed=42
        )

        rr_warmup = CreditPhaseConfig(
            phase=CreditPhase.WARMUP,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=50,
            concurrency=8,
            request_rate=10.0,
            arrival_pattern=ArrivalPattern.POISSON,
        )
        runner = _make_runner(rr_warmup, src)
        # REQUEST_RATE warmup untouched (the re-anchor is AGENTIC_REPLAY-specific).
        assert runner._config.total_expected_requests == 50


class TestWarmupFailureAbortGate:
    """``PhaseRunner._report_warmup_failures`` wiring: agentic WARMUP terminal failures must abort the benchmark before PROFILING starts."""

    def _make_warmup_runner(self, strategy_obj) -> PhaseRunner:
        md = _make_dataset_metadata({f"t{i}": 5 for i in range(4)})
        sampler = MagicMock()
        sampler.next_conversation_id.side_effect = [
            c.conversation_id for c in md.conversations
        ]
        src = TrajectorySource(
            dataset_metadata=md, dataset_sampler=sampler, concurrency=4, random_seed=42
        )
        return _make_runner(_warmup_config(concurrency=4), src)

    async def test_warmup_failures_raise_and_abort(self) -> None:
        """A strategy with recorded warmup failures raises through the gate."""
        from aiperf.common.scenario.base import TrajectoryWarmupFailedError

        strategy = MagicMock()
        strategy.report_warmup_failures = MagicMock(
            side_effect=TrajectoryWarmupFailedError(["t0", "t2"])
        )
        runner = self._make_warmup_runner(strategy)
        with pytest.raises(TrajectoryWarmupFailedError):
            runner._report_warmup_failures(strategy)
        strategy.report_warmup_failures.assert_called_once()

    async def test_warmup_no_failures_is_noop(self) -> None:
        """A WARMUP strategy with no failures calls report but does not raise."""
        strategy = MagicMock()
        strategy.report_warmup_failures = MagicMock(return_value=None)
        runner = self._make_warmup_runner(strategy)
        runner._report_warmup_failures(strategy)
        strategy.report_warmup_failures.assert_called_once()

    async def test_backstop_skipped_when_live_abort_wired(self) -> None:
        """When the live early-abort is wired (on_warmup_abort is not None), the teardown backstop must NOT fire and double-abort."""
        strategy = MagicMock()
        strategy.report_warmup_failures = MagicMock()
        runner = self._make_warmup_runner(strategy)
        runner._callback_handler.on_warmup_abort = AsyncMock()  # live path wired

        assert runner._should_fire_warmup_backstop(strategy) is False

    async def test_backstop_fires_when_live_abort_unwired(self) -> None:
        """The backstop is the only abort path when on_warmup_abort is None."""
        strategy = MagicMock()
        strategy.report_warmup_failures = MagicMock()
        runner = self._make_warmup_runner(strategy)
        runner._callback_handler.on_warmup_abort = None
        runner._was_cancelled = False

        assert runner._should_fire_warmup_backstop(strategy) is True

    async def test_backstop_skipped_when_runner_already_cancelled(self) -> None:
        """A cancelled runner skips the backstop (the cancel already aborted)."""
        strategy = MagicMock()
        strategy.report_warmup_failures = MagicMock()
        runner = self._make_warmup_runner(strategy)
        runner._callback_handler.on_warmup_abort = None
        runner._was_cancelled = True

        assert runner._should_fire_warmup_backstop(strategy) is False

    async def test_profiling_phase_does_not_report(self) -> None:
        """The gate is WARMUP-only: PROFILING must never call report_warmup_failures."""
        md = _make_dataset_metadata({f"t{i}": 5 for i in range(8)})
        sampler = MagicMock()
        sampler.next_conversation_id.side_effect = [
            c.conversation_id for c in md.conversations
        ]
        src = TrajectorySource(
            dataset_metadata=md, dataset_sampler=sampler, concurrency=8, random_seed=42
        )
        profiling = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENTIC_REPLAY,
            total_expected_requests=100,
            expected_duration_sec=900,
            concurrency=8,
            request_rate=None,
            arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        )
        strategy = MagicMock()
        strategy.report_warmup_failures = MagicMock()
        runner = _make_runner(profiling, src)
        runner._report_warmup_failures(strategy)
        strategy.report_warmup_failures.assert_not_called()

    async def test_strategy_without_hook_is_noop(self) -> None:
        """Strategies that do not implement report_warmup_failures are skipped."""

        class _NoHook:
            pass

        runner = self._make_warmup_runner(_NoHook())
        runner._report_warmup_failures(_NoHook())  # must not raise

    async def test_run_invokes_gate_and_aborts_on_warmup_failure(self) -> None:
        """End-to-end through the real ``run()``/``_run_strategy`` path: a WARMUP strategy with recorded failures calls report_warmup_failures at teardown and propagates the abort, pinning the call-site wiring."""
        from aiperf.common.scenario.base import TrajectoryWarmupFailedError

        md = _make_dataset_metadata({f"t{i}": 5 for i in range(2)})
        sampler = MagicMock()
        sampler.next_conversation_id.side_effect = [
            c.conversation_id for c in md.conversations
        ]
        src = TrajectorySource(
            dataset_metadata=md, dataset_sampler=sampler, concurrency=2, random_seed=42
        )
        runner = _make_runner(_warmup_config(concurrency=2), src)

        strategy = MagicMock()
        strategy.setup_phase = AsyncMock()
        strategy.execute_phase = AsyncMock()
        strategy.finalize_phase = AsyncMock()
        strategy.report_warmup_failures = MagicMock(
            side_effect=TrajectoryWarmupFailedError(["t0"])
        )

        # Drive the real _run_strategy past its credit-flow waits without a live
        # bus: stub the blocking awaits + ramper build + background-task spawn so
        # control reaches the warmup-failure gate at the end of _run_strategy.
        runner._build_strategy = MagicMock(return_value=strategy)
        # This test pins the TEARDOWN BACKSTOP, which only fires when the live
        # warmup early-abort is NOT wired. Force the un-wired condition (the
        # MagicMock callback_handler would otherwise auto-create a truthy
        # ``on_warmup_abort`` and the backstop would correctly be skipped).
        runner._callback_handler.on_warmup_abort = None
        # The gate is strategy-only; bypass the live branch orchestrator so the
        # pre-session-branch dispatch doesn't need a real credit bus.
        runner._branch_orchestrator = None
        runner._credit_router.wait_for_workers = AsyncMock()
        runner._create_rampers = MagicMock(
            side_effect=lambda _s: setattr(runner, "_rampers", [])
        )
        runner._wait_for_sending_complete = AsyncMock()
        runner._wait_for_returning_complete = AsyncMock()

        with (
            patch.object(runner, "execute_async", return_value=MagicMock()),
            pytest.raises(TrajectoryWarmupFailedError),
        ):
            await runner.run(is_final_phase=True)

        strategy.report_warmup_failures.assert_called_once()


class TestAgenticReplayWarmupTargetIntegrationWithCounter:
    """Sanity-check that the warmup target makes the counter fire ``is_final_credit``."""

    @pytest.mark.parametrize(
        "concurrency,pool_size,expected_count",
        [
            (4, 10, 4),  # below pool size
            (10, 10, 10),  # at pool size
        ],
    )
    async def test_counter_fires_final_credit_on_last_trajectory(
        self,
        concurrency: int,
        pool_size: int,
        expected_count: int,
    ) -> None:
        """The counter flips ``is_final_credit`` exactly on the last (in-budget) trajectory's credit, which is what unblocks the runner's wait."""
        from aiperf.credit.structs import TurnToSend
        from aiperf.timing.phase.credit_counter import CreditCounter

        turn_counts: dict[str, int] = {f"t{i}": 5 for i in range(pool_size)}
        md = _make_dataset_metadata(turn_counts)
        sampler = MagicMock()
        sampler.next_conversation_id.side_effect = [
            c.conversation_id for c in md.conversations
        ]
        src = TrajectorySource(
            dataset_metadata=md,
            dataset_sampler=sampler,
            concurrency=concurrency,
            random_seed=42,
        )
        assert len(src.trajectories) == expected_count

        config = _warmup_config(concurrency=concurrency)
        runner = _make_runner(config, src)
        assert runner._config.total_expected_requests == expected_count

        counter = CreditCounter(runner._config)
        is_final_seen = False
        for i in range(expected_count):
            turn = TurnToSend(
                conversation_id=f"t{i}",
                x_correlation_id=f"x{i}",
                turn_index=0,
                num_turns=5,
                agent_depth=0,
            )
            _, is_final = counter.increment_sent(turn)
            if i == expected_count - 1:
                assert is_final is True, (
                    f"Last warmup credit (i={i}) must flip is_final_credit; "
                    f"otherwise warmup hangs at {expected_count}/{concurrency}."
                )
                is_final_seen = True
            else:
                assert is_final is False
        assert is_final_seen
