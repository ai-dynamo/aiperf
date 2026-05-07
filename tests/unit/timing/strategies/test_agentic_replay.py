# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AgenticReplayStrategy.

Tests the phase-aware trajectory dispatch (WARMUP) and resume-at-k+1 + recycle
(PROFILING) behaviors specified in agentx-mvp Spec §4.2.
"""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CacheBustTarget, ConversationBranchMode, CreditPhase
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.scenario.base import TrajectoryWarmupFailedError
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import (
    Trajectory,
    TrajectorySource,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_dataset(num_traces: int, turns_per_trace: int) -> DatasetMetadata:
    convs = []
    for i in range(num_traces):
        turns = [
            TurnMetadata(timestamp_ms=None, delay_ms=None)
            for _ in range(turns_per_trace)
        ]
        convs.append(ConversationMetadata(conversation_id=f"trace_{i}", turns=turns))
    return DatasetMetadata(
        conversations=convs, sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )


def _build_real_trajectory_source(
    num_traces: int,
    turns_per_trace: int,
    trajectories: list[Trajectory],
) -> TrajectorySource:
    """Build a real TrajectorySource with deterministic trajectories.

    We construct the source via __new__ + manual init so we control the
    trajectories exactly (avoid randomization in tests).
    """
    ds = _make_dataset(num_traces, turns_per_trace)

    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src._random_seed = 0
    src._target_size = len(trajectories)
    src.trajectories = list(trajectories)
    return src


def _make_strategy(
    *,
    phase: CreditPhase,
    trajectories: list[Trajectory],
    num_traces: int = 5,
    turns_per_trace: int = 4,
    issuer: AsyncMock | None = None,
    scheduler: MagicMock | None = None,
    user_config: object | None = None,
) -> tuple[AgenticReplayStrategy, AsyncMock, MagicMock, TrajectorySource]:
    src = _build_real_trajectory_source(num_traces, turns_per_trace, trajectories)
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = len(trajectories)
    issuer = issuer if issuer is not None else AsyncMock()
    scheduler = scheduler if scheduler is not None else MagicMock()
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        user_config=user_config,
    )
    return strategy, issuer, scheduler, src


def _make_credit(
    *,
    conversation_id: str,
    x_correlation_id: str = "xcorr",
    turn_index: int,
    num_turns: int,
    phase: CreditPhase = CreditPhase.PROFILING,
) -> Credit:
    return Credit(
        id=0,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        branch_mode=ConversationBranchMode.FORK,
    )


# =============================================================================
# Constructor validation
# =============================================================================


def test_constructor_rejects_unknown_phase():
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    src = _build_real_trajectory_source(1, 2, trajectories)
    cfg = MagicMock()
    cfg.phase = "unknown"
    cfg.concurrency = 1
    with pytest.raises(ValueError):
        AgenticReplayStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=AsyncMock(),
            lifecycle=MagicMock(),
        )


def test_constructor_rejects_non_trajectory_source():
    """ConversationSource that is not a TrajectorySource is rejected."""
    cfg = MagicMock()
    cfg.phase = CreditPhase.WARMUP
    cfg.concurrency = 1
    plain_src = MagicMock()  # not a TrajectorySource instance
    with pytest.raises(TypeError):
        AgenticReplayStrategy(
            config=cfg,
            conversation_source=plain_src,
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=AsyncMock(),
            lifecycle=MagicMock(),
        )


def test_constructor_accepts_warmup_and_profiling():
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    for phase in (CreditPhase.WARMUP, CreditPhase.PROFILING):
        strategy, *_ = _make_strategy(phase=phase, trajectories=trajectories)
        assert strategy.config.phase == phase


# =============================================================================
# WARMUP phase
# =============================================================================


@pytest.mark.asyncio
async def test_warmup_dispatches_one_credit_per_trajectory():
    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectories
    )
    await strategy.setup_phase()
    await strategy.execute_phase()
    assert issuer.issue_credit.await_count == 3


@pytest.mark.asyncio
async def test_warmup_dispatch_uses_start_turn_index():
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_1", start_turn_index=1),
        Trajectory(conversation_id="trace_2", start_turn_index=2),
    ]
    issued_turn_indices: list[int] = []

    async def capture(turn):
        issued_turn_indices.append(turn.turn_index)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        turns_per_trace=4,
        issuer=issuer,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()
    assert sorted(issued_turn_indices) == [0, 1, 2]


@pytest.mark.asyncio
async def test_warmup_handle_credit_return_is_noop():
    """In WARMUP, handle_credit_return must not dispatch follow-up turns."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectories, turns_per_trace=4
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()

    credit = _make_credit(
        conversation_id="trace_0",
        turn_index=0,
        num_turns=4,
        phase=CreditPhase.WARMUP,
    )
    await strategy.handle_credit_return(credit)

    assert issuer.issue_credit.await_count == 0


def test_report_warmup_failures_raises_when_failures_present():
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy, *_ = _make_strategy(phase=CreditPhase.WARMUP, trajectories=trajectories)
    strategy.record_warmup_failure("trace_0")
    strategy.record_warmup_failure("trace_3")
    with pytest.raises(TrajectoryWarmupFailedError) as exc_info:
        strategy.report_warmup_failures()
    assert exc_info.value.failed_trace_ids == ["trace_0", "trace_3"]


def test_report_warmup_failures_silent_when_no_failures():
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy, *_ = _make_strategy(phase=CreditPhase.WARMUP, trajectories=trajectories)
    strategy.report_warmup_failures()  # must not raise


# =============================================================================
# PROFILING phase: setup_phase + execute_phase
# =============================================================================


@pytest.mark.asyncio
async def test_profiling_setup_seeds_recycle_queue_with_full_pool():
    """PROFILING setup seeds the recycle queue with the FULL dataset pool
    (including trajectory trace_ids). The pop loop in
    ``_spawn_from_recycle_or_id`` skips trace_ids whose session is currently
    active, so duplicate concurrent sessions are still avoided."""
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_2", start_turn_index=1),
    ]
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=5,  # trace_0..trace_4
    )
    await strategy.setup_phase()

    queue = strategy._recycle_queue
    assert queue is not None
    queued: list[str] = []
    while not queue.empty():
        queued.append(queue.get_nowait())

    # Full pool in iteration order from dataset_metadata.conversations.
    assert queued == ["trace_0", "trace_1", "trace_2", "trace_3", "trace_4"]


@pytest.mark.asyncio
async def test_profiling_phase_resumes_trajectory_at_k_plus_one():
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),  # resume at 1
        Trajectory(conversation_id="trace_1", start_turn_index=2),  # resume at 3
    ]
    captured: list[tuple[str, int]] = []

    async def capture(turn):
        captured.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        turns_per_trace=5,
        issuer=issuer,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    assert sorted(captured) == [("trace_0", 1), ("trace_1", 3)]


@pytest.mark.asyncio
async def test_profiling_skips_trajectory_at_last_turn_and_recycles():
    """If k_i is already the last turn, k_i+1 is out of range. Recycle immediately."""
    trajectories = [
        Trajectory(
            conversation_id="trace_0", start_turn_index=3
        ),  # turns_per_trace=4 -> last index
    ]
    captured: list[tuple[str, int]] = []

    async def capture(turn):
        captured.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=3,
        turns_per_trace=4,
        issuer=issuer,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    # No resume issued for the trajectory; instead a recycle session at turn 0.
    assert all(idx == 0 for _, idx in captured)
    assert len(captured) == 1
    # With the full-pool recycle queue, the head is "trace_0" (iteration
    # order from dataset_metadata.conversations). The trajectory's session
    # is discarded from _active_traces inside _spawn_from_recycle_or_id
    # before the pop loop runs, so trace_0 is popped and dispatched at
    # turn 0 as the first recycled session.
    assert captured[0][0] == "trace_0"


# =============================================================================
# PROFILING handle_credit_return: continuation + recycle
# =============================================================================


@pytest.mark.asyncio
async def test_handle_credit_return_dispatches_next_turn_when_not_final():
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issuer = AsyncMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        turns_per_trace=4,
        issuer=issuer,
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()

    credit = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=4)
    await strategy.handle_credit_return(credit)

    assert issuer.issue_credit.await_count == 1
    issued_turn = issuer.issue_credit.await_args.args[0]
    assert issued_turn.turn_index == 2
    assert issued_turn.conversation_id == "trace_0"


@pytest.mark.asyncio
async def test_handle_credit_return_honors_delay_ms_via_scheduler():
    """When next turn has delay_ms, dispatch is scheduled, not immediate."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issuer = AsyncMock()
    scheduler = MagicMock()

    # Build a dataset where turn_index=2 has a delay_ms.
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace_0",
                turns=[
                    TurnMetadata(timestamp_ms=None, delay_ms=None),
                    TurnMetadata(timestamp_ms=None, delay_ms=None),
                    TurnMetadata(timestamp_ms=None, delay_ms=500),
                    TurnMetadata(timestamp_ms=None, delay_ms=None),
                ],
            )
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src._random_seed = 0
    src._target_size = 1
    src.trajectories = list(trajectories)

    cfg = MagicMock()
    cfg.phase = CreditPhase.PROFILING
    cfg.concurrency = 1
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()
    scheduler.schedule_later.reset_mock()

    credit = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=4)
    await strategy.handle_credit_return(credit)

    # No direct issue; one scheduled dispatch with delay 0.5s.
    assert issuer.issue_credit.await_count == 0
    assert scheduler.schedule_later.call_count == 1
    delay_arg = scheduler.schedule_later.call_args.args[0]
    assert delay_arg == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_handle_credit_return_recycles_on_final_turn():
    """Last turn of a session -> trace_id put back; new session pulled FIFO."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issued_sessions: list[tuple[str, int]] = []

    async def capture(turn):
        issued_sessions.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=3,
        turns_per_trace=4,
        issuer=issuer,
    )
    await strategy.setup_phase()

    # Recycle queue should currently be the full pool ["trace_0", "trace_1", "trace_2"].
    initial_queue_size = strategy._recycle_queue.qsize()
    assert initial_queue_size == 3

    # Register the in-flight session's lane bookkeeping (normally done by
    # _execute_profiling); handle_credit_return's recycle path now requires
    # finished_correlation_id to be in _correlation_to_lane. Also seed
    # _active_traces so the new full-pool pop loop's skip-active-on-pop
    # logic mirrors a real run.
    strategy._correlation_to_lane["xcorr"] = 0
    strategy._active_traces.add("trace_0")

    issuer.issue_credit.reset_mock()
    issued_sessions.clear()

    # trace_0 finishes its last turn (index 3 of 4).
    final_credit = _make_credit(conversation_id="trace_0", turn_index=3, num_turns=4)
    await strategy.handle_credit_return(final_credit)

    # Spawn flow: discard trace_0 from active (was alive); push trace_0 to
    # tail of [trace_0, trace_1, trace_2] -> [trace_0, trace_1, trace_2, trace_0];
    # pop head trace_0 (not active anymore, just discarded), dispatch at turn 0.
    assert issued_sessions == [("trace_0", 0)]
    # Queue now contains [trace_1, trace_2, trace_0] (head trace_0 popped).
    remaining: list[str] = []
    while not strategy._recycle_queue.empty():
        remaining.append(strategy._recycle_queue.get_nowait())
    assert remaining == ["trace_1", "trace_2", "trace_0"]


@pytest.mark.asyncio
async def test_handle_credit_return_reuses_finished_trace_when_queue_empty():
    """Single-trace dataset: just-finished trace_id is reused immediately.

    With the full-pool recycle queue, a single-trace dataset means the queue
    holds [trace_0] at setup; the trajectory's session is still alive there
    (tracked in _active_traces), so the only available pop after re-enqueue
    is the just-finished trace_0 itself.
    """
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issued_sessions: list[tuple[str, int]] = []

    async def capture(turn):
        issued_sessions.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=1,  # single-trace dataset
        turns_per_trace=3,
        issuer=issuer,
    )
    await strategy.setup_phase()
    # Full pool: queue is [trace_0] at setup.
    assert strategy._recycle_queue.qsize() == 1

    # Register the in-flight session's lane (normally done by _execute_profiling).
    # Also seed _active_traces so the new pop loop skips trace_0 while it is
    # nominally alive — discard happens at the top of _spawn_from_recycle_or_id.
    strategy._correlation_to_lane["xcorr"] = 0
    strategy._active_traces.add("trace_0")

    issuer.issue_credit.reset_mock()
    issued_sessions.clear()

    final_credit = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=3)
    await strategy.handle_credit_return(final_credit)

    # trace_0 discarded from active, pushed to tail, immediately popped and
    # dispatched at turn 0.
    assert issued_sessions == [("trace_0", 0)]


@pytest.mark.asyncio
async def test_spawn_from_recycle_prunes_marker_dicts_on_stop_checker_reject():
    """Early-return paths in _spawn_from_recycle_or_id must still prune marker/lane dicts.

    Regression: previously the pop only happened on the success path, so a finished
    session whose recycle attempt hit any early return (stop-checker reject, queue
    empty without a put because _recycle_queue is None, missing metadata) would
    leak its entry into _session_marker and _correlation_to_lane for the rest of
    the phase.
    """
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issuer = AsyncMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=2,
        turns_per_trace=3,
        issuer=issuer,
    )
    await strategy.setup_phase()

    # Simulate an in-flight session for trace_0: lane assigned, marker minted.
    finished_corr_id = "xcorr-finished"
    strategy._correlation_to_lane[finished_corr_id] = 0
    strategy._session_marker[finished_corr_id] = None

    # Force the stop_checker early-return path - cooldown reached, no new sessions.
    strategy.stop_checker.can_start_new_session = MagicMock(return_value=False)

    issuer.issue_credit.reset_mock()
    await strategy._spawn_from_recycle_or_id(
        "trace_0", finished_correlation_id=finished_corr_id
    )

    # Early return must still have pruned both bookkeeping dicts.
    assert finished_corr_id not in strategy._session_marker
    assert finished_corr_id not in strategy._correlation_to_lane
    # No new credit issued because of the early-return.
    assert issuer.issue_credit.await_count == 0


@pytest.mark.asyncio
async def test_handle_credit_return_warmup_phase_is_noop_for_final_turn():
    """In WARMUP, even a final-turn credit return must not trigger recycle."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issuer = AsyncMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        turns_per_trace=2,
        issuer=issuer,
    )
    await strategy.setup_phase()
    assert strategy._recycle_queue is None

    issuer.issue_credit.reset_mock()
    final_credit = _make_credit(
        conversation_id="trace_0",
        turn_index=1,
        num_turns=2,
        phase=CreditPhase.WARMUP,
    )
    await strategy.handle_credit_return(final_credit)
    assert issuer.issue_credit.await_count == 0


# =============================================================================
# WARMUP setup_phase: no recycle queue built
# =============================================================================


@pytest.mark.asyncio
async def test_warmup_setup_does_not_build_recycle_queue():
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy, *_ = _make_strategy(phase=CreditPhase.WARMUP, trajectories=trajectories)
    await strategy.setup_phase()
    assert strategy._recycle_queue is None


# =============================================================================
# Warmup signals sending-complete after dispatch
# =============================================================================
#
# Belt-and-suspenders alongside total_expected_requests=loadgen.concurrency:
# _execute_warmup must call lifecycle.mark_sending_complete() AFTER the
# cohort dispatch loop. Without it, when pool_size < concurrency the count
# target is never reached and the cohort barrier holds forever. Must be
# called exactly once, after all credits are issued.


@pytest.mark.asyncio
async def test_warmup_marks_sending_complete():
    """``_execute_warmup`` signals sending-complete once after dispatching
    all trajectory credits.

    ``mark_sending_complete`` is a guarded fallback now that PhaseRunner re-anchors
    ``total_expected_requests`` to the actual trajectory count: when the count-based
    path wins the race, the strategy's call is skipped via the
    ``is_sending_complete`` guard. Force the guard to evaluate ``False`` so this
    legacy behavioral assertion still applies.
    """
    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectories
    )
    strategy.lifecycle.is_sending_complete = False
    await strategy.setup_phase()
    await strategy.execute_phase()

    assert strategy.lifecycle.mark_sending_complete.call_count == 1
    # Sanity: dispatch happened for each trajectory.
    assert issuer.issue_credit.await_count == 3


@pytest.mark.asyncio
async def test_warmup_marks_sending_complete_after_dispatch():
    """``mark_sending_complete`` is called AFTER all credits are issued,
    not before — otherwise ``SendingCompleteStopCondition`` can fire
    mid-dispatch."""
    call_order: list[str] = []

    async def record_issue(_turn) -> bool:
        call_order.append("issue_credit")
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = record_issue

    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectories, issuer=issuer
    )
    strategy.lifecycle.is_sending_complete = False

    def record_mark() -> None:
        call_order.append("mark_sending_complete")

    strategy.lifecycle.mark_sending_complete.side_effect = record_mark

    await strategy.setup_phase()
    await strategy.execute_phase()

    assert call_order == [
        "issue_credit",
        "issue_credit",
        "issue_credit",
        "mark_sending_complete",
    ]


@pytest.mark.asyncio
async def test_warmup_skips_mark_sending_complete_when_already_complete():
    """When ``CreditCounter.is_final_credit`` already fired (and PhaseRunner's
    ``CreditIssuer`` already advanced the lifecycle into SENDING_COMPLETE),
    the strategy must NOT re-call ``mark_sending_complete``. Without this
    guard the strategy double-transitions the state machine -> ValueError.

    This is the regression guard for the warmup-hang fix: PhaseRunner now
    re-anchors ``total_expected_requests`` to the actual trajectory count,
    so the count-based path is the primary signal and the strategy's call
    becomes a guarded fallback.
    """
    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectories
    )
    # Simulate the count-based path having already won the race.
    strategy.lifecycle.is_sending_complete = True

    await strategy.setup_phase()
    await strategy.execute_phase()

    strategy.lifecycle.mark_sending_complete.assert_not_called()


# =============================================================================
# Cache-bust marker minting (Task 5)
# =============================================================================
#
# Per spec §4.5, AgenticReplayStrategy mints one marker per session keyed by
# x_correlation_id, reuses it across the warmup k_i / profile k_i+1 boundary,
# and rotates it on recycle (recycle_pass increments). Lane (trajectory_index)
# is stable per slot so marker digests change only across recycle passes.

_RID_RE = re.compile(r"\[rid:[0-9a-f]{12}\]")


def _make_user_config(
    *, target: CacheBustTarget, benchmark_id: str = "bench-fixed"
) -> SimpleNamespace:
    """Lightweight stand-in for UserConfig; only the two attributes the
    strategy reads are exposed (avoids spinning up real Pydantic config)."""
    return SimpleNamespace(
        input=SimpleNamespace(
            prompt=SimpleNamespace(cache_bust=SimpleNamespace(target=target))
        ),
        benchmark_id=benchmark_id,
    )


def _extract_rid(marker: str | None) -> str | None:
    if marker is None:
        return None
    m = _RID_RE.search(marker)
    return m.group(0) if m else None


@pytest.mark.asyncio
async def test_warmup_session_marker_reused_in_profile_resume():
    """Trajectory's warmup turn k_i and profile turn k_i+1 share the same
    marker (recycle_pass=0, same lane index, same benchmark_id, same
    trace_id; phase deliberately NOT in the digest tuple per spec
    warmup-coherence requirement)."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=2)]
    user_config = _make_user_config(target=CacheBustTarget.SYSTEM_PREFIX)

    # WARMUP phase mints first.
    issuer = AsyncMock()
    warmup_turns: list = []

    async def capture_warmup(turn):
        warmup_turns.append(turn)
        return True

    issuer.issue_credit.side_effect = capture_warmup

    strategy_w, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        turns_per_trace=5,
        issuer=issuer,
        user_config=user_config,
    )
    await strategy_w.setup_phase()
    await strategy_w.execute_phase()

    # PROFILING phase (constructed fresh like PhaseRunner does).
    issuer2 = AsyncMock()
    profile_turns: list = []

    async def capture_profile(turn):
        profile_turns.append(turn)
        return True

    issuer2.issue_credit.side_effect = capture_profile

    strategy_p, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        turns_per_trace=5,
        issuer=issuer2,
        user_config=user_config,
    )
    await strategy_p.setup_phase()
    await strategy_p.execute_phase()

    assert len(warmup_turns) == 1
    assert len(profile_turns) == 1
    warmup_rid = _extract_rid(warmup_turns[0].cache_bust_marker)
    profile_rid = _extract_rid(profile_turns[0].cache_bust_marker)
    assert warmup_rid is not None
    assert warmup_rid == profile_rid, (
        "Spec requires warmup-coherence: the digest tuple "
        "(benchmark_id, recycle_pass, trajectory_index, trace_id) is "
        "phase-agnostic, so WARMUP turn k_i and PROFILING turn k_i+1 must "
        "render the same marker so warmup KV-cache work transfers to profile."
    )
    assert warmup_turns[0].cache_bust_target == CacheBustTarget.SYSTEM_PREFIX
    assert profile_turns[0].cache_bust_target == CacheBustTarget.SYSTEM_PREFIX


@pytest.mark.asyncio
async def test_recycle_increments_pass_and_rotates_marker():
    """Spawn for traceA, finish, recycle traceA — markers differ because
    recycle_pass increments. Single-trace dataset so the just-finished
    trace_id is reused immediately on recycle."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    user_config = _make_user_config(target=CacheBustTarget.SYSTEM_PREFIX)
    issued_turns: list = []

    async def capture(turn):
        issued_turns.append(turn)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture

    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=1,  # forces queue empty -> reuse on recycle
        turns_per_trace=2,
        issuer=issuer,
        user_config=user_config,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()
    assert len(issued_turns) == 1
    initial_marker = issued_turns[0].cache_bust_marker
    initial_rid = _extract_rid(initial_marker)
    initial_xcorr = issued_turns[0].x_correlation_id

    # Final-turn credit return triggers recycle of trace_0.
    final_credit = _make_credit(
        conversation_id="trace_0",
        x_correlation_id=initial_xcorr,
        turn_index=1,
        num_turns=2,
    )
    await strategy.handle_credit_return(final_credit)

    assert len(issued_turns) == 2
    recycled_rid = _extract_rid(issued_turns[1].cache_bust_marker)
    assert recycled_rid is not None
    assert recycled_rid != initial_rid


@pytest.mark.asyncio
async def test_two_trajectories_same_starting_trace_get_distinct_markers():
    """Two trajectories at lane 0 and lane 1 mint different markers because
    trajectory_index differs. (TrajectorySource itself rejects duplicate
    trace_ids in trajectories, so we model 'same trace' as recycle reuse:
    trajectory[0] starts on trace_x; later trajectory[1]'s recycle pulls
    trace_x. We assert markers differ via the lane component instead.)"""
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_1", start_turn_index=0),
    ]
    user_config = _make_user_config(target=CacheBustTarget.SYSTEM_PREFIX)
    issued_turns: list = []

    async def capture(turn):
        issued_turns.append(turn)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture

    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        turns_per_trace=3,
        issuer=issuer,
        user_config=user_config,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    assert len(issued_turns) == 2
    rid0 = _extract_rid(issued_turns[0].cache_bust_marker)
    rid1 = _extract_rid(issued_turns[1].cache_bust_marker)
    assert rid0 is not None
    assert rid1 is not None
    # Different lane (trajectory_index) -> different digest, even with same
    # benchmark_id and same recycle_pass=0.
    assert rid0 != rid1


@pytest.mark.asyncio
async def test_target_none_emits_no_marker():
    """With target=NONE (or no user_config plumbed), cache_bust_marker is
    None and cache_bust_target is NONE on every issued turn."""
    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(2)
    ]
    user_config = _make_user_config(target=CacheBustTarget.NONE)
    issued_turns: list = []

    async def capture(turn):
        issued_turns.append(turn)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture

    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        turns_per_trace=3,
        issuer=issuer,
        user_config=user_config,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    assert len(issued_turns) == 2
    for turn in issued_turns:
        assert turn.cache_bust_marker is None
        assert turn.cache_bust_target == CacheBustTarget.NONE


@pytest.mark.asyncio
async def test_two_traces_at_same_pass_and_lane_get_distinct_markers():
    """Two different trace_ids landing on the same (recycle_pass, lane) tuple
    must mint distinct markers. Regression bar for the collision-free fix:
    the marker tuple now includes ``trace_id`` so cross-trace collisions on
    the same (pass, lane) are eliminated by construction.

    Setup: single-lane (concurrency=1) PROFILING run starting on trace_A.
    When trace_A finishes its only profile turn, the empty recycle queue
    forces FIFO reuse — but we seed a second trajectory by directly
    inspecting the strategy's marker state via the per-session minting path.
    Cleaner: drive two sessions on lane 0 explicitly via the mint helper and
    assert the digests differ. ``recycle_pass`` is per-trace_id so both
    start at 0; ``trajectory_index`` is fixed at 0; only trace_id differs.
    """
    trajectories = [Trajectory(conversation_id="trace_A", start_turn_index=0)]
    user_config = _make_user_config(target=CacheBustTarget.SYSTEM_PREFIX)

    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=2,
        turns_per_trace=2,
        user_config=user_config,
    )
    # Mint markers for two distinct trace_ids both at lane 0, both at
    # recycle_pass=0 (their first incarnation).
    marker_a = strategy._mint_marker_for_session(
        x_correlation_id="xcorr_a", trace_id="trace_A", trajectory_index=0
    )
    marker_b = strategy._mint_marker_for_session(
        x_correlation_id="xcorr_b", trace_id="trace_B", trajectory_index=0
    )
    rid_a = _extract_rid(marker_a)
    rid_b = _extract_rid(marker_b)
    assert rid_a is not None
    assert rid_b is not None
    assert rid_a != rid_b, (
        "Two distinct traces at the same (recycle_pass=0, lane=0) must "
        "produce distinct markers — collision-free uniqueness depends on "
        "trace_id being part of the digest tuple."
    )


# =============================================================================
# Signature lock: _spawn_from_recycle_or_id requires finished_correlation_id
# =============================================================================


def test_spawn_from_recycle_or_id_requires_finished_correlation_id() -> None:
    """``finished_correlation_id`` must be a required keyword-only parameter
    so the lane bookkeeping pop has a valid key on every code path."""
    import inspect

    sig = inspect.signature(AgenticReplayStrategy._spawn_from_recycle_or_id)
    param = sig.parameters["finished_correlation_id"]
    assert param.default is inspect.Parameter.empty
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.asyncio
async def test_spawn_from_recycle_or_id_pops_lane_and_marker_for_correlation() -> None:
    """The finished session's lane and marker entries are popped from the
    bookkeeping dicts so memory stays bounded by live concurrency."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    user_config = _make_user_config(target=CacheBustTarget.SYSTEM_PREFIX)
    strategy, *_ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=2,
        user_config=user_config,
    )

    correlation_id = "xcorr-finished"
    strategy._correlation_to_lane[correlation_id] = 7
    strategy._session_marker[correlation_id] = "[rid:abc123]"

    # Force early-return after the unconditional cleanup pop, isolating the
    # bookkeeping behavior from the spawn path.
    strategy.stop_checker.can_start_new_session = MagicMock(return_value=False)
    strategy._recycle_queue = asyncio.Queue()

    await strategy._spawn_from_recycle_or_id(
        "trace_0", finished_correlation_id=correlation_id
    )

    assert correlation_id not in strategy._correlation_to_lane
    assert correlation_id not in strategy._session_marker
