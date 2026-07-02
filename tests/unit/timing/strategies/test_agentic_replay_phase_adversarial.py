# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial unit tests for AgenticReplayStrategy phase-branching.

Spec §8.4.5 - attacks the phase-branching surface past the happy-path tests in
``test_agentic_replay.py``: defensive constructor checks, empty-trajectory handling,
warmup failure surfacing, no embedded wall-clock timeout, defensive pin for
PROFILING-without-WARMUP, mid-turn duration stops, in-WARMUP subagent dispatch
semantics, and the multi-construction defensive pin.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.scenario.base import TrajectoryWarmupFailedError
from aiperf.credit.structs import Credit
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import (
    Trajectory,
    TrajectorySource,
)

# =============================================================================
# Helpers (mirror test_agentic_replay.py patterns; kept local for isolation)
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
    ds = _make_dataset(num_traces, turns_per_trace)
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    _roots = [
        c.conversation_id
        for c in src._dataset_metadata.conversations
        if getattr(c, "is_root", True)
    ]
    src._dataset_sampler = SequentialSampler(_roots) if _roots else MagicMock()
    src._pool_size = len(_roots)
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
    timing_mode=None,
) -> tuple[AgenticReplayStrategy, AsyncMock, MagicMock, TrajectorySource]:
    src = _build_real_trajectory_source(num_traces, turns_per_trace, trajectories)
    cfg = MagicMock()
    cfg.phase = phase
    cfg.timing_mode = (
        timing_mode if timing_mode is not None else TimingMode.AGENTIC_REPLAY
    )
    cfg.concurrency = max(1, len(trajectories))
    issuer = issuer if issuer is not None else AsyncMock()
    scheduler = scheduler if scheduler is not None else MagicMock()
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )
    return strategy, issuer, scheduler, src


def _make_credit(
    *,
    conversation_id: str,
    turn_index: int,
    num_turns: int,
    phase: CreditPhase = CreditPhase.PROFILING,
    x_correlation_id: str = "xcorr",
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
# Test 1: WARMUP phase + non-AGENTIC_REPLAY timing_mode is a defensive case
# =============================================================================


def test_warmup_phase_with_non_agentic_timing_mode_pins_current_behavior():
    """Test 1: ``config.phase = WARMUP`` with ``config.timing_mode != AGENTIC_REPLAY``.

    The constructor today only validates ``config.phase``, not ``config.timing_mode``.
    This is technically a defensive gap - PhaseRunner builds the config so this
    should never happen in production. We pin the current behavior here so a
    future tightening (raise on mismatched timing_mode) flips this test, prompting
    a docs/CHANGELOG update rather than a silent escape.
    """
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    src = _build_real_trajectory_source(1, 2, trajectory)
    cfg = MagicMock()
    cfg.phase = CreditPhase.WARMUP
    cfg.timing_mode = TimingMode.REQUEST_RATE  # mismatched on purpose
    cfg.concurrency = 1
    # PINNED: today, the constructor accepts this without error. If a future
    # commit tightens this guard to ``raise ValueError``, this assertion will
    # fail and the corresponding negative test (rejection) should be added.
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=MagicMock(),
        stop_checker=MagicMock(),
        credit_issuer=AsyncMock(),
        lifecycle=MagicMock(),
    )
    assert strategy.config.timing_mode == TimingMode.REQUEST_RATE
    assert strategy.config.phase == CreditPhase.WARMUP


# =============================================================================
# Test 2: WARMUP empty trajectory -> no credits; PROFILING aborts with clear error
# =============================================================================


@pytest.mark.asyncio
async def test_warmup_empty_trajectories_emits_no_credits():
    """Test 2a: Empty trajectory during WARMUP -> strategy issues zero credits."""
    strategy, issuer, _, _ = _make_strategy(phase=CreditPhase.WARMUP, trajectories=[])
    await strategy.setup_phase()
    await strategy.execute_phase()
    assert issuer.issue_credit.await_count == 0


@pytest.mark.asyncio
async def test_profiling_empty_trajectories_aborts_setup_with_clear_error():
    """Test 2b: PROFILING phase with empty trajectory raises a clear error.

    The strategy MUST refuse to start PROFILING on an empty trajectory. Otherwise
    the recycle queue runs from an empty seed and quietly produces zero load.
    """
    strategy, _, _, _ = _make_strategy(phase=CreditPhase.PROFILING, trajectories=[])
    with pytest.raises(RuntimeError) as exc_info:
        await strategy.setup_phase()
    msg = str(exc_info.value)
    assert "trajectory" in msg.lower()
    assert "empty" in msg.lower() or "warmup" in msg.lower()


# =============================================================================
# Test 3: WARMUP credit terminal failure -> TrajectoryWarmupFailedError
#         and PROFILING never runs (the strategy contract).
# =============================================================================


@pytest.mark.asyncio
async def test_warmup_terminal_failure_blocks_profiling():
    """Test 3: ``record_warmup_failure`` accumulates; ``report_warmup_failures``
    raises ``TrajectoryWarmupFailedError`` so the orchestrator does not advance to
    PROFILING. We additionally pin that handle_credit_return remains a no-op
    in WARMUP regardless of failure state (failure routing is the issuer's job)."""
    trajectory = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectory
    )
    await strategy.setup_phase()
    await strategy.execute_phase()
    issuer.issue_credit.reset_mock()

    # Two trajectories fail terminally; one succeeds.
    strategy.record_warmup_failure("trace_0")
    strategy.record_warmup_failure("trace_2")

    # Even after recording failures, in-WARMUP credit-return remains a no-op.
    failed_credit = _make_credit(
        conversation_id="trace_0",
        turn_index=0,
        num_turns=3,
        phase=CreditPhase.WARMUP,
    )
    await strategy.handle_credit_return(failed_credit)
    assert issuer.issue_credit.await_count == 0

    # Reporting must raise so PhaseRunner aborts before PROFILING construction.
    with pytest.raises(TrajectoryWarmupFailedError) as exc_info:
        strategy.report_warmup_failures()
    assert exc_info.value.failed_trace_ids == ["trace_0", "trace_2"]


# =============================================================================
# Test 4: WARMUP exceeds 5 minutes wall-clock - strategy has no embedded timeout
# =============================================================================


@pytest.mark.asyncio
async def test_warmup_no_embedded_wallclock_abort():
    """Test 4: Strategy MUST NOT enforce its own wall-clock timeout.

    Spec §8.4.5: "WARMUP exceeds 5 minutes wall-clock - INFO log fires once;
    no abort." The 5-minute INFO log lives at the lifecycle layer (or higher);
    at the strategy layer we pin that ``execute_phase`` returns deterministically
    after dispatching trajectory credits and does NOT poll a deadline of any kind.
    Concretely: dispatch happens once and finishes; nothing in the strategy
    aborts a long-running warmup.
    """
    trajectory = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(2)
    ]
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectory
    )
    await strategy.setup_phase()
    await strategy.execute_phase()
    # Strategy dispatched all trajectory credits and returned without raising.
    assert issuer.issue_credit.await_count == 2
    # No internal deadline / cancellation state set on strategy.
    assert not hasattr(strategy, "_warmup_deadline")
    assert not hasattr(strategy, "_warmup_aborted")


# =============================================================================
# Test 5: PROFILING without preceding WARMUP -> strategy is operator-trusting
# =============================================================================


@pytest.mark.asyncio
async def test_profiling_without_preceding_warmup_does_not_self_enforce():
    """Test 5: PROFILING with a populated trajectory but no recorded WARMUP completion
    is permitted by the strategy. Ordering enforcement lives at PhaseRunner /
    config build time (the 'no warmup config' error is a config concern). The
    strategy itself is operator-trusting on phase ordering; we pin that here
    so the responsibility split is documented in tests.

    A degenerate case where PROFILING starts on an empty trajectory is the
    *signal* that something is wrong - that case is covered by Test 2b.
    """
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy, issuer, _, src = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        num_traces=3,
        turns_per_trace=4,
    )
    # No prior WARMUP strategy ran, no record_warmup_failure invoked: PROFILING
    # setup + execute must succeed.
    await strategy.setup_phase()
    await strategy.execute_phase()
    # One resume credit at k_i + 1 = 1.
    assert issuer.issue_credit.await_count == 1
    issued = issuer.issue_credit.await_args.args[0]
    assert issued.turn_index == 1
    assert issued.conversation_id == "trace_0"


# =============================================================================
# Test 6: PROFILING DurationStopCondition mid-turn -> in-flight finishes; metrics include it
# =============================================================================


@pytest.mark.asyncio
async def test_profiling_credit_return_after_stop_dispatches_next_turn():
    """Test 6: When ``DurationStopCondition`` has fired, an in-flight trajectory
    member returning mid-session still triggers ``handle_credit_return`` -> next
    turn issuance. The strategy does NOT short-circuit on its own; whether the
    issuer ultimately admits or rejects the new credit (because sending is
    complete) is an issuer/lifecycle concern. This pins the existing aiperf
    semantic that an in-flight request's response is *included* in metrics."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issuer = AsyncMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        num_traces=3,
        turns_per_trace=4,
        issuer=issuer,
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()

    # Simulate a stop-condition firing (lifecycle marked "sending complete").
    # Strategy must NOT consult lifecycle.is_sending_complete to drop the next
    # turn - that's the issuer's job. So a mid-session credit return after stop
    # still drives a next-turn dispatch.
    strategy.lifecycle.is_sending_complete = True

    in_flight_return = _make_credit(
        conversation_id="trace_0", turn_index=1, num_turns=4
    )
    await strategy.handle_credit_return(in_flight_return)
    assert issuer.issue_credit.await_count == 1
    next_turn = issuer.issue_credit.await_args.args[0]
    assert next_turn.turn_index == 2
    assert next_turn.conversation_id == "trace_0"


# =============================================================================
# Test 7: Subagent SPAWN during WARMUP -> strategy does not branch; orchestrator handles it
# =============================================================================


@pytest.mark.asyncio
async def test_warmup_credit_return_does_not_self_spawn_subagents():
    """Test 7: When a trajectory warmup turn ``k_i`` happens to be a turn flagged for
    SPAWN, the spawn is dispatched by ``BranchOrchestrator`` (independent of
    strategy). The strategy's own ``handle_credit_return`` is a no-op in WARMUP
    so it MUST NOT issue any follow-up credit, even when the returning credit
    carries SPAWN-relevant flags (``has_forks=True``,
    ``branch_mode=SPAWN``). The spawned credit's phase tagging and barrier
    accounting is the orchestrator + issuer's responsibility.

    Pin: a WARMUP credit returning with ``has_forks=True`` + branch_mode=SPAWN
    yields zero strategy-level dispatches.
    """
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issuer = AsyncMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectory,
        turns_per_trace=4,
        issuer=issuer,
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()

    # Build a credit that mimics a trajectory warmup credit at turn k_i with SPAWN
    # branch semantics (the kind a DAG turn might have).
    spawning_credit = Credit(
        id=0,
        phase=CreditPhase.WARMUP,
        conversation_id="trace_0",
        x_correlation_id="xcorr",
        turn_index=0,
        num_turns=4,
        issued_at_ns=0,
        branch_mode=ConversationBranchMode.SPAWN,
        has_forks=True,
    )
    await strategy.handle_credit_return(spawning_credit)
    assert issuer.issue_credit.await_count == 0


# =============================================================================
# Test 8: Multiple constructions within one phase -> independent instances (PINNED)
# =============================================================================


def test_strategy_constructed_multiple_times_within_one_phase_is_independent():
    """Test 8: PhaseRunner is contractually expected to construct the strategy
    exactly once per phase, but the strategy class today does NOT enforce a
    singleton - each construction yields a fresh, independent instance that
    shares the trajectory source state.

    We pin: two AgenticReplayStrategy instances built for the same PROFILING
    phase against the same trajectory source share trajectory + metadata state but have
    independent recycle queues and independent failure accumulators. A future
    commit that adds a class-level construction guard will flip this assertion
    and prompt a CHANGELOG entry.
    """
    trajectory = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_1", start_turn_index=1),
    ]
    src = _build_real_trajectory_source(3, 4, trajectory)

    def _build():
        cfg = MagicMock()
        cfg.phase = CreditPhase.PROFILING
        cfg.timing_mode = TimingMode.AGENTIC_REPLAY
        cfg.concurrency = 2
        return AgenticReplayStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=AsyncMock(),
            lifecycle=MagicMock(),
        )

    s1 = _build()
    s2 = _build()

    assert s1 is not s2
    assert s1.conversation_source is s2.conversation_source
    # Independent failure accumulators.
    s1.record_warmup_failure("trace_0")
    assert s1._failed_warmup_traces == ["trace_0"]
    assert s2._failed_warmup_traces == []
    # Independent double-recycle guard sets.
    s1._in_flight_recycled.add("x")
    assert "x" not in s2._in_flight_recycled


@pytest.mark.asyncio
async def test_strategy_setup_twice_within_one_phase_is_idempotent():
    """Calling ``setup_phase`` twice on the same instance MUST be safe (no
    error, recycle still works). Recycle state now lives in the shared dataset
    sampler, not a per-setup queue, so there is nothing to leak or duplicate.
    """
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        num_traces=3,
        turns_per_trace=4,
    )
    await strategy.setup_phase()
    await strategy.setup_phase()  # must not raise

    # Recycle still resolves a root from the sampler.
    assert strategy.conversation_source.next_recycle_conversation_id() is not None


# =============================================================================
# Bonus pin: warmup INFO log on long elapsed time would live OUTSIDE the
# strategy. This explicit no-op test guards against a regression where
# someone adds a strategy-level long-warmup logger that fires per-credit
# (which would spam logs at high trajectory sizes).
# =============================================================================


@pytest.mark.asyncio
async def test_warmup_execute_does_not_emit_per_credit_long_warmup_log(caplog):
    """The strategy's WARMUP execute path must not emit a long-warmup INFO log
    per trajectory credit. (Spec §8.4.5: log fires once - if at all - and not from
    inside the dispatch loop.)"""
    trajectory = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(5)
    ]
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectory
    )
    with caplog.at_level(logging.INFO, logger="AgenticReplayTiming"):
        await strategy.setup_phase()
        await strategy.execute_phase()
    long_warmup_logs = [
        r
        for r in caplog.records
        if "5 minutes" in r.getMessage() or "exceeded" in r.getMessage().lower()
    ]
    assert long_warmup_logs == []
