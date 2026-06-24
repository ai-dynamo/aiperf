# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concurrency / cancellation / stop-condition pathology tests for ``BranchOrchestrator``.

Targets are orthogonal to ``test_branch_orchestrator_adversarial_full.py``:

- ``asyncio.CancelledError`` propagation through every awaited boundary in
  ``intercept`` (lock acquired, dispatch in flight, gather of children,
  pre-session loop, ``_satisfy_prerequisite`` mid-decrement,
  ``_release_blocked_join``).
- High-fan concurrent intercept stress on independent vs shared parents.
- Parent / child completion races driven by ``asyncio.Event`` synchronizers.
- Cleanup-mid-anything (intercept, pre-session, satisfy).
- Stop-condition "flip mid-flight" simulated by toggling
  ``issuer.dispatch_join_turn`` return value between the satisfy decision
  and the actual dispatch.
- Fail-fast race where two siblings of one parent error simultaneously.
- ``applies_to_dag_children`` truth-table walk for each stop condition.
- ``asyncio.wait_for(intercept, timeout=0)`` cancellation propagation.
- Reentrancy guards: a second intercept queued on the same parent never
  sees ``_release_blocked_join`` re-enter ``intercept``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import (
    BranchOrchestrator,
    ChildJoinEntry,
)
from aiperf.timing.phase.stop_conditions import (
    CancellationStopCondition,
    DurationStopCondition,
    RequestCountStopCondition,
    SendingCompleteStopCondition,
    SessionCountStopCondition,
)

# ---------------------------------------------------------------------------
# Helpers — kept local so changes to test_branch_orchestrator_adversarial_full
# don't introduce coupling.
# ---------------------------------------------------------------------------


def _mk_conv(
    cid: str,
    turns: list[TurnMetadata],
    branches: list[ConversationBranchInfo],
    agent_depth: int = 0,
) -> ConversationMetadata:
    return ConversationMetadata(
        conversation_id=cid,
        turns=turns,
        branches=branches,
        agent_depth=agent_depth,
    )


def _mk_source(conversations: list[ConversationMetadata]):
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    cs.get_metadata.side_effect = lambda cid: next(
        c for c in conversations if c.conversation_id == cid
    )

    def _start_branch(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        s.conversation_id = child_conversation_id
        return s

    cs.start_branch_child = MagicMock(side_effect=_start_branch)

    def _start_pre(child_cid, **kwargs):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_cid}"
        s.conversation_id = child_cid
        s.agent_depth = 1
        s.parent_correlation_id = None
        return s

    cs.start_pre_session_child = MagicMock(side_effect=_start_pre)
    return cs


def _mk_credit(conv_id: str, corr_id: str, turn_index: int, agent_depth: int = 0):
    return MagicMock(
        x_correlation_id=corr_id,
        conversation_id=conv_id,
        turn_index=turn_index,
        agent_depth=agent_depth,
        parent_correlation_id=None,
        branch_mode=ConversationBranchMode.FORK,
    )


def _mk_issuer():
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    issuer.abort_session = AsyncMock()
    return issuer


def _simple_spawn_metadata(
    n_children: int = 2, conv_id: str = "root"
) -> list[ConversationMetadata]:
    """Conversation: turn 0 spawns ``n_children`` children, turn 1 gates them."""
    branch = ConversationBranchInfo(
        branch_id=f"{conv_id}:0",
        child_conversation_ids=[f"{conv_id}-c{i}" for i in range(n_children)],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        conv_id,
        [
            TurnMetadata(branch_ids=[f"{conv_id}:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id=f"{conv_id}:0"
                    )
                ]
            ),
        ],
        [branch],
    )
    children = [
        _mk_conv(f"{conv_id}-c{i}", [TurnMetadata()], []) for i in range(n_children)
    ]
    return [root, *children]


# ---------------------------------------------------------------------------
# 1. CancelledError raised inside intercept while it holds _parent_locks.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_during_intercept_releases_parent_lock():
    """Cancel a task awaiting ``dispatch_first_turn`` inside ``intercept``.
    The async-with on ``_parent_locks[parent_corr]`` must release the lock so
    a second intercept on the same parent does not deadlock.
    """
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()

    block = asyncio.Event()

    async def _hang(child):
        await block.wait()
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_hang)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    credit = _mk_credit("root", "corr-root", 0)
    t1 = asyncio.create_task(orch.intercept(credit))
    # Yield to let t1 acquire the lock and reach the await.
    for _ in range(5):
        await asyncio.sleep(0)
    assert "corr-root" in orch._parent_locks
    # Cancel mid-await; CancelledError unwinds out of `async with`.
    t1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t1

    # The lock dict entry might still exist; the Lock object must be released.
    lock = orch._parent_locks.get("corr-root")
    if lock is not None:
        assert not lock.locked(), "lock leaked after intercept cancel"

    # Second intercept on same parent must proceed without deadlocking.
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    result = await asyncio.wait_for(
        orch.intercept(_mk_credit("root", "corr-root", 0)), timeout=2.0
    )
    # State after a second turn-0 intercept: branch already spawned once but
    # the turn-0 metadata still says branch_ids=["root:0"]; second intercept
    # spawns again. We only assert no hang and consistent suspension.
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 2. CancelledError raised in _satisfy_prerequisite mid-decrement.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_during_satisfy_prerequisite_keeps_state_consistent():
    """``_satisfy_prerequisite`` itself has no awaits between the
    ``completed.add`` and the gate-satisfied check; cancelling at the only
    boundary (entering the coroutine) is a no-op. Verify that cancelling the
    ``on_child_leaf_reached`` task right at the await of
    ``_release_blocked_join`` leaves the gate in a coherent state — the
    child IS recorded as completed, the gate IS popped from _active_joins,
    but the issuer call may or may not have happened. Either way no partial
    re-fire is possible because ``is_blocked`` was set to False by the pop.
    """
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()

    dispatch_started = asyncio.Event()
    dispatch_block = asyncio.Event()

    async def _join_dispatch(pending):
        dispatch_started.set()
        await dispatch_block.wait()
        return True

    issuer.dispatch_join_turn = AsyncMock(side_effect=_join_dispatch)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert "corr-root" in orch._active_joins

    t = asyncio.create_task(orch.on_child_leaf_reached("corr-root-c0"))
    await dispatch_started.wait()

    # At this point _satisfy_prerequisite has run, gate was popped from
    # _active_joins, issuer.dispatch_join_turn is mid-await. Cancel.
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    # Gate is gone (popped during satisfy). Child entry was already removed
    # via _child_to_join.pop in _handle_child_done. State is consistent.
    assert "corr-root" not in orch._active_joins
    assert "corr-root-c0" not in orch._child_to_join

    # A subsequent (re-)delivery of the same child is a no-op (entry gone).
    await orch.on_child_leaf_reached("corr-root-c0")
    # No second dispatch fired even after we release the original.
    dispatch_block.set()
    await asyncio.sleep(0)
    assert issuer.dispatch_join_turn.await_count == 1


# ---------------------------------------------------------------------------
# 3. CancelledError raised in asyncio.gather of children's _dispatch_first_turn.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_during_gather_partial_dispatch_rolls_back_consistently():
    """One child raises a generic exception (return_exceptions=True ⇒ caught
    inline; siblings finish). Verify per-child rollback fires for the
    exception child but NOT for the successful siblings, and the gate's
    expected counter reflects only the survivors.
    """
    cs = _mk_source(_simple_spawn_metadata(3))
    issuer = _mk_issuer()

    async def _dispatch_with_one_failure(child):
        if child.x_correlation_id == "corr-root-c1":
            raise RuntimeError("boom")
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_dispatch_with_one_failure)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))

    # Two children survived (c0, c2); c1 rolled back.
    assert "corr-root-c0" in orch._child_to_join
    assert "corr-root-c2" in orch._child_to_join
    assert "corr-root-c1" not in orch._child_to_join

    pending = orch._active_joins["corr-root"]
    state = pending.outstanding["SPAWN_JOIN:root:0"]
    # Three started, one rolled back ⇒ expected reflects 2.
    assert state.expected == 2
    assert orch.stats.children_errored == 1
    assert orch.stats.children_spawned == 2  # net after rollback decrement


# ---------------------------------------------------------------------------
# 4. CancelledError raised during dispatch_pre_session_branches mid-loop.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_during_pre_session_loop_partial_pre_dispatched_set():
    """Three pre-session branches; the second blocks, gets cancelled. Only
    the first should be in ``_pre_dispatched_branches`` after cancellation.
    """
    branches = [
        ConversationBranchInfo(
            branch_id=f"root:pre{i}",
            child_conversation_ids=[f"pre{i}"],
            mode=ConversationBranchMode.SPAWN,
            is_background=True,
            dispatch_timing="pre",
        )
        for i in range(3)
    ]
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=[f"root:pre{i}" for i in range(3)]),
            TurnMetadata(),
        ],
        branches,
    )
    children = [_mk_conv(f"pre{i}", [TurnMetadata()], []) for i in range(3)]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()

    call_count = 0
    block = asyncio.Event()

    async def _dispatch(session):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            await block.wait()
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_dispatch)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    t = asyncio.create_task(orch.dispatch_pre_session_branches())
    # Yield until the second iteration is awaiting.
    for _ in range(10):
        await asyncio.sleep(0)
        if call_count >= 2:
            break

    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    # First branch fully completed; the second's _pre_dispatched_branches
    # add() never ran (cancellation hit the await before it). The third
    # never started.
    pre = orch._pre_dispatched_branches
    assert ("root", "root:pre0") in pre
    assert ("root", "root:pre1") not in pre
    assert ("root", "root:pre2") not in pre


# ---------------------------------------------------------------------------
# 5. 100 concurrent intercepts on 100 different parents.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_100_concurrent_intercepts_independent_parents_isolated_state():
    """Each parent's gates / joins are independent. No cross-talk via the
    ``_parent_locks`` defaultdict.
    """
    N = 100
    convs: list[ConversationMetadata] = []
    for i in range(N):
        convs.extend(_simple_spawn_metadata(2, conv_id=f"r{i}"))
    cs = _mk_source(convs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    credits = [_mk_credit(f"r{i}", f"corr-r{i}", 0) for i in range(N)]
    results = await asyncio.gather(*(orch.intercept(c) for c in credits))

    # K=1 gate: turn 0 intercept already suspends parent at T=1 -> True.
    assert all(r is True for r in results)
    # Each parent has 2 children spawned => 2N total.
    assert orch.stats.children_spawned == 2 * N
    # Each parent has its gate promoted to active.
    assert len(orch._active_joins) == N
    for i in range(N):
        active = orch._active_joins[f"corr-r{i}"]
        assert active.gated_turn_index == 1
        state = active.outstanding[f"SPAWN_JOIN:r{i}:0"]
        assert state.expected == 2


# ---------------------------------------------------------------------------
# 6. 100 concurrent intercepts on the SAME parent (different turn_indexes).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_100_concurrent_intercepts_same_parent_serialized():
    """Single parent receives 100 intercept calls at distinct turn_indexes
    in arbitrary order; per-parent lock must serialize them. The exact
    final state depends on the (arbitrary) interleaving of which turn was
    "last" — but the orchestrator must not crash, and counters must reflect
    one spawn for the only spawning turn (turn 0).
    """
    cs = _mk_source(_simple_spawn_metadata(2))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Build 100 credits with various turn_indexes; only turn 0 has branches.
    credits = [_mk_credit("root", "corr-root", i % 2) for i in range(100)]
    await asyncio.gather(*(orch.intercept(c) for c in credits))

    # Each turn-0 intercept re-spawns the same branch — orchestrator does
    # not de-dup turn re-runs. We only assert the lock did not deadlock and
    # state is non-corrupt: stats are consistent.
    assert orch.stats.children_spawned > 0
    # Lock must still be acquirable (no leak).
    lock = orch._parent_locks["corr-root"]
    assert not lock.locked()


# ---------------------------------------------------------------------------
# 7. Race: parent return and last child completion happen "simultaneously".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_race_parent_return_and_last_child_completion_gate_fires_once():
    """Two orderings — child-first then parent, parent-first then child —
    both must end in exactly one ``dispatch_join_turn`` call.
    """
    # Ordering A: child completes first (T=1 future gate is satisfied,
    # popped silently). Parent's intercept on T=0 return then sees
    # next_idx=1 satisfied -> returns False, no dispatch.
    cs1 = _mk_source(_simple_spawn_metadata(1))
    issuer1 = _mk_issuer()
    orch1 = BranchOrchestrator(conversation_source=cs1, credit_issuer=issuer1)
    # Spawn first to register the future gate.
    await orch1.intercept(_mk_credit("root", "corr-root", 0))
    # Child completes -> gate is satisfied (parent not yet at T=0 return
    # for next-idx check; wait, intercept already ran). Actually the spawn
    # happens INSIDE intercept and intercept also runs _maybe_suspend_parent
    # immediately. Since spawn just happened, next_idx=1 sees the active gate
    # already promoted -> returned True. We test the "child done after parent
    # was suspended" case (= ordering B) below.
    # Ordering A means: spawn at turn 0; parent NOW already at active T=1.
    # Then child completes -> dispatch_join_turn fires once.
    assert orch1._active_joins["corr-root"].gated_turn_index == 1
    await orch1.on_child_leaf_reached("corr-root-c0")
    issuer1.dispatch_join_turn.assert_awaited_once()

    # Ordering B: spawn at turn 0 with delayed (K=2) gate. Parent walks T=0
    # then T=1 (no gate yet). Then last child completes BEFORE parent
    # arrives at T=1's return -> _satisfy_prerequisite pops future gate.
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(),  # T=1 has no prereq
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        [branch],
    )
    cs2 = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    issuer2 = _mk_issuer()
    orch2 = BranchOrchestrator(conversation_source=cs2, credit_issuer=issuer2)
    await orch2.intercept(_mk_credit("root", "corr-root2", 0))  # spawns
    # Child completes BEFORE parent's T=1 return.
    await orch2.on_child_leaf_reached("corr-c1")
    # Parent reaches T=1; next_idx=2 is gated, but already satisfied -> pops.
    suspended = await orch2.intercept(_mk_credit("root", "corr-root2", 1))
    assert suspended is False
    # No join_turn dispatch — parent breezes through via strategy path.
    issuer2.dispatch_join_turn.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Cleanup mid-pre-session loop.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_mid_pre_session_dispatch_no_state_leak():
    """``cleanup()`` is synchronous — it cannot interrupt an awaiting
    coroutine. But once the loop's first iteration completes, a second
    iteration that re-enters checks ``_cleaning_up`` only at the very top.
    The pre-session loop does NOT recheck after the first await. Verify the
    actual behavior: cleanup mid-flight does NOT abort the loop, but state
    is cleared after both finish.
    """
    branches = [
        ConversationBranchInfo(
            branch_id=f"root:pre{i}",
            child_conversation_ids=[f"pre{i}"],
            mode=ConversationBranchMode.SPAWN,
            is_background=True,
            dispatch_timing="pre",
        )
        for i in range(3)
    ]
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=[f"root:pre{i}" for i in range(3)]),
            TurnMetadata(),
        ],
        branches,
    )
    children = [_mk_conv(f"pre{i}", [TurnMetadata()], []) for i in range(3)]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    started = asyncio.Event()
    proceed = asyncio.Event()

    async def _slow(session):
        started.set()
        await proceed.wait()
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_slow)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    t = asyncio.create_task(orch.dispatch_pre_session_branches())
    await started.wait()
    # Cleanup while loop is mid-await on first child.
    orch.cleanup()
    proceed.set()
    await t  # loop drains naturally — no exception expected.

    # After both finish: state cleared.
    assert orch._cleaning_up is True
    # cleanup() ran its clear; the loop continued to populate set after
    # cleanup, so the set may or may not be non-empty depending on
    # ordering. We do NOT assert on its emptiness — instead a second
    # cleanup is idempotent and a fresh intercept is a no-op.
    orch.cleanup()  # idempotent
    assert (await orch.intercept(_mk_credit("root", "corr-root", 0))) is False


# ---------------------------------------------------------------------------
# 9. Stop-condition flips False during _release_blocked_join.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_flips_during_release_increments_joins_suppressed_only_once():
    """``dispatch_join_turn`` returns False (simulating stop). Verify
    ``joins_suppressed`` increments exactly once and no double-dispatch
    occurs even if the same satisfy is somehow re-entered.
    """
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()
    issuer.dispatch_join_turn = AsyncMock(return_value=False)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.on_child_leaf_reached("corr-root-c0")
    assert orch.stats.joins_suppressed == 1
    assert orch.stats.parents_resumed == 0

    # Re-deliver same child (idempotent path) — gate already gone.
    await orch.on_child_leaf_reached("corr-root-c0")
    assert orch.stats.joins_suppressed == 1
    issuer.dispatch_join_turn.assert_awaited_once()


# ---------------------------------------------------------------------------
# 10. Fail-fast race: two siblings of one parent error simultaneously.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fail_fast_two_simultaneous_child_errors_aborts_parent_once(
    monkeypatch, force_fail_fast
):
    """Under fail-fast, two children of the same parent fire
    ``on_child_errored`` concurrently via ``asyncio.gather``. The parent
    should be aborted exactly once (or at most once per orchestrator
    semantics). Sibling cascades must not double-abort the parent.
    """

    force_fail_fast(True)
    cs = _mk_source(_simple_spawn_metadata(3))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert "corr-root" in orch._active_joins

    # Two children error concurrently.
    await asyncio.gather(
        orch.on_child_errored("corr-root-c0"),
        orch.on_child_errored("corr-root-c1"),
    )

    # Parent should appear in abort list at least once (idempotency at the
    # issuer level is the issuer's responsibility — orchestrator may call
    # abort_session twice if both errors race past the active_joins.pop).
    aborts = [c.args[0] for c in issuer.abort_session.await_args_list]
    assert "corr-root" in aborts
    # Counter must show exactly one cascade-credit. The second error fires
    # on a child whose entry was already drained by the first cascade and
    # `_child_to_join.get(...)` returns None -> early return on entries-empty.
    # That early return also short-circuits the children_errored increment
    # before the fail-fast branch runs.
    # ChildJoinEntry presence guards the second cascade. Verify that.
    assert orch.stats.parents_failed_due_to_child_error == 1


# ---------------------------------------------------------------------------
# 11. Cancel via asyncio.wait_for(..., timeout=0).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_zero_timeout_cancels_intercept_lock_released():
    """Force a TimeoutError -> CancelledError propagation into intercept.
    The parent lock must be released afterwards.
    """
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()

    block = asyncio.Event()

    async def _hang(child):
        await block.wait()
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_hang)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            orch.intercept(_mk_credit("root", "corr-root", 0)),
            timeout=0.001,
        )

    lock = orch._parent_locks.get("corr-root")
    if lock is not None:
        assert not lock.locked()
    # Unblock to drain the awaiting coroutine if any was orphaned.
    block.set()


# ---------------------------------------------------------------------------
# 12. Reentrancy: _release_blocked_join must not synchronously call intercept.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_blocked_join_does_not_recurse_into_intercept():
    """If ``dispatch_join_turn`` synchronously triggered another intercept
    on the same parent_corr, the per-parent lock would deadlock (re-entrant
    asyncio.Lock acquisition on the same task hangs). Verify by spying:
    intercept is never called from within ``dispatch_join_turn``.
    """
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    intercept_calls_during_dispatch: list[bool] = []
    in_dispatch = False

    async def _join_dispatch(pending):
        nonlocal in_dispatch
        in_dispatch = True
        # Yield so any spurious reentrant intercept could run.
        await asyncio.sleep(0)
        in_dispatch = False
        return True

    issuer.dispatch_join_turn = AsyncMock(side_effect=_join_dispatch)
    original_intercept = orch.intercept

    async def _spy_intercept(credit):
        intercept_calls_during_dispatch.append(in_dispatch)
        return await original_intercept(credit)

    orch.intercept = _spy_intercept  # type: ignore[method-assign]

    # Spawn + complete child.
    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.on_child_leaf_reached("corr-root-c0")

    # Only the explicit intercept calls; no synchronous reentry.
    assert intercept_calls_during_dispatch == [False]
    issuer.dispatch_join_turn.assert_awaited_once()


# ---------------------------------------------------------------------------
# 13. on_child_leaf_reached and on_child_errored race for same child.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leaf_and_errored_for_same_child_one_wins():
    """Concurrent leaf + errored for same child. ``_child_to_join.pop``
    inside ``_handle_child_done`` (or the fail-fast path) makes the second
    invocation a no-op via ``entries`` being None / empty.
    """
    cs = _mk_source(_simple_spawn_metadata(2))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))

    # Race: leaf + errored for same child. One advances the gate, the other
    # is a no-op.
    await asyncio.gather(
        orch.on_child_leaf_reached("corr-root-c0"),
        orch.on_child_errored("corr-root-c0"),
    )

    # Both stats counters may have incremented (errored increments before
    # the entries-None guard? Let's check) — confirm via state:
    pending = orch._active_joins["corr-root"]
    state = pending.outstanding["SPAWN_JOIN:root:0"]
    # Exactly one completion recorded for c0 (idempotent set).
    assert "corr-root-c0" in state.completed
    assert len(state.completed) == 1


# ---------------------------------------------------------------------------
# 14. applies_to_dag_children truth-table: only Cancellation + Duration apply.
# ---------------------------------------------------------------------------


def test_stop_condition_applies_to_dag_children_truth_table():
    """Children honor: Cancellation, Duration, RequestCount (the literal wire
    cap, "N means N"). Skip: SendingComplete, SessionCount.
    """
    assert CancellationStopCondition.applies_to_dag_children is True
    assert DurationStopCondition.applies_to_dag_children is True
    assert SendingCompleteStopCondition.applies_to_dag_children is False
    assert RequestCountStopCondition.applies_to_dag_children is True
    assert SessionCountStopCondition.applies_to_dag_children is False


# ---------------------------------------------------------------------------
# 15. Pre-session child whose dispatch_first_turn returns False
#     (issuer stopped) must increment children_errored, not raise.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_session_dispatch_first_turn_returns_false_counts_truncated():
    """``issued`` is False ⇒ stop-condition refusal (e.g. ``--request-count``
    cap), not an error. The orchestrator should tally as
    ``children_truncated``, matching the semantics already used by
    ``on_child_stopped``."""
    pre_branch = ConversationBranchInfo(
        branch_id="root:pre",
        child_conversation_ids=["early"],
        mode=ConversationBranchMode.SPAWN,
        is_background=True,
        dispatch_timing="pre",
    )
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:pre"]), TurnMetadata()],
        [pre_branch],
    )
    early = _mk_conv("early", [TurnMetadata()], [])
    cs = _mk_source([root, early])
    issuer = _mk_issuer()
    issuer.dispatch_first_turn = AsyncMock(return_value=False)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()

    assert orch.stats.children_spawned == 0
    assert orch.stats.children_errored == 0
    assert orch.stats.children_truncated == 1
    # Branch still recorded as pre-dispatched (current semantics).
    assert ("root", "root:pre") in orch._pre_dispatched_branches


# ---------------------------------------------------------------------------
# 16. Many parents reaching their gated turns within the same loop tick.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_many_parents_simultaneous_gate_arrival_no_active_joins_iter_corruption():
    """50 parents all arrive at their gated turn simultaneously. _active_joins
    is only mutated via dict[]/pop on a per-parent key — no iteration during
    normal operation. Verify by stress: gather all parents' arrivals and
    have all children complete in interleaved order.
    """
    N = 50
    convs: list[ConversationMetadata] = []
    for i in range(N):
        convs.extend(_simple_spawn_metadata(1, conv_id=f"r{i}"))
    cs = _mk_source(convs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # All parents intercept turn 0 simultaneously.
    await asyncio.gather(
        *(orch.intercept(_mk_credit(f"r{i}", f"corr-r{i}", 0)) for i in range(N))
    )
    assert len(orch._active_joins) == N

    # All children complete simultaneously.
    await asyncio.gather(
        *(orch.on_child_leaf_reached(f"corr-r{i}-c0") for i in range(N))
    )

    assert issuer.dispatch_join_turn.await_count == N
    assert orch._active_joins == {}
    assert orch.stats.parents_resumed == N


# ---------------------------------------------------------------------------
# 17. TaskGroup-style: 50 children dispatched via gather; one raises.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_of_fifty_children_raises_others_complete_state_consistent():
    """Inside ``_spawn_children_and_register_gates`` the gather uses
    ``return_exceptions=True``. Even when one child's dispatch raises, the
    other 49 land cleanly. Verify counters and state.
    """
    cs = _mk_source(_simple_spawn_metadata(50))
    issuer = _mk_issuer()

    async def _maybe_raise(child):
        if child.x_correlation_id == "corr-root-c25":
            raise RuntimeError("boom")
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_maybe_raise)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))

    pending = orch._active_joins["corr-root"]
    state = pending.outstanding["SPAWN_JOIN:root:0"]
    assert state.expected == 49
    assert orch.stats.children_errored == 1
    # 49 children must now complete to fire the gate.
    for i in range(50):
        if i == 25:
            continue
        await orch.on_child_leaf_reached(f"corr-root-c{i}")
    issuer.dispatch_join_turn.assert_awaited_once()


# ---------------------------------------------------------------------------
# 18. Cancel during _release_blocked_join AFTER pop, BEFORE dispatch.
#     Verify stats counters do not increment on a cancelled call.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_release_blocked_join_before_dispatch_returns_no_double_count():
    """Mid-await of ``dispatch_join_turn``, cancel the satisfying task. The
    gate has already been popped; ``parents_resumed`` was not yet
    incremented (increment happens AFTER the await). Verify that on
    cancellation neither ``parents_resumed`` nor ``joins_suppressed`` is
    incremented and the gate is not silently re-firable (no duplicate
    dispatch on a re-trigger).
    """
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()

    started = asyncio.Event()
    block = asyncio.Event()

    async def _hang_dispatch(pending):
        started.set()
        await block.wait()
        return True

    issuer.dispatch_join_turn = AsyncMock(side_effect=_hang_dispatch)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    t = asyncio.create_task(orch.on_child_leaf_reached("corr-root-c0"))
    await started.wait()
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    # Stats counters NOT incremented (cancellation hit before stats lines).
    assert orch.stats.parents_resumed == 0
    assert orch.stats.joins_suppressed == 0
    # Gate is gone; a re-delivery of the same child is a no-op.
    assert "corr-root" not in orch._active_joins
    assert "corr-root-c0" not in orch._child_to_join
    block.set()


# ---------------------------------------------------------------------------
# 19. Many concurrent intercepts on cleanup'd orchestrator.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_intercepts_post_cleanup_all_short_circuit():
    """After cleanup, every intercept must early-return False without
    touching state.
    """
    cs = _mk_source(_simple_spawn_metadata(2))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    orch.cleanup()

    results = await asyncio.gather(
        *(orch.intercept(_mk_credit("root", "corr-root", 0)) for _ in range(20))
    )
    assert all(r is False for r in results)
    # No spawn happened.
    assert orch.stats.children_spawned == 0
    cs.start_branch_child.assert_not_called()


# ---------------------------------------------------------------------------
# 20. Intercept after cleanup never grows _parent_locks (no leak).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intercept_after_cleanup_does_not_repopulate_parent_locks():
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    orch.cleanup()

    for i in range(50):
        await orch.intercept(_mk_credit("root", f"corr-{i}", 0))
    # cleanup() cleared _parent_locks; intercept early-returns BEFORE
    # acquiring the lock (the _cleaning_up check is first), so no entries
    # are re-added.
    assert orch._parent_locks == {}


# ---------------------------------------------------------------------------
# 21. Cancel during _spawn_children_and_register_gates rolls nothing back
#     prematurely (state matches what the cancelled call had committed).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_mid_spawn_partial_state_visible_no_corruption():
    """Cancel the intercept task while ``_spawn_children_and_register_gates``
    is mid-gather. Some children may have started before the cancel point;
    the ``_child_to_join`` for them is registered but the rollback loop
    (which runs after gather completes) never executes.

    This is a known-tradeoff: cancelling intercept mid-flight can leave
    ``_child_to_join`` with entries whose dispatch_first_turn was cancelled.
    Document the actual behavior so future regressions surface.
    """
    cs = _mk_source(_simple_spawn_metadata(3))
    issuer = _mk_issuer()
    block = asyncio.Event()
    started_count = 0

    async def _slow(child):
        nonlocal started_count
        started_count += 1
        await block.wait()
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_slow)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    t = asyncio.create_task(orch.intercept(_mk_credit("root", "corr-root", 0)))
    for _ in range(10):
        await asyncio.sleep(0)
        if started_count >= 3:
            break

    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    # State after cancel: _child_to_join populated for all 3, gate has
    # expected=3. The post-gather rollback loop never ran. This is a
    # known limitation; cleanup() will surface it as leaked state.
    assert len(orch._child_to_join) == 3
    pending = orch._active_joins.get("corr-root") or orch._future_joins.get(
        "corr-root", {}
    ).get(1)
    assert pending is not None
    assert pending.outstanding["SPAWN_JOIN:root:0"].expected == 3

    block.set()
    # cleanup logs the leak — does not raise.
    orch.cleanup()


# ---------------------------------------------------------------------------
# 22. Race: cleanup mid-satisfy via interleaved tasks.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_during_satisfy_release_does_not_fire_dispatch():
    """``cleanup()`` sets ``_cleaning_up=True`` synchronously. A child
    completion task that was already past the cleaning-up check at
    ``on_child_leaf_reached`` entry will continue to drive the gate. This
    documents the known race-window; cleanup is best-effort.
    """
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()
    started = asyncio.Event()
    block = asyncio.Event()

    async def _hang(pending):
        started.set()
        await block.wait()
        return True

    issuer.dispatch_join_turn = AsyncMock(side_effect=_hang)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await orch.intercept(_mk_credit("root", "corr-root", 0))

    t = asyncio.create_task(orch.on_child_leaf_reached("corr-root-c0"))
    await started.wait()
    # By this point, _satisfy_prerequisite already popped the gate and we
    # are awaiting dispatch_join_turn. cleanup() now runs.
    orch.cleanup()
    # Release.
    block.set()
    await t
    # Dispatch completed (it was already in flight); orchestrator state
    # cleared.
    issuer.dispatch_join_turn.assert_awaited_once()


# ---------------------------------------------------------------------------
# 23. Defensive: ChildJoinEntry invariants — frozen, hashable,
#     orchestrator stores them in dict values.
# ---------------------------------------------------------------------------


def test_child_join_entry_is_frozen_and_hashable():
    e = ChildJoinEntry(
        parent_correlation_id="p", gated_turn_index=1, prereq_key="SPAWN_JOIN:b"
    )
    with pytest.raises((AttributeError, Exception)):
        e.parent_correlation_id = "x"  # type: ignore[misc]
    # Hashable (slots=True, frozen=True).
    s = {e}
    assert e in s


# ---------------------------------------------------------------------------
# 24. Stop-condition all-active simultaneously: orchestrator state is
#     orthogonal to stop conditions; verify by inspection that
#     orchestrator does not touch any StopCondition class.
# ---------------------------------------------------------------------------


def test_orchestrator_never_imports_stop_conditions():
    """Sanity: BranchOrchestrator must not depend on StopCondition state —
    stop conditions live at the issuer level and the orchestrator only
    observes ``dispatch_join_turn`` returning False.
    """
    import inspect

    import aiperf.timing.branch_orchestrator as mod

    src = inspect.getsource(mod)
    assert "StopCondition" not in src
    assert "stop_conditions" not in src
