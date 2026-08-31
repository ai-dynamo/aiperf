# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-session-TREE session-slot accounting for agentic replay.

A session in agentic replay is not a single root conversation -- it is a whole
TREE: a depth-0 root plus every subagent it spawns, recursively (children,
subchildren, background ``::fa:`` flat-async streams, ``::aux:`` sidecars,
SPAWN/FORK children, gated joins). ``--concurrency N`` means N such trees live
at any instant.

The historical model held a session slot per ROOT CREDIT: acquired on the root's
first turn, released on the root's final turn. Background subagents routinely
outlive their root, so the slot freed while the tree was still running and a
fresh root filled it -- pushing live trees above N. The symmetric undershoot
(rootless / gated lanes that held no slot) was patched separately with a "lane
credit".

``SessionTreeRegistry`` replaces both with one rule, keyed by the tree's
``root_correlation_id``:

    Exactly one session slot per live tree, released exactly once, when the tree
    DRAINS: the root has sent its terminal turn (``root_pending`` cleared) AND
    every descendant has terminally completed (``outstanding == 0``).

The physical slot is still ACQUIRED by the credit issuer / concurrency manager
(blocking semantics unchanged); the registry owns only the RELEASE decision and,
on release, fires a drain callback so the strategy can recycle the freed lane.

Wiring (agentic replay only; ``None`` everywhere else so non-agentic paths keep
their exact legacy behavior):
  - ``CreditIssuer`` -> :meth:`open_tree` after a root/lane session slot is
    acquired.
  - ``BranchOrchestrator`` -> :meth:`register_descendants` at every spawn/seed
    and :meth:`on_descendant_done` at every descendant-terminal point.
  - ``CreditCallbackHandler`` -> :meth:`on_root_terminal` after the root's final
    turn return is intercepted (so final-turn spawns are registered first), and
    :meth:`release_all` at phase teardown.
  - ``AgenticReplayStrategy`` -> :meth:`set_drain_callback` to recycle a drained
    lane.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.phase import PhaseRuntimeKey

_logger = AIPerfLogger(__name__)


@dataclass(slots=True)
class _TreeState:
    """Liveness of one session tree."""

    phase: PhaseRuntimeKey
    """Runtime key the tree's slot was acquired under; release targets the same key."""
    root_pending: bool
    """True until the root's terminal turn returns. Always False for a rootless
    lane (no root credit will ever run -- it drains on descendants alone)."""
    outstanding: int = 0
    """Descendants currently live or registered-pending (any depth)."""
    released: bool = False
    """Set once the slot has been released; guards against double release."""

    @property
    def drained(self) -> bool:
        """True when the root is done and no descendant work remains."""
        return not self.root_pending and self.outstanding <= 0


class SessionTreeRegistry:
    """Owns session-slot release per session tree (keyed by root_correlation_id).

    See the module docstring for the invariant and wiring. Single-threaded
    asyncio: every method runs to completion between awaits, so the counter
    mutations are atomic without locking.
    """

    def __init__(self, concurrency_manager) -> None:
        """Args:
        concurrency_manager: provides ``release_session_slot(phase)`` -- the
            same semaphore the issuer acquires from.
        """
        self._concurrency_manager = concurrency_manager
        self._trees: dict[str, _TreeState] = {}
        self._on_drain: Callable[[str, PhaseRuntimeKey], None] | None = None
        # Descendants registered before their tree is opened (the agentic-replay
        # snapshot path seeds a lane's pre-t* subagents BEFORE acquiring the
        # lane/root slot). Buffered here and folded into ``outstanding`` at
        # ``open_tree`` so the tree opens already aware of its in-flight
        # subagents -- otherwise it would open at outstanding=0 and drain on the
        # FIRST child completion while siblings still run (premature drain).
        self._pending_descendants: dict[str, int] = {}
        # Peak simultaneously-open trees == peak session-slot occupancy. The
        # whole point of per-tree accounting is that this never exceeds the
        # configured concurrency; logged at teardown so any overshoot is visible.
        self._peak_open: int = 0
        # A non-zero count means a credit returned for a tree whose slot was
        # already released -- i.e. the tree drained while that work was still in
        # flight (premature drain). Surfaces the span-overhang that inflates the
        # swim-lane lane count above the true (capped) slot occupancy.
        self._late_events: int = 0

    def set_drain_callback(
        self, callback: Callable[[str, PhaseRuntimeKey], None] | None
    ) -> None:
        """Register the callback fired (with ``root_corr, phase``) when a tree
        drains and its slot is released during normal operation.

        Used by the strategy to recycle the freed lane into a fresh root. NOT
        fired by :meth:`release_all` (teardown must not start new sessions).
        """
        self._on_drain = callback

    def open_tree(
        self, root_corr: str, phase: PhaseRuntimeKey, *, root_pending: bool
    ) -> None:
        """Record a newly admitted tree after its session slot was acquired.

        Args:
            root_corr: the tree's ``root_correlation_id`` (depth-0 root's
                x_correlation_id, or a rootless lane's synthetic root id).
            phase: runtime key the slot was acquired under (phase index when
                set, otherwise the CreditPhase enum).
            root_pending: True when a root credit is in flight that will reach a
                terminal turn; False for a rootless lane that holds only a bare
                lane credit and drains on descendant completion alone.
        """
        existing = self._trees.get(root_corr)
        if existing is not None and not existing.released:
            # Duplicate open for a still-live tree: keep the original (its
            # outstanding count is authoritative). Should not happen with
            # correct wiring; log so a regression is visible.
            _logger.warning(
                lambda: (
                    f"open_tree for already-open tree root_corr={root_corr!r}; "
                    "ignoring duplicate"
                )
            )
            return
        state = _TreeState(phase=phase, root_pending=root_pending)
        # Fold in any descendants registered before this tree was opened (the
        # snapshot path seeds subagents before the lane/root slot is acquired).
        state.outstanding += self._pending_descendants.pop(root_corr, 0)
        self._trees[root_corr] = state
        if len(self._trees) > self._peak_open:
            self._peak_open = len(self._trees)

    @property
    def peak_open(self) -> int:
        """Maximum simultaneously-open trees seen (== peak session-slot occupancy)."""
        return self._peak_open

    @property
    def late_events(self) -> int:
        """Count of returns for already-released trees (premature-drain evidence)."""
        return self._late_events

    def has_tree(self, root_corr: str) -> bool:
        """True when this registry is tracking ``root_corr`` (engagement gate)."""
        return root_corr in self._trees

    def register_descendants(self, root_corr: str, n: int = 1) -> None:
        """Add ``n`` descendants (spawned or snapshot-seeded) to a tree.

        Called BEFORE the spawning turn's completion is accounted (under the
        orchestrator's per-parent lock) and at snapshot-seed time, so the
        outstanding count can never transiently hit zero with a spawn pending.
        Tolerant of unknown trees (e.g. dag_jsonl pre-session spawns whose tree
        is not yet open): a logged no-op.
        """
        if n <= 0:
            return
        state = self._trees.get(root_corr)
        if state is None:
            # Tree not opened yet (snapshot seeds descendants before the lane
            # slot is acquired). Buffer so open_tree folds them in -- NOT a
            # no-op, or the tree would open unaware of these subagents and drain
            # the moment the first one completes.
            self._pending_descendants[root_corr] = (
                self._pending_descendants.get(root_corr, 0) + n
            )
            return
        state.outstanding += n

    def on_descendant_done(self, root_corr: str) -> bool:
        """Account one descendant terminally completing (leaf / error / stopped /
        dispatch-rollback). Releases the slot iff the tree is now drained.

        Returns True if the slot was released by this call.
        """
        state = self._trees.get(root_corr)
        if state is None:
            # Tree not open yet: the descendant was buffered by
            # register_descendants (snapshot seeds subagents before the lane
            # slot is acquired). A subagent at offset 0 can complete before the
            # lane's root -- scheduled later -- opens the tree. Decrement the
            # pending buffer so open_tree folds in the remaining count; otherwise
            # it folds the stale full count and the tree never drains (a
            # descendant that already completed cannot complete again), leaking
            # the session slot until phase teardown. A completion with no pending
            # buffer is a genuine late event (tree already released / untracked).
            pending = self._pending_descendants.get(root_corr, 0)
            if pending > 1:
                self._pending_descendants[root_corr] = pending - 1
            elif pending == 1:
                # Last buffered descendant done before the tree opened: drop the
                # entry so it can't linger for a tree that never acquires a slot.
                del self._pending_descendants[root_corr]
            else:
                self._late_events += 1
            return False
        if state.outstanding > 0:
            state.outstanding -= 1
        return self._maybe_release(root_corr, state)

    def on_root_terminal(self, root_corr: str) -> bool:
        """Account the root's terminal turn returning (or root cancel/truncate).

        Clears ``root_pending``; releases the slot iff the tree is now drained.
        Must be called AFTER the returning root credit's intercept has run, so
        any children spawned on the final turn are already registered.

        Returns True if the slot was released by this call. False both when the
        tree is held (descendants remain) AND when ``root_corr`` is untracked --
        callers gate the legacy release path on :meth:`has_tree`, not on this
        return value.
        """
        state = self._trees.get(root_corr)
        if state is None:
            return False
        state.root_pending = False
        return self._maybe_release(root_corr, state)

    def _maybe_release(self, root_corr: str, state: _TreeState) -> bool:
        if state.released or not state.drained:
            return False
        state.released = True
        self._trees.pop(root_corr, None)
        self._concurrency_manager.release_session_slot(state.phase)
        if self._on_drain is not None:
            self._on_drain(root_corr, state.phase)
        return True

    def release_all(self, phase: PhaseRuntimeKey) -> int:
        """Release every still-open tree's slot for ``phase`` at teardown.

        Replaces the callback handler's ``in_flight_sessions`` release loop when
        the registry is engaged. Does NOT fire the drain callback (teardown must
        not recycle). Returns the number of slots released.
        """
        to_release = [
            root_corr
            for root_corr, state in self._trees.items()
            if state.phase == phase and not state.released
        ]
        for root_corr in to_release:
            state = self._trees.pop(root_corr, None)
            if state is None or state.released:
                continue
            state.released = True
            self._concurrency_manager.release_session_slot(phase)
        return len(to_release)

    def open_count(self, phase: PhaseRuntimeKey | None = None) -> int:
        """Number of trees currently holding a slot (optionally for one phase)."""
        if phase is None:
            return len(self._trees)
        return sum(1 for state in self._trees.values() if state.phase == phase)
