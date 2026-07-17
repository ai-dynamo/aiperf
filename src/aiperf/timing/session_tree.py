# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-session-TREE liveness bookkeeping for DAG-lineage finality.

A DAG session is not a single root conversation -- it is a whole TREE: a
depth-0 root plus every child it spawns, recursively (FORK/SPAWN children and
their subchildren). ``SessionTreeRegistry`` tracks the liveness of each such
tree, keyed by the tree's ``root_correlation_id`` (the depth-0 root's
x_correlation_id), so the credit issuer can stamp two conservative facts on
every emitted ``Credit`` at issue time:

  - ``is_parent_final`` -- the credit's parent had already returned its terminal
    turn (v1: determinable only when the parent IS the root).
  - ``is_tree_final`` -- this credit is provably the last request the whole tree
    will ever send.

This registry is **finality bookkeeping only**. Unlike the agentic-replay
variant it was ported from, it does NOT own session-slot release: on main the
credit callback handler releases the root's session slot on the root's terminal
turn (``_release_slots_for_return``) exactly as before, and DAG children inherit
the root's slot. The registry never touches the concurrency manager and fires no
drain callback -- it only answers the two finality queries below from live tree
state.

Wiring (DAG datasets only; the registry is None on non-DAG paths, so
``_finality_for_issue`` returns the conservative ``(None, False)``):
  - ``CreditIssuer._open_session_tree`` -> :meth:`open_tree` when a root
    session's first credit is issued (turn 0 session-slot acquisition).
  - ``BranchOrchestrator`` -> :meth:`register_descendants` when a parent turn
    spawns children, :meth:`on_descendant_done` at every descendant-terminal
    point, and :meth:`on_root_terminal` from ``intercept`` after the root's
    final-turn return is processed (so final-turn spawns register first).
  - ``BranchOrchestrator.cleanup`` -> :meth:`release_all` at phase teardown to
    drop any still-open trees.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CreditPhase

_logger = AIPerfLogger(__name__)


@dataclass(slots=True)
class _TreeState:
    """Liveness of one session tree."""

    phase: CreditPhase
    """Phase the tree was opened under; teardown targets the same phase."""
    root_pending: bool
    """True until the root's terminal turn returns."""
    outstanding: int = 0
    """Descendants currently live or registered-pending (any depth)."""
    released: bool = False
    """Set once the tree has been retired; guards against double retire."""

    @property
    def drained(self) -> bool:
        """True when the root is done and no descendant work remains."""
        return not self.root_pending and self.outstanding <= 0


class SessionTreeRegistry:
    """Tracks per-session-tree liveness for DAG-lineage finality queries.

    See the module docstring for the invariant and wiring. Single-threaded
    asyncio: every method runs to completion between awaits, so the counter
    mutations are atomic without locking. Finality bookkeeping only -- this
    class never acquires or releases a concurrency slot.
    """

    def __init__(self) -> None:
        self._trees: dict[str, _TreeState] = {}
        # Descendants registered before their tree is opened. On main's flow a
        # root's tree is opened at turn-0 issue, before any child can spawn (a
        # spawn only happens on a parent turn RETURN), so this buffer is a
        # defensive belt that keeps the branch's state transitions identical.
        self._pending_descendants: dict[str, int] = {}
        # Roots whose tree already RETIRED (drained), mapped to the phase they
        # were opened under. A descendant's final-turn SPAWN registers its
        # grandchildren AFTER that descendant's own on_descendant_done drained
        # the tree (callback order: the child-completion decrement precedes the
        # return-intercept that spawns). Without this ledger those grandchildren
        # would buffer into a retired root that nothing drains, and is_tree_final
        # could never answer True for them. register_descendants consults it to
        # RESURRECT the tree root-terminal instead. Cleared at teardown.
        self._retired_roots: dict[str, CreditPhase] = {}
        # Peak simultaneously-open trees == peak tree concurrency; logged at
        # teardown so any overshoot is visible.
        self._peak_open: int = 0
        # A non-zero count means a descendant returned for a tree that was
        # already retired -- i.e. the tree drained while that work was still in
        # flight (premature drain). Surfaces span-overhang.
        self._late_events: int = 0

    def open_tree(
        self, root_corr: str, phase: CreditPhase, *, root_pending: bool
    ) -> None:
        """Record a newly admitted tree when its root's first credit is issued.

        Args:
            root_corr: the tree's ``root_correlation_id`` (depth-0 root's
                x_correlation_id).
            phase: phase the tree was opened under.
            root_pending: True when a root credit is in flight that will reach a
                terminal turn (always True on main -- open_tree fires only at a
                root session start).
        """
        existing = self._trees.get(root_corr)
        if existing is not None and not existing.released:
            # Duplicate open for a still-live tree: keep the original (its
            # outstanding count is authoritative). Should not happen with
            # correct wiring; log so a regression is visible.
            _logger.warning(
                lambda: f"open_tree for already-open tree root_corr={root_corr!r}; "
                "ignoring duplicate"
            )
            return
        state = _TreeState(phase=phase, root_pending=root_pending)
        state.outstanding += self._pending_descendants.pop(root_corr, 0)
        # A freshly-opened tree supersedes any stale retired record for the same
        # id (correlation ids are unique, so this is defensive).
        self._retired_roots.pop(root_corr, None)
        self._trees[root_corr] = state
        if len(self._trees) > self._peak_open:
            self._peak_open = len(self._trees)

    @property
    def peak_open(self) -> int:
        """Maximum simultaneously-open trees seen (== peak tree concurrency)."""
        return self._peak_open

    @property
    def late_events(self) -> int:
        """Count of returns for already-retired trees (premature-drain evidence)."""
        return self._late_events

    def has_tree(self, root_corr: str) -> bool:
        """True when this registry is tracking ``root_corr`` (engagement gate)."""
        return root_corr in self._trees

    def root_terminal(self, root_correlation_id: str) -> bool | None:
        """True when the tree's root has returned its terminal turn; None if unknown.

        Queried at credit-issue time while the tree is still live (a descendant
        being issued keeps it in ``_trees``). A fully drained tree has been
        retired and reads as unknown/None.
        """
        state = self._trees.get(root_correlation_id)
        if state is None:
            return None
        return not state.root_pending

    def is_last_tree_request(
        self,
        root_correlation_id: str,
        *,
        is_final_turn: bool,
        is_root_credit: bool,
        has_branches: bool,
    ) -> bool:
        """Conservative: True only when this credit is provably the tree's last request.

        ``has_branches`` means "this turn declares ANY branch (FORK or SPAWN)
        and will therefore spawn descendants on its return". It must NOT be the
        FORK-only ``has_forks`` flag: SPAWN children register with this registry
        only at return-intercept, AFTER the issuer stamped finality, so a
        spawning final turn would otherwise read as last while its children are
        pending (wrong-True, violating the conservative contract).
        """
        if not is_final_turn or has_branches:
            return False
        state = self._trees.get(root_correlation_id)
        if state is None:
            return False
        if is_root_credit:
            return state.outstanding <= 0
        return (not state.root_pending) and state.outstanding == 1

    def register_descendants(self, root_corr: str, n: int = 1) -> None:
        """Add ``n`` descendants (spawned under one root) to a tree.

        Called when a parent turn spawns children, keyed by the tree's root id.
        Three cases when the tree is not live:
          - RETIRED root (in ``_retired_roots``): resurrect it root-terminal.
            A descendant's final-turn SPAWN registers its grandchildren after
            that descendant's own ``on_descendant_done`` drained the tree; the
            root was terminal at retire (``drained`` requires ``not
            root_pending``), so recreate the tree with ``root_pending=False``
            and these descendants outstanding -- keeping ``is_tree_final``
            answerable and the count coherent (the grandchildren re-drain it
            when they finish).
          - Unopened root: buffer the count so a later ``open_tree`` folds it in
            (defensive -- on main the tree is always open first).
        """
        if n <= 0:
            return
        state = self._trees.get(root_corr)
        if state is not None:
            state.outstanding += n
            return
        retired_phase = self._retired_roots.pop(root_corr, None)
        if retired_phase is not None:
            self._trees[root_corr] = _TreeState(
                phase=retired_phase, root_pending=False, outstanding=n
            )
            if len(self._trees) > self._peak_open:
                self._peak_open = len(self._trees)
            return
        self._pending_descendants[root_corr] = (
            self._pending_descendants.get(root_corr, 0) + n
        )

    def on_descendant_done(self, root_corr: str) -> bool:
        """Account one descendant terminally completing (leaf / error / stopped /
        dispatch-rollback). Retires the tree iff it is now drained.

        Returns True if the tree was retired by this call.
        """
        state = self._trees.get(root_corr)
        if state is None:
            pending = self._pending_descendants.get(root_corr, 0)
            if pending > 0:
                self._pending_descendants[root_corr] = pending - 1
            else:
                self._late_events += 1
            return False
        if state.outstanding > 0:
            state.outstanding -= 1
        return self._retire_if_drained(root_corr, state)

    def on_root_terminal(self, root_corr: str) -> bool:
        """Account the root's terminal turn returning (or root cancel/truncate).

        Clears ``root_pending``; retires the tree iff it is now drained. Must be
        called AFTER the returning root credit's intercept has run, so any
        children spawned on the final turn are already registered.

        Returns True if the tree was retired by this call. False both when the
        tree is held (descendants remain) AND when ``root_corr`` is untracked.
        """
        state = self._trees.get(root_corr)
        if state is None:
            return False
        state.root_pending = False
        return self._retire_if_drained(root_corr, state)

    def _retire_if_drained(self, root_corr: str, state: _TreeState) -> bool:
        """Pop a drained tree from tracking. Finality-only: releases no slot."""
        if state.released or not state.drained:
            return False
        state.released = True
        self._trees.pop(root_corr, None)
        # Remember the retired root (with its phase) so a late final-turn SPAWN
        # can RESURRECT it instead of silently buffering. See register_descendants.
        self._retired_roots[root_corr] = state.phase
        return True

    def release_all(self, phase: CreditPhase | None = None) -> int:
        """Retire every still-open tree at teardown (optionally one phase).

        Finality-only: drops the tracking entries; releases no concurrency slot
        (the callback handler owns slot release on main). The registry is
        created per-phase, so ``phase=None`` (retire all) is the common
        teardown call. Returns the number of trees retired.
        """
        to_release = [
            root_corr
            for root_corr, state in self._trees.items()
            if (phase is None or state.phase == phase) and not state.released
        ]
        for root_corr in to_release:
            state = self._trees.pop(root_corr, None)
            if state is None or state.released:
                continue
            state.released = True
        # Drop the transient buffers: with the trees above retired, any
        # pending/retired descendant accounting refers to no live tree. On a
        # full teardown (``phase is None`` -- the common per-phase call, since
        # the registry is created per-phase) clear both; a phase-scoped call
        # clears only that phase's retired roots (``_pending_descendants``
        # carries no phase and is left untouched then).
        if phase is None:
            self._pending_descendants.clear()
            self._retired_roots.clear()
        else:
            self._retired_roots = {
                r: p for r, p in self._retired_roots.items() if p != phase
            }
        return len(to_release)

    def open_count(self, phase: CreditPhase | None = None) -> int:
        """Number of trees currently tracked (optionally for one phase)."""
        if phase is None:
            return len(self._trees)
        return sum(1 for state in self._trees.values() if state.phase == phase)
