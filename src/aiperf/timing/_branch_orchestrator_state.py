# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State dataclasses for ``BranchOrchestrator``.

Split out of ``branch_orchestrator.py`` to keep the orchestrator file
within the project's per-file ergonomics cap. ``PrereqState``,
``PendingBranchJoin``, and ``ChildJoinEntry`` are pure data — the
orchestrator owns all mutation logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from aiperf.common.enums import ConversationBranchMode


@dataclass
class PrereqState:
    """Runtime gate state for one ``TurnPrerequisite`` (one entry in ``PendingBranchJoin.outstanding``).

    Distinct from ``aiperf.common.models.prerequisites.TurnPrerequisite``,
    which is the *static* author-supplied prereq descriptor. ``PrereqState``
    tracks ``expected``/``completed``/``registered`` as children dispatch
    and join at runtime.

    Tracks the number of expected child completions (``expected``) and the
    set of child correlation ids that have already reported (``completed``).
    The set form gives idempotent double-delivery protection; the counter
    form lets multiple spawn points contribute to the same ``prereq_key``
    (fan-in) without requiring the orchestrator to know every child
    correlation id at registration time.

    ``registered`` is False until the spawning turn actually fires and
    ``expected`` has been incremented for at least one child. Fan-in
    requires the gate to be seeded with every declared prereq_key at
    pending-join-creation time so a prereq that fires-and-completes before
    the sibling prereq registers doesn't prematurely satisfy the gate.
    """

    expected: int = 0
    completed: set[str] = field(default_factory=set)
    registered: bool = False

    @property
    def is_done(self) -> bool:
        """True once the prereq has been registered and every expected
        completion has landed. Unregistered prereqs are never done — even
        with expected==0 — because some future spawning turn will increment
        ``expected``.
        """
        return self.registered and len(self.completed) >= self.expected


@dataclass
class PendingBranchJoin:
    """Join state for a parent session awaiting outstanding children.

    Holds everything the credit issuer needs to build the parent's gated
    TurnToSend without re-entering the conversation source, so the orchestrator
    stays the single source of truth for join bookkeeping.

    Phase 3 uses ``outstanding: dict[prereq_key, PrereqState]`` where each
    ``PrereqState`` carries an ``expected`` counter and a ``completed`` set.
    A single gated turn may have multiple prereq keys (fan-in); all must be
    done for ``is_satisfied`` to be True.
    """

    parent_x_correlation_id: str
    parent_conversation_id: str
    parent_num_turns: int
    parent_agent_depth: int = 0
    parent_parent_correlation_id: str | None = None
    gated_turn_index: int | None = None
    outstanding: dict[str, PrereqState] = field(default_factory=dict)
    parent_branch_mode: ConversationBranchMode = ConversationBranchMode.FORK
    parent_has_forks_on_gated_turn: bool = False
    is_blocked: bool = False
    created_at_ns: int = field(default_factory=time.monotonic_ns)

    @property
    def is_satisfied(self) -> bool:
        """True when every prereq's expected completions have all arrived."""
        return all(s.is_done for s in self.outstanding.values())

    @property
    def total_outstanding(self) -> int:
        """Total outstanding children across all prereqs (for diagnostics)."""
        return sum(
            max(0, s.expected - len(s.completed)) for s in self.outstanding.values()
        )


@dataclass(slots=True, frozen=True)
class ChildJoinEntry:
    """Tracks which parent pending-join a blocking child belongs to.

    ``prereq_key`` is ``None`` for background children (no gate); they still
    appear in ``_child_to_join`` so ``has_pending_branch_work`` and cleanup
    see them, but satisfying the entry skips gate bookkeeping.
    """

    parent_correlation_id: str
    """x_correlation_id of the parent session this child contributes to."""

    gated_turn_index: int | None
    """Parent turn index whose dispatch this child gates; ``None`` for background (ungated) children."""

    prereq_key: str | None
    """Pending-join's prereq key (``"SPAWN_JOIN:<branch_id>"``); ``None`` iff the child is background. Always None or non-None in lockstep with ``gated_turn_index``."""
