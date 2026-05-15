# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-function helpers for ``BranchOrchestrator``.

Split out of ``branch_orchestrator.py`` for ergonomics. Everything here is
side-effect free or operates only on its arguments — no orchestrator state
is mutated outside the explicit returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.enums import PrerequisiteKind
from aiperf.timing._branch_orchestrator_state import ChildJoinEntry

if TYPE_CHECKING:
    from aiperf.common.models.dataset_models import ConversationMetadata


def build_prereq_index(
    conversations: list[ConversationMetadata],
) -> tuple[
    dict[tuple[str, int], list[tuple[str, int, str]]],
    dict[tuple[str, int], set[str]],
]:
    """Build the (conv_id, spawning_turn_idx) -> [(branch_id, gated_idx,
    prereq_key)] index AND the (conv_id, gated_idx) -> {prereq_keys} fan-in
    seed map from the dataset's conversations.

    Returns ``(prereq_index, gated_turn_prereq_keys)``.
    """
    prereq_index: dict[tuple[str, int], list[tuple[str, int, str]]] = {}
    gated_turn_prereq_keys: dict[tuple[str, int], set[str]] = {}
    for conv in conversations:
        # Resolve each SPAWN_JOIN prereq to the spawning turn that
        # declared the referenced branch_id.
        branch_declaration_turn: dict[str, int] = {}
        for turn_idx, turn in enumerate(conv.turns):
            for b_id in turn.branch_ids or []:
                branch_declaration_turn.setdefault(b_id, turn_idx)
        for gated_idx, turn in enumerate(conv.turns):
            for prereq in turn.prerequisites:
                if prereq.kind != PrerequisiteKind.SPAWN_JOIN:
                    continue
                if prereq.branch_id is None:
                    continue
                spawning_idx = branch_declaration_turn.get(prereq.branch_id)
                if spawning_idx is None:
                    continue
                prereq_key = f"SPAWN_JOIN:{prereq.branch_id}"
                key = (conv.conversation_id, spawning_idx)
                bucket = prereq_index.setdefault(key, [])
                bucket.append((prereq.branch_id, gated_idx, prereq_key))
                gated_turn_prereq_keys.setdefault(
                    (conv.conversation_id, gated_idx), set()
                ).add(prereq_key)
    return prereq_index, gated_turn_prereq_keys


def gate_for_branch_map(
    prereq_entries: list[tuple[str, int, str]],
) -> dict[str, list[tuple[int, str]]]:
    """Group prereq-index entries by branch_id.

    Phase 3 multi-consumer: a branch may appear under multiple gate entries
    — each ``(gated_idx, prereq_key)`` forms its own independent gate.
    """
    out: dict[str, list[tuple[int, str]]] = {}
    for branch_id, gated_idx, prereq_key in prereq_entries:
        out.setdefault(branch_id, []).append((gated_idx, prereq_key))
    return out


def any_child_tracked_for_parent(
    child_to_join: dict[str, list[ChildJoinEntry]], parent_corr: str
) -> bool:
    """Return True if any child in ``child_to_join`` belongs to ``parent_corr``.

    Module-level helper (rather than method) because it is called from inside
    the orchestrator's spawn-and-register routine to decide whether all
    children rolled back and no per-parent state should remain reserved.
    """
    return any(
        any(e.parent_correlation_id == parent_corr for e in ents)
        for ents in child_to_join.values()
    )
