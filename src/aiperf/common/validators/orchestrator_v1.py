# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load-time validator for constructs the v1 BranchOrchestrator honors.

Every unsupported construct raises ``NotImplementedError`` with a message
pointing at the deferred feature. Loaders call this from the end of
``load_dataset`` so misconfigurations surface before any credit is issued.
"""

from __future__ import annotations

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import DatasetMetadata, TurnPrerequisite


def _check_prereq_fields(prereq: TurnPrerequisite, loc: str) -> None:
    if prereq.kind != PrerequisiteKind.SPAWN_JOIN:
        raise NotImplementedError(
            f"{loc}: prerequisite kind '{prereq.kind}' not supported by v1 orchestrator; "
            "only SPAWN_JOIN is implemented"
        )
    if prereq.child_conversation_ids is not None:
        raise NotImplementedError(
            f"{loc}: per-child prerequisite subsets not supported by v1 orchestrator; "
            "remove child_conversation_ids from the TurnPrerequisite"
        )
    if prereq.barrier_id is not None:
        raise NotImplementedError(
            f"{loc}: barrier-based prerequisites (runtime-diamond joins) not supported by v1 orchestrator"
        )
    if prereq.timer_seconds is not None:
        raise NotImplementedError(
            f"{loc}: timer-based prerequisites not supported by v1 orchestrator"
        )
    if prereq.event_name is not None:
        raise NotImplementedError(
            f"{loc}: event-based prerequisites not supported by v1 orchestrator"
        )


def validate_for_orchestrator_v1(metadata: DatasetMetadata) -> None:
    """Raise NotImplementedError for any construct v1 cannot honor.

    Centralized so every loader emits the same error shapes.
    """
    supported_modes = {ConversationBranchMode.FORK, ConversationBranchMode.SPAWN}
    all_conversation_ids = {c.conversation_id for c in metadata.conversations}

    for conv in metadata.conversations:
        branch_ids_by_turn: dict[int, list[str]] = {}
        for idx, turn in enumerate(conv.turns):
            if turn.branch_ids:
                branch_ids_by_turn[idx] = list(turn.branch_ids)

        # Duplicate-branch-id-per-turn check (Phase 2 authoring guardrail):
        # declaring the same branch_id twice on a single parent turn is
        # always an authoring bug — the orchestrator would spawn children
        # under that branch twice and double-register the gate.
        for idx, branch_ids in branch_ids_by_turn.items():
            seen: set[str] = set()
            for b_id in branch_ids:
                if b_id in seen:
                    raise NotImplementedError(
                        f"conversation '{conv.conversation_id}' turn {idx}: "
                        f"branch_id '{b_id}' declared multiple times on the "
                        f"same turn; each branch_id must be unique per turn"
                    )
                seen.add(b_id)

        branches_by_id = {b.branch_id: b for b in conv.branches}

        # Map each branch_id to the earliest turn that declares it, for
        # enforcing strictly-prior-turn spawn references below.
        branch_declaration_turn: dict[str, int] = {}
        for turn_idx_ in range(len(conv.turns)):
            for b_id in conv.turns[turn_idx_].branch_ids or []:
                branch_declaration_turn.setdefault(b_id, turn_idx_)

        for branch in conv.branches:
            if branch.mode not in supported_modes:
                raise NotImplementedError(
                    f"conversation '{conv.conversation_id}' branch '{branch.branch_id}': "
                    f"branch mode '{branch.mode}' not supported by v1 orchestrator"
                )
            # Every child_conversation_ids entry must resolve to a real
            # conversation; otherwise the orchestrator cannot start the child
            # session at runtime.
            for child_id in branch.child_conversation_ids:
                if child_id not in all_conversation_ids:
                    raise NotImplementedError(
                        f"conversation '{conv.conversation_id}' branch "
                        f"'{branch.branch_id}': child_conversation_id '{child_id}' "
                        f"does not reference an existing conversation in the dataset"
                    )

            # Phase 2b: dispatch_timing="pre" restrictions. The pre-session
            # hook runs before any parent credit has been issued, so it
            # cannot support FORK (needs real parent session) or blocking
            # branches (cannot gate against a non-existent parent).
            # The declaring conversation must also be a root with the
            # branch attached to turn 0.
            if getattr(branch, "dispatch_timing", "post") == "pre":
                if branch.mode == ConversationBranchMode.FORK:
                    raise NotImplementedError(
                        f"conversation '{conv.conversation_id}' branch "
                        f"'{branch.branch_id}': pre-session dispatch requires "
                        f"SPAWN mode (FORK requires real parent session)"
                    )
                if not branch.is_background:
                    raise NotImplementedError(
                        f"conversation '{conv.conversation_id}' branch "
                        f"'{branch.branch_id}': pre-session dispatch requires "
                        f"is_background=True (cannot gate against non-existent parent)"
                    )
                if getattr(conv, "agent_depth", 0) > 0:
                    raise NotImplementedError(
                        f"conversation '{conv.conversation_id}' branch "
                        f"'{branch.branch_id}': pre-session dispatch requires a "
                        f"root conversation (agent_depth=0), got "
                        f"agent_depth={getattr(conv, 'agent_depth', 0)}"
                    )
                # Locate the turn that declared this branch. It must be turn 0.
                decl_idx = branch_declaration_turn.get(branch.branch_id)
                if decl_idx is None:
                    raise NotImplementedError(
                        f"conversation '{conv.conversation_id}' branch "
                        f"'{branch.branch_id}': pre-session dispatch branch is "
                        f"not attached to any turn's branch_ids"
                    )
                if decl_idx != 0:
                    raise NotImplementedError(
                        f"conversation '{conv.conversation_id}' branch "
                        f"'{branch.branch_id}': pre-session dispatch must be "
                        f"declared on turn 0, got turn {decl_idx}"
                    )

        # Per-turn prerequisite checks.
        for idx, turn in enumerate(conv.turns):
            loc = f"conversation '{conv.conversation_id}' turn {idx}"
            seen_prereq_branch_ids: set[str] = set()
            for prereq in turn.prerequisites:
                _check_prereq_fields(prereq, loc)
                # Duplicate-prereq check: two TurnPrerequisite entries on the
                # same gated turn referencing the same branch_id is always an
                # authoring bug — the orchestrator's prereq index would
                # otherwise carry duplicate (branch_id, gated_turn_idx) tuples.
                if (
                    prereq.branch_id is not None
                    and prereq.branch_id in seen_prereq_branch_ids
                ):
                    raise ValueError(
                        f"{loc}: duplicate SPAWN_JOIN prerequisite for "
                        f"branch_id '{prereq.branch_id}' on the same gated "
                        f"turn; each branch_id may appear at most once in a "
                        f"turn's prerequisites"
                    )
                if prereq.branch_id is not None:
                    seen_prereq_branch_ids.add(prereq.branch_id)
                # SPAWN_JOIN must reference a branch on an earlier turn of the same conversation.
                if prereq.branch_id is None or prereq.branch_id not in branches_by_id:
                    raise NotImplementedError(
                        f"{loc}: prerequisite branch_id '{prereq.branch_id}' does not "
                        f"reference a prior branch of this conversation"
                    )
                # v1 requires the referenced branch to be declared on a turn
                # strictly earlier than the consuming turn; same-turn or
                # forward references cannot be gated at runtime.
                decl_idx = branch_declaration_turn.get(prereq.branch_id)
                if decl_idx is None or decl_idx >= idx:
                    raise NotImplementedError(
                        f"{loc}: prerequisite branch_id '{prereq.branch_id}' "
                        f"references a branch declared on turn {decl_idx} which "
                        f"is not earlier than this turn; v1 requires strictly-"
                        f"prior spawn turns"
                    )
                branch = branches_by_id[prereq.branch_id]
                if branch.is_background:
                    raise NotImplementedError(
                        f"{loc}: branch '{branch.branch_id}' is background but is "
                        f"referenced by a SPAWN_JOIN prerequisite"
                    )

        # Phase 3: multi-source gates (multiple SPAWN_JOIN prereqs on the
        # same turn referencing different branches) are now supported via
        # ``PendingBranchJoin.outstanding: dict[prereq_key, PrereqState]``.
        # Phase 3: multi-consumer branches (one branch_id referenced by
        # prereqs on multiple gated turns) are now supported; each
        # (gated_turn_idx) installs its own pending join keyed independently.

    # Global FORK single-parent invariant (defense-in-depth). The loader's
    # _resolve_and_validate already enforces this for jsonl input, but
    # hand-authored DatasetMetadata that bypasses the loader could still
    # ship two FORK branches across different conversations claiming the
    # same child. FORK semantics inherit a single parent context, so two
    # FORK parents would produce ambiguous seed messages at the child.
    fork_claims: dict[str, list[tuple[str, str]]] = {}
    for conv in metadata.conversations:
        for branch in conv.branches:
            if branch.mode != ConversationBranchMode.FORK:
                continue
            for child_id in branch.child_conversation_ids:
                fork_claims.setdefault(child_id, []).append(
                    (conv.conversation_id, branch.branch_id)
                )
    for child_id, claimants in fork_claims.items():
        if len(claimants) > 1:
            joined = ", ".join(f"conversation '{c}' branch '{b}'" for c, b in claimants)
            raise NotImplementedError(
                f"child conversation '{child_id}' is claimed by multiple FORK "
                f"branches ({joined}); FORK-mode children require a single "
                f"parent across the entire dataset"
            )
