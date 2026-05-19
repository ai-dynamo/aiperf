# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load-time validator for constructs the v1 BranchOrchestrator honors.

Every unsupported construct raises ``NotImplementedError`` with a message
pointing at the deferred feature. Loaders call this from the end of
``load_dataset`` so misconfigurations surface before any credit is issued.
"""

from __future__ import annotations

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnPrerequisite,
)
from aiperf.common.models.branch import ConversationBranchInfo

_SUPPORTED_BRANCH_MODES = {
    ConversationBranchMode.FORK,
    ConversationBranchMode.SPAWN,
}


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


def _check_unique_branch_ids_per_turn(
    conv: ConversationMetadata, branch_ids_by_turn: dict[int, list[str]]
) -> None:
    """Declaring the same branch_id twice on a single parent turn is always
    an authoring bug — the orchestrator would spawn children under that
    branch twice and double-register the gate."""
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


def _check_pre_session_branch(
    conv: ConversationMetadata,
    branch: ConversationBranchInfo,
    branch_declaration_turn: dict[str, int],
) -> None:
    """The pre-session hook runs before any parent credit is issued, so it
    cannot support FORK (needs real parent session). The branch's field
    validator already rejects mode=FORK + dispatch_timing="pre" at
    construction time, but we re-check here for hand-authored DatasetMetadata
    that might bypass the loader. The declaring conversation must also be a
    root with the branch attached to turn 0."""
    if branch.mode == ConversationBranchMode.FORK:
        raise NotImplementedError(
            f"conversation '{conv.conversation_id}' branch "
            f"'{branch.branch_id}': pre-session dispatch requires "
            f"SPAWN mode (FORK requires real parent session)"
        )
    is_root = getattr(conv, "is_root", True)
    agent_depth = getattr(conv, "agent_depth", 0)
    if not is_root or agent_depth > 0:
        raise NotImplementedError(
            f"conversation '{conv.conversation_id}' branch "
            f"'{branch.branch_id}': pre-session dispatch requires a "
            f"root conversation (is_root=True, agent_depth=0), got "
            f"is_root={is_root} agent_depth={agent_depth}"
        )
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


def _check_branches(
    conv: ConversationMetadata,
    all_conversation_ids: set[str],
    branch_declaration_turn: dict[str, int],
) -> None:
    for branch in conv.branches:
        if branch.mode not in _SUPPORTED_BRANCH_MODES:
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
        if getattr(branch, "dispatch_timing", "post") == "pre":
            _check_pre_session_branch(conv, branch, branch_declaration_turn)


def _check_turn_prereq(
    conv: ConversationMetadata,
    idx: int,
    prereq: TurnPrerequisite,
    *,
    branches_by_id: dict[str, ConversationBranchInfo],
    branch_declaration_turn: dict[str, int],
) -> None:
    loc = f"conversation '{conv.conversation_id}' turn {idx}"
    _check_prereq_fields(prereq, loc)
    # SPAWN_JOIN must reference a branch on an earlier turn of the same
    # conversation.
    if prereq.branch_id is None or prereq.branch_id not in branches_by_id:
        raise NotImplementedError(
            f"{loc}: prerequisite branch_id '{prereq.branch_id}' does not "
            f"reference a prior branch of this conversation"
        )
    # v1 requires the referenced branch to be declared on a turn strictly
    # earlier than the consuming turn; same-turn or forward references cannot
    # be gated at runtime.
    decl_idx = branch_declaration_turn.get(prereq.branch_id)
    if decl_idx is None:
        raise NotImplementedError(
            f"{loc}: prerequisite branch_id '{prereq.branch_id}' is not declared "
            f"on any turn of conversation '{conv.conversation_id}'; check for a "
            f"typo or add the branch declaration to an earlier turn"
        )
    if decl_idx >= idx:
        raise NotImplementedError(
            f"{loc}: prerequisite branch_id '{prereq.branch_id}' is declared on "
            f"turn {decl_idx}, which is not strictly earlier than this turn ({idx}); "
            f"v1 requires the spawn turn to precede the join turn"
        )
    branch = branches_by_id[prereq.branch_id]
    # Pre-session SPAWN branches are fire-and-forget (no parent session
    # yet exists at dispatch time), so they cannot be gated by a SPAWN_JOIN.
    if getattr(branch, "dispatch_timing", "post") == "pre":
        raise NotImplementedError(
            f"{loc}: branch '{branch.branch_id}' is a pre-session "
            f"(fire-and-forget) SPAWN but is referenced by a "
            f"SPAWN_JOIN prerequisite"
        )


def _check_prereqs(
    conv: ConversationMetadata,
    branches_by_id: dict[str, ConversationBranchInfo],
    branch_declaration_turn: dict[str, int],
) -> None:
    for idx, turn in enumerate(conv.turns):
        loc = f"conversation '{conv.conversation_id}' turn {idx}"
        seen_prereq_branch_ids: set[str] = set()
        for prereq in turn.prerequisites:
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
            _check_turn_prereq(
                conv,
                idx,
                prereq,
                branches_by_id=branches_by_id,
                branch_declaration_turn=branch_declaration_turn,
            )


def _check_global_fork_single_parent(metadata: DatasetMetadata) -> None:
    """Defense-in-depth across conversations. The loader's
    _resolve_and_validate already enforces this for jsonl input, but
    hand-authored DatasetMetadata that bypasses the loader could still ship
    two FORK branches across different conversations claiming the same child.
    FORK semantics inherit a single parent context, so two FORK parents would
    produce ambiguous seed messages at the child."""
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


def _validate_conversation(
    conv: ConversationMetadata, all_conversation_ids: set[str]
) -> None:
    branch_ids_by_turn: dict[int, list[str]] = {}
    for idx, turn in enumerate(conv.turns):
        if turn.branch_ids:
            branch_ids_by_turn[idx] = list(turn.branch_ids)
    _check_unique_branch_ids_per_turn(conv, branch_ids_by_turn)

    branches_by_id = {b.branch_id: b for b in conv.branches}
    # Map each branch_id to the earliest turn that declares it, for
    # enforcing strictly-prior-turn spawn references below.
    branch_declaration_turn: dict[str, int] = {}
    for turn_idx_ in range(len(conv.turns)):
        for b_id in conv.turns[turn_idx_].branch_ids or []:
            branch_declaration_turn.setdefault(b_id, turn_idx_)

    _check_branches(conv, all_conversation_ids, branch_declaration_turn)
    _check_prereqs(conv, branches_by_id, branch_declaration_turn)


def validate_for_orchestrator_v1(metadata: DatasetMetadata) -> None:
    """Gate a DatasetMetadata against the v1 BranchOrchestrator's capability set.

    Call from the end of any loader's ``load_dataset`` so unsupported
    constructs surface before any credit is issued. Already wired into
    ``DagJsonlLoader.load_dataset``; hand-built ``DatasetMetadata`` (e.g.
    in tests or external integrations) must call this explicitly:

        >>> from aiperf.common.models import DatasetMetadata
        >>> from aiperf.common.validators import validate_for_orchestrator_v1
        >>> validate_for_orchestrator_v1(metadata)  # raises on first violation

    Rules enforced (each violation raises ``NotImplementedError`` with a
    ``"conversation '<id>' turn <N>: <reason>"`` location prefix):

    - prerequisite kinds other than ``SPAWN_JOIN`` (TIMER, EXTERNAL_EVENT, ...)
    - per-child / barrier / timer / event reserved fields on ``TurnPrerequisite``
    - branch modes other than ``FORK`` / ``SPAWN``
    - ``child_conversation_id`` not present in the dataset
    - duplicate ``branch_id`` on the same parent turn
    - ``SPAWN_JOIN`` whose referenced branch is on the same or later turn
    - pre-session SPAWN (``dispatch_timing='pre'``) on a non-root or non-turn-0
    - multiple FORK parents claiming the same child across the dataset

    Raises:
        NotImplementedError: First unsupported construct found (fail-fast).
        ValueError: Duplicate ``SPAWN_JOIN`` prereq for the same branch on one turn.
    """
    all_conversation_ids = {c.conversation_id for c in metadata.conversations}
    for conv in metadata.conversations:
        _validate_conversation(conv, all_conversation_ids)
    _check_global_fork_single_parent(metadata)
