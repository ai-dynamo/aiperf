# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load-time validator for constructs the v1 BranchOrchestrator honors.

dag3 v1 honors FORK branches with empty per-turn prerequisites only. Every
other construct raises ``NotImplementedError`` with a message pointing at
the deferred feature, so misconfigurations surface before any credit is
issued.

The "v1" in the name means *current-minimum-supported features*. This
function is **relaxed**, not replaced, as the orchestrator gains
capabilities: when SPAWN_JOIN's runtime ships, the corresponding rejection
clause is removed; the function name and contract stay. There is no v2.
"""

from __future__ import annotations

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models import DatasetMetadata


def validate_for_orchestrator_v1(metadata: DatasetMetadata) -> None:
    """Raise NotImplementedError for any construct v1 cannot honor."""
    all_conversation_ids = {c.conversation_id for c in metadata.conversations}

    for conv in metadata.conversations:
        # Branch-mode + child-resolution checks.
        for branch in conv.branches:
            if branch.mode != ConversationBranchMode.FORK:
                raise NotImplementedError(
                    f"conversation '{conv.conversation_id}' branch '{branch.branch_id}': "
                    f"branch mode '{branch.mode}' not supported by v1 orchestrator "
                    f"(dag3 ships FORK-only; SPAWN re-enters via TurnPrerequisite "
                    f"once the orchestrator catches up)"
                )
            if not branch.child_conversation_ids:
                raise NotImplementedError(
                    f"conversation '{conv.conversation_id}' branch "
                    f"'{branch.branch_id}': declares no child_conversation_ids; "
                    f"empty branches are rejected"
                )
            for child_id in branch.child_conversation_ids:
                if child_id not in all_conversation_ids:
                    raise NotImplementedError(
                        f"conversation '{conv.conversation_id}' branch "
                        f"'{branch.branch_id}': child_conversation_id '{child_id}' "
                        f"does not reference an existing conversation in the dataset"
                    )

        declared_branch_ids = {b.branch_id for b in conv.branches}

        # Per-turn checks: branch_id uniqueness + resolution, prereq rejection.
        for idx, turn in enumerate(conv.turns):
            seen: set[str] = set()
            for b_id in turn.branch_ids:
                if b_id in seen:
                    raise NotImplementedError(
                        f"conversation '{conv.conversation_id}' turn {idx}: "
                        f"branch_id '{b_id}' declared multiple times on the same turn"
                    )
                seen.add(b_id)
                if b_id not in declared_branch_ids:
                    raise NotImplementedError(
                        f"conversation '{conv.conversation_id}' turn {idx}: "
                        f"references undeclared branch_id '{b_id}' "
                        f"(no matching ConversationBranchInfo on this conversation)"
                    )
            if turn.prerequisites:
                raise NotImplementedError(
                    f"conversation '{conv.conversation_id}' turn {idx}: "
                    f"prerequisites are not supported by v1 orchestrator "
                    f"(dag3 ships FORK-only; SPAWN_JOIN gating arrives with SPAWN)"
                )

    # Cross-conversation FORK single-parent invariant.
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
                f"branches ({joined}); FORK-mode children require a single parent"
            )
