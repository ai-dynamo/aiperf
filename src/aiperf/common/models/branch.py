# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import msgspec

from aiperf.common.enums import ConversationBranchMode


class ConversationBranchInfo(
    msgspec.Struct,
    kw_only=True,
    frozen=True,
    forbid_unknown_fields=True,
    omit_defaults=True,
):
    """Describes a DAG branch from a parent turn to one or more child conversations.

    One primitive unifies aiperf's native FORK semantics (child inherits
    parent turn_list by seeding from the parent's session; same-worker
    pinning is inert in v1) with pre-session SPAWN semantics (fresh context,
    free routing, optionally dispatched before the parent's first turn). The
    ``mode`` field discriminates the two; the ``dispatch_timing`` field gates
    pre-session SPAWN.

    Disambiguation note: this "branch" is a DAG conversation branch (a
    parent turn fanning out to one or more child conversations). Not a
    git branch. The same DAG-branch concept is tracked at runtime by
    ``BranchOrchestrator`` and counted in ``BranchStats``.
    """

    branch_id: str
    """Deterministic branch ID emitted by the dag_jsonl loader. Five shapes
    are produced: ``<sid>:<turn>`` for single-mode fork or single-group
    spawn; ``<sid>:<turn>:fork`` when fork+spawn coexist; ``<sid>:<turn>:spawn``
    for the first spawn group on a mixed/multi-group turn;
    ``<sid>:<turn>:spawn<N>`` for additional spawn groups on the same turn;
    ``<sid>:pre`` for the pre-session SPAWN marker (always turn 0).
    Hand-authored DatasetMetadata may supply any opaque string as long as
    it round-trips through ``turn_idx_from_branch_id``."""

    child_conversation_ids: list[str]
    """Child conversation_ids dispatched when this branch fires."""

    mode: ConversationBranchMode
    """FORK = child inherits parent context; SPAWN = fresh context."""

    dispatch_timing: Literal["pre", "post"] = "post"
    """When the children dispatch relative to the parent's first turn.
    ``post`` (default) fires after the parent turn that declares the
    branch completes - both FORK and SPAWN children. ``pre`` fires the
    children before the parent's first turn - reserved for SPAWN. The
    ``__post_init__`` validator rejects ``pre`` when mode is FORK."""

    background: bool = False
    """If True (FORK only), the parent's must-be-last-turn rule is waived
    for this branch and the parent continues running its remaining turns
    after the fork dispatches. Ignored for SPAWN-mode branches."""

    def __post_init__(self) -> None:
        if self.dispatch_timing == "pre" and self.mode == ConversationBranchMode.FORK:
            raise ValueError(
                "dispatch_timing='pre' is reserved for SPAWN-mode branches "
                "(background pre-session sub-agent dispatch). FORK children "
                "inherit the parent's context and must dispatch after the "
                "parent turn - drop dispatch_timing or change mode to SPAWN."
            )
