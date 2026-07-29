# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from pydantic import Field, ValidationInfo, field_validator

from aiperf.common.enums import ConversationBranchMode, SubagentType
from aiperf.common.models.base_models import AIPerfBaseModel


class ConversationBranchInfo(AIPerfBaseModel):
    """Describes a DAG branch from a parent turn to one or more child conversations.

    One primitive unifies aiperf's native FORK semantics (child inherits
    parent turn_list + sticky-routes to parent worker) with SPAWN semantics
    (fresh context, sticky co-locate while the parent entry is live, no
    SPAWN refcount bump; optionally dispatched before the parent's first
    turn). The ``mode`` field discriminates the two; the ``dispatch_timing``
    field gates pre-session SPAWN.

    Disambiguation note: this "branch" is a DAG conversation branch (a
    parent turn fanning out to one or more child conversations). Not a
    git branch. The same DAG-branch concept is tracked at runtime by
    ``BranchOrchestrator`` and counted in ``BranchStats``.
    """

    branch_id: str = Field(
        description=(
            "Deterministic branch ID emitted by the dag_jsonl loader. "
            "Five shapes are produced:\n"
            "  - '<sid>:<turn>' for single-mode fork OR single-group spawn\n"
            "  - '<sid>:<turn>:fork' when fork+spawn coexist on the same turn (mixed)\n"
            "  - '<sid>:<turn>:spawn' for the first spawn group on a mixed/multi-group turn\n"
            "  - '<sid>:<turn>:spawn<N>' (N>=2) for additional spawn groups on the same turn\n"
            "  - '<sid>:pre' for the pre-session SPAWN marker (always turn 0)\n"
            "Note: '<sid>' may itself contain ':' characters; the inverse "
            "parser ``turn_idx_from_branch_id`` anchors on the trailing "
            "numeric (with optional mode suffix) or the literal ':pre' tail. "
            "Hand-authored DatasetMetadata may supply any opaque string here "
            "as long as it round-trips through that parser."
        ),
    )
    child_conversation_ids: list[str] = Field(
        description="Child conversation_ids dispatched when this branch fires.",
    )
    mode: ConversationBranchMode = Field(
        description="FORK = child inherits parent context; SPAWN = fresh context.",
    )
    dispatch_timing: Literal["pre", "post"] = Field(
        default="post",
        description="When the children dispatch relative to the parent's "
        "first turn. ``post`` (default) fires after the parent turn that "
        "declares the branch completes - both FORK and SPAWN children. "
        "``pre`` fires the children before the parent's first turn - "
        "reserved for SPAWN (background pre-session sub-agent dispatch); "
        "the field validator rejects ``pre`` when mode is FORK.",
    )
    is_background: bool = Field(
        default=False,
        description="Fire-and-forget branch: the parent's must-be-last-turn "
        "rule is waived and the parent continues running its remaining turns "
        "after this branch dispatches. For FORK branches the children still "
        "inherit context and sticky-route to the parent's worker (only the "
        "parent's termination is changed; emitted by the dag_jsonl loader for "
        "background forks). For SPAWN branches it marks a fire-and-forget "
        "sub-agent (fresh context, no rejoin). Default False. Renamed from "
        "``background`` to align with the agentx engine reader "
        "(``BranchOrchestrator``/``TrajectorySource`` read ``is_background``).",
    )
    subagent_type: SubagentType | None = Field(
        default=None,
        description="SPAWN-mode classification used for agentic-benchmark bucket "
        "metrics. Must be None when mode=FORK (FORK children inherit the "
        "parent's role).",
    )
    start_timestamp_ms: int | float | None = Field(
        default=None,
        ge=0,
        description=(
            "Absolute wall-clock timestamp, in milliseconds, when this branch's "
            "children entered the live trace timeline. Used by agentic replay "
            "to reconstruct in-flight subagents when sampling a mid-trace "
            "snapshot."
        ),
    )

    @field_validator("dispatch_timing")
    @classmethod
    def _validate_pre_requires_spawn(
        cls, v: Literal["pre", "post"], info: ValidationInfo
    ) -> Literal["pre", "post"]:
        if v == "pre" and info.data.get("mode") == ConversationBranchMode.FORK:
            raise ValueError(
                "dispatch_timing='pre' is reserved for SPAWN-mode branches "
                "(background pre-session sub-agent dispatch). FORK children "
                "inherit the parent's context and must dispatch after the "
                "parent turn - drop dispatch_timing or change mode to SPAWN."
            )
        return v

    # ``is_background=True`` is valid for FORK branches: the parent keeps
    # running while the child executes. ``subagent_type`` remains SPAWN-only,
    # which preserves the classification boundary without rejecting background
    # forks.
    @field_validator("subagent_type")
    @classmethod
    def _validate_subagent_type(
        cls, v: SubagentType | None, info: ValidationInfo
    ) -> SubagentType | None:
        if v is not None and info.data.get("mode") == ConversationBranchMode.FORK:
            raise ValueError(
                "subagent_type is a SPAWN-only classification (used for "
                "agentic-benchmark bucket metrics). FORK children inherit the "
                "parent's role; they have no subagent_type. Drop the field or "
                "change mode to SPAWN."
            )
        return v
