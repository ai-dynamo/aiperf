# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import Field, field_validator

from aiperf.common.enums import ConversationBranchMode, SubagentType
from aiperf.common.models.base_models import AIPerfBaseModel


class ConversationBranchInfo(AIPerfBaseModel):
    """Describes a DAG branch from a parent turn to one or more child conversations.

    A single primitive unifies aiperf's native FORK semantics (child inherits
    parent turn_list + sticky-routes to parent worker) with
    kv-cache-tester-v2-style SPAWN semantics (fresh context, free routing).
    The ``mode`` field discriminates the two; 95% of the orchestration code is
    mode-agnostic.
    """

    branch_id: str = Field(
        description="Deterministic branch ID, shape '<parent_session_id>:<parent_turn_index>'.",
    )
    child_conversation_ids: list[str] = Field(
        description="Child conversation_ids to dispatch when this branch triggers.",
    )
    mode: ConversationBranchMode = Field(
        description="FORK = child inherits parent context; SPAWN = fresh context.",
    )
    is_background: bool = Field(
        default=False,
        description="SPAWN-mode only: fire-and-forget. Must be False when mode=FORK.",
    )
    subagent_type: SubagentType | None = Field(
        default=None,
        description="SPAWN-mode classification. Must be None when mode=FORK.",
    )
    dispatch_timing: Literal["pre", "post"] = Field(
        default="post",
        description=(
            "When the branch's children dispatch relative to the parent's spawning turn. "
            "'post' (default) fires after the parent turn returns. "
            "'pre' fires before the parent's turn 0 is issued; "
            "restricted to background SPAWN branches on root conversations."
        ),
    )

    @field_validator("is_background")
    @classmethod
    def _validate_background(cls, v, info):
        if v and info.data.get("mode") == ConversationBranchMode.FORK:
            raise ValueError("is_background=True is invalid for FORK mode")
        return v

    @field_validator("subagent_type")
    @classmethod
    def _validate_subagent_type(cls, v, info):
        if v is not None and info.data.get("mode") == ConversationBranchMode.FORK:
            raise ValueError("subagent_type cannot be set for FORK mode")
        return v
