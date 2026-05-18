# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import ConfigDict, Field

from aiperf.common.enums import PrerequisiteKind
from aiperf.common.models.base_models import AIPerfBaseModel


class TurnPrerequisite(AIPerfBaseModel):
    """A condition that must be satisfied before the turn it is attached to dispatches.

    Lives on the gated (consuming) turn. The v1 orchestrator honors only the
    ``SPAWN_JOIN`` kind; all other kinds and the per-child/barrier/timer/event
    reserved fields raise ``NotImplementedError`` at load time via
    ``validate_for_orchestrator_v1`` with pointers to the deferred feature.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PrerequisiteKind = Field(description="Prerequisite type.")

    branch_id: str | None = Field(
        default=None,
        description=(
            "For SPAWN_JOIN: the branch_id whose children must complete. Must "
            "reference a branch declared on an earlier turn of the same conversation."
        ),
    )

    child_conversation_ids: list[str] | None = Field(
        default=None,
        description=(
            "Optional per-child subset: if set, only these specific children must "
            "complete. Reserved; v1 orchestrator rejects at load time."
        ),
    )

    barrier_id: str | None = Field(
        default=None,
        description=(
            "Optional runtime barrier ID for prereqs from multiple parent sessions "
            "to synchronize on a shared runtime session (runtime-diamond topology). "
            "Reserved; v1 orchestrator rejects at load time."
        ),
    )

    timer_seconds: float | None = Field(
        default=None,
        description=(
            "For TIMER: wall-clock seconds this turn waits before dispatching. "
            "Reserved; v1 orchestrator rejects at load time."
        ),
    )

    event_name: str | None = Field(
        default=None,
        description=(
            "For EXTERNAL_EVENT: named signal to await. Reserved; v1 orchestrator "
            "rejects at load time."
        ),
    )
