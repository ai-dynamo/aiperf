# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypedDict

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel


class BranchStatsDict(TypedDict):
    """Stable shape produced by ``BranchStats.stats_dict()`` for exporters."""

    children_spawned: int
    children_completed: int
    children_errored: int
    children_truncated: int
    parents_suspended: int
    parents_resumed: int
    parents_failed_due_to_child_error: int
    joins_suppressed: int


class BranchStats(AIPerfBaseModel):
    """Counters for DAG branch orchestration observability.

    Exported as part of ``ProfileResults.branch_stats`` so DAG-shaped runs
    (FORK or SPAWN mode) can be inspected (how many children dispatched,
    how many parents resumed after joins, etc.). Stats are mode-agnostic.
    """

    children_spawned: int = Field(
        default=0,
        description="Number of DAG child sessions that were successfully dispatched.",
    )
    children_completed: int = Field(
        default=0,
        description="Number of DAG child sessions that reached their leaf turn "
        "and were joined back.",
    )
    children_errored: int = Field(
        default=0,
        description="Number of DAG child sessions that terminated with an error.",
    )
    children_truncated: int = Field(
        default=0,
        description="Number of DAG child sessions whose continuation was "
        "blocked by a stop condition (typically the --request-count cap). "
        "The child completed at least one turn but its remaining turns did "
        "not dispatch; tallied separately from children_completed so "
        "observability stays accurate.",
    )
    parents_suspended: int = Field(
        default=0,
        description="Number of parent sessions that paused to await an outstanding "
        "branch join.",
    )
    parents_resumed: int = Field(
        default=0,
        description="Number of parent sessions that resumed with a join turn after "
        "all children completed.",
    )
    parents_failed_due_to_child_error: int = Field(
        default=0,
        description="Number of parent sessions that were aborted because a child "
        "errored under AIPERF_DAG_FAIL_FAST=true.",
    )
    joins_suppressed: int = Field(
        default=0,
        description="Number of joins released without firing because a stop "
        "condition (typically the --request-count cap) blocked the gated child "
        "from dispatching. Counts each join once. Reportable but not a failure.",
    )

    def stats_dict(self) -> BranchStatsDict:
        """Snapshot the counters as a plain dict (stable shape for exporters)."""
        return self.model_dump()
