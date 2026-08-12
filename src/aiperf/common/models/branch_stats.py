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
    children_delayed: int
    parents_suspended: int
    parents_resumed: int
    parents_failed_due_to_child_error: int
    joins_suppressed: int
    graphs_admitted: int
    graphs_completed_to_end: int


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
    children_delayed: int = Field(
        default=0,
        ge=0,
        description="Number of SPAWN child sessions whose turn-0 dispatch was "
        "scheduled at its recorded offset from the branch spawn (child turn-0 "
        "timestamp past the branch start) instead of dispatching immediately.",
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
    graphs_admitted: int = Field(
        default=0,
        ge=0,
        description="Number of request-free orchestrator graph instances admitted "
        "(turn 0 fired). One per sampled orchestrator firing.",
    )
    graphs_completed_to_end: int = Field(
        default=0,
        ge=0,
        description="Number of orchestrator graph instances that reached END (the "
        "terminal request-free gate completed after all rounds). "
        "``graphs_admitted - graphs_completed_to_end`` is the count of graphs that "
        "did not finish, so completion status is reconstructable from the export.",
    )

    def stats_dict(self) -> BranchStatsDict:
        """Snapshot the counters as a plain dict (stable shape for exporters)."""
        return self.model_dump()
