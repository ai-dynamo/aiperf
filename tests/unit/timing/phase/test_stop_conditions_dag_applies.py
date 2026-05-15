# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""StopCondition.applies_to_dag_children classification.

DAG children (``agent_depth > 0``) are dispatched by the
BranchOrchestrator at credit-return time, NOT by the phase's
TimingStrategy loop. Some stop conditions still apply to them
(--request-count is a literal wire-request cap, time/cancellation
are user-facing guarantees) and others do not (--num-conversations
is a sampler-plan target keyed to root sessions). Each concrete
StopCondition subclass declares its intent via a class-level
``applies_to_dag_children`` flag.
"""

from aiperf.timing.phase.stop_conditions import (
    RequestCountStopCondition,
    SessionCountStopCondition,
)


class TestAppliesToDagChildren:
    def test_request_count_applies_to_children(self) -> None:
        # --request-count N is a literal wire cap: every HTTP request
        # counts, so DAG children must honor it just like roots.
        assert RequestCountStopCondition.applies_to_dag_children is True

    def test_session_count_root_only(self) -> None:
        # --num-conversations N targets the sampler plan — N full
        # conversation TREES — and DAG offspring belong to a tree
        # they did not start. They must bypass this gate.
        assert SessionCountStopCondition.applies_to_dag_children is False
