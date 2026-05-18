# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BranchStats publication on CreditPhaseCompleteMessage.

DAG-shaped runs surface BranchOrchestrator's running counters via the
phase-complete message so the records pipeline can splice them into
profile_export_aiperf.json. Non-DAG runs leave the field unset.
"""

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import CreditPhaseStats
from aiperf.common.models.branch_stats import BranchStats
from aiperf.credit.messages import CreditPhaseCompleteMessage


@pytest.fixture
def phase_stats() -> CreditPhaseStats:
    return CreditPhaseStats(phase=CreditPhase.PROFILING)


class TestCreditPhaseCompleteBranchStats:
    def test_field_present(self, phase_stats: CreditPhaseStats) -> None:
        stats = BranchStats(
            children_spawned=4,
            children_completed=4,
            children_errored=0,
            joins_suppressed=0,
        )
        msg = CreditPhaseCompleteMessage(
            service_id="timing-1",
            stats=phase_stats,
            branch_stats=stats,
        )
        assert msg.branch_stats == stats

    def test_default_none_for_non_dag_runs(self, phase_stats: CreditPhaseStats) -> None:
        msg = CreditPhaseCompleteMessage(
            service_id="timing-1",
            stats=phase_stats,
        )
        assert msg.branch_stats is None

    def test_round_trip(self, phase_stats: CreditPhaseStats) -> None:
        stats = BranchStats(
            children_spawned=2,
            children_completed=2,
            children_errored=1,
            joins_suppressed=1,
        )
        msg = CreditPhaseCompleteMessage(
            service_id="timing-1",
            stats=phase_stats,
            branch_stats=stats,
        )
        rebuilt = CreditPhaseCompleteMessage.model_validate(msg.model_dump())
        assert rebuilt.branch_stats == stats


class TestBranchOrchestratorSnapshot:
    def test_snapshot_branch_stats_returns_copy_of_current_counters(self) -> None:
        from aiperf.timing.branch_orchestrator import BranchOrchestrator

        orch = BranchOrchestrator.__new__(BranchOrchestrator)
        orch.stats = BranchStats(
            children_spawned=7,
            children_completed=5,
            children_errored=1,
            joins_suppressed=2,
        )

        snap = orch.snapshot_branch_stats()

        assert snap.children_spawned == 7
        assert snap.children_completed == 5
        assert snap.children_errored == 1
        assert snap.joins_suppressed == 2

    def test_snapshot_is_independent_copy(self) -> None:
        from aiperf.timing.branch_orchestrator import BranchOrchestrator

        orch = BranchOrchestrator.__new__(BranchOrchestrator)
        orch.stats = BranchStats(children_spawned=3)

        snap = orch.snapshot_branch_stats()
        orch.stats.children_spawned = 999

        assert snap.children_spawned == 3
