# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end pin for ``DagFork.background=True`` (fork-and-continue).

Drives the full credit/orchestrator pipeline using ``FakeTransport`` (no
real HTTP) against a 3-turn parent that BG-forks on turn 0. Asserts:

  - parent dispatches all 3 of its turns (parent is NOT terminated by
    the BG fork on turn 0 — the must-be-last-turn rule is correctly
    waived for ``background=True`` branches);
  - child session inherits parent's accumulated context at the spawn
    point AND runs to its own leaf;
  - exact wire count (3 parent turns + 2 child turns = 5);
  - branch_stats.children_spawned/completed both tick to 1; no
    truncation, no errors, no parent suspension (BG fork doesn't
    generate a SPAWN_JOIN gate);
  - no hang.

Topology: ``tests/fixtures/dag/background_fork.dag.jsonl`` — 1 root with
3 turns, BG-forks 1 child with 2 turns from turn 0.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.component_integration.conftest import AIPerfRunnerResultWithSharedBus
from tests.component_integration.timing.conftest import defaults
from tests.harness.analyzers import CreditFlowAnalyzer
from tests.harness.utils import AIPerfCLI

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "dag"
    / "background_fork.dag.jsonl"
)


def _build_command(input_file: Path) -> str:
    return f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --custom-dataset-type dag_jsonl \
            --input-file {input_file} \
            --concurrency 1 \
            --num-conversations 1 \
            --record-processor-service-count 1 \
            --workers-max 2 \
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui}
    """


@pytest.mark.component_integration
class TestBackgroundForkParentContinues:
    """A parent that BG-forks on a non-final turn must run all of its
    own remaining turns AND the child must drain. The 3-turn parent
    plus 2-turn child must produce exactly 5 wires."""

    def test_parent_runs_all_turns_and_child_drains(self, cli: AIPerfCLI) -> None:
        result = cli.run_sync(
            _build_command(FIXTURE),
            timeout=30.0,
            assert_success=True,
        )
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.total_credits == 5, (
            f"expected 5 wire credits (3 parent turns + 2 child turns); "
            f"got {analyzer.total_credits}"
        )
        assert analyzer.credits_balanced(), (
            f"credit leak: {analyzer.total_credits} issued, "
            f"{analyzer.total_returns} returned"
        )

    def test_branch_stats_reports_one_spawn_one_completion(
        self, cli: AIPerfCLI
    ) -> None:
        """``branch_stats`` must reflect the BG fork: one child spawned,
        one completed. No suspensions (no SPAWN_JOIN gate). No truncation."""
        result = cli.run_sync(
            _build_command(FIXTURE),
            timeout=30.0,
            assert_success=True,
        )
        assert result.json is not None
        bs = result.json.branch_stats
        assert bs is not None
        assert bs.children_spawned == 1, (
            f"BG-forked one child; expected children_spawned=1, got "
            f"{bs.children_spawned}"
        )
        assert bs.children_completed == 1, (
            f"child must drain to its leaf; expected children_completed=1, got "
            f"{bs.children_completed}"
        )
        assert bs.children_truncated == 0
        assert bs.children_errored == 0
        assert bs.parents_suspended == 0, (
            "BG fork does not generate a SPAWN_JOIN gate, so the parent must "
            f"never suspend; got parents_suspended={bs.parents_suspended}"
        )
        assert bs.joins_suppressed == 0
