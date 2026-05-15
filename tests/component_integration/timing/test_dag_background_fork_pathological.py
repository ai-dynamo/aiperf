# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial / pathological end-to-end pins for ``DagFork.background``.

These cases drive the full credit/orchestrator pipeline against fixtures
designed to stress the parts that quietly broke under the v1 design (and
caught the ``root_requests_sent`` counter bug):

  - Nested BG-fork (root → BG fork → A → BG fork → B): does the parent's
    full-turn-list dispatch correctly at every depth?
  - Fan-out BG (root BG-forks 5 children at once): orchestrator must
    track all 5 spawn-and-drain edges concurrently with the parent's
    later turns.
  - BG-fork + SPAWN_JOIN coexistence: parent BG-forks on turn 0
    (fire-and-forget), THEN suspends on a SPAWN_JOIN gate at turn 1, then
    runs turn 2 after the gate fires. Tests the runtime correctly
    distinguishes the two scheduling modes on the same conversation.
  - Concurrency=4 + BG-fork: drain-observer interaction. Multiple BG
    parents in flight, each with their own children fanning out. Pre-fix
    this exposed the same race the dag_hard_cap suite caught."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.component_integration.conftest import AIPerfRunnerResultWithSharedBus
from tests.component_integration.timing.conftest import defaults
from tests.harness.analyzers import CreditFlowAnalyzer
from tests.harness.utils import AIPerfCLI

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "dag"
NESTED = FIXTURES / "bg_fork_nested.dag.jsonl"
FANOUT = FIXTURES / "bg_fork_fanout.dag.jsonl"
JOIN_COEX = FIXTURES / "bg_fork_with_spawn_join.dag.jsonl"


def _cmd(input_file: Path, **kwargs) -> str:
    extra = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in kwargs.items())
    return f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --custom-dataset-type dag_jsonl \
            --input-file {input_file} \
            --record-processor-service-count 1 \
            --workers-max 4 \
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui} \
            {extra}
    """


@pytest.mark.component_integration
class TestNestedBgFork:
    """Nested BG fork: a BG-forked child can itself BG-fork (or
    plain-FORK) a grandchild. The orchestrator's ``intercept`` runs at
    every depth — the prior ``agent_depth > 0`` short-circuit was
    overly conservative and silently dropped grandchildren.

    Topology: r (2 turns, BG-forks ``a`` on t0) → a (2 turns, BG-forks
    ``b`` on t0) → b (2 turns).
    Wire count: r(2) + a(2) + b(2) = 6.
    """

    def test_nested_bg_fork_grandchild_dispatches(self, cli: AIPerfCLI) -> None:
        """Grandchild fires; full tree's wires (r+a+b) all run."""
        result = cli.run_sync(
            _cmd(NESTED, concurrency=1, num_conversations=1),
            timeout=30.0,
            assert_success=True,
        )
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.total_credits == 6, (
            f"nested BG-fork (r→a→b, each 2 turns): expected 6 wires, "
            f"got {analyzer.total_credits}"
        )
        assert analyzer.credits_balanced()

    def test_nested_bg_fork_branch_stats_counts_both_edges(
        self, cli: AIPerfCLI
    ) -> None:
        """Both spawn edges (r→a and a→b) counted in BranchStats."""
        result = cli.run_sync(
            _cmd(NESTED, concurrency=1, num_conversations=1),
            timeout=30.0,
            assert_success=True,
        )
        bs = result.json.branch_stats
        assert bs is not None
        assert bs.children_spawned == 2
        assert bs.children_completed == 2
        assert bs.parents_suspended == 0


@pytest.mark.component_integration
class TestBgForkFanOut:
    """One parent BG-forks 5 children at once. The orchestrator's
    descendant tracking must handle the concurrent spawn-and-drain across
    all 5 children in parallel with the parent's later turns.

    Topology: r (2 turns, BG-forks 5 children on t0) → c1..c5 (1 turn each).
    Wire count: r(2) + 5×1 = 7.
    """

    def test_fanout_5_bg_children(self, cli: AIPerfCLI) -> None:
        result = cli.run_sync(
            _cmd(FANOUT, concurrency=1, num_conversations=1),
            timeout=30.0,
            assert_success=True,
        )
        analyzer = CreditFlowAnalyzer(result.runner_result)
        assert analyzer.total_credits == 7, (
            f"r(2) + 5 single-turn BG children = 7 wires; got {analyzer.total_credits}"
        )
        assert analyzer.credits_balanced()

    def test_fanout_branch_stats_all_five_completed(self, cli: AIPerfCLI) -> None:
        result = cli.run_sync(
            _cmd(FANOUT, concurrency=1, num_conversations=1),
            timeout=30.0,
            assert_success=True,
        )
        bs = result.json.branch_stats
        assert bs is not None
        assert bs.children_spawned == 5
        assert bs.children_completed == 5
        assert bs.children_truncated == 0
        assert bs.parents_suspended == 0


@pytest.mark.component_integration
class TestBgForkCoexistsWithSpawnJoinE2E:
    """A parent that BG-forks on turn 0 (fire-and-forget) AND runs a
    SPAWN with auto-join on turn 1 must:
      - dispatch the BG child concurrently with the parent's continuation
      - SUSPEND at turn 2 (next-turn after spawn) until the SPAWN child drains
      - resume turn 2 after the gate fires
      - eventually drain the BG child too

    Topology: r (3 turns; t0 BG-forks side, t1 SPAWNs sync) → side (2
    turns) + sync (1 turn). Wire count: r(3) + side(2) + sync(1) = 6.
    """

    def test_bg_and_spawn_join_coexist_e2e(self, cli: AIPerfCLI) -> None:
        result = cli.run_sync(
            _cmd(JOIN_COEX, concurrency=1, num_conversations=1),
            timeout=30.0,
            assert_success=True,
        )
        analyzer = CreditFlowAnalyzer(result.runner_result)
        assert analyzer.total_credits == 6, (
            f"r(3) + side BG(2) + sync SPAWN(1) = 6 wires; got {analyzer.total_credits}"
        )
        assert analyzer.credits_balanced()

    def test_bg_and_spawn_join_branch_stats(self, cli: AIPerfCLI) -> None:
        """BG fork is one spawn; SPAWN+auto-join is another. Both complete.
        Parent suspends ONCE (at the SPAWN's auto-join gate) and resumes."""
        result = cli.run_sync(
            _cmd(JOIN_COEX, concurrency=1, num_conversations=1),
            timeout=30.0,
            assert_success=True,
        )
        bs = result.json.branch_stats
        assert bs is not None
        assert bs.children_spawned == 2
        assert bs.children_completed == 2
        assert bs.parents_suspended == 1, (
            f"parent should suspend exactly once on the SPAWN_JOIN gate "
            f"(BG fork doesn't generate one); got "
            f"parents_suspended={bs.parents_suspended}"
        )
        assert bs.parents_resumed == 1


@pytest.mark.component_integration
class TestBgForkUnderConcurrency:
    """Concurrency-stress: 4 BG-forking parents in flight simultaneously
    means 4 BG children fan out concurrently with each parent's later
    turns. Pre-drain-observer-fix this race-class hung; this is a
    regression pin specifically against the BG-fork interaction."""

    def test_bg_fork_concurrency_4_no_hang(self, cli: AIPerfCLI) -> None:
        """4 root sessions, each runs 2 turns + BG-forks 5 children (1
        turn each). Total per root: 2 + 5 = 7 wires. 4 roots × 7 = 28 wires.
        With ``--concurrency 4`` all 4 roots are simultaneously in flight.
        Pre-fix the conc=4 + BG-fork combination hung; post-fix completes."""
        result = cli.run_sync(
            _cmd(FANOUT, concurrency=4, num_conversations=4),
            timeout=30.0,
            assert_success=True,
        )
        analyzer = CreditFlowAnalyzer(result.runner_result)
        assert analyzer.total_credits == 28, (
            f"4 roots × (2 root turns + 5 BG children) = 28 wires; "
            f"got {analyzer.total_credits}"
        )
        assert analyzer.credits_balanced()

    def test_bg_fork_request_count_truncation(self, cli: AIPerfCLI) -> None:
        """``--request-count 10`` against the 5-child fanout topology
        truncates somewhere mid-tree. The cap must hold exactly; the run
        must not hang. (``--num-conversations`` is rejected when paired
        with ``--request-count`` so we drop it; the dataset recycles to
        fill the cap.)"""
        result = cli.run_sync(
            _cmd(FANOUT, concurrency=1, request_count=10),
            timeout=30.0,
            assert_success=True,
        )
        analyzer = CreditFlowAnalyzer(result.runner_result)
        assert analyzer.total_credits == 10, (
            f"--request-count 10 must cap at exactly 10 wires; "
            f"got {analyzer.total_credits}"
        )
        assert analyzer.credits_balanced()

    def test_bg_fork_request_count_concurrency_4(self, cli: AIPerfCLI) -> None:
        """The full stress: --concurrency 4 + --request-count 12 + 5-child
        BG fan-out. Multiple BG parents truncating concurrently — exactly
        the shape that exposed the dag_hard_cap drain-observer race.
        Post-fix it must complete with exactly 12 wires."""
        result = cli.run_sync(
            _cmd(FANOUT, concurrency=4, request_count=12),
            timeout=30.0,
            assert_success=True,
        )
        analyzer = CreditFlowAnalyzer(result.runner_result)
        assert analyzer.total_credits == 12
        assert analyzer.credits_balanced()
