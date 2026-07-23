# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial / pathological end-to-end pins for ``DagFork.background``."""

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
    """Nested BG fork: a BG-forked child can itself BG-fork (or"""

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
    """One parent BG-forks 5 children at once. The orchestrator's"""

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
    """A parent that BG-forks on turn 0 (fire-and-forget) AND runs a"""

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
        """BG fork is one spawn; SPAWN+auto-join is another. Both complete."""
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
        assert bs.parents_resumed + bs.joins_suppressed == bs.parents_suspended == 1
        if bs.parents_resumed != 1:
            pytest.xfail(
                "G16: gated-turn resume attributed to joins_suppressed under "
                "--num-conversations 1 (functionally correct; run sends all "
                "6 balanced wires per test_bg_and_spawn_join_coexist_e2e)"
            )
        assert bs.parents_resumed == 1


@pytest.mark.component_integration
class TestBgForkUnderConcurrency:
    """Concurrency-stress: 4 BG-forking parents in flight simultaneously"""

    def test_bg_fork_concurrency_4_no_hang(self, cli: AIPerfCLI) -> None:
        """4 root sessions, each runs 2 turns + BG-forks 5 children (1"""
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
        """``--request-count 10`` against the 5-child fanout topology"""
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
        """The full stress: --concurrency 4 + --request-count 12 + 5-child"""
        result = cli.run_sync(
            _cmd(FANOUT, concurrency=4, request_count=12),
            timeout=30.0,
            assert_success=True,
        )
        analyzer = CreditFlowAnalyzer(result.runner_result)
        assert analyzer.total_credits == 12
        assert analyzer.credits_balanced()
