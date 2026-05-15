# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression pin: multi-root, single-turn-everywhere DAG must NOT trip the
FORK-routing invariant.

When every conversation in a ``dag_jsonl`` file is single-turn,
``DatasetManager._preformat_payloads`` is eligible to flip the storage
format to ``PAYLOAD_BYTES`` (the worker fast path). PAYLOAD_BYTES drops
branch metadata: when a child later lands on a worker and the worker
rebuilds the parent's ``Conversation`` from cached payload bytes, the
``branches`` list is empty. The session cache's evict path consults
``is_fork_parent`` — which checks ``conversation.branches`` for any FORK —
and decides the parent isn't a FORK parent at all. The parent is popped
immediately on its own credit return, so by the time the first FORK
child arrives at the same worker the parent's session is gone and the
runtime aborts with::

    RuntimeError: FORK routing invariant violated: parent session
    '<id>' not found on this worker

The fix: ``DatasetManager._preformat_payloads`` bails out whenever any
conversation declares a FORK branch. The format stays ``CONVERSATION``,
the worker uses the slow path, the parent's full Conversation (branches
intact) lands in the cache, and ``is_fork_parent`` correctly pins it
until children arrive.

This file is the end-to-end regression pin. It uses the
all-single-turn / multi-root fixture deliberately, because that's the
ONLY shape that flips the format to PAYLOAD_BYTES — fixtures with any
multi-turn child (like ``full.dag.jsonl``) silently dodge the bug.
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
    / "multi_root_single_turn.dag.jsonl"
)


@pytest.mark.component_integration
class TestDagMultiRootSingleTurnPayloadBytes:
    """Multi-root / all-single-turn DAG runs must complete without tripping
    the FORK-routing invariant."""

    @pytest.mark.parametrize(
        "concurrency,num_conversations",
        [(1, 1), (1, 2), (2, 2), (2, 4)],
        ids=[
            "conc=1-one-root",
            "conc=1-both-roots-sequential",
            "conc=2-both-roots-concurrent",
            "conc=2-roots-recycled",
        ],
    )
    def test_no_fork_routing_violation(
        self,
        cli: AIPerfCLI,
        concurrency: int,
        num_conversations: int,
    ) -> None:
        """Run completes; every issued credit returns; no leaks."""
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --custom-dataset-type dag_jsonl \
                --input-file {FIXTURE} \
                --concurrency {concurrency} \
                --num-conversations {num_conversations} \
                --record-processor-service-count 1 \
                --workers-max 2 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui}
        """
        result = cli.run_sync(cmd, timeout=60.0, assert_success=True)
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.total_credits > 0, "run produced no credits"
        assert analyzer.credits_balanced(), (
            f"Credit leak: {analyzer.total_credits} issued, "
            f"{analyzer.total_returns} returned"
        )
