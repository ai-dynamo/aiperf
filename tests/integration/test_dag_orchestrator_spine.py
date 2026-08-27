# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration test for the request-free orchestrator SPINE.

An orchestrator conversation with ``rounds: N`` synthesizes a chained,
request-free spine: each round spawns its branches, and the next round
AND-waits (join=all) for ALL of the round's branches to complete before firing.

This drives the full aiperf subprocess against the mock server and asserts:

- The orchestrator spine issues **zero** HTTP requests (every wire record is a
  spawned child; the spine turns are ``no_request``).
- Each round fires both branches, and the run completes with the exact child
  count for ``rounds`` rounds -- i.e. the join gates fire and the session
  neither hangs nor over-fires.
- A multi-turn branch (branch-a, 2 turns) runs to full depth each round.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "dag"
    / "orchestrator_spine.dag.jsonl"
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestDagOrchestratorSpineEndToEnd:
    async def test_request_free_spine_gates_each_round(
        self,
        cli: AIPerfCLI,
        tests.aiperf_mock_server: AIPerfMockServer,
    ):
        assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"

        result = await cli.run(
            f"""
            aiperf profile \
                --model test-model \
                --url {tests.aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {FIXTURE} \
                --custom-dataset-type dag_jsonl \
                --num-conversations 1 \
                --concurrency 1 \
                --workers-max 2 \
                --export-level raw \
                --ui simple
            """,
            timeout=300.0,
        )

        assert result.raw_records is not None, (
            "profile_export_raw.jsonl must exist when --export-level raw is set"
        )

        # One spine firing, rounds=2: branch-a (2 turns) + branch-b (1 turn) per
        # round => 2 * (2 + 1) = 6 real wire requests. The orchestrator spine
        # itself sends nothing.
        assert len(result.raw_records) == 6, (
            f"Expected 6 child requests (2 rounds x [a:2 turns + b:1 turn]), "
            f"got {len(result.raw_records)}"
        )

        # The orchestrator sends NO request: every wire record is a spawned
        # child (parent_correlation_id set). No depth-0 orchestrator record.
        for rec in result.raw_records:
            assert rec.metadata.parent_correlation_id is not None, (
                "a request-free orchestrator spine must not appear on the wire; "
                "every record must be a spawned child"
            )

        # Branch accounting: 2 branches x 2 rounds = 4 spawned children, all
        # completed, none errored. If a round's join failed to gate, the counts
        # would differ or the run would hang.
        assert result.json is not None, "profile_export_aiperf.json must exist"
        assert result.json.branch_stats is not None
        assert result.json.branch_stats.children_spawned == 4
        assert result.json.branch_stats.children_completed == 4
        assert result.json.branch_stats.children_errored == 0
