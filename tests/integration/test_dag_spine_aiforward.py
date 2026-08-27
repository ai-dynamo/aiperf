# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: full 'AI-Forward' orchestrator spine against the real mock server.

Exercises the complete feature: a request-free coordinator that, for N rounds,
fans out an ASYMMETRIC pair of branches (branch-a: 4 turns, branch-b: 2 turns),
AND-waits both to complete, applies a per-round SAMPLED (lognormal) think-time,
then fires the next round. Asserts the spine issues no HTTP itself and that every
round runs both branches to full depth.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "dag"
    / "orchestrator_spine_aiforward.dag.jsonl"
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_aiforward_spine_runs_all_rounds_request_free(
    cli: AIPerfCLI, tests.aiperf_mock_server: AIPerfMockServer
):
    result = await cli.run(
        f"""
        aiperf profile \
            --model test-model \
            --url {tests.aiperf_mock_server.url} \
            --endpoint-type chat \
            --input-file {FIXTURE} \
            --custom-dataset-type dag_jsonl \
            --num-conversations 1 \
            --concurrency 4 \
            --workers-max 2 \
            --random-seed 1234 \
            --export-level raw \
            --ui simple
        """,
        timeout=300.0,
    )

    recs = result.raw_records or []
    # 3 rounds x [branch-a: 4 turns + branch-b: 2 turns] = 18 real wire requests.
    assert len(recs) == 18, f"expected 18 child requests, got {len(recs)}"
    # The request-free spine issues nothing: every wire record is a spawned child.
    assert all(r.metadata.parent_correlation_id is not None for r in recs), (
        "orchestrator spine must not appear on the wire"
    )
    # 2 branches x 3 rounds, all completed (no hang, no over-fire); think-time
    # applied per round via the join gate.
    bs = result.json.branch_stats if result.json else None
    assert bs is not None
    assert bs.children_spawned == 6
    assert bs.children_completed == 6
    assert bs.children_errored == 0
