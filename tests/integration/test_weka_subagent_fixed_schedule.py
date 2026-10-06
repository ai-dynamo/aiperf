# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-schedule replay of a weka trace containing a subagent (AIP-1978).

Three defects lived on this path, each hidden by the one above it:

1. A weka trace could not enter fixed schedule at all -- rejected at config
   resolution -- so nothing below was reachable.
2. Once it could, the fixed schedule also scheduled DAG children that the
   BranchOrchestrator already dispatches, so every subagent request ran twice
   and the duplicates consumed the credit budget, which left the parent's
   gated turn refused and every parent turn after the spawn silently dropped.
3. Once those turns ran, they fired the moment their children finished rather
   than at their recorded timestamp, so the replay compressed the very trace
   it exists to reproduce.

All three report success, so only assertions on request identity *and* timing
catch them. Schedule-construction unit tests cannot: they never issue a credit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

FIXTURE = Path(__file__).parent.parent / "fixtures/weka_subagent"

# (source_outer_idx, source_inner_idx) -> recorded offset in seconds.
RECORDED = {
    (0, None): 0.0,
    (1, 0): 1.5,
    (1, 1): 2.5,
    (2, None): 5.0,
    (3, None): 8.0,
}

# Scheduling is not exact: credit issuance, worker pickup and mock-server
# latency all add jitter. Generous enough not to flake, far tighter than the
# 2.5s error the join-timing defect produced.
TOLERANCE_S = 0.75


@pytest.mark.integration
@pytest.mark.asyncio
class TestWekaSubagentFixedSchedule:
    async def test_subagent_trace_replays_recorded_requests_on_schedule(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
    ):
        assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"

        result = await cli.run(
            f"""
            aiperf profile \
                --model test-model \
                --tokenizer builtin \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --custom-dataset-type weka_trace \
                --input-file {FIXTURE} \
                --fixed-schedule \
                --workers-max 2 \
                --ui simple
            """,
            timeout=300.0,
        )

        assert result.exit_code == 0, f"run failed: {result.exit_code}"
        records = result.jsonl
        assert records, "no profile export records"

        seen = []
        for record in records:
            meta = record.metadata
            seen.append(
                (
                    meta.request_start_ns,
                    meta.source_outer_idx,
                    meta.source_inner_idx,
                )
            )

        identities = sorted((o, i) for _, o, i in seen)

        # Cardinality and identity: no duplicated children, no dropped parents.
        assert identities == sorted(RECORDED), (
            f"replayed {identities}, expected {sorted(RECORDED)}. Duplicated "
            "entries mean children were dispatched by both the schedule and "
            "the orchestrator; missing (2, None)/(3, None) mean the parent's "
            "gated turn was refused and later turns stranded behind it."
        )

        base = min(ts for ts, _, _ in seen)
        offsets = {(o, i): (ts - base) / 1e9 for ts, o, i in seen}

        # The parent turn gated on the subagent must wait for BOTH its children
        # and its own recorded timestamp. Releasing on children alone fired it
        # 2.5s early here.
        gated = offsets[(2, None)]
        assert gated >= RECORDED[(2, None)] - TOLERANCE_S, (
            f"joined parent fired at t+{gated:.2f}s but was recorded at "
            f"t+{RECORDED[(2, None)]}s -- a join released on child completion "
            "alone, ignoring the recorded schedule"
        )

        # Every parent turn keeps its recorded time.
        for key in [(0, None), (2, None), (3, None)]:
            actual, expected = offsets[key], RECORDED[key]
            assert abs(actual - expected) <= TOLERANCE_S, (
                f"parent {key} fired at t+{actual:.2f}s, recorded t+{expected}s"
            )

        # Children keep their documented spawn-relative offsets rather than
        # absolute recorded times, so they are deliberately NOT asserted
        # against RECORDED -- only their ordering is contractual.
        assert offsets[(1, 0)] <= offsets[(1, 1)] + TOLERANCE_S, (
            "subagent inner requests replayed out of order"
        )
