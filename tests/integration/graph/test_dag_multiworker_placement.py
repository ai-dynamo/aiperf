# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DAG instance co-placement over the full multiprocess stack.

The live A/B for graph INSTANCE co-placement: a real ``aiperf profile`` run
of TWO independent spawn+join dag trees at concurrency 2 with TWO workers
must place every trajectory (session) of one dag tree instance on ONE
worker. The spawned child sessions mint their own ``x_correlation_id``s, so
without the router keying graph sessions on the instance ``trace_id`` they
would balance least-loaded onto the OTHER tree's worker -- and the join
turn's splice would miss the worker-local dynamic pool (``pool_missing``
trace-stop), which single-worker component tests can never observe.

Tree instances are reconstructed from the records themselves: every child
record carries ``parent_correlation_id`` (the parent session's corr), so a
dag tree is a connected component over the corr <- parent-corr edges. The
decisive assertions: each component's records span exactly ONE worker, and
at least one component really contains multiple sessions (anti-vacuous).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "dag"
    / "two_spawn_trees.dag.jsonl"
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestDagMultiworkerPlacement:
    async def test_each_dag_tree_pins_to_one_worker(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
    ) -> None:
        result = await cli.run(
            f"""
            aiperf profile \
                --model spawn-model \
                --url {aiperf_mock_server.url} \
                --input-file {_FIXTURE} \
                --graph-format dag_jsonl \
                --endpoint-type chat \
                --tokenizer builtin \
                --random-seed 1234 \
                --num-conversations 2 \
                --concurrency 2 \
                --workers-max 2 \
                --export-level raw \
                --ui simple
            """,
            timeout=300.0,
            assert_success=True,
        )
        assert result.exit_code == 0, result.stderr[-2000:]

        raw = result.raw_records
        assert raw is not None and raw, "no raw records exported"

        # Union-find over corr <- parent-corr edges: one component per dag
        # tree instance (spawned children carry parent_correlation_id).
        root: dict[str, str] = {}

        def find(x: str) -> str:
            root.setdefault(x, x)
            while root[x] != x:
                root[x] = root[root[x]]
                x = root[x]
            return x

        def union(a: str, b: str) -> None:
            root[find(a)] = find(b)

        for rec in raw:
            corr = rec.metadata.x_correlation_id
            assert corr, f"record missing x_correlation_id: {rec.metadata}"
            find(corr)
            if rec.metadata.parent_correlation_id:
                union(corr, rec.metadata.parent_correlation_id)

        workers_by_tree: dict[str, set[str]] = defaultdict(set)
        sessions_by_tree: dict[str, set[str]] = defaultdict(set)
        for rec in raw:
            worker = rec.metadata.worker_id
            assert worker is not None, f"record missing worker_id: {rec.metadata}"
            tree = find(rec.metadata.x_correlation_id)
            workers_by_tree[tree].add(worker)
            sessions_by_tree[tree].add(rec.metadata.x_correlation_id)

        # THE co-placement invariant: no dag tree instance spans workers.
        scattered = {t: w for t, w in workers_by_tree.items() if len(w) > 1}
        assert not scattered, f"dag trees scattered across workers: {scattered}"

        # Anti-vacuous: at least one tree really had multiple sessions (the
        # spawn actually produced child trajectories with their own corrs).
        assert any(len(s) > 1 for s in sessions_by_tree.values()), (
            f"expected a multi-session dag tree, got per-tree session counts "
            f"{[len(s) for s in sessions_by_tree.values()]}"
        )
