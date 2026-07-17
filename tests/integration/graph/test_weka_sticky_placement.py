# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-trajectory sticky placement over the full multiprocess stack.

The live A/B for sticky routing: a real `aiperf profile` run
over the 2-trace weka multigraph directory against the in-repo mock server, with
two workers, must place ALL of a trajectory's credits on ONE worker.

Graph credits mint `turn_index` per node and key their router session on the
instance `trace_id`; without the session the router would scatter a trace's
nodes across workers (least-loaded per credit). The instance session pins
them; the decisive proof is worker consolidation grouped by the template
`conversation_id` (one instance per template in this single-pass run), which
accounting-only assertions cannot see.

This is also the non-regression gate for sticky routing: the weka corpus
still runs clean and still produces ISL-bearing records under sticky routing.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

MULTIGRAPH_DIR = Path(__file__).parent / "fixtures" / "weka_multigraph_dir"


@pytest.mark.integration
@pytest.mark.asyncio
class TestWekaStickyPlacement:
    async def test_each_trace_pins_to_one_worker(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        result = await cli.run(
            f"""
            aiperf profile \
                --model claude-opus-4-5-20251101 \
                --url {aiperf_mock_server.url} \
                --input-file {MULTIGRAPH_DIR} \
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

        # Group every request by its trajectory template id (graph credits
        # stamp conversation_id = the nonce-less trajectory template; this
        # single-pass run mints exactly one instance per template).
        by_trace: dict[str, set[str]] = defaultdict(set)
        multi_node_traces = defaultdict(int)
        for rec in raw:
            conv = rec.metadata.conversation_id
            worker = rec.metadata.worker_id
            assert conv is not None, f"record missing conversation_id: {rec.metadata}"
            assert worker is not None, f"record missing worker_id: {rec.metadata}"
            by_trace[conv].add(worker)
            multi_node_traces[conv] += 1

        # THE sticky invariant: no trace instance spans more than one worker.
        scattered = {c: w for c, w in by_trace.items() if len(w) > 1}
        assert not scattered, f"traces scattered across workers: {scattered}"

        # The check is non-vacuous: at least one trace issued multiple credits
        # (so single-worker placement is a real consolidation, not one request).
        assert any(n > 1 for n in multi_node_traces.values()), (
            f"expected a multi-node trace to make consolidation meaningful, "
            f"got per-trace request counts {dict(multi_node_traces)}"
        )

    async def test_weka_still_runs_clean_under_sticky(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-regression: sticky routing does not break weka ISL/records."""

        result = await cli.run(
            f"""
            aiperf profile \
                --model claude-opus-4-5-20251101 \
                --url {aiperf_mock_server.url} \
                --input-file {MULTIGRAPH_DIR} \
                --endpoint-type chat \
                --tokenizer builtin \
                --random-seed 1234 \
                --num-conversations 2 \
                --concurrency 2 \
                --workers-max 2 \
                --ui simple
            """,
            timeout=300.0,
            assert_success=True,
        )
        assert result.exit_code == 0, result.stderr[-2000:]
        assert result.json is not None
        assert result.json.input_sequence_length is not None
        assert result.json.input_sequence_length.avg is not None
        assert result.jsonl is not None and result.jsonl
        for rec in result.jsonl:
            isl = rec.metrics.get("input_sequence_length")
            assert isl is not None and isl.value > 0
