# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-mp E2E: the unified segment store drives a full profile run.

Runs the full multiprocess ``aiperf`` stack (subprocess ``python -m aiperf
profile`` + live mock server + real Worker + HTTP) over the 2-trace weka
multigraph directory with a pinned seed and asserts the unified-store Worker
path actually dispatches and produces per-record ISL metrics.

The unified store is the SOLE trie store shape (the legacy split
segment+delta stores are retired); the streaming-vs-eager unified store A/B
lives in ``tests/unit/graph/test_hf_streaming_trie_stores.py::
test_streaming_unified_store_byte_matches_eager_interned``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

MULTIGRAPH_DIR = Path(__file__).parent / "fixtures" / "weka_multigraph_dir"

# Pinned seed so the synthesized weka content is deterministic
# (the weka content seed derives from run.random_seed).
_SEED = "1234"


async def _run(
    cli: AIPerfCLI,
    mock: AIPerfMockServer,
    monkeypatch: pytest.MonkeyPatch,
):
    """Run one full profile pass over the weka multigraph directory."""
    return await cli.run(
        f"""
        aiperf profile \
            --model claude-opus-4-5-20251101 \
            --url {mock.url} \
            --endpoint-type chat \
            --input-file {MULTIGRAPH_DIR} \
            --tokenizer builtin \
            --random-seed {_SEED} \
            --num-conversations 2 \
            --concurrency 2 \
            --workers-max 2 \
            --export-level raw \
            --ui simple
        """,
        timeout=300.0,
        assert_success=True,
    )


@pytest.mark.integration
@pytest.mark.asyncio
class TestWekaUnifiedStoreAB:
    async def test_unified_store_runs_end_to_end(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The unified Worker path actually dispatches and produces records."""
        result = await _run(cli, aiperf_mock_server, monkeypatch)
        assert result.exit_code == 0, result.stderr[-2000:]

        assert result.request_count > 0
        assert result.jsonl is not None and result.jsonl, "no profiling records"
        for rec in result.jsonl:
            isl = rec.metrics.get("input_sequence_length")
            assert isl is not None, f"record missing ISL: {rec.metadata}"
            assert isl.value > 0
