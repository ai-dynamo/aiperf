# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-mp E2E locks for two general weka-export fidelity behaviors on the trie.

Runs the full multiprocess ``aiperf`` stack over the 2-trace weka multigraph
directory against the in-repo mock server (segment-trie IR default) and asserts,
on the EXPORTED artifacts, two endpoint-agnostic behaviors that the trie path
carries verbatim:

* **ISL present:** ``input_sequence_length`` appears in the JSON export and
  ``token_counts.input`` is populated on every profiling record, tracking the
  block-aligned covered-token count of the wire prompt (builtin/o200k tokenization,
  modulo small client-side boundary drift).

* **Non-chat rejected up front:** ``--endpoint-type completions`` over a weka
  workload exits non-zero with a clear ``GraphEndpointUnsupportedError`` message at
  configure time, NOT a per-request 422.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

MULTIGRAPH_DIR = Path(__file__).parent / "fixtures" / "weka_multigraph_dir"


async def _run(
    cli: AIPerfCLI,
    mock: AIPerfMockServer,
    monkeypatch,
    *extra: str,
    assert_success: bool = True,
):
    # Pin the seed so the synthesized weka content -- and thus the block-aligned
    # ISL multiset asserted below -- is deterministic across runs and store paths.
    # Without a pinned seed, unseeded runs draw different random content and the
    # exact ISL floor drifts +/-1 token at block boundaries (see the A/B parity
    # test, which pins the same seed to prove legacy == unified byte-for-byte).
    return await cli.run(
        f"""
        aiperf profile \
            --model claude-opus-4-5-20251101 \
            --url {mock.url} \
            --input-file {MULTIGRAPH_DIR} \
            --tokenizer builtin \
            --random-seed 1234 \
            --num-conversations 2 \
            --concurrency 2 \
            --workers-max 2 \
            --export-level raw \
            --ui simple \
            {" ".join(extra)}
        """,
        timeout=300.0,
        assert_success=assert_success,
    )


@pytest.mark.integration
@pytest.mark.asyncio
class TestWekaIslEndpointOslE2E:
    async def test_isl_present_in_exports(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ISL is present (block-aligned covered-count) on every record."""
        result = await _run(
            cli, aiperf_mock_server, monkeypatch, "--endpoint-type", "chat"
        )
        assert result.exit_code == 0, result.stderr[-2000:]

        # Aggregate ISL is present (was entirely absent before the fix).
        assert result.json is not None and result.json.input_sequence_length is not None
        assert result.json.input_sequence_length.avg is not None

        # Every profiling metric record carries an input token count.
        assert result.jsonl is not None and result.jsonl
        for rec in result.jsonl:
            isl = rec.metrics.get("input_sequence_length")
            assert isl is not None, f"record missing ISL: {rec.metadata}"
            assert isl.value > 0

        # The trie IR materializes BLOCK-ALIGNED prompts: a node's prompt is the
        # message-unit concatenation of its covered whole blocks
        # (``min(len(hash_ids), in // block_size)`` of them, block_size 64), with
        # the recorded ``in % block_size`` partial tail deliberately excluded (see
        # ``_assemble_messages`` / ``compute_turn_block_geometry`` and the
        # ``_assert_isl`` covered-count gate). So the recovered ISLs are the
        # block-aligned covered-token counts, NOT the raw recorded ``in``:
        # turn-0 (in=180/200, 2 covered blocks) -> 128; the deeper turns cover
        # 3 or 4 blocks (~192/~256) modulo small builtin (o200k) decode/re-encode
        # boundary drift. The minimum is the 2-block floor (128), never the raw 180.
        isls = sorted(
            rec.metrics["input_sequence_length"].value for rec in result.jsonl
        )
        assert min(isls) == 128  # 2 covered blocks * block_size (64), no partial tail
        assert (
            max(isls) <= 265
        )  # largest is the 4-block prompt (~256) + tokenizer drift

    async def test_non_chat_endpoint_rejected_up_front(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A completions endpoint is rejected at configure time, not per-request."""
        result = await _run(
            cli,
            aiperf_mock_server,
            monkeypatch,
            "--endpoint-type",
            "completions",
            assert_success=False,
        )
        assert result.exit_code != 0
        combined = (result.stderr + result.stdout + (result.log or "")).lower()
        assert (
            "wekaendpointunsupported" in combined or "only supports a chat" in combined
        ), f"expected up-front endpoint rejection, got: {combined[-1500:]}"
        # It must fail UP FRONT at configure, not via a per-request 422 (the mock's
        # "Field required: prompt" rejection of a chat body at /v1/completions).
        assert "field required" not in combined, (
            "non-chat endpoint reached per-request dispatch (422) instead of the "
            "configure-time guard"
        )
