# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-mp E2E: a Dynamo agent-trace runs end-to-end and produces records.

Runs the full multiprocess ``aiperf`` stack (subprocess ``python -m aiperf
profile`` + live mock server + real Worker + HTTP) over a current-schema
``dynamo.request.trace.v1`` directory: ONE root session (three clean linear
recorded-hash turns) + ONE subagent child (two recorded-hash turns) spliced at
the parent's turn 3. This is the FIRST proof dynamo produces records through
the real build+schedule pipeline -- if the build-plane and schedule-plane
ordinals disagree, every dispatch dies with ``GraphEnvelopeMissing`` and no
record lands.

The fixture records carry replay metadata, so the adapter uses the recorded
``input_sequence_hashes`` (recorded-when-present, no mode selection). The
determinism env (start-ratio pin) mirrors ``test_weka_unified_store_ab.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

DYN_DIR = Path(__file__).parent / "fixtures" / "dynamo_dir"

# Pinned seed so the synthesized recorded-hash content is deterministic.
_SEED = "1234"


async def _run(
    cli: AIPerfCLI,
    mock: AIPerfMockServer,
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    """Run one full profile pass over the dynamo directory."""
    return await cli.run(
        f"""
        aiperf profile \
            --model m \
            --url {mock.url} \
            --endpoint-type chat \
            --input-file {DYN_DIR} \
            --tokenizer builtin \
            --random-seed {_SEED} \
            --num-conversations 1 \
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
class TestDynamoE2E:
    async def test_dynamo_produces_records(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dynamo dispatches and produces records with valid ISL (no GraphEnvelopeMissing).

        The flat trie lowering emits every dispatch node with a per-node scratch
        ``{node_id}_out`` output channel (TEXT / overwrite, scalar-tolerant), so
        the schedule-plane executor's scalar placeholder write never hits a list
        reducer. The prompt content comes from the segment store via
        ``prompt_segment_ids`` stamped at parse time by the shared LCP
        segment-trie core.
        """
        r = await _run(cli, aiperf_mock_server, monkeypatch)
        assert r.exit_code == 0, r.stderr[-2000:]

        assert r.request_count > 0, "dynamo produced no profiling records"
        assert r.jsonl, "no per-record metrics captured"

        isls: list[int] = []
        for rec in r.jsonl:
            isl = rec.metrics.get("input_sequence_length")
            assert isl is not None, f"record missing ISL: {rec.metadata}"
            assert isl.value > 0, f"record has non-positive ISL: {isl.value}"
            isls.append(int(isl.value))

        # Content-parity guard: the fix is dispatch-completion only -- prompts are
        # still materialized from the segment store (``prompt_segment_ids``), NOT
        # the runtime channel. The fixture's root session accumulates strictly-
        # nested replay hashes ([111,222] < [..,333] < [..,444] < [..,555]),
        # so store-sourced prompts grow turn over turn: the recorded ISLs MUST be
        # multiple distinct, growing values. Had the scalar placeholder ("") leaked
        # into the prompt content (a content regression, not a dispatch-completion
        # fix), every ISL would collapse to a single tiny constant. Requiring >=3
        # distinct ISLs spanning a real range proves the accumulating store content
        # reached the wire unchanged.
        distinct = sorted(set(isls))
        assert len(distinct) >= 3, (
            "expected multiple distinct store-sourced ISLs (accumulating prompts); "
            f"got {distinct} -- a single/tiny constant would mean placeholder content "
            "leaked past dispatch"
        )
        assert max(isls) > min(isls), (
            f"ISLs must grow with the accumulating conversation; got {distinct}"
        )
