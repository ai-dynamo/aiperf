# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: per-round-authored orchestrator spine against the real mock server.

Exercises the ``rounds`` LIST form (AIP-1105): a request-free coordinator whose
rounds each fan out their OWN distinct branch sessions (not one repeated
template), with per-round think-time. Asserts the spine issues no HTTP itself,
that each round's DISTINCT authored payloads reach the wire, and that a
``message_array_with_responses`` branch sends each turn's authored array with no
accumulation of prior turns/responses (payload isolation).
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer, RawRecordInfo

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "dag"
    / "orchestrator_spine_per_round.dag.jsonl"
)


def _payload_text(rec: RawRecordInfo) -> str:
    return orjson.dumps(rec.payload).decode() if rec.payload is not None else ""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_per_round_spine_fires_distinct_payloads_request_free(
    cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
):
    result = await cli.run(
        f"""
        aiperf profile \
            --model test-model \
            --url {aiperf_mock_server.url} \
            --endpoint-type chat \
            --input-file {FIXTURE} \
            --custom-dataset-type dag_jsonl \
            --num-conversations 2 \
            --concurrency 2 \
            --workers-max 2 \
            --random-seed 1234 \
            --export-level raw \
            --ui simple
        """,
        timeout=300.0,
    )

    recs = result.raw_records or []
    # 2 instances x 2 rounds x [branch-a: 4 turns + branch-b: 2 turns] = 24 requests.
    assert len(recs) == 24, f"expected 24 child requests, got {len(recs)}"
    # The request-free spine issues nothing: every wire record is a spawned child.
    assert all(r.metadata.parent_correlation_id is not None for r in recs), (
        "orchestrator spine must not appear on the wire"
    )

    # Concurrent identity: the two graph instances are separable by
    # root_correlation_id, and every request is uniquely attributable to
    # (instance, round-branch, node) even though conversation_id repeats across
    # instances -- so per-round latency stays reconstructable with no shared state.
    roots = {r.metadata.root_correlation_id for r in recs}
    assert len(roots) == 2, f"expected 2 distinct graph instances, got {len(roots)}"
    attribution = {
        (
            r.metadata.root_correlation_id,
            r.metadata.conversation_id,
            r.metadata.turn_index,
        )
        for r in recs
    }
    assert len(attribution) == 24, (
        f"every request must be uniquely attributable; got {len(attribution)} keys"
    )

    texts = [_payload_text(r) for r in recs]
    joined = " ".join(texts)
    # Each round fired its OWN distinct authored payloads -- NOT one repeated
    # template (which would emit only t0-* content twice).
    for sentinel in ("t0-a-u0", "t1-a-u0", "t0-b-u0", "t1-b-u0"):
        assert sentinel in joined, f"missing per-round payload {sentinel!r} on the wire"

    # Payload isolation (message_array_with_responses): the turn carrying u1 sends
    # only its own authored array -- it must NOT accumulate turn 0's u0.
    u1_texts = [t for t in texts if "t0-a-u1" in t]
    assert u1_texts, "expected a wire record for branch t0-a turn 1"
    assert all("t0-a-u0" not in t for t in u1_texts), (
        "payload isolation violated: turn 1 accumulated turn 0's content"
    )

    # Per-node system prompts: a LATER turn's OWN distinct system prompt reaches
    # the wire (each round-turn authors its own; not just turn 0).
    assert any("t0-a-sys3" in t for t in texts), (
        "per-turn system prompt on a non-first turn did not reach the wire"
    )

    # Verbatim multimodal: a typed projection_embedding block is sent as a
    # STRUCTURED object (not stringified / field-dropped), field-complete.
    def _proj_blocks(rec):
        for m in (rec.payload or {}).get("messages", []):
            content = m.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "projection_embedding":
                        yield c

    projs = [c for r in recs for c in _proj_blocks(r)]
    assert projs, "no projection_embedding block reached the wire"
    # Full deep-equal: the typed block is preserved byte-for-byte with no field
    # loss (projection_model / inputs[].{name,dtype,shape,data} / kwargs).
    assert projs[0] == {
        "type": "projection_embedding",
        "projection_model": "visual_tokens_binarized",
        "inputs": [
            {
                "name": "visual_tokens",
                "dtype": "float32",
                "shape": [512, 512],
                "data": "AAAA",
            }
        ],
        "kwargs": {"input_dimension": [512, 512]},
    }

    # Deep-equal (acceptance criterion): a turn's wire messages == its authored
    # array exactly -- no accumulation, no field loss, no injected messages.
    t0a_turn1 = next(
        r
        for r in recs
        if r.metadata.conversation_id == "t0-a" and r.metadata.turn_index == 1
    )
    assert t0a_turn1.payload["messages"] == [
        {"role": "system", "content": "t0-a-sys1"},
        {"role": "user", "content": "t0-a-u1"},
    ]

    # 2 instances x 2 rounds x 2 branches = 8 children, all completed (no hang/over-fire).
    bs = result.json.branch_stats if result.json else None
    assert bs is not None
    assert bs.children_spawned == 8
    assert bs.children_completed == 8
    assert bs.children_errored == 0
    # Graph-admission / END events: both instances admitted and both reached END
    # (equal counts => every admitted graph completed; completion is reconstructable).
    assert bs.graphs_admitted == 2
    assert bs.graphs_completed_to_end == 2
