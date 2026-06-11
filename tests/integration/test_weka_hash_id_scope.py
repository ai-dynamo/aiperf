# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end benchmark that stress-tests the Weka ``hash_id_scope: "local"``
contract on the wire.

A Weka trace declares one hash_id namespace per trace FILE: the same hash_id
must decode to identical tokens across the parent conversation and every
subagent (spawn-mode child) conversation of that trace. This is what lets
replay reproduce the cross-agent shared prefixes a real inference server serves
from KV cache.

The crafted trace below has a parent turn and TWO sibling subagents whose inner
requests reference the EXACT same hash_id blocks as the parent's first turn
(``[10, 11, 12]``), with ``tool_tokens == system_tokens == 0`` and
``in == n*block_size`` so each first-turn prompt is purely the decoded blocks.
Under the correct (shared) scope, all three first-turn requests render to
byte-identical prompt text on the wire. A per-child decode scope (the bug this
guards against) would decode the shared blocks under different seeds, so the
sibling payloads -- and the parent vs child payloads -- would diverge.

We run the real ``aiperf profile`` subprocess against the mock server, export
raw records (``--export-level raw``), and assert on the ACTUAL request payloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

BLOCK_SIZE = 64
SHARED_HASH_IDS = [10, 11, 12]
SHARED_IN = BLOCK_SIZE * len(SHARED_HASH_IDS)  # exact tile -> no partial tail


def _normal(t, in_tokens, hash_ids, *, stop="end_turn", out=32):
    return {
        "t": t,
        "type": "n",
        "model": "test-model",
        "in": in_tokens,
        "out": out,
        "hash_ids": hash_ids,
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": stop,
        "api_time": 1.0,
        "think_time": 0.0,
    }


def _subagent(agent_id, t):
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "Explore",
        "duration_ms": 1000,
        "total_tokens": 100,
        "tool_use_count": 1,
        "status": "completed",
        "requests": [_normal(0.0, SHARED_IN, SHARED_HASH_IDS)],
        "models": ["test-model"],
        "tool_tokens": 0,
        "system_tokens": 0,
    }


def _text_of(msg: dict) -> str | None:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = [
            p["text"]
            for p in c
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        ]
        return "".join(parts) if parts else None
    return None


def _last_user_text(messages: list[dict]) -> str | None:
    users = [m for m in messages if m.get("role") == "user"]
    return _text_of(users[-1]) if users else None


@pytest.mark.integration
@pytest.mark.asyncio
class TestWekaHashIdScopeEndToEnd:
    async def test_subagents_share_parent_hash_id_scope_on_the_wire(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Isolate the content-addressed mmap dataset cache to this run. The cache
        # key is (input bytes, tokenizer, prompt/input settings) with no source
        # component, so a global cache could otherwise replay a dataset built by
        # a DIFFERENT loader version for the same trace -- masking a scope
        # regression. A per-test dir forces a fresh load of the current code.
        monkeypatch.setenv(
            "AIPERF_DATASET_MMAP_CACHE_DIR", str(tmp_path / "mmap_cache")
        )

        trace = {
            "id": "scope_stress",
            "models": ["test-model"],
            "block_size": BLOCK_SIZE,
            "hash_id_scope": "local",
            "tool_tokens": 0,
            "system_tokens": 0,
            "requests": [
                _normal(0.0, SHARED_IN, SHARED_HASH_IDS, stop="tool_use"),
                _subagent("agent_001", 2.0),
                _subagent("agent_002", 3.0),
                # A later parent turn so the subagents join (are dispatched)
                # rather than being dropped for lack of a following turn.
                _normal(20.0, BLOCK_SIZE * 4, [10, 11, 12, 13]),
            ],
        }
        trace_file = tmp_path / "scope_stress.json"
        trace_file.write_text(json.dumps(trace))

        result = await cli.run(
            f"""
            aiperf profile \
                --model test-model \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {trace_file} \
                --custom-dataset-type weka_trace \
                --request-count 1 \
                --concurrency 1 \
                --workers-max 1 \
                --export-level raw \
                --ui simple
            """,
            timeout=300.0,
        )

        assert result.raw_records is not None, (
            "profile_export_raw.jsonl must exist when --export-level raw is set"
        )

        roots = [
            r for r in result.raw_records if r.metadata.parent_correlation_id is None
        ]
        kids = [
            r
            for r in result.raw_records
            if r.metadata.parent_correlation_id is not None
        ]

        # Both sibling subagents must have been dispatched as spawn children.
        assert len(kids) == 2, (
            f"expected 2 spawn-child records, got {len(kids)}: "
            f"{[_last_user_text(r.payload.get('messages', [])) for r in kids]}"
        )

        # CORE SCOPE GUARD: the two siblings reference identical hash_id blocks,
        # so under one shared trace scope they render byte-identical payloads.
        # A per-child decode scope would make these diverge.
        assert kids[0].payload["messages"] == kids[1].payload["messages"], (
            "sibling subagents referencing the same hash_ids must render "
            "identical prompts -- they share the parent trace's hash_id scope"
        )

        # PARENT<->CHILD SHARING: the subagents reference the parent's turn-0
        # blocks exactly, so the child's user prompt must equal the parent's
        # turn-0 user prompt (the shared blocks decode identically).
        parent_turn0 = min(roots, key=lambda r: len(r.payload["messages"]))
        assert _last_user_text(kids[0].payload["messages"]) == _last_user_text(
            parent_turn0.payload["messages"]
        ), (
            "a subagent reusing the parent's hash_id blocks must decode them to "
            "the same prompt text as the parent (shared hash_id scope)"
        )
