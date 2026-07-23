# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end benchmark that stress-tests the Weka ``hash_id_scope: "local"``"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

BLOCK_SIZE = 64
SHARED_HASH_IDS = [10, 11, 12]
SHARED_IN = BLOCK_SIZE * len(SHARED_HASH_IDS)


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
                _normal(4.0, BLOCK_SIZE * 4, [10, 11, 12, 13]),
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
                --no-fixed-schedule \
                --benchmark-duration 12 \
                --benchmark-grace-period 20 \
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

        roots_by_corr: dict[str, list] = defaultdict(list)
        kids_by_parent: dict[str, list] = defaultdict(list)
        for r in result.raw_records:
            md = r.metadata
            if md.parent_correlation_id is None:
                assert md.x_correlation_id is not None
                roots_by_corr[md.x_correlation_id].append(r)
            else:
                kids_by_parent[md.parent_correlation_id].append(r)

        complete_plays = [
            (corr, kids_by_parent[corr])
            for corr in roots_by_corr
            if len(kids_by_parent.get(corr, [])) == 2
        ]
        assert complete_plays, (
            "no play dispatched both sibling subagents as spawn children; "
            f"roots={len(roots_by_corr)}, "
            f"child-counts={ {c: len(k) for c, k in kids_by_parent.items()} }"
        )

        for corr, kids in complete_plays:
            roots = roots_by_corr[corr]

            assert kids[0].payload["messages"] == kids[1].payload["messages"], (
                "sibling subagents referencing the same hash_ids must render "
                "identical prompts -- they share the parent trace's hash_id scope"
            )

            parent_turn0 = min(roots, key=lambda r: len(r.payload["messages"]))
            assert _last_user_text(kids[0].payload["messages"]) == _last_user_text(
                parent_turn0.payload["messages"]
            ), (
                "a subagent reusing the parent's hash_id blocks must decode them "
                "to the same prompt text as the parent (shared hash_id scope)"
            )
