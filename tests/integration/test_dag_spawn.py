# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration test for SPAWN-mode DAG branches.

Unlike FORK mode (which inherits the parent's accumulated messages and pins
the child to the parent's worker), SPAWN-mode children:

- Start with an EMPTY accumulator (no parent context merged in).
- Route freely (no sticky pin to the parent's worker).

This test drives the full aiperf subprocess over a minimal root+spawn-child
fixture and asserts both invariants on the wire payloads and run stats.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "dag" / "spawn_minimal.dag.jsonl"
)

ROOT_SYS = "root-sys"
ROOT_USER = "root-u"
SPAWN_SYS = "spawn-sys"
SPAWN_USER = "spawn-u"


def _text_of(msg: dict) -> str | None:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts) if parts else None
    return None


def _roles_contents(messages: list[dict]) -> list[tuple[str, str | None]]:
    return [(m.get("role"), _text_of(m)) for m in messages]


@pytest.mark.integration
@pytest.mark.asyncio
class TestDagSpawnEndToEnd:
    """End-to-end DAG benchmark exercising SPAWN-mode (fresh-context) branches."""

    async def test_spawn_child_has_fresh_context_and_is_not_sticky_pinned(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
    ):
        assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"

        result = await cli.run(
            f"""
            aiperf profile \
                --model test-model \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {FIXTURE} \
                --custom-dataset-type dag_jsonl \
                --num-conversations 1 \
                --concurrency 1 \
                --workers-max 2 \
                --export-level raw \
                --ui simple
            """,
            timeout=300.0,
        )

        assert result.raw_records is not None, (
            "profile_export_raw.jsonl must exist when --export-level raw is set"
        )
        # Exactly 2 wire requests: root + one spawn-mode child.
        assert len(result.raw_records) == 2, (
            f"Expected 2 raw records, got {len(result.raw_records)}: "
            f"{[r.payload.get('messages', [])[0] for r in result.raw_records]}"
        )

        # Classify by distinguishing system prompt.
        root_rec = None
        child_rec = None
        for rec in result.raw_records:
            first_sys = _text_of(rec.payload.get("messages", [{}])[0])
            if first_sys == ROOT_SYS:
                root_rec = rec
            elif first_sys == SPAWN_SYS:
                child_rec = rec
        assert root_rec is not None, "root record not found"
        assert child_rec is not None, "spawn-mode child record not found"

        # Root's payload is untouched: just its own [sys, user].
        assert _roles_contents(root_rec.payload["messages"]) == [
            ("system", ROOT_SYS),
            ("user", ROOT_USER),
        ]

        # Critical: SPAWN child must NOT inherit root's context. Its messages
        # are exactly its own [sys, user] with no root-* entries and no
        # captured assistant text from root.
        assert _roles_contents(child_rec.payload["messages"]) == [
            ("system", SPAWN_SYS),
            ("user", SPAWN_USER),
        ], (
            "SPAWN-mode child must start with a fresh context (no parent "
            "turn_list inherited)"
        )

        # Parent linkage is still stamped on the child (via Credit.parent_
        # correlation_id) — mode only changes context-inheritance and routing,
        # not the tree-shape bookkeeping.
        assert root_rec.metadata.parent_correlation_id is None
        assert (
            child_rec.metadata.parent_correlation_id
            == root_rec.metadata.x_correlation_id
        )

        # Stats are mode-agnostic: the orchestrator counted one dispatched
        # child and one completed.
        assert result.json is not None, "profile_export_aiperf.json must exist"
        assert result.json.branch_stats is not None
        assert result.json.branch_stats.children_spawned == 1
        assert result.json.branch_stats.children_completed == 1
        assert result.json.branch_stats.children_errored == 0
