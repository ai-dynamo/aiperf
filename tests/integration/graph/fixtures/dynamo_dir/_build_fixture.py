# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproducibly emit the integration dynamo e2e fixture (``trace.jsonl.gz``).

Run to regenerate::

    uv run python tests/integration/graph/fixtures/dynamo_dir/_build_fixture.py

Shape: ONE root session (``parent``, three clean linear ``request_end`` turns
with strictly-nested replay hashes) + ONE subagent (``child``, two nested-hash
turns) spliced at the parent's turn 3 via a ``:subagent:child:invoke`` tool call.
Exactly one root -> avoids the Task-8 multi-root gate.

Each replay hash covers one 16-token block, so ``input_length == 16 * len(hashes)``
stays block-aligned for the ``_assert_block_aligned_isl`` gate at
``trace_block_size=16`` (``(n-1)*16 < input_length <= n*16``). This mirrors the
component-integration fixture in
``tests/component_integration/graph/test_dynamo_e2e_materialize.py``.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import orjson

FIXTURE_DIR = Path(__file__).resolve().parent
MODEL = "m"
_BLOCK_SIZE = 16


def _request_end(
    *,
    ts: int,
    session_id: str,
    hashes: list[int],
    parent_session_id: str | None = None,
) -> dict[str, Any]:
    """One ``request_end`` carrying replay hashes (one 16-token block each)."""
    ctx: dict[str, Any] = {"session_id": session_id}
    if parent_session_id is not None:
        ctx["parent_session_id"] = parent_session_id
    input_length = _BLOCK_SIZE * len(hashes)
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": ctx,
        "request": {
            "request_id": f"{session_id}-{ts}",
            "model": MODEL,
            "input_tokens": input_length,
            "output_tokens": 16,
            "cached_tokens": 0,
            "replay": {
                "trace_block_size": _BLOCK_SIZE,
                "input_length": input_length,
                "input_sequence_hashes": hashes,
            },
        },
    }


def _tool_event(
    *, ts: int, session_id: str, event_type: str, tool_call_id: str
) -> dict[str, Any]:
    tool: dict[str, Any] = {"tool_call_id": tool_call_id, "tool_class": "search"}
    if event_type == "tool_end":
        tool["duration_ms"] = 40.0
        tool["status"] = "succeeded"
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": event_type,
        "event_time_unix_ms": ts,
        "event_source": "harness",
        "agent_context": {"session_id": session_id},
        "tool": tool,
    }


def build_records() -> list[dict[str, Any]]:
    return [
        _request_end(ts=1000, session_id="parent", hashes=[111, 222]),
        _request_end(ts=1100, session_id="parent", hashes=[111, 222, 333]),
        _request_end(ts=1200, session_id="parent", hashes=[111, 222, 333, 444]),
        _tool_event(
            ts=1220,
            session_id="parent",
            event_type="tool_start",
            tool_call_id=":subagent:child:invoke",
        ),
        _tool_event(
            ts=1260,
            session_id="parent",
            event_type="tool_end",
            tool_call_id=":subagent:child:invoke",
        ),
        _request_end(
            ts=1280,
            session_id="child",
            parent_session_id="parent",
            hashes=[900, 901],
        ),
        _request_end(
            ts=1300,
            session_id="child",
            parent_session_id="parent",
            hashes=[900, 901, 902],
        ),
        _request_end(ts=1400, session_id="parent", hashes=[111, 222, 333, 444, 555]),
    ]


def write_fixture() -> Path:
    # Segment naming (`<prefix>.NNNNNN.jsonl.gz`) so `DynamoTraceAdapter.can_load`
    # autoroutes the DIRECTORY input (the plain `*.jsonl.gz` dir glob is only
    # accepted by `discover_segments`, not by the dir can_load sniff).
    out = FIXTURE_DIR / "trace.000000.jsonl.gz"
    records = sorted(build_records(), key=lambda r: r["event_time_unix_ms"])
    with gzip.open(out, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")
    return out


if __name__ == "__main__":
    print(write_fixture())
