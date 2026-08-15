# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproducibly emit synthetic Dynamo nested-subagent trace fixtures."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

MODEL = "synthetic-llm"

FIXTURES_DIR = Path(__file__).resolve().parent


# --- record builders ------------------------------------------------------


def _ctx(*, session_id: str, parent_session_id: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "session_id": session_id,
    }
    if parent_session_id is not None:
        out["parent_session_id"] = parent_session_id
    return out


def request_end(
    *,
    ts: int,
    session_id: str,
    parent_session_id: str | None = None,
    request_id: str | None = None,
    input_tokens: int = 32,
    output_tokens: int = 16,
    cached_tokens: int = 0,
    ttft_ms: float = 50.0,
) -> dict[str, Any]:
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": _ctx(
            session_id=session_id, parent_session_id=parent_session_id
        ),
        "request": {
            "request_id": request_id or f"{session_id}-r{ts}",
            "model": MODEL,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "ttft_ms": ttft_ms,
        },
    }


def tool_start(
    *,
    ts: int,
    session_id: str,
    parent_session_id: str | None = None,
    tool_call_id: str = "tc",
    tool_class: str = "search",
) -> dict[str, Any]:
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "tool_start",
        "event_time_unix_ms": ts,
        "event_source": "harness",
        "agent_context": _ctx(
            session_id=session_id, parent_session_id=parent_session_id
        ),
        "tool": {
            "tool_call_id": tool_call_id,
            "tool_class": tool_class,
            "started_at_unix_ms": ts,
            "status": "running",
        },
    }


def tool_end(
    *,
    ts: int,
    session_id: str,
    parent_session_id: str | None = None,
    tool_call_id: str = "tc",
    tool_class: str = "search",
    duration_ms: float = 30.0,
    status: str = "succeeded",
) -> dict[str, Any]:
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "tool_end",
        "event_time_unix_ms": ts,
        "event_source": "harness",
        "agent_context": _ctx(
            session_id=session_id, parent_session_id=parent_session_id
        ),
        "tool": {
            "tool_call_id": tool_call_id,
            "tool_class": tool_class,
            "ended_at_unix_ms": ts,
            "duration_ms": duration_ms,
            "status": status,
        },
    }


# --- declarative chain helpers --------------------------------------------


@dataclass(frozen=True)
class SessionSpec:
    """Declarative description of one session's request_end chain."""

    session_id: str
    parent_session_id: str | None
    base_ts: int
    n_turns: int


def request_ends_for(spec: SessionSpec) -> list[dict[str, Any]]:
    """Emit ``n_turns`` request_end records 100 ms apart from ``base_ts``."""
    return [
        request_end(
            ts=spec.base_ts + 100 * i,
            session_id=spec.session_id,
            parent_session_id=spec.parent_session_id,
        )
        for i in range(spec.n_turns)
    ]


def write_fixture(name: str, records: list[dict[str, Any]]) -> Path:
    """Sort by event_time (Dynamo doesn't guarantee global order) and write."""
    out = FIXTURES_DIR / name
    records_sorted = sorted(records, key=lambda r: r["event_time_unix_ms"])
    with gzip.open(out, "wb") as f:
        for r in records_sorted:
            f.write(orjson.dumps(r))
            f.write(b"\n")
    return out


# --- fixtures -------------------------------------------------------------


def build_nested_2_level() -> Path:
    """A: 3 turns. B (parent=A): 2 turns; B's first request_end inside A.K=2 window."""
    a = SessionSpec(
        session_id="sess_A",
        parent_session_id=None,
        base_ts=1000,
        n_turns=3,
    )
    b_records = [
        request_end(ts=1150, session_id="sess_B", parent_session_id="sess_A"),
        request_end(ts=1170, session_id="sess_B", parent_session_id="sess_A"),
    ]
    records = request_ends_for(a) + b_records
    return write_fixture("nested_2_level.jsonl.gz", records)


def build_nested_3_level() -> Path:
    """A -> B -> C; B at A.K=2, C at B.K=1."""
    a_records = request_ends_for(
        SessionSpec(
            session_id="sess_A",
            parent_session_id=None,
            base_ts=1000,
            n_turns=3,
        )
    )
    b_records = [
        request_end(ts=1150, session_id="sess_B", parent_session_id="sess_A"),
        request_end(ts=1180, session_id="sess_B", parent_session_id="sess_A"),
    ]
    c_records = [
        request_end(ts=1160, session_id="sess_C", parent_session_id="sess_B"),
    ]
    return write_fixture("nested_3_level.jsonl.gz", a_records + b_records + c_records)


def build_mixed_turn() -> Path:
    """A.K=2 has BOTH A's own tool events AND a subagent B's first request_end."""
    a_records = request_ends_for(
        SessionSpec(
            session_id="sess_A",
            parent_session_id=None,
            base_ts=1000,
            n_turns=3,
        )
    )
    # A.K=2 window = (1100, 1200)
    parent_tool_events = [
        tool_start(
            ts=1110,
            session_id="sess_A",
            tool_call_id="local_search_call_xyz",
            tool_class="local_search",
        ),
        tool_end(
            ts=1140,
            session_id="sess_A",
            tool_call_id="local_search_call_xyz",
            tool_class="local_search",
            duration_ms=30.0,
        ),
    ]
    b_records = [
        request_end(ts=1150, session_id="sess_B", parent_session_id="sess_A"),
        request_end(ts=1170, session_id="sess_B", parent_session_id="sess_A"),
    ]
    records = a_records + parent_tool_events + b_records
    return write_fixture("mixed_turn.jsonl.gz", records)


def build_cycle_AB_A() -> Path:
    """Malformed: A claims parent=B, B claims parent=A."""
    records: list[dict[str, Any]] = []
    for i in range(3):
        records.append(
            request_end(
                ts=1000 + i * 100,
                session_id="sess_A",
                parent_session_id="sess_B",
            )
        )
    for i in range(2):
        records.append(
            request_end(
                ts=1150 + i * 30,
                session_id="sess_B",
                parent_session_id="sess_A",
            )
        )
    return write_fixture("cycle_AB_A.jsonl.gz", records)


def build_parallel_subagents() -> Path:
    """A invokes B and C from the same parent turn (K=2)."""
    a_records = request_ends_for(
        SessionSpec(
            session_id="sess_A",
            parent_session_id=None,
            base_ts=1000,
            n_turns=3,
        )
    )
    b_records = [
        request_end(ts=1130, session_id="sess_B", parent_session_id="sess_A"),
        request_end(ts=1160, session_id="sess_B", parent_session_id="sess_A"),
    ]
    c_records = [
        request_end(ts=1140, session_id="sess_C", parent_session_id="sess_A"),
        request_end(ts=1170, session_id="sess_C", parent_session_id="sess_A"),
    ]
    return write_fixture(
        "parallel_subagents.jsonl.gz", a_records + b_records + c_records
    )


def build_tool_call_id_linkage() -> Path:
    """A invokes B, with parent A's tool_start.tool_call_id naming B's session_id."""
    a_records = request_ends_for(
        SessionSpec(
            session_id="sess_A",
            parent_session_id=None,
            base_ts=1000,
            n_turns=3,
        )
    )
    parent_tool_events = [
        tool_start(
            ts=1110,
            session_id="sess_A",
            tool_call_id="subagent:sess_B:invoke",
            tool_class="subagent_call",
        ),
        tool_end(
            ts=1190,
            session_id="sess_A",
            tool_call_id="subagent:sess_B:invoke",
            tool_class="subagent_call",
            duration_ms=80.0,
        ),
    ]
    b_records = [
        request_end(ts=1130, session_id="sess_B", parent_session_id="sess_A"),
        request_end(ts=1160, session_id="sess_B", parent_session_id="sess_A"),
    ]
    return write_fixture(
        "tool_call_id_linkage.jsonl.gz",
        a_records + parent_tool_events + b_records,
    )


def build_parallel_two_root() -> Path:
    """Two parentless (root) sessions in one file."""
    a_records = request_ends_for(
        SessionSpec(
            session_id="sess_A",
            parent_session_id=None,
            base_ts=1000,
            n_turns=2,
        )
    )
    b_records = request_ends_for(
        SessionSpec(
            session_id="sess_B",
            parent_session_id=None,
            base_ts=1010,
            n_turns=2,
        )
    )
    return write_fixture("parallel_two_root.jsonl.gz", a_records + b_records)


def build_all() -> list[Path]:
    return [
        build_nested_2_level(),
        build_nested_3_level(),
        build_mixed_turn(),
        build_cycle_AB_A(),
        build_parallel_subagents(),
        build_tool_call_id_linkage(),
        build_parallel_two_root(),
    ]


if __name__ == "__main__":
    paths = build_all()
    for p in paths:
        print(p)
