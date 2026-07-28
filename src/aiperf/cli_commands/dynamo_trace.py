# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for converting Dynamo request traces into Weka traces."""

from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson
from cyclopts import App
from rich.console import Console

from aiperf.common.finite import is_finite_value
from aiperf.dataset.loader.weka_trace_models import WekaTrace

app = App(name="dynamo-trace")

_DYNAMO_TRACE_V1 = "dynamo.request.trace.v1"


@app.default
def dynamo_trace(input_file: Path, *, output: Path) -> None:
    """Convert a native Dynamo request trace v1 into Weka trace files.

    Args:
        input_file: Native Dynamo JSONL or JSONL.GZ request trace.
        output: Empty directory for the generated Weka trace files.
    """
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Output path must be a directory: {output}")
        if any(output.iterdir()):
            raise ValueError(f"Output directory must be empty: {output}")

    traces = _dynamo_traces_to_weka(input_file)
    output.mkdir(parents=True, exist_ok=True)
    for index, trace in enumerate(traces):
        (output / f"trace_{index:06d}.json").write_bytes(
            orjson.dumps(
                trace.model_dump(by_alias=True, exclude_none=True),
                option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
            )
        )
    Console().print(f"[green]Weka traces: {len(traces)} written to {output}[/green]")


def _non_negative_int(value: Any, field: str, line_number: int) -> int:
    """Return a required non-negative integer trace field."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Line {line_number}: {field} must be a non-negative integer")
    return value


def _non_negative_number(
    value: Any, field: str, line_number: int, *, required: bool = False
) -> float | None:
    """Return an optional finite non-negative timing field."""
    if value is None:
        if required:
            raise ValueError(f"Line {line_number}: {field} is required")
        return None
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not is_finite_value(value)
        or value < 0
    ):
        raise ValueError(
            f"Line {line_number}: {field} must be a finite non-negative number"
        )
    return float(value)


def _dynamo_event(record: Any) -> dict[str, Any] | None:
    """Extract a native event from raw and wrapped Dynamo JSONL rows."""
    if not isinstance(record, dict):
        return None
    event = record.get("event", record)
    return event if isinstance(event, dict) else None


def _session_ids(
    context: Any, request: dict[str, Any], line_number: int
) -> tuple[str, str | None]:
    """Return validated Dynamo session linkage or a context-free request ID."""
    if context is not None and not isinstance(context, dict):
        raise ValueError(f"Line {line_number}: agent_context must be an object")
    session_id = context.get("session_id") if context else None
    if session_id is not None and (not isinstance(session_id, str) or not session_id):
        raise ValueError(f"Line {line_number}: session_id must be a non-empty string")
    parent_session_id = context.get("parent_session_id") if context else None
    if parent_session_id is not None and (
        not isinstance(parent_session_id, str) or not parent_session_id
    ):
        raise ValueError(
            f"Line {line_number}: parent_session_id must be a non-empty string"
        )
    if session_id is None:
        request_id = request.get("request_id")
        session_id = (
            f"request-{request_id}"
            if isinstance(request_id, str) and request_id
            else f"request-line-{line_number}"
        )
    return session_id, parent_session_id


def _parse_dynamo_row(record: Any, line_number: int) -> dict[str, Any] | None:
    """Convert one replayable Dynamo request-end event into an internal row."""
    event = _dynamo_event(record)
    if event is None or event.get("event_type") != "request_end":
        return None
    if event.get("schema") != _DYNAMO_TRACE_V1:
        raise ValueError(f"Line {line_number}: unsupported Dynamo trace schema")

    request = event.get("request")
    replay = request.get("replay") if isinstance(request, dict) else None
    if not isinstance(request, dict) or not isinstance(replay, dict):
        raise ValueError(f"Line {line_number}: request is missing replay metadata")

    session_id, parent_session_id = _session_ids(
        event.get("agent_context"), request, line_number
    )

    block_size = _non_negative_int(
        replay.get("trace_block_size"), "trace_block_size", line_number
    )
    if block_size == 0:
        raise ValueError(f"Line {line_number}: trace_block_size must be positive")
    input_length = _non_negative_int(
        replay.get("input_length"), "input_length", line_number
    )
    output_length = _non_negative_int(
        request.get("output_tokens"), "output_tokens", line_number
    )
    hashes = replay.get("input_sequence_hashes")
    if not isinstance(hashes, list) or not all(
        isinstance(hash_id, int) and not isinstance(hash_id, bool) for hash_id in hashes
    ):
        raise ValueError(f"Line {line_number}: input_sequence_hashes must be integers")
    expected_hashes = (input_length + block_size - 1) // block_size
    if len(hashes) != expected_hashes:
        raise ValueError(
            f"Line {line_number}: input_length requires {expected_hashes} replay hashes"
        )

    return {
        "session_id": session_id,
        "parent_session_id": parent_session_id,
        "received_ms": _non_negative_number(
            request.get("request_received_ms"),
            "request_received_ms",
            line_number,
            required=True,
        ),
        "block_size": block_size,
        "input_length": input_length,
        "output_length": output_length,
        "hashes": hashes,
        "model": request.get("model")
        if isinstance(request.get("model"), str) and request["model"]
        else "dynamo-trace",
        "total_time_ms": _non_negative_number(
            request.get("total_time_ms"), "total_time_ms", line_number
        ),
    }


def _load_dynamo_rows(input_file: Path) -> list[dict[str, Any]]:
    """Load replayable request-end rows from one Dynamo JSONL file."""
    if not input_file.is_file():
        raise ValueError(f"Dynamo trace does not exist: {input_file}")

    rows: list[dict[str, Any]] = []
    open_file = gzip.open if input_file.suffix == ".gz" else Path.open
    with open_file(input_file, "rb") as trace_file:
        for line_number, line in enumerate(trace_file, 1):
            if not line.strip():
                continue
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {input_file}"
                ) from error
            row = _parse_dynamo_row(record, line_number)
            if row is not None:
                rows.append(row)
    if not rows:
        raise ValueError(f"No replayable requests found in {input_file}")
    return sorted(rows, key=lambda row: row["received_ms"])


def _weka_request(
    row: dict[str, Any], *, origin_ms: float, final: bool
) -> dict[str, Any]:
    """Convert one Dynamo request into a Weka normal request."""
    request = {
        "t": round((row["received_ms"] - origin_ms) / 1000, 6),
        "type": "n",
        "model": row["model"],
        "in": row["input_length"],
        "out": row["output_length"],
        "hash_ids": row["hashes"],
        "stop": "end_turn" if final else "tool_use",
    }
    if row["total_time_ms"] is not None:
        request["api_time"] = row["total_time_ms"] / 1000
    return request


def _weka_subagent(
    session_id: str, rows: list[dict[str, Any]], *, origin_ms: float
) -> dict[str, Any]:
    """Convert one direct Dynamo child session into a Weka subagent."""
    requests = [
        _weka_request(row, origin_ms=origin_ms, final=index == len(rows) - 1)
        for index, row in enumerate(rows)
    ]
    end_ms = max(row["received_ms"] + (row["total_time_ms"] or 0) for row in rows)
    return {
        "t": requests[0]["t"],
        "type": "subagent",
        "agent_id": session_id,
        "subagent_type": "agent",
        "duration_ms": round(end_ms - rows[0]["received_ms"]),
        "total_tokens": sum(row["input_length"] + row["output_length"] for row in rows),
        "status": "completed",
        "requests": requests,
        "models": sorted({row["model"] for row in rows}),
    }


def _dynamo_traces_to_weka(input_file: Path) -> list[WekaTrace]:
    """Convert roots and direct ``agent_context`` children into Weka traces."""
    rows = _load_dynamo_rows(input_file)
    origin_ms = rows[0]["received_ms"]
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sessions[row["session_id"]].append(row)

    parents: dict[str, str | None] = {}
    for session_id, session_rows in sessions.items():
        parent_ids = {row["parent_session_id"] for row in session_rows}
        if len(parent_ids) != 1:
            raise ValueError(f"Session has inconsistent parents: {session_id}")
        parent_session_id = next(iter(parent_ids))
        if parent_session_id == session_id:
            raise ValueError(f"Session cannot parent itself: {session_id}")
        if parent_session_id is not None and parent_session_id not in sessions:
            raise ValueError(
                f"Session {session_id} references missing parent {parent_session_id}"
            )
        parents[session_id] = parent_session_id

    for session_id, parent_session_id in parents.items():
        if parent_session_id is not None and parents[parent_session_id] is not None:
            raise ValueError(
                "Nested Dynamo subagents are not representable in a Weka trace: "
                f"{session_id}"
            )

    traces: list[WekaTrace] = []
    root_ids = sorted(
        (
            session_id
            for session_id, parent_session_id in parents.items()
            if parent_session_id is None
        ),
        key=lambda session_id: sessions[session_id][0]["received_ms"],
    )
    for root_id in root_ids:
        child_ids = sorted(
            (
                session_id
                for session_id, parent_session_id in parents.items()
                if parent_session_id == root_id
            ),
            key=lambda session_id: sessions[session_id][0]["received_ms"],
        )
        selected_rows = [
            row for session_id in [root_id, *child_ids] for row in sessions[session_id]
        ]
        block_sizes = {row["block_size"] for row in selected_rows}
        if len(block_sizes) != 1:
            raise ValueError(f"Lineage has multiple trace block sizes: {root_id}")
        block_size = next(iter(block_sizes))
        timeline = [
            (
                row["received_ms"],
                0,
                index,
                _weka_request(
                    row,
                    origin_ms=origin_ms,
                    final=index == len(sessions[root_id]) - 1,
                ),
            )
            for index, row in enumerate(sessions[root_id])
        ]
        timeline.extend(
            (
                sessions[child_id][0]["received_ms"],
                1,
                index,
                _weka_subagent(child_id, sessions[child_id], origin_ms=origin_ms),
            )
            for index, child_id in enumerate(child_ids)
        )
        traces.append(
            WekaTrace.model_validate(
                {
                    "id": root_id,
                    "models": sorted({row["model"] for row in selected_rows}),
                    "block_size": block_size,
                    "hash_id_scope": "local",
                    "requests": [entry[3] for entry in sorted(timeline)],
                }
            )
        )
    return traces
