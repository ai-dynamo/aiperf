# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`aiperf dynamo trace-report` against the REAL `dynamo.request.trace.v1` schema.

The original report was written against a nonexistent workflow/program schema
(WK1): it passed a `workflow_id=` kwarg `iter_trace_records` does not accept
and read `agent_context.workflow_id`/`.program_id` fields `AgentContext` does
not have -- dead on arrival for every trace the in-repo reader parses. These
tests drive `aggregate_by_session` end-to-end over real minimal trace files:
session grouping, parent linkage, replay-only skip accounting, the reader-level
session filter, and nearest-rank percentile math.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import orjson
import pytest
from pytest import param

from aiperf.cli_commands.dynamo_trace_report import (
    _format_csv,
    _format_json,
    _percentiles,
    aggregate_by_session,
)


def _request_end(
    *,
    session_id: str | None,
    parent_session_id: str | None = None,
    model: str | None = "test-model",
    ts_ms: int = 1_000,
    ttft_ms: float | None = None,
    input_tokens: int | None = None,
    replay_hashes: list[int] | None = None,
) -> dict[str, Any]:
    """A minimal bare `request_end` record the reader accepts.

    `session_id=None` omits `agent_context` entirely (a replay-only record).
    """
    request: dict[str, Any] = {"request_id": f"r-{ts_ms}"}
    if model is not None:
        request["model"] = model
    if ttft_ms is not None:
        request["ttft_ms"] = ttft_ms
    if input_tokens is not None:
        request["input_tokens"] = input_tokens
    if replay_hashes is not None:
        request["replay"] = {
            "trace_block_size": 16,
            "input_length": 32,
            "input_sequence_hashes": replay_hashes,
        }
    record: dict[str, Any] = {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts_ms,
        "request": request,
    }
    if session_id is not None:
        ctx: dict[str, Any] = {"session_id": session_id}
        if parent_session_id is not None:
            ctx["parent_session_id"] = parent_session_id
        record["agent_context"] = ctx
    return record


def _write_trace(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    trace = tmp_path / "trace.jsonl"
    trace.write_bytes(b"\n".join(orjson.dumps(r) for r in records) + b"\n")
    return trace


def test_aggregate_by_session_groups_and_links_parent(tmp_path: Path) -> None:
    """Minimal valid request_end records -> per-session rows with parent links."""
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-root", ts_ms=1_000, ttft_ms=5.0),
            _request_end(session_id="s-root", ts_ms=3_000, ttft_ms=7.0),
            _request_end(
                session_id="s-child",
                parent_session_id="s-root",
                ts_ms=2_000,
                ttft_ms=9.0,
                replay_hashes=[11, 22],
            ),
        ],
    )
    report = aggregate_by_session(trace)

    assert report.skipped_no_agent_context == 0
    assert [row["session_id"] for row in report.rows] == ["s-child", "s-root"]

    child, root = report.rows
    assert root["parent_session_id"] is None
    assert root["request_count"] == 2
    assert root["child_session_count"] == 1
    assert root["time_range_ms"] == [1_000, 3_000]
    assert root["models"] == ["test-model"]
    assert root["metrics"]["ttft_ms"]["count"] == 2.0
    assert root["metrics"]["ttft_ms"]["mean"] == pytest.approx(6.0)

    assert child["parent_session_id"] == "s-root"
    assert child["parent_session_id_conflict"] is False
    assert child["child_session_count"] == 0
    assert child["replay_records"] == 1
    assert child["unique_replay_hashes"] == 2


def test_duplicate_record_across_dual_sink_files_folded_once(tmp_path: Path) -> None:
    """Dynamo's dual file sinks can write the SAME record into two files of one
    capture dir (`discover_segments` reads ALL *.jsonl + *.jsonl.gz); the
    aggregate must fold it once and count the duplicate, matching the chain
    parser's ("request_end", session_id, request_id) dedup identity."""
    rec = _request_end(session_id="s-a", ts_ms=1_000, ttft_ms=5.0)
    envelope = orjson.dumps({"timestamp": 5, "event": rec}) + b"\n"
    (tmp_path / "trace.jsonl").write_bytes(envelope)
    with gzip.open(tmp_path / "trace.000000.jsonl.gz", "wb") as f:
        f.write(envelope)

    report = aggregate_by_session(tmp_path)
    assert [row["session_id"] for row in report.rows] == ["s-a"]
    assert report.rows[0]["request_count"] == 1
    assert report.rows[0]["metrics"]["ttft_ms"]["count"] == 1.0
    assert report.duplicate_records == 1
    assert orjson.loads(_format_json(report))["duplicate_records"] == 1


def test_distinct_request_ids_are_not_deduped(tmp_path: Path) -> None:
    """Two records of one session with different request_ids both fold in."""
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-a", ts_ms=1_000),
            _request_end(session_id="s-a", ts_ms=2_000),
        ],
    )
    report = aggregate_by_session(trace)
    assert report.duplicate_records == 0
    assert report.rows[0]["request_count"] == 2


def test_replay_only_records_skipped_with_counter(tmp_path: Path) -> None:
    """Records without agent_context produce no rows but are counted."""
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id=None, ts_ms=1_000),
            _request_end(session_id=None, ts_ms=2_000),
            _request_end(session_id="s-live", ts_ms=3_000),
        ],
    )
    report = aggregate_by_session(trace)
    assert report.skipped_no_agent_context == 2
    assert [row["session_id"] for row in report.rows] == ["s-live"]


def test_session_filter_uses_reader_session_id_param(tmp_path: Path) -> None:
    """The --session-id filter is pushed down to the reader's parse-time filter."""
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-a", ts_ms=1_000),
            _request_end(session_id="s-b", ts_ms=2_000),
        ],
    )
    report = aggregate_by_session(trace, session_id="s-a")
    assert [row["session_id"] for row in report.rows] == ["s-a"]
    assert report.rows[0]["request_count"] == 1


def test_model_none_excluded_from_model_set(tmp_path: Path) -> None:
    """`request.model` is optional in the schema; None must not enter the model set."""
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-a", model=None, ts_ms=1_000),
            _request_end(session_id="s-a", model="test-model", ts_ms=2_000),
        ],
    )
    report = aggregate_by_session(trace)
    assert report.rows[0]["models"] == ["test-model"]


def test_limit_stops_new_sessions_but_keeps_existing(tmp_path: Path) -> None:
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-a", ts_ms=1_000),
            _request_end(session_id="s-b", ts_ms=2_000),
            _request_end(session_id="s-a", ts_ms=3_000),
        ],
    )
    report = aggregate_by_session(trace, limit=1)
    assert [row["session_id"] for row in report.rows] == ["s-a"]
    assert report.rows[0]["request_count"] == 2


@pytest.mark.parametrize(
    "values, stat, expected",
    [
        param([1.0, 2.0, 3.0, 4.0], "p50", 2.0, id="p50_even_n_nearest_rank"),
        param([1.0, 2.0, 3.0, 4.0], "p90", 4.0, id="p90_even_n"),
        param([1.0, 2.0, 3.0], "p50", 2.0, id="p50_odd_n"),
        param([1.0, 2.0, 3.0, 4.0, 5.0], "p99", 5.0, id="p99_top_rank"),
        param([7.0], "p50", 7.0, id="single_value_all_stats"),
        param([1.0, 2.0, 3.0, 4.0], "min", 1.0, id="min"),
        param([1.0, 2.0, 3.0, 4.0], "max", 4.0, id="max"),
        param([1.0, 2.0, 3.0, 4.0], "mean", 2.5, id="mean"),
        param([1.0, 2.0, 3.0, 4.0], "count", 4.0, id="count"),
    ],
)  # fmt: skip
def test_percentiles_nearest_rank(
    values: list[float], stat: str, expected: float
) -> None:
    """Nearest-rank percentiles: rank = ceil(p/100 * n), 1-based.

    The original implementation used int(p/100 * n) as a 0-based index, which
    is off by one (p50 of [1,2,3,4] returned 3 instead of 2).
    """
    assert _percentiles(values)[stat] == pytest.approx(expected)


def test_percentiles_empty_returns_empty_dict() -> None:
    assert _percentiles([]) == {}


def test_format_json_envelope_carries_skip_counter(tmp_path: Path) -> None:
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-a", ts_ms=1_000),
            _request_end(session_id=None, ts_ms=2_000),
        ],
    )
    report = aggregate_by_session(trace)
    payload = orjson.loads(_format_json(report))
    assert payload["skipped_no_agent_context"] == 1
    assert [s["session_id"] for s in payload["sessions"]] == ["s-a"]


def test_format_csv_headers_and_rows(tmp_path: Path) -> None:
    trace = _write_trace(
        tmp_path,
        [_request_end(session_id="s-a", ts_ms=1_000, ttft_ms=5.0, input_tokens=10)],
    )
    report = aggregate_by_session(trace)
    lines = _format_csv(report.rows).strip().splitlines()
    assert len(lines) == 2
    header = lines[0].split(",")
    assert header[0] == "session_id"
    assert "parent_session_id" in header
    assert "ttft_ms_p50" in header
    assert lines[1].startswith("s-a,")
