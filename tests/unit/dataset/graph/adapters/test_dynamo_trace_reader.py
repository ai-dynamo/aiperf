# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Dynamo agent-trace reader: JSONL/gzip iteration, segment discovery, filtering, sink-envelope parity, and malformed-input rejection."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace_reader import (
    AgentContext,
    AgentReplayMetrics,
    AgentTraceRecord,
    DynamoTraceReadError,
    discover_segments,
    iter_raw_records,
    iter_trace_records,
)
from tests.unit.dataset.graph.adapters.conftest import write_jsonl, write_jsonl_gz


def _request_end(
    *,
    event_time_unix_ms: int = 1_000,
    session_id: str = "sess-1",
    request_id: str = "req-1",
    model: str = "my-model",
    replay: dict | None = None,
) -> dict:
    """Build a minimal well-formed ``request_end`` trace record."""
    rec: dict = {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": event_time_unix_ms,
        "event_source": "dynamo",
        "agent_context": {
            "session_id": session_id,
        },
        "request": {"request_id": request_id, "model": model},
    }
    if replay is not None:
        rec["request"]["replay"] = replay
    return rec


def _tool_event(
    *,
    event_type: str,
    event_time_unix_ms: int = 1_500,
    session_id: str = "sess-1",
    tool_call_id: str = "call-1",
    tool_class: str = "web_search",
    status: str | None = None,
) -> dict:
    """Build a minimal well-formed harness-sourced tool trace record."""
    rec: dict = {
        "schema": "dynamo.request.trace.v1",
        "event_type": event_type,
        "event_time_unix_ms": event_time_unix_ms,
        "event_source": "harness",
        "agent_context": {
            "session_id": session_id,
        },
        "tool": {"tool_call_id": tool_call_id, "tool_class": tool_class},
    }
    if status is not None:
        rec["tool"]["status"] = status
    return rec


def _wrap(rec: dict, ts: int = 12) -> dict:
    """Wrap a record the way both dynamo file sinks do (telemetry/jsonl_gz.rs)."""
    return {"timestamp": ts, "event": rec}


def _write_three_gz_segments(seg_dir: Path) -> None:
    """Lay down ``trace.00000{0,1,2}.jsonl.gz`` each holding one record ``r{n}``."""
    seg_dir.mkdir(parents=True, exist_ok=True)
    for n in range(3):
        write_jsonl_gz(
            seg_dir / f"trace.{n:06d}.jsonl.gz", [_request_end(request_id=f"r{n}")]
        )


def _request_ids(records: list[AgentTraceRecord]) -> list[str]:
    """Request ids of every record that carries a request payload."""
    return [r.request.request_id for r in records if r.request]


def test_iter_plain_jsonl_round_trips_three_records(tmp_path: Path) -> None:
    """Plain .jsonl iteration preserves order and parses request and tool payloads."""
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        [
            _request_end(),
            _tool_event(event_type="tool_start"),
            _tool_event(event_type="tool_end", status="succeeded"),
        ],
    )
    out = list(iter_trace_records(p))
    assert len(out) == 3
    assert out[0].event_type == "request_end"
    assert out[0].request is not None
    assert out[0].request.model == "my-model"
    assert out[1].event_type == "tool_start"
    assert out[1].tool is not None and out[1].tool.tool_call_id == "call-1"
    assert out[2].event_type == "tool_end"
    assert out[2].tool is not None and out[2].tool.status == "succeeded"


def test_iter_gzipped_jsonl_round_trips(tmp_path: Path) -> None:
    """A gzipped .jsonl.gz file yields the same event sequence as its plain form."""
    p = write_jsonl_gz(
        tmp_path / "trace.jsonl.gz",
        [
            _request_end(),
            _tool_event(event_type="tool_start"),
            _tool_event(event_type="tool_end", status="succeeded"),
        ],
    )
    out = list(iter_trace_records(p))
    assert [r.event_type for r in out] == ["request_end", "tool_start", "tool_end"]


def test_segmented_directory_iterates_in_segment_order(tmp_path: Path) -> None:
    """Pointing the reader at a segment directory iterates segments in index order."""
    seg_dir = tmp_path / "segdir"
    _write_three_gz_segments(seg_dir)
    assert _request_ids(list(iter_trace_records(seg_dir))) == ["r0", "r1", "r2"]


def test_segmented_prefix_iterates_in_segment_order(tmp_path: Path) -> None:
    """A bare segment prefix discovers all segments in index order and iterates them."""
    seg_dir = tmp_path / "segdir"
    _write_three_gz_segments(seg_dir)

    prefix = seg_dir / "trace"
    segments = discover_segments(prefix)
    assert [s.name for s in segments] == [
        "trace.000000.jsonl.gz",
        "trace.000001.jsonl.gz",
        "trace.000002.jsonl.gz",
    ]
    assert _request_ids(list(iter_trace_records(prefix))) == ["r0", "r1", "r2"]


@pytest.mark.parametrize(
    "event_filter,expected_count",
    [
        param({"request_end"}, 2, id="request_end_only"),
        param({"tool_start", "tool_end"}, 2, id="tool_pair"),
        param({"tool_error"}, 1, id="tool_error_only"),
    ],
)  # fmt: skip
def test_filter_by_event_type(
    tmp_path: Path, event_filter: set[str], expected_count: int
) -> None:
    """``event_types`` keeps exactly the records whose event type is in the filter set."""
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        [
            _request_end(request_id="a"),
            _tool_event(event_type="tool_start"),
            _request_end(request_id="b"),
            _tool_event(event_type="tool_end", status="succeeded"),
            _tool_event(event_type="tool_error", status="error"),
        ],
    )
    out = list(iter_trace_records(p, event_types=event_filter))
    assert len(out) == expected_count
    assert {r.event_type for r in out} <= event_filter


@pytest.mark.parametrize(
    "sessions,wanted,expected_ids",
    [
        param(
            [("sess-A", "a1"), ("sess-B", "b1"), ("sess-A", "a2"), ("sess-B", "b2")],
            "sess-A",
            ["a1", "a2"],
            id="interleaved_sessions",
        ),
        param(
            [("planner", "p1"), ("researcher", "r1"), ("planner", "p2")],
            "planner",
            ["p1", "p2"],
            id="subagent_sessions",
        ),
    ],
)  # fmt: skip
def test_filter_by_session_id_keeps_subset(
    tmp_path: Path,
    sessions: list[tuple[str, str]],
    wanted: str,
    expected_ids: list[str],
) -> None:
    """``session_id`` keeps only that session's records, in file order."""
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        [_request_end(session_id=s, request_id=rid) for s, rid in sessions],
    )
    assert _request_ids(list(iter_trace_records(p, session_id=wanted))) == expected_ids


def test_filter_by_time_range_inclusive(tmp_path: Path) -> None:
    """``time_range_ms`` is inclusive on both bounds."""
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        [
            _request_end(event_time_unix_ms=ms, request_id=f"t{ms}")
            for ms in (500, 1000, 2000, 3000, 4000)
        ],
    )
    out = list(iter_trace_records(p, time_range_ms=(1000, 3000)))
    assert _request_ids(out) == ["t1000", "t2000", "t3000"]


def _schema_mismatch() -> dict:
    rec = _request_end()
    rec["schema"] = "other.schema.v1"
    return rec


def _missing_session_id() -> dict:
    rec = _request_end()
    del rec["agent_context"]["session_id"]
    return rec


def _negative_replay_hash() -> dict:
    # A negative recorded hash collides with the virtual negative-id namespace.
    return _request_end(
        replay={
            "trace_block_size": 16,
            "input_length": 32,
            "input_sequence_hashes": [1, -2, 3],
        }
    )


@pytest.mark.parametrize(
    "make_record",
    [
        param(_schema_mismatch, id="schema_mismatch"),
        param(_missing_session_id, id="agent_context_missing_session_id"),
        param(_negative_replay_hash, id="negative_replay_hash"),
    ],
)  # fmt: skip
def test_malformed_record_raises_read_error(
    tmp_path: Path, make_record: Callable[[], dict]
) -> None:
    """Records violating the v1 schema surface as DynamoTraceReadError, not raw pydantic errors."""
    p = write_jsonl(tmp_path / "trace.jsonl", [make_record()])
    with pytest.raises(DynamoTraceReadError):
        list(iter_trace_records(p))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        param(
            {
                "trace_block_size": 16,
                "input_length": 32,
                "input_sequence_hashes": [1, -2, 3],
            },
            "non-negative",
            id="negative_replay_hash",
        ),
        param(
            {"trace_block_size": 0, "input_length": 4, "input_sequence_hashes": [1]},
            "trace_block_size",
            id="zero_trace_block_size",
        ),
    ],
)  # fmt: skip
def test_replay_metrics_field_validation_rejects(kwargs: dict, match: str) -> None:
    """AgentReplayMetrics guards its own fields so direct constructors get the same rejection as the reader."""
    with pytest.raises(ValueError, match=match):
        AgentReplayMetrics(**kwargs)


def test_replay_metrics_round_trip(tmp_path: Path) -> None:
    """Replay metrics survive a read round trip, including hashes above 2^63."""
    big_hash = 14_879_255_164_371_896_291  # > 2^63, valid u64 from Dynamo
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        [
            _request_end(
                replay={
                    "trace_block_size": 64,
                    "input_length": 128,
                    "input_sequence_hashes": [1, 2, 3, big_hash],
                }
            )
        ],
    )
    out = list(iter_trace_records(p))
    assert len(out) == 1
    assert out[0].request is not None
    replay = out[0].request.replay
    assert replay is not None
    assert replay.trace_block_size == 64
    assert replay.input_length == 128
    assert replay.input_sequence_hashes == [1, 2, 3, big_hash]


def test_schema_alias_round_trip_via_model_dump() -> None:
    """Dumping by alias re-emits the reserved ``schema`` key, never the ``schema_`` field name."""
    parsed = AgentTraceRecord.model_validate(_request_end())
    dumped = parsed.model_dump(by_alias=True, exclude_none=True)
    assert dumped["schema"] == "dynamo.request.trace.v1"
    assert "schema_" not in dumped


@pytest.mark.parametrize(
    "setup",
    [
        param(lambda d: d, id="directory_with_no_trace_files"),
        param(lambda d: d / "missing-prefix", id="prefix_matching_nothing"),
    ],
)  # fmt: skip
def test_discover_segments_with_no_match_raises(
    tmp_path: Path, setup: Callable[[Path], Path]
) -> None:
    """Segment discovery that resolves to zero files raises instead of yielding nothing."""
    seg_dir = tmp_path / "segdir"
    seg_dir.mkdir()
    with pytest.raises(DynamoTraceReadError):
        discover_segments(setup(seg_dir))


def _write_invalid_json_line(p: Path) -> None:
    p.write_bytes(b'{"schema": "dynamo.request.trace.v1"}\nnot-json\n')


def _write_truncated_gzip_member(p: Path) -> None:
    write_jsonl_gz(p, [_request_end()])
    data = p.read_bytes()
    p.write_bytes(data[: len(data) - 8])  # chop the gzip trailer mid-member


def _write_corrupt_deflate(p: Path) -> None:
    # Valid 10-byte gzip header followed by bytes that are not a deflate stream:
    # reading raises zlib.error (not EOFError / BadGzipFile).
    p.write_bytes(
        b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + b"this-is-not-deflate-data"
    )


def _write_non_utf8(p: Path) -> None:
    p.write_bytes(b"\xff\xfe\x00garbage-bytes")


@pytest.mark.parametrize(
    "name,writer,match",
    [
        param("trace.jsonl", _write_invalid_json_line, None, id="invalid_json_line"),
        param(
            "trace.jsonl.gz",
            _write_truncated_gzip_member,
            "truncated or corrupt gzip",
            id="truncated_final_gzip_member",
        ),
        param(
            "trace.jsonl.gz",
            _write_corrupt_deflate,
            "truncated or corrupt gzip",
            id="corrupt_gzip_deflate",
        ),
        param("trace.jsonl", _write_non_utf8, "not valid UTF-8", id="non_utf8_bytes"),
    ],
)  # fmt: skip
def test_unreadable_payload_raises_read_error(
    tmp_path: Path, name: str, writer: Callable[[Path], None], match: str | None
) -> None:
    """Decode-level failures (bad JSON, corrupt gzip, non-UTF-8) all wrap into DynamoTraceReadError."""
    p = tmp_path / name
    writer(p)
    with pytest.raises(DynamoTraceReadError, match=match):
        list(iter_raw_records(p))


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    """Empty and whitespace-only lines between records are ignored, not parse errors."""
    p = tmp_path / "trace.jsonl"
    p.write_bytes(
        orjson.dumps(_request_end())
        + b"\n\n   \n"
        + orjson.dumps(_request_end(request_id="r2"))
        + b"\n"
    )
    out = list(iter_trace_records(p))
    assert len(out) == 2


def test_agent_context_empty_string_passes_consumer_side(tmp_path: Path) -> None:
    """Consumer-side parity: an empty session_id is accepted because Dynamo's deserialize_non_empty_string guard is server-side only."""
    rec = _request_end()
    rec["agent_context"]["session_id"] = ""
    p = write_jsonl(tmp_path / "trace.jsonl", [rec])
    out = list(iter_trace_records(p))
    assert len(out) == 1
    assert out[0].agent_context.session_id == ""


def test_extra_fields_ignored() -> None:
    """Unknown forward-compatible keys at record and request level are dropped, not fatal."""
    rec = _request_end()
    rec["future_field"] = {"new": "thing"}
    rec["request"]["another_future"] = 42
    parsed = AgentTraceRecord.model_validate(rec)
    assert not hasattr(parsed, "future_field")
    assert parsed.request is not None


def test_agent_context_parent_session_id_optional() -> None:
    """``parent_session_id`` defaults to None and round-trips when supplied."""
    ac = AgentContext(session_id="z")
    assert ac.parent_session_id is None
    ac2 = AgentContext(
        session_id="z",
        parent_session_id="parent",
    )
    assert ac2.parent_session_id == "parent"


# --- dynamo file-sink envelope + discovery parity -----------------------------


def test_sink_envelope_unwrapped_plain_jsonl(tmp_path: Path) -> None:
    """A ``{timestamp, event}`` sink envelope in plain JSONL is unwrapped to the inner record."""
    p = write_jsonl(
        tmp_path / "trace.jsonl", [_wrap(_request_end(request_id="req-w1"))]
    )
    out = list(iter_trace_records(p))
    assert len(out) == 1
    assert out[0].request is not None and out[0].request.request_id == "req-w1"


def test_sink_envelope_unwrapped_gz_segments(tmp_path: Path) -> None:
    """Sink envelopes are unwrapped across gzipped segments discovered from a prefix."""
    write_jsonl_gz(
        tmp_path / "trace.000000.jsonl.gz",
        [_wrap(_request_end(event_time_unix_ms=1_000, request_id="a"))],
    )
    write_jsonl_gz(
        tmp_path / "trace.000001.jsonl.gz",
        [_wrap(_request_end(event_time_unix_ms=2_000, request_id="b"), ts=99)],
    )
    out = list(iter_trace_records(tmp_path / "trace"))
    assert [r.request.request_id for r in out] == ["a", "b"]


def test_bare_and_wrapped_records_mix(tmp_path: Path) -> None:
    """Bare and sink-wrapped records may interleave in one file."""
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        [_request_end(request_id="bare"), _wrap(_request_end(request_id="wrapped"))],
    )
    assert [r.request.request_id for r in iter_trace_records(p)] == ["bare", "wrapped"]


def test_prefix_with_output_path_suffix_is_stripped(tmp_path: Path) -> None:
    """A configured output path is accepted verbatim as a segment prefix, suffixes stripped."""
    # DYN_REQUEST_TRACE_OUTPUT_PATH=/x/trace.jsonl.gz produces /x/trace.000000.jsonl.gz;
    # passing the configured path verbatim must resolve the segments.
    write_jsonl_gz(tmp_path / "trace.000000.jsonl.gz", [_request_end()])
    for given in ("trace.jsonl.gz", "trace.jsonl", "trace"):
        segs = discover_segments(tmp_path / given)
        assert [s.name for s in segs] == ["trace.000000.jsonl.gz"]


def test_directory_segment_order_is_numeric_past_six_digits(tmp_path: Path) -> None:
    """Segment ordering is numeric, so index 1000000 sorts after 999999 rather than lexically."""
    for name, ms in (
        ("t.1000000.jsonl.gz", 3_000),
        ("t.999999.jsonl.gz", 2_000),
        ("t.000001.jsonl.gz", 1_000),
    ):
        write_jsonl_gz(tmp_path / name, [_request_end(event_time_unix_ms=ms)])
    segs = discover_segments(tmp_path)
    assert [s.name for s in segs] == [
        "t.000001.jsonl.gz",
        "t.999999.jsonl.gz",
        "t.1000000.jsonl.gz",
    ]


def test_uppercase_gz_suffix_opens_as_gzip(tmp_path: Path) -> None:
    """Detection lowercases suffixes, so the segment opener must match .JSONL.GZ too."""
    p = write_jsonl_gz(tmp_path / "trace.JSONL.GZ", [_request_end(request_id="upper")])
    assert _request_ids(list(iter_trace_records(p))) == ["upper"]
