# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`aggregate_by_session` end-to-end over real minimal `dynamo.request.trace.v1` trace files."""

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
    """A minimal bare `request_end` record the reader accepts."""
    # session_id=None omits agent_context entirely, i.e. a replay-only record.
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
            "input_length": len(replay_hashes) * 16,
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
    """Materialize the records as one bare JSONL trace file."""
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


def test_duplicate_record_across_dual_sink_files_folded_once(tmp_path: Path) -> None:
    """A record duplicated across a capture dir's two sink files folds once and increments the duplicate counter."""
    # discover_segments reads ALL *.jsonl + *.jsonl.gz in the dir, so dual sinks
    # surface the same record twice; dedup identity is
    # ("request_end", session_id, request_id), matching the chain parser.
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


def test_replay_only_records_synthesize_root_sessions(tmp_path: Path) -> None:
    """Context-free request_end records become deterministic root sessions."""
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id=None, ts_ms=1_000),
            _request_end(session_id=None, ts_ms=2_000),
            _request_end(session_id="s-live", ts_ms=3_000),
        ],
    )
    report = aggregate_by_session(trace)
    assert report.skipped_no_agent_context == 0
    assert [row["session_id"] for row in report.rows] == [
        "request-r-1000",
        "request-r-2000",
        "s-live",
    ]


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
    """``limit`` caps how many distinct sessions are admitted, but later records of an admitted session still fold in."""
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-z", ts_ms=1_000),
            _request_end(session_id="s-a", ts_ms=2_000),
            _request_end(session_id="s-z", ts_ms=3_000),
        ],
    )
    report = aggregate_by_session(trace, limit=1)
    assert [row["session_id"] for row in report.rows] == ["s-z"]
    assert report.rows[0]["request_count"] == 2


def test_limit_does_not_lower_sessions_beyond_cap(tmp_path: Path) -> None:
    """Parent cycles outside the admitted set cannot poison a bounded report."""
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-keep", ts_ms=1_000),
            _request_end(
                session_id="s-a",
                parent_session_id="s-b",
                ts_ms=2_000,
            ),
            _request_end(
                session_id="s-b",
                parent_session_id="s-a",
                ts_ms=3_000,
            ),
        ],
    )

    report = aggregate_by_session(trace, limit=1)

    assert [row["session_id"] for row in report.rows] == ["s-keep"]
    assert report.skipped_over_limit == 2


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
    """Percentiles use 1-based nearest rank, rank = ceil(p/100 * n)."""
    # A 0-based int(p/100 * n) index is off by one: p50 of [1,2,3,4] gives 3, not 2.
    assert _percentiles(values)[stat] == pytest.approx(expected)


def test_percentiles_empty_returns_empty_dict() -> None:
    """No samples yields an empty stat dict rather than zeros or NaNs."""
    assert _percentiles([]) == {}


def test_format_json_envelope_carries_skip_counter(tmp_path: Path) -> None:
    """The JSON envelope exposes the replay-only skip count alongside the session rows."""
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-a", ts_ms=1_000),
            _request_end(session_id=None, ts_ms=2_000),
        ],
    )
    report = aggregate_by_session(trace)
    payload = orjson.loads(_format_json(report))
    assert payload["skipped_no_agent_context"] == 0
    assert [s["session_id"] for s in payload["sessions"]] == [
        "request-r-2000",
        "s-a",
    ]


def test_corpus_pools_raw_samples_not_per_session_percentiles(
    tmp_path: Path,
) -> None:
    """Corpus percentiles pool raw records; a median-of-medians would differ."""
    # s-a contributes 1,2,3 and s-b contributes 100. The pooled p50 over
    # [1,2,3,100] is 2; the median of the per-session p50s (2 and 100) is 51.
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-a", ts_ms=1_000, ttft_ms=1.0),
            _request_end(session_id="s-a", ts_ms=2_000, ttft_ms=2.0),
            _request_end(session_id="s-a", ts_ms=3_000, ttft_ms=3.0),
            _request_end(session_id="s-b", ts_ms=4_000, ttft_ms=100.0),
        ],
    )
    report = aggregate_by_session(trace)
    assert report.corpus.request_count == 4
    assert report.corpus.session_count == 2
    assert report.corpus.metrics["ttft_ms"]["count"] == 4.0
    assert report.corpus.metrics["ttft_ms"]["p50"] == pytest.approx(2.0)
    assert report.corpus.metrics["ttft_ms"]["max"] == pytest.approx(100.0)


def test_corpus_counts_blocks_shared_across_sessions(tmp_path: Path) -> None:
    """Blocks under >1 session are shared; repeats within one session are not."""
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-a", ts_ms=1_000, replay_hashes=[1, 2, 3]),
            _request_end(session_id="s-a", ts_ms=2_000, replay_hashes=[1, 2, 9]),
            _request_end(session_id="s-b", ts_ms=3_000, replay_hashes=[1, 7]),
        ],
    )
    corpus = aggregate_by_session(trace).corpus
    assert corpus.distinct_hashes == 5  # {1,2,3,9,7}
    assert corpus.shared_hashes == 1  # only hash 1 spans s-a and s-b
    assert corpus.replay_records == 3
    assert corpus.block_sizes == [16]


def test_corpus_dedup_ratio_from_per_session_entries(tmp_path: Path) -> None:
    """Dedup ratio is 1 - distinct/sum-of-per-session-distinct."""
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-a", ts_ms=1_000, replay_hashes=[1, 2]),
            _request_end(session_id="s-b", ts_ms=2_000, replay_hashes=[1, 2]),
        ],
    )
    corpus = aggregate_by_session(trace).corpus
    # per-session entries = 2 + 2 = 4, distinct = 2 -> half the entries are dupes.
    assert corpus.cross_session_dedup_ratio == pytest.approx(0.5)


def test_theoretical_hit_rate_is_leading_prefix_of_seen_blocks(
    tmp_path: Path,
) -> None:
    """Hit rate is the index of the first never-before-seen block over block count."""
    trace = _write_trace(
        tmp_path,
        [
            # First record sees nothing: first block is unseen -> 0/2.
            _request_end(session_id="s-a", ts_ms=1_000, replay_hashes=[1, 2]),
            # Second reuses [1,2] then adds 3 -> first unseen at index 2 -> 2/3.
            _request_end(session_id="s-a", ts_ms=2_000, replay_hashes=[1, 2, 3]),
        ],
    )
    corpus = aggregate_by_session(trace).corpus
    assert corpus.hit_rate_stats["count"] == 2.0
    assert corpus.theoretical_hit_rate == pytest.approx((0.0 + 2 / 3) / 2)


def test_mixed_block_sizes_are_all_reported(tmp_path: Path) -> None:
    """A capture mixing trace_block_size values surfaces every distinct size."""
    records = [
        _request_end(session_id="s-a", ts_ms=1_000, replay_hashes=[1]),
        _request_end(session_id="s-b", ts_ms=2_000, replay_hashes=[2]),
    ]
    records[1]["request"]["replay"]["trace_block_size"] = 64
    trace = _write_trace(tmp_path, records)
    assert aggregate_by_session(trace).corpus.block_sizes == [16, 64]


def test_limit_dropped_records_counted_in_total(tmp_path: Path) -> None:
    """Records dropped by ``limit`` stay in the denominator of the skip ratios."""
    # Without this the ratio silently inflates: the dropped records vanish from
    # total_records and every reported percentage is over a smaller corpus.
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-a", ts_ms=1_000),
            _request_end(session_id="s-b", ts_ms=2_000),
            _request_end(session_id="s-c", ts_ms=3_000),
            _request_end(session_id=None, ts_ms=4_000),
        ],
    )
    report = aggregate_by_session(trace, limit=1)
    assert report.skipped_over_limit == 3
    assert report.skipped_no_agent_context == 0
    assert report.corpus.request_count == 1
    assert report.total_records == 4


def test_corpus_includes_replay_only_sessions(tmp_path: Path) -> None:
    """A replay-only request contributes to the synthetic root corpus."""
    trace = _write_trace(tmp_path, [_request_end(session_id=None, ts_ms=1_000)])
    report = aggregate_by_session(trace)
    assert [row["session_id"] for row in report.rows] == ["request-r-1000"]
    assert report.corpus.request_count == 1
    assert report.corpus.session_count == 1
    assert report.corpus.cross_session_dedup_ratio == 0.0
    assert report.corpus.theoretical_hit_rate == 0.0
    assert report.total_records == 1


def test_format_json_envelope_carries_corpus_and_totals(tmp_path: Path) -> None:
    """The JSON envelope exposes the corpus rollup and every skip counter."""
    trace = _write_trace(
        tmp_path,
        [
            _request_end(session_id="s-a", ts_ms=1_000, ttft_ms=5.0),
            _request_end(session_id=None, ts_ms=2_000),
        ],
    )
    payload = orjson.loads(_format_json(aggregate_by_session(trace)))
    assert payload["total_records"] == 2
    assert payload["skipped_over_limit"] == 0
    corpus = payload["corpus"]
    assert corpus["request_count"] == 2
    assert corpus["metrics"]["ttft_ms"]["p50"] == pytest.approx(5.0)


def test_optional_trie_analysis_is_emitted_in_json(tmp_path: Path) -> None:
    """The report can expose analysis computed by the existing Dynamo trie."""
    records = [
        _request_end(
            session_id="s-a",
            ts_ms=1_000,
            input_tokens=32,
            replay_hashes=[1, 2],
        ),
        _request_end(
            session_id="s-a",
            ts_ms=2_000,
            input_tokens=48,
            replay_hashes=[1, 2, 3],
        ),
    ]
    records[1]["request"]["replay"]["input_length"] = 48
    trace = _write_trace(tmp_path, records)

    report = aggregate_by_session(trace)
    corpus = orjson.loads(_format_json(report))["corpus"]

    assert corpus["metrics"]["input_tokens"]["count"] == 2
    assert corpus["metrics"]["context_length"]["mean"] == pytest.approx(16.0)
    assert "trie_analysis" not in corpus


def test_format_csv_headers_and_rows(tmp_path: Path) -> None:
    """CSV output emits one header line plus one row per session, with flattened metric columns."""
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


def test_aggregate_by_session_limit_zero_returns_zeroed_corpus(tmp_path: Path) -> None:
    """No admitted session (limit=0) yields an empty corpus rollup instead of raising."""
    trace = _write_trace(tmp_path, [_request_end(session_id="s-a", ts_ms=1_000)])

    report = aggregate_by_session(trace, limit=0)

    assert report.rows == []
    assert report.corpus.request_count == 0
    assert orjson.loads(_format_json(report))["corpus"]["request_count"] == 0
