# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI subcommand: `aiperf dynamo trace-report <path>`.

Aggregates Dynamo agent-trace metrics per session (`agent_context.session_id`).

Streams an `AgentTraceRecord` JSONL/JSONL.gz/segmented trace via
`dynamo.trace_reader.iter_trace_records`, groups `request_end` records by
`agent_context.session_id` (the unit of work in the current
`dynamo.request.trace.v1` schema; `parent_session_id` links subagent sessions
to their parent), and emits per-session percentile aggregates for token
counts, timings, kv_hit_rate, and queue_depth in json/table/csv format.
Replay-only records (no `agent_context`) carry no session identity and are
skipped with a counter. Duplicated `request_end` records (dynamo's dual file
sinks can write the SAME record into two files of one capture dir) are folded
once and counted separately, matching the chain parser's dedup identity.
"""

from __future__ import annotations

import csv as _csv
import io
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import orjson
from cyclopts import App, Parameter
from rich import box
from rich.console import Console
from rich.table import Table

from aiperf.dataset.graph.adapters.dynamo.trace import _record_identity
from aiperf.dataset.graph.adapters.dynamo.trace_reader import (
    AgentTraceRecord,
    iter_trace_records,
)

OutFormat = Literal["json", "table", "csv"]

app = App(
    name="trace-report",
    help=(
        "Aggregate Dynamo agent-trace metrics per session. Streams "
        "`request_end` records, groups them by `agent_context.session_id` "
        "(`parent_session_id` gives the subagent hierarchy), and prints "
        "percentile aggregates for token counts, timings, kv_hit_rate, and "
        "queue_depth."
    ),
)


_NUMERIC_FIELDS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "prefill_wait_time_ms",
    "prefill_time_ms",
    "ttft_ms",
    "total_time_ms",
    "avg_itl_ms",
    "kv_hit_rate",
    "kv_transfer_estimated_latency_ms",
    "queue_depth",
)

_PCT_STATS: tuple[str, ...] = (
    "count",
    "min",
    "p50",
    "p90",
    "p95",
    "p99",
    "max",
    "mean",
)


@dataclass(slots=True)
class _SessionAggregate:
    """In-memory accumulator for one `agent_context.session_id` group."""

    session_id: str
    """The session id this group aggregates."""
    parent_session_id: str | None = None
    """Parent session id (subagent hierarchy); None for root sessions."""
    parent_session_id_conflict: bool = False
    """True when records of this session disagree on their parent session."""
    request_count: int = 0
    """Number of `request_end` records folded into this session."""
    child_session_count: int = 0
    """Number of OTHER aggregated sessions whose parent is this session."""
    model_set: set[str] = field(default_factory=set)
    """Distinct model names seen (records with `model: null` contribute none)."""
    decode_workers: set[int] = field(default_factory=set)
    """Distinct decode worker ids seen."""
    prefill_workers: set[int] = field(default_factory=set)
    """Distinct prefill worker ids seen."""
    time_lo: int | None = None
    """Earliest `event_time_unix_ms` seen."""
    time_hi: int | None = None
    """Latest `event_time_unix_ms` seen."""
    metrics: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    """Raw per-record samples per numeric field, for percentile aggregation."""
    replay_records: int = 0
    """Number of records carrying replay (KV block-hash) metadata."""
    unique_hashes: set[int] = field(default_factory=set)
    """Distinct replay input-sequence block hashes seen."""


@dataclass(slots=True)
class SessionTraceReport:
    """Per-session aggregation result for one Dynamo agent trace."""

    rows: list[dict[str, Any]]
    """Per-session aggregate dicts, sorted by session_id."""
    skipped_no_agent_context: int
    """`request_end` records skipped for lacking `agent_context` (replay-only)."""
    duplicate_records: int
    """Duplicated `request_end` records skipped (same session_id + request_id).

    Dynamo's dual file sinks can hold the SAME record twice in one capture dir;
    the dedup identity matches the chain parser's (`trace._record_identity`).
    """


def _percentiles(values: list[float]) -> dict[str, float]:
    """Compute count/min/p50/p90/p95/p99/max/mean. Empty input returns empty dict."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def pct(p: float) -> float:
        # Nearest-rank percentile: the 1-based rank is ceil(p/100 * n).
        return sorted_vals[max(0, math.ceil(p / 100.0 * n) - 1)]

    return {
        "count": float(n),
        "min": sorted_vals[0],
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": sorted_vals[-1],
        "mean": sum(sorted_vals) / n,
    }


def _record_parent(agg: _SessionAggregate, parent: str | None) -> None:
    """Track the session's parent link, flagging conflicting parent claims."""
    if parent is None:
        return
    if agg.parent_session_id is None:
        agg.parent_session_id = parent
    elif agg.parent_session_id != parent:
        agg.parent_session_id_conflict = True


def _record_workers(agg: _SessionAggregate, rec: AgentTraceRecord) -> None:
    """Fold model name and prefill/decode worker ids into the aggregate."""
    model = rec.request.model
    if model is not None:
        agg.model_set.add(model)
    worker = rec.request.worker
    if worker is None:
        return
    if worker.decode_worker_id is not None:
        agg.decode_workers.add(worker.decode_worker_id)
    if worker.prefill_worker_id is not None:
        agg.prefill_workers.add(worker.prefill_worker_id)


def _record_metrics(agg: _SessionAggregate, rec: AgentTraceRecord) -> None:
    """Fold the record's timestamps, numeric metrics, and replay hashes."""
    ts = rec.event_time_unix_ms
    agg.time_lo = ts if agg.time_lo is None else min(agg.time_lo, ts)
    agg.time_hi = ts if agg.time_hi is None else max(agg.time_hi, ts)

    for f in _NUMERIC_FIELDS:
        v = getattr(rec.request, f, None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            agg.metrics[f].append(float(v))

    if rec.request.replay is not None:
        agg.replay_records += 1
        agg.unique_hashes.update(rec.request.replay.input_sequence_hashes)


def _count_children(aggs: dict[str, _SessionAggregate]) -> None:
    """Fold child-session counts onto each parent present in the aggregate set."""
    for agg in aggs.values():
        parent = agg.parent_session_id
        if parent is None or parent == agg.session_id:
            continue
        parent_agg = aggs.get(parent)
        if parent_agg is not None:
            parent_agg.child_session_count += 1


def _to_dict(agg: _SessionAggregate) -> dict[str, Any]:
    """Serialize a `_SessionAggregate` into a JSON-safe dict."""
    return {
        "session_id": agg.session_id,
        "parent_session_id": agg.parent_session_id,
        "parent_session_id_conflict": agg.parent_session_id_conflict,
        "child_session_count": agg.child_session_count,
        "request_count": agg.request_count,
        "models": sorted(agg.model_set),
        "decode_worker_count": len(agg.decode_workers),
        "prefill_worker_count": len(agg.prefill_workers),
        "decode_workers": sorted(agg.decode_workers),
        "prefill_workers": sorted(agg.prefill_workers),
        "time_range_ms": [agg.time_lo, agg.time_hi],
        "replay_records": agg.replay_records,
        "unique_replay_hashes": len(agg.unique_hashes),
        "metrics": {f: _percentiles(vs) for f, vs in agg.metrics.items()},
    }


def aggregate_by_session(
    path: str | Path,
    *,
    session_id: str | None = None,
    limit: int | None = None,
) -> SessionTraceReport:
    """Stream a Dynamo agent-trace and return per-session aggregates.

    Public-API library entry point. The CLI is a thin wrapper.

    Args:
        path: Path to `.jsonl`, `.jsonl.gz`, segmented prefix, or directory.
        session_id: When set, only records whose `agent_context.session_id`
            matches are read (pushed down to the reader's parse-time filter).
        limit: When set, stop accumulating new sessions once `limit` distinct
            session ids have been seen. Records belonging to already-seen
            sessions continue to accumulate; child-session counts only cover
            the accumulated set.

    Returns:
        A `SessionTraceReport` with per-session rows (sorted by session_id),
        the count of skipped context-less (replay-only) records, and the count
        of skipped duplicate records.
    """
    aggs: dict[str, _SessionAggregate] = {}
    skipped_no_agent_context = 0
    duplicate_records = 0
    seen: set[tuple] = set()
    for rec in iter_trace_records(
        path, event_types={"request_end"}, session_id=session_id
    ):
        if rec.request is None:
            continue
        ctx = rec.agent_context
        if ctx is None:
            skipped_no_agent_context += 1
            continue
        # Dynamo's dual file sinks can write the SAME record into two files of
        # one capture dir; fold each identity once, mirroring the chain path.
        identity = _record_identity(rec)
        if identity is not None:
            if identity in seen:
                duplicate_records += 1
                continue
            seen.add(identity)
        agg = aggs.get(ctx.session_id)
        if agg is None:
            if limit is not None and len(aggs) >= limit:
                continue
            agg = aggs[ctx.session_id] = _SessionAggregate(session_id=ctx.session_id)
        agg.request_count += 1
        _record_parent(agg, ctx.parent_session_id)
        _record_workers(agg, rec)
        _record_metrics(agg, rec)
    _count_children(aggs)
    return SessionTraceReport(
        rows=[_to_dict(aggs[k]) for k in sorted(aggs)],
        skipped_no_agent_context=skipped_no_agent_context,
        duplicate_records=duplicate_records,
    )


def _format_json(report: SessionTraceReport) -> str:
    return orjson.dumps(
        {
            "sessions": report.rows,
            "skipped_no_agent_context": report.skipped_no_agent_context,
            "duplicate_records": report.duplicate_records,
        },
        option=orjson.OPT_INDENT_2,
    ).decode("utf-8")


def _format_csv(rows: list[dict[str, Any]]) -> str:
    """One row per session; columns are flat session facts plus per-metric percentiles."""
    buf = io.StringIO()
    base_cols = [
        "session_id",
        "parent_session_id",
        "parent_session_id_conflict",
        "child_session_count",
        "request_count",
        "models",
        "decode_worker_count",
        "prefill_worker_count",
        "time_range_ms_lo",
        "time_range_ms_hi",
        "replay_records",
        "unique_replay_hashes",
    ]
    pct_cols = [f"{f}_{stat}" for f in _NUMERIC_FIELDS for stat in _PCT_STATS]
    writer = _csv.writer(buf)
    writer.writerow(base_cols + pct_cols)
    for row in rows:
        time_lo, time_hi = row["time_range_ms"]
        flat = [
            row["session_id"],
            row["parent_session_id"] or "",
            row["parent_session_id_conflict"],
            row["child_session_count"],
            row["request_count"],
            ";".join(row["models"]),
            row["decode_worker_count"],
            row["prefill_worker_count"],
            time_lo if time_lo is not None else "",
            time_hi if time_hi is not None else "",
            row["replay_records"],
            row["unique_replay_hashes"],
        ]
        metrics = row["metrics"]
        for f in _NUMERIC_FIELDS:
            mvals = metrics.get(f, {})
            for stat in _PCT_STATS:
                flat.append(mvals.get(stat, ""))
        writer.writerow(flat)
    return buf.getvalue()


def _format_table(rows: list[dict[str, Any]], console: Console) -> None:
    """Render a Rich table with one column-block per metric (p50/p95/mean)."""
    if not rows:
        console.print("[dim](no request_end records)[/dim]")
        return

    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.SIMPLE_HEAVY,
        title="Dynamo trace aggregate",
    )
    table.add_column("session_id", style="cyan", overflow="fold")
    table.add_column("parent", style="yellow", overflow="fold")
    table.add_column("requests", justify="right")
    table.add_column("children", justify="right")
    table.add_column("models", overflow="fold")
    table.add_column("decode_w", justify="right")
    table.add_column("prefill_w", justify="right")
    metric_columns = (
        ("ttft_ms", "p50"),
        ("ttft_ms", "p95"),
        ("total_time_ms", "p50"),
        ("total_time_ms", "p95"),
        ("kv_hit_rate", "mean"),
        ("prefill_wait_time_ms", "p95"),
        ("queue_depth", "p95"),
    )
    for f, stat in metric_columns:
        table.add_column(f"{f}.{stat}", justify="right")
    table.add_column("replay/hashes", justify="right")

    for row in rows:
        models = ",".join(row["models"])
        cells = [
            row["session_id"],
            (row["parent_session_id"] or "-")
            + (" [red](mixed)[/red]" if row["parent_session_id_conflict"] else ""),
            str(row["request_count"]),
            str(row["child_session_count"]),
            models,
            str(row["decode_worker_count"]),
            str(row["prefill_worker_count"]),
        ]
        for f, stat in metric_columns:
            v = row["metrics"].get(f, {}).get(stat)
            cells.append("-" if v is None else _fmt_num(v))
        cells.append(f"{row['replay_records']}/{row['unique_replay_hashes']}")
        table.add_row(*cells)
    console.print(table)


def _fmt_num(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


@app.default
def dynamo_trace_report(
    path: Annotated[
        Path,
        Parameter(
            help=(
                "Path to a Dynamo agent-trace `.jsonl`, `.jsonl.gz`, segmented "
                "prefix, or directory of segments."
            )
        ),
    ],
    *,
    out_format: Annotated[
        Literal["json", "table", "csv"],
        Parameter(
            name="--format",
            help="Output format. Default: table.",
        ),
    ] = "table",
    session_id: Annotated[
        str | None,
        Parameter(
            name="--session-id",
            help="Restrict to records matching this agent_context.session_id.",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        Parameter(
            name="--limit",
            help="Stop accumulating new sessions once N distinct session ids are seen.",
        ),
    ] = None,
) -> None:
    """Aggregate Dynamo agent-trace metrics per agent_context.session_id."""
    report = aggregate_by_session(path, session_id=session_id, limit=limit)

    if out_format == "json":
        sys.stdout.write(_format_json(report))
        sys.stdout.write("\n")
        return
    if out_format == "csv":
        sys.stdout.write(_format_csv(report.rows))
        if report.skipped_no_agent_context:
            sys.stderr.write(
                f"skipped {report.skipped_no_agent_context} request_end records "
                "without agent_context (replay-only)\n"
            )
        if report.duplicate_records:
            sys.stderr.write(
                f"skipped {report.duplicate_records} duplicate request_end "
                "records (dual file sinks)\n"
            )
        return
    console = Console()
    _format_table(report.rows, console)
    if report.skipped_no_agent_context:
        console.print(
            f"[dim](skipped {report.skipped_no_agent_context} request_end "
            "records without agent_context -- replay-only)[/dim]"
        )
    if report.duplicate_records:
        console.print(
            f"[dim](skipped {report.duplicate_records} duplicate request_end "
            "records -- dual file sinks)[/dim]"
        )


__all__ = [
    "SessionTraceReport",
    "aggregate_by_session",
    "app",
    "dynamo_trace_report",
]
