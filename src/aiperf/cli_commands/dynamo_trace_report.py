# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI subcommand: `aiperf dynamo trace-report <path>`.

Aggregates Dynamo agent-trace metrics per session and trace-wide. Context-free
request records use deterministic synthetic root session ids.

Loads an `AgentTraceRecord` JSONL/JSONL.gz/segmented trace through the same
Dynamo chain and trie path used by graph replay. `request_end` records are
grouped by `agent_context.session_id` (or a synthetic `request-{request_id}`
root when context is absent), and percentile aggregates plus trie-derived
prefix statistics are emitted in json/table/csv format.

Two views over the same fold:

* the DEFAULT trace-wide rollup (:class:`CorpusStats`) -- pooled percentiles
  over per-record samples plus statistics from the existing replay trie;
* the per-session listing behind ``--per-session`` -- one row per session.

Corpus percentiles pool RAW samples rather than reducing the per-session
percentiles, because a median-of-medians is a different, weaker statistic that
cannot be corrected after the per-session reduction has run.

The replay block hashes are analyzed by the same trie and causal prefix-cache
pass that stamps the replay graph. The recorded `kv_hit_rate` mean is printed
next to the trie-derived theoretical rate so the two can be compared directly.

Replay-only `request_end` records without `agent_context` receive deterministic
`request-{request_id}` root sessions, matching the graph adapter. Context-free
non-request records remain skipped. Duplicated
`request_end` records (dynamo's dual file sinks can write the SAME record into
two files of one capture dir) are folded once and counted separately, matching
the chain parser's dedup identity.
"""

from __future__ import annotations

import csv as _csv
import io
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import orjson
from cyclopts import App, Parameter
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from aiperf.common.environment import Environment
from aiperf.common.finite import scrub_non_finite
from aiperf.dataset.graph.adapters.dynamo.trace import (
    _Chain,
    _collect_chains,
    _guard_chain_forest,
    _records_to_chain,
    analyze_dynamo_chains_trie,
)
from aiperf.dataset.graph.adapters.dynamo.trace_reader import (
    AgentTraceRecord,
    EmptyDynamoTraceError,
    iter_session_records,
    load_ingest_sidecar,
    resolve_parent,
    scan_dynamo_trace,
    write_ingest_sidecar,
)

OutFormat = Literal["json", "table", "csv"]

app = App(
    name="trace-report",
    help=(
        "Aggregate Dynamo agent-trace metrics. Streams `request_end` records, "
        "groups them by session (`parent_session_id` gives "
        "the subagent hierarchy), and prints a trace-wide rollup: pooled "
        "percentiles for token counts, timings, kv_hit_rate and queue_depth, "
        "plus KV-block prefix reuse and a theoretical infinite-cache hit rate. "
        "Use `--per-session` for the one-row-per-session listing."
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


@dataclass(slots=True)
class _CorpusAggregate:
    """Trace-wide accumulator: pooled raw samples plus replay-hash reuse facts.

    Pooling happens HERE, over raw per-record samples, because a corpus
    percentile cannot be recovered from per-session percentiles after the fact
    (a median-of-medians is a different, weaker statistic).
    """

    metrics: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    """Raw per-record samples per numeric field, pooled across every session."""
    replay_records: int = 0
    """Records carrying replay metadata."""
    block_sizes: set[int] = field(default_factory=set)
    """Distinct `trace_block_size` values seen across replay records."""


@dataclass(slots=True)
class CorpusStats:
    """Trace-wide rollup over every `request_end` record with a session."""

    request_count: int
    """Total `request_end` records folded (sum over sessions)."""
    session_count: int
    """Distinct sessions aggregated."""
    metrics: dict[str, dict[str, float]]
    """Pooled percentiles per numeric field, over raw samples across sessions."""
    replay_records: int
    """Records carrying replay (KV block-hash) metadata."""
    distinct_hashes: int
    """Distinct replay block hashes across the whole trace."""
    shared_hashes: int
    """Block hashes seen under more than one session id."""
    cross_session_dedup_ratio: float
    """Fraction of per-session hash entries removed by global dedup (0.0-1.0)."""
    theoretical_hit_rate: float
    """Mean per-record infinite-cache hit rate (0.0-1.0)."""
    hit_rate_stats: dict[str, float]
    """Percentiles of the per-record theoretical hit rate."""
    block_sizes: list[int]
    """Distinct `trace_block_size` values; >1 means the trace mixes block sizes."""


class SessionTraceRow(TypedDict):
    """Stable serialized schema for one session in a Dynamo trace report."""

    session_id: str
    parent_session_id: str | None
    parent_session_id_conflict: bool
    child_session_count: int
    request_count: int
    models: list[str]
    decode_worker_count: int
    prefill_worker_count: int
    decode_workers: list[int]
    prefill_workers: list[int]
    time_range_ms: list[int | None]
    replay_records: int
    metrics: dict[str, dict[str, float]]


@dataclass(slots=True)
class SessionTraceReport:
    """Per-session aggregation result for one Dynamo agent trace."""

    rows: list[SessionTraceRow]
    """Per-session aggregate dicts, sorted by session_id."""
    skipped_no_agent_context: int
    """Context-free non-request records skipped by the canonical fold."""
    duplicate_records: int
    """Duplicated `request_end` records skipped (same session_id + request_id).

    Dynamo's dual file sinks can hold the SAME record twice in one capture dir;
    the dedup identity IS the chain parser's -- both run the same shared fold
    (`trace_reader.iter_session_records` / `trace_reader.record_identity`).
    """
    corpus: CorpusStats
    """Trace-wide rollup over the same records the per-session rows cover."""
    skipped_over_limit: int = 0
    """Records dropped because `limit` distinct sessions were already seen.

    Counted so the skip ratios stay honest: without this the denominator would
    silently exclude everything `--limit` discarded.
    """

    @property
    def total_records(self) -> int:
        """Report denominator across retained request_end and skipped classes."""
        return (
            self.corpus.request_count
            + self.skipped_no_agent_context
            + self.duplicate_records
            + self.skipped_over_limit
        )


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
    """Fold the record's timestamps and numeric metrics."""
    ts = rec.event_time_unix_ms
    agg.time_lo = ts if agg.time_lo is None else min(agg.time_lo, ts)
    agg.time_hi = ts if agg.time_hi is None else max(agg.time_hi, ts)

    for f in _NUMERIC_FIELDS:
        v = getattr(rec.request, f, None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            agg.metrics[f].append(float(v))

    if rec.request.replay is not None:
        agg.replay_records += 1


def _fold_corpus(corpus: _CorpusAggregate, rec: AgentTraceRecord) -> None:
    """Fold one record's raw metric samples into the corpus."""
    for f in _NUMERIC_FIELDS:
        v = getattr(rec.request, f, None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            corpus.metrics[f].append(float(v))

    replay = rec.request.replay
    if replay is None:
        return
    corpus.replay_records += 1
    corpus.block_sizes.add(replay.trace_block_size)


_EMPTY_TRIE_ANALYSIS: dict[str, Any] = {
    "total_requests": 0,
    "cache_hit_rate": 0.0,
    "input_length_stats": {},
    "output_length_stats": {},
    "context_length_stats": {},
    "unique_prompt_length_stats": {},
    "hit_rate_stats": {},
    "distinct_hashes": 0,
    "shared_hashes": 0,
    "cross_session_dedup_ratio": 0.0,
}


def _merge_trie_analysis(analyses: list[Any]) -> dict[str, Any]:
    """Merge per-tree trie analysis into the trace-report JSON shape.

    Returns the zeroed shape when no tree was analyzed (no admitted session),
    so callers can subscript the result unconditionally.
    """
    if not analyses:
        return dict(_EMPTY_TRIE_ANALYSIS)
    merged = analyses[0]
    for analysis in analyses[1:]:
        merged.merge(analysis)
    input_lengths = merged.input_lengths
    output_lengths = merged.output_lengths
    context_lengths = merged.context_lengths
    unique_prompt_lengths = merged.unique_prompt_lengths
    hit_rates = merged.hit_rates
    return {
        "total_requests": len(input_lengths),
        "cache_hit_rate": (sum(hit_rates) / len(hit_rates) if hit_rates else 0.0),
        "input_length_stats": _percentiles([float(v) for v in input_lengths]),
        "output_length_stats": _percentiles([float(v) for v in output_lengths]),
        "context_length_stats": _percentiles([float(v) for v in context_lengths]),
        "unique_prompt_length_stats": _percentiles(
            [float(v) for v in unique_prompt_lengths]
        ),
        "hit_rate_stats": _percentiles(hit_rates),
        "distinct_hashes": merged.distinct_hashes,
        "shared_hashes": merged.shared_hashes,
        "cross_session_dedup_ratio": (
            1.0 - merged.distinct_hashes / merged.per_owner_hash_entries
            if merged.per_owner_hash_entries
            else 0.0
        ),
    }


def _build_corpus_stats(
    corpus: _CorpusAggregate,
    aggs: dict[str, _SessionAggregate],
    *,
    trie_analysis: dict[str, Any],
) -> CorpusStats:
    """Reduce the corpus accumulator to the serializable rollup."""
    return CorpusStats(
        request_count=sum(a.request_count for a in aggs.values()),
        session_count=len(aggs),
        metrics={
            **{f: _percentiles(vs) for f, vs in corpus.metrics.items()},
            "input_tokens": trie_analysis["input_length_stats"],
            "output_tokens": trie_analysis["output_length_stats"],
            "context_length": trie_analysis["context_length_stats"],
            "unique_prompt_length": trie_analysis["unique_prompt_length_stats"],
        },
        replay_records=corpus.replay_records,
        distinct_hashes=trie_analysis["distinct_hashes"],
        shared_hashes=trie_analysis["shared_hashes"],
        cross_session_dedup_ratio=trie_analysis["cross_session_dedup_ratio"],
        theoretical_hit_rate=trie_analysis["cache_hit_rate"],
        hit_rate_stats=trie_analysis["hit_rate_stats"],
        block_sizes=sorted(corpus.block_sizes),
    )


def _count_children(aggs: dict[str, _SessionAggregate]) -> None:
    """Fold child-session counts onto each parent present in the aggregate set."""
    for agg in aggs.values():
        parent = agg.parent_session_id
        if parent is None or parent == agg.session_id:
            continue
        parent_agg = aggs.get(parent)
        if parent_agg is not None:
            parent_agg.child_session_count += 1


def _to_dict(agg: _SessionAggregate) -> SessionTraceRow:
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
        "metrics": {f: _percentiles(vs) for f, vs in agg.metrics.items()},
    }


def _raise_empty_trace(path: str | Path, skipped_no_context: int) -> None:
    """Raise the same precise empty-input error as the shared chain collector."""
    if skipped_no_context:
        raise EmptyDynamoTraceError(
            f"{path}: {skipped_no_context:,} non-request records had no "
            "agent_context and could not be assigned a session"
        )
    raise EmptyDynamoTraceError(f"{path}: no trace records found")


def _chains_from_limited_records(
    records: dict[str, list[AgentTraceRecord]],
    parent_link: dict[str, str],
) -> dict[str, _Chain]:
    """Sort retained records and validate only their selected session forest."""
    chains: dict[str, _Chain] = {}
    for sid, session_records in records.items():
        session_records.sort(key=lambda record: record.event_time_unix_ms)
        chain = _records_to_chain(
            session_records,
            session_id=sid,
            parent_session_id=parent_link.get(sid),
        )
        if chain.turns:
            chains[sid] = chain
    selected_parent_link = {
        sid: parent for sid, parent in parent_link.items() if sid in chains
    }
    _guard_chain_forest(
        chains,
        selected_parent_link,
        max_depth=Environment.DYNAMO.MAX_SUBAGENT_DEPTH,
    )
    return chains


def _collect_limited_chains(
    path: str | Path,
    *,
    session_id: str | None,
    limit: int,
) -> tuple[dict[str, _Chain], int, int, int]:
    """Retain request records only for the first ``limit`` sessions seen."""
    records: dict[str, list[AgentTraceRecord]] = defaultdict(list)
    parent_link: dict[str, str] = {}
    admitted: set[str] = set()
    skipped_over_limit = 0
    skipped_no_context = 0
    duplicate_records = 0
    request_end_seen = 0

    def _count_no_context(_record: AgentTraceRecord) -> None:
        nonlocal skipped_no_context
        skipped_no_context += 1

    def _count_duplicate(_record: AgentTraceRecord) -> None:
        nonlocal duplicate_records
        duplicate_records += 1

    for ctx, record in iter_session_records(
        path,
        session_id=session_id,
        on_no_context=_count_no_context,
        on_duplicate=_count_duplicate,
        synthesize_contextless_requests=True,
    ):
        parent = resolve_parent(ctx)
        if parent is not None and ctx.session_id not in parent_link:
            parent_link[ctx.session_id] = parent
        if record.event_type != "request_end":
            continue
        request_end_seen += 1
        if ctx.session_id not in admitted:
            if len(admitted) >= limit:
                skipped_over_limit += 1
                continue
            admitted.add(ctx.session_id)
        records[ctx.session_id].append(record)

    if not request_end_seen:
        _raise_empty_trace(path, skipped_no_context)

    chains = _chains_from_limited_records(records, parent_link)
    return chains, skipped_no_context, duplicate_records, skipped_over_limit


def aggregate_by_session(
    path: str | Path,
    *,
    session_id: str | None = None,
    limit: int | None = None,
    progress: Progress | None = None,
    progress_task: int | None = None,
    progress_finalize: bool = False,
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
        the count of unusable context-free records, the count of
        skipped duplicate records, and the trace-wide `corpus` rollup.
    """
    aggs: dict[str, _SessionAggregate] = {}
    corpus = _CorpusAggregate()
    if limit is None:
        duplicate_count: list[int] = []
        skipped_no_context: list[int] = []
        chains = _collect_chains(
            path,
            session_id,
            max_depth=Environment.DYNAMO.MAX_SUBAGENT_DEPTH,
            duplicate_out=duplicate_count,
            skipped_out=skipped_no_context,
        )
        duplicate_records = duplicate_count[0] if duplicate_count else 0
        skipped_no_agent_context = skipped_no_context[0] if skipped_no_context else 0
        skipped_over_limit = 0
    else:
        (
            chains,
            skipped_no_agent_context,
            duplicate_records,
            skipped_over_limit,
        ) = _collect_limited_chains(path, session_id=session_id, limit=limit)
    admitted = set(chains)
    for sid in sorted(admitted):
        chain = chains[sid]
        agg = _SessionAggregate(
            session_id=sid,
            parent_session_id=chain.parent_session_id,
        )
        aggs[sid] = agg
        for turn in chain.turns:
            rec = turn.record
            agg.request_count += 1
            if rec.agent_context is not None:
                _record_parent(agg, resolve_parent(rec.agent_context))
            _record_workers(agg, rec)
            _record_metrics(agg, rec)
            _fold_corpus(corpus, rec)
            if progress is not None and progress_task is not None:
                progress.advance(progress_task)
    if progress is not None and progress_task is not None and progress_finalize:
        progress.update(progress_task, description="Building replay trie(s)")
    _count_children(aggs)
    trie_analysis = _merge_trie_analysis(
        analyze_dynamo_chains_trie({sid: chains[sid] for sid in admitted})
    )
    if progress is not None and progress_task is not None and progress_finalize:
        progress.update(progress_task, description="Finalizing trace(s)")
    report = SessionTraceReport(
        rows=[_to_dict(aggs[k]) for k in sorted(aggs)],
        skipped_no_agent_context=skipped_no_agent_context,
        duplicate_records=duplicate_records,
        corpus=_build_corpus_stats(corpus, aggs, trie_analysis=trie_analysis),
        skipped_over_limit=skipped_over_limit,
    )
    if progress is not None and progress_task is not None and progress_finalize:
        progress.advance(progress_task)
    return report


def _corpus_to_dict(c: CorpusStats) -> dict[str, Any]:
    """Serialize the corpus rollup into a JSON-safe dict."""
    return {
        "request_count": c.request_count,
        "session_count": c.session_count,
        "replay_records": c.replay_records,
        "distinct_hashes": c.distinct_hashes,
        "shared_hashes": c.shared_hashes,
        "cross_session_dedup_ratio": c.cross_session_dedup_ratio,
        "theoretical_hit_rate": c.theoretical_hit_rate,
        "hit_rate_stats": c.hit_rate_stats,
        "block_sizes": c.block_sizes,
        "metrics": c.metrics,
    }


def _format_json(report: SessionTraceReport) -> str:
    return orjson.dumps(
        scrub_non_finite(
            {
                "corpus": _corpus_to_dict(report.corpus),
                "sessions": report.rows,
                "skipped_no_agent_context": report.skipped_no_agent_context,
                "duplicate_records": report.duplicate_records,
                "skipped_over_limit": report.skipped_over_limit,
                "total_records": report.total_records,
            }
        ),
        option=orjson.OPT_INDENT_2,
    ).decode("utf-8")


def _format_csv(rows: list[SessionTraceRow]) -> str:
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
        ]
        metrics = row["metrics"]
        for f in _NUMERIC_FIELDS:
            mvals = metrics.get(f, {})
            for stat in _PCT_STATS:
                flat.append(mvals.get(stat, ""))
        writer.writerow(flat)
    return buf.getvalue()


def _format_corpus(report: SessionTraceReport, console: Console) -> None:
    """Render the trace-wide rollup: headline facts, prefix reuse, metric table.

    Metrics are rows and percentiles are columns (transposed relative to the
    per-session listing), matching `aiperf analyze-trace`'s report shape.
    """
    c = report.corpus
    if not c.request_count:
        console.print("[dim](no request_end records with agent_context)[/dim]")
        return

    console.print()
    console.print("[bold]Dynamo Trace Report[/bold]")
    console.print(f"Sessions:         {c.session_count:,}")
    console.print(f"Requests:         {c.request_count:,}")
    console.print(
        f"Replay records:   {c.replay_records:,}"
        + (
            "  [dim](all)[/dim]"
            if c.replay_records == c.request_count
            else "  [yellow](some records fall back to virtual hashes)[/yellow]"
        )
    )
    if c.block_sizes:
        sizes = ", ".join(str(b) for b in c.block_sizes)
        warn = "  [yellow](mixed)[/yellow]" if len(c.block_sizes) > 1 else ""
        console.print(f"Block size:       {sizes}{warn}")
    console.print()
    console.print(f"Distinct blocks:  {c.distinct_hashes:,}")
    console.print(
        f"Shared blocks:    {c.shared_hashes:,}"
        f"  [dim]({_pct(c.shared_hashes, c.distinct_hashes)} seen in >1 session)[/dim]"
    )
    console.print(
        f"Cross-session dedup: {c.cross_session_dedup_ratio * 100:.1f}%"
        "  [dim](per-session hash entries removed by global dedup)[/dim]"
    )
    console.print(
        f"Theoretical hit rate: {c.theoretical_hit_rate:.4f}"
        "  [dim](infinite cache, prefix-only)[/dim]"
    )
    recorded = c.metrics.get("kv_hit_rate", {}).get("mean")
    if recorded is not None:
        console.print(
            f"Recorded kv_hit_rate: {recorded:.4f}"
            f"  [dim](mean; delta {recorded - c.theoretical_hit_rate:+.4f})[/dim]"
        )
    console.print()

    table = Table(title="Corpus Statistics", box=box.SIMPLE_HEAVY)
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    for stat in _PCT_STATS:
        table.add_column(stat, justify="right", style="green", no_wrap=True)

    ordered = [f for f in _NUMERIC_FIELDS if f in c.metrics]
    ordered.extend(
        f for f in ("context_length", "unique_prompt_length") if f in c.metrics
    )
    for f in ordered:
        vals = c.metrics[f]
        table.add_row(f, *[_fmt_num(vals[s]) for s in _PCT_STATS])
    if c.hit_rate_stats:
        table.add_row(
            "theoretical_hit_rate",
            *[_fmt_num(c.hit_rate_stats[s]) for s in _PCT_STATS],
        )
    console.print(table)


def _pct(part: int, whole: int) -> str:
    """Format `part` as a percentage of `whole`; empty whole reads as 0%."""
    return f"{(100.0 * part / whole) if whole else 0.0:.1f}%"


def _format_skips(report: SessionTraceReport, console: Console) -> None:
    """Print the skip counters as a share of every record seen."""
    total = report.total_records
    if report.skipped_no_agent_context:
        console.print(
            f"[dim](skipped {report.skipped_no_agent_context:,} of {total:,} "
            f"request_end records "
            f"({_pct(report.skipped_no_agent_context, total)}) without "
            "request identity)[/dim]"
        )
    if report.duplicate_records:
        console.print(
            f"[dim](skipped {report.duplicate_records:,} of {total:,} duplicate "
            f"request_end records ({_pct(report.duplicate_records, total)}) "
            "-- dual file sinks)[/dim]"
        )
    if report.skipped_over_limit:
        console.print(
            f"[yellow](--limit dropped {report.skipped_over_limit:,} of {total:,} "
            f"records ({_pct(report.skipped_over_limit, total)}) belonging to "
            "sessions past the cap; corpus stats cover the capped subset only)"
            "[/yellow]"
        )


def _format_table(rows: list[SessionTraceRow], console: Console) -> None:
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
    table.add_column("session_id", style="cyan", overflow="ellipsis", no_wrap=True)
    table.add_column("parent", style="yellow", overflow="ellipsis", no_wrap=True)
    table.add_column("requests", justify="right")
    table.add_column("children", justify="right")
    table.add_column("models", overflow="ellipsis", no_wrap=True)
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
        cells.append(str(row["replay_records"]))
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
    per_session: Annotated[
        bool,
        Parameter(
            name="--per-session",
            help=(
                "Print the per-session listing (one row per session) instead of "
                "the trace-wide rollup. Ignored for json/csv."
            ),
        ),
    ] = False,
    output_file: Annotated[
        Path | None,
        Parameter(
            name="--output-file",
            help="Also write the full report as JSON to this path.",
        ),
    ] = None,
    write_ingest_sidecar_flag: Annotated[
        bool,
        Parameter(
            name="--write-ingest-sidecar",
            help="Write reusable Dynamo ingest metadata beside the trace.",
        ),
    ] = False,
) -> None:
    """Aggregate Dynamo agent-trace metrics per agent_context.session_id."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=Console(file=sys.stderr),
    )
    progress_task = progress.add_task("Scanning trace(s)", total=None)
    with progress:
        scan = load_ingest_sidecar(path) if session_id is None else None
        if write_ingest_sidecar_flag and session_id is None:
            scan = scan or scan_dynamo_trace(
                path,
                session_id=session_id,
                on_record=lambda _record: progress.advance(progress_task),
            )
            write_ingest_sidecar(path, scan)
        progress.update(progress_task, description="Loading trace(s)")
        report = aggregate_by_session(
            path,
            session_id=session_id,
            limit=limit,
            progress=progress,
            progress_task=progress_task,
            progress_finalize=True,
        )

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(_format_json(report), encoding="utf-8")

    if out_format == "json":
        sys.stdout.write(_format_json(report))
        sys.stdout.write("\n")
        return
    if out_format == "csv":
        sys.stdout.write(_format_csv(report.rows))
        total = report.total_records
        if report.skipped_no_agent_context:
            sys.stderr.write(
                f"skipped {report.skipped_no_agent_context} of {total} trace "
                f"records ({_pct(report.skipped_no_agent_context, total)}) "
                "without request identity\n"
            )
        if report.duplicate_records:
            sys.stderr.write(
                f"skipped {report.duplicate_records} of {total} duplicate "
                f"request_end records ({_pct(report.duplicate_records, total)}) "
                "(dual file sinks)\n"
            )
        return
    # The corpus table is a fixed 9 columns, so pin it like `analyze-trace`.
    # The per-session listing is 15 columns wide and only readable on a real
    # terminal, so let Rich size that one to the actual width.
    if per_session:
        console = Console()
        _format_table(report.rows, console)
    else:
        console = Console(width=120)
        _format_corpus(report, console)
    _format_skips(report, console)
    if output_file is not None:
        console.print(f"Report saved to {output_file}")


__all__ = [
    "CorpusStats",
    "SessionTraceRow",
    "SessionTraceReport",
    "aggregate_by_session",
    "app",
    "dynamo_trace_report",
]
