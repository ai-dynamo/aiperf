# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused read+build process-parallel session-tree build for the Dynamo adapter.

This SUPERSEDES the earlier ship-collected-trees parallel path. That path read
every record in the PARENT and then shipped the raw ``input_sequence_hashes``
(~900 ints per record, tens of millions per file) across the process boundary
to build workers -- the IPC serialization of those arrays was the wall (only
~1.35x on real captures). Here the raw hashes NEVER cross a process boundary:
they are parsed only inside the worker that builds them, and only the compact
built graph (no hashes) returns.

Three phases, all reusing the shared pool lifecycle
(:func:`~aiperf.dataset.graph.adapters.shared.pool.run_pool_streaming`
-- forkserver context, parent-built shared-memory corpus, bounded ordered
window, graceful shutdown) with dynamo's own ``DYNAMO_GRAPH_PARALLEL_*`` tuning:

1. **Grouping scan** (parent, :func:`_scan_grouping`): reuse a valid ingest
   sidecar, else decode and validate every line via :func:`scan_dynamo_trace`
   with the serial reader's JSON, envelope, schema, and identity semantics.
   This yields the session -> tree-root assignment (the SAME walk
   :func:`~aiperf.dataset.graph.adapters.dynamo.trace.root_of_sessions` runs for
   the serial build), the pinned block size, and a per-session byte-length build
   weight.

2. **Shuffle** (parent, :func:`_shuffle_to_batch_files`): decode and validate
   every segment a second time, then append each raw record line VERBATIM to
   a per-batch gzip temp file, routed by the line's session id -> its batch.
   Streaming (bounded memory), skipping schema-less marker lines and records
   with no session (dropped exactly as the serial path drops no-``agent_context``
   records). Each batch temp file is self-contained: every record of every tree
   assigned to that batch, regardless of which source segment it came from, so a
   subagent whose records are split across two files still lands whole in ONE
   batch.

3. **Fused build** (workers, :func:`_build_batch_file_to_blob`): each worker
   READS its batch temp file (:func:`_collect_chains`) and BUILDS it
   (:func:`_build_trees_sequential`, the inner build STRICTLY sequential -- no
   nested pool) with the parent-pinned seed + block size + shared-memory corpus.
   The shuffle makes every session-tree BATCH-LOCAL, so a worker always sees
   COMPLETE trees and returns its batch's LIST of per-tree single-graph
   ``ParsedGraph`` blobs (each keyed by its root id). The parent decodes each
   frame in input (batch) order and FLATTENS all workers' per-tree graphs, then
   the caller merges them via
   :func:`~aiperf.dataset.graph.merge.merge_parsed_graphs`.

Contiguous batching over the globally sorted-by-root tree list keeps the
parent's flattened per-tree list byte-identical to
:func:`~aiperf.dataset.graph.adapters.dynamo.trace._build_trees_sequential` over
the same capture, so the merged multi-graph workload is identical: same per-tree
node keys/order, same edge set, same content-addressed ``segment_pool``.
"""

from __future__ import annotations

import functools
import gzip
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import IO, Any, TypeVar

import msgspec

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.dataset.graph.adapters.dynamo.trace_reader import (
    AgentTraceRecord,
    DynamoIngestScan,
    DynamoTraceAdapterError,
    DynamoTraceReadError,
    discover_segments,
    load_ingest_sidecar,
    normalize_dynamo_record,
    parse_trace_line,
    scan_dynamo_trace,
)
from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
    DEFAULT_VIRTUAL_BLOCK_SIZE,
)
from aiperf.dataset.graph.adapters.shared.selection import (
    SelectionStats,
    log_selection_summary,
)
from aiperf.dataset.graph.codecs import (
    decode_parsed_graph_msgpack,
    encode_parsed_graph_msgpack,
)
from aiperf.dataset.graph.models import ParsedGraph
from aiperf.dataset.graph.parser import GraphParseError
from aiperf.dataset.graph.segment_trie.store_builder import TraceSegmentPayload

_T = TypeVar("_T")

# Batch-result frame codecs: ``[encode_parsed_graph_msgpack(pg) bytes, tags]``.
# The nested typed ``ParsedGraph`` blob keeps the existing cross-process codec
# for the graph + pool; the outer plain frame carries the trace tags alongside.
_FRAME_ENCODER = msgspec.msgpack.Encoder()
_FRAME_DECODER = msgspec.msgpack.Decoder()
_logger = AIPerfLogger(__name__)

_TIMEOUT_HINT = (
    "dynamo graph build worker produced no result for one session-tree batch: "
    "the worker process was most likely killed mid-build (OOM kill / external "
    "SIGKILL) -- a raw multiprocessing Pool cannot complete a killed worker's "
    "in-flight task. Reduce worker count "
    "(AIPERF_DATASET_DYNAMO_GRAPH_PARALLEL_WORKERS) to lower peak memory, or "
    "raise AIPERF_DATASET_DYNAMO_GRAPH_PARALLEL_ITEM_TIMEOUT_SECONDS if a single "
    "batch legitimately builds slower than the timeout."
)


def _dynamo_threshold() -> int:
    from aiperf.common.environment import Environment

    return max(0, Environment.DATASET.DYNAMO_GRAPH_PARALLEL_THRESHOLD)


def _dynamo_workers(*, item_count: int) -> int:
    """Resolve the worker count from ``DYNAMO_GRAPH_PARALLEL_*``, capped at trees.

    0 = auto (``min(cpu_count - 1, DYNAMO_GRAPH_PARALLEL_AUTO_MAX_WORKERS)``);
    a positive value pins it. Always capped at ``item_count`` (the tree count)
    so a two-tree capture never spawns sixteen idle workers.
    """
    from aiperf.common.environment import Environment

    configured = Environment.DATASET.DYNAMO_GRAPH_PARALLEL_WORKERS
    if configured > 0:
        resolved = configured
    else:
        cpu = os.cpu_count() or 1
        resolved = min(
            max(cpu - 1, 1),
            Environment.DATASET.DYNAMO_GRAPH_PARALLEL_AUTO_MAX_WORKERS,
        )
    return max(1, min(resolved, item_count))


def _io_threads() -> int:
    """Thread count for the parallel shuffle's decompression round.

    gzip decompression releases the GIL, so threads parallelize the segment
    reads without a process pool. Capped at the CPU count (and, inside each
    round, at the segment count) -- more decompress threads than cores buys
    nothing once every segment already owns one.
    """
    return max(1, os.cpu_count() or 1)


def _handle_budget() -> int:
    """Max simultaneously-open temp-file writers, from the fd soft limit.

    Phase 2 keeps one gzip writer open per batch; capping the batch count at a
    fraction of ``RLIMIT_NOFILE`` (leaving headroom for the input read, the pool,
    stdio and worker fds) keeps a capture with tens of thousands of trees from
    exhausting file descriptors. ``resource`` is Unix-only (the forkserver pool
    is Linux/macOS anyway); a conservative constant is used where it is absent.
    """
    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ImportError, ValueError, OSError):
        return 256
    if soft <= 0:
        return 256
    return max(8, soft - 256)


def _contiguous_weight_batches(
    items: Sequence[_T], weights: Sequence[int], *, num_batches: int
) -> list[list[_T]]:
    """Split ``items`` into <= ``num_batches`` CONTIGUOUS weight-balanced batches.

    Contiguous (order-preserving) so a downstream in-order union reproduces the
    global item order exactly. Each batch accumulates items until its cumulative
    weight crosses the next ``k * total / num_batches`` boundary, always
    reserving at least one item for every still-unopened batch, so batches carry
    roughly equal total weight; a single item heavier than the target closes its
    batch alone (items are atomic). ``weights`` must be positive (the caller
    floors them at 1) so a zero-weight run cannot collapse every item into one
    batch.
    """
    if not items:
        return []
    n = max(1, min(num_batches, len(items)))
    if n == 1:
        return [list(items)]

    total = sum(weights)
    batches: list[list[_T]] = []
    current: list[_T] = []
    cumulative = 0
    for index, (item, weight) in enumerate(zip(items, weights, strict=True)):
        current.append(item)
        cumulative += weight
        batches_unopened = n - len(batches) - 1
        items_left = len(items) - (index + 1)
        boundary = (len(batches) + 1) * total / n
        crossed = cumulative >= boundary
        room_for_rest = items_left >= batches_unopened
        if batches_unopened >= 1 and crossed and room_for_rest:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


# --- Phase 1: cheap grouping scan (no hash parse) --------------------------


@dataclass(slots=True)
class _GroupingScan(DynamoIngestScan):
    """Result of the parent's serial-equivalent grouping scan over every segment.

    ``request_end_sessions`` are the sessions that carry at least one
    ``request_end`` (the serial build's ``chains`` keys); ``parent_link`` maps a
    session to its first non-self parent (``parent_trajectory_id`` preferred over
    ``parent_session_id``, first occurrence wins -- identical to
    :func:`~aiperf.dataset.graph.adapters.dynamo.trace._collect_records`);
    ``session_weight`` is the summed record-line byte length per session (a cheap
    proxy for the recorded hash volume the worker will parse, needing no hash
    parse); ``block_size`` is the single recorded ``trace_block_size`` across the
    whole capture (fail-loud on a mix), defaulting to the virtual size when no
    replay is present.
    """

    request_end_sessions: set[str] = field(default_factory=set)
    record_count: int = 0
    parent_link: dict[str, str] = field(default_factory=dict)
    session_weight: dict[str, int] = field(default_factory=dict)
    session_peak: dict[str, int] = field(default_factory=dict)
    session_start_ms: dict[str, int] = field(default_factory=dict)


def _open_raw(path: Path) -> IO[bytes]:
    """Open a segment for raw BYTE line iteration (gzip transparently)."""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _parse_typed_line(line: bytes, *, segment: Path) -> AgentTraceRecord | None:
    """Apply the serial reader's wire and model semantics to one raw line."""
    raw = parse_trace_line(line, source=str(segment))
    if raw is None:
        return None
    try:
        return AgentTraceRecord.model_validate(raw)
    except Exception as exc:
        raise DynamoTraceReadError(
            f"failed to parse trace record: {exc!s} (raw keys: {sorted(raw.keys())})"
        ) from exc


def _scan_grouping(
    path: str | Path, *, threads: int, capture_peak: bool = False
) -> _GroupingScan:
    """Phase 1: scan canonical records in serial source order.

    ``threads`` remains part of the call contract, but global first-seen
    de-duplication requires that aggregation occur before any metadata update.
    Parsing and validating each line also keeps envelope, escaping, and schema
    semantics identical to the serial reader.
    """
    started = perf_counter()
    _logger.info(f"Dynamo load: parallel grouping scan started for {path}")
    _ = threads
    cached = load_ingest_sidecar(path) if not capture_peak else None
    base = cached or scan_dynamo_trace(path, capture_peak=capture_peak)
    merged = _GroupingScan(
        physical_record_count=base.physical_record_count,
        canonical_record_count=base.canonical_record_count,
        request_end_count=base.request_end_count,
        duplicate_count=base.duplicate_count,
        skipped_record_count=base.skipped_record_count,
        synthetic_session_count=base.synthetic_session_count,
        block_size=base.block_size,
        source_fingerprint=base.source_fingerprint,
        segments=base.segments,
        block_sizes=set(base.block_sizes),
        sessions=dict(base.sessions),
        record_count=base.physical_record_count,
    )
    merged.request_end_sessions = {
        session_id
        for session_id, summary in merged.sessions.items()
        if summary.request_end_count
    }
    merged.parent_link = {
        session_id: summary.parent_session_id
        for session_id, summary in merged.sessions.items()
        if summary.parent_session_id is not None
    }
    merged.session_weight = {
        session_id: max(1, summary.byte_weight)
        for session_id, summary in merged.sessions.items()
    }
    merged.session_peak = {
        session_id: summary.peak_context
        for session_id, summary in merged.sessions.items()
        if summary.peak_context
    }
    merged.session_start_ms = {
        session_id: summary.first_request_end_ms
        for session_id, summary in merged.sessions.items()
    }

    if len(merged.block_sizes) > 1:
        raise DynamoTraceAdapterError(
            f"mixed replay trace_block_size values are not supported: "
            f"{sorted(merged.block_sizes)}"
        )
    merged.block_size = next(iter(merged.block_sizes), DEFAULT_VIRTUAL_BLOCK_SIZE)
    _logger.info(
        f"Dynamo load: parallel grouping scan complete in "
        f"{perf_counter() - started:.2f}s ({merged.physical_record_count:,} "
        f"physical records, {merged.request_end_count:,} request_end records, "
        f"{len(merged.request_end_sessions):,} sessions, "
        f"{merged.duplicate_count:,} duplicate records)"
    )
    return merged


# --- Phase 2: ordered raw-line shuffle to per-batch temp files -------------


_SHUFFLE_SUFFIX = ".jsonl.gz"


def _shuffle_produce_segment(
    segment_index: int,
    segment: Path,
    session_to_batch: dict[str, int],
    tmpdir: Path,
) -> dict[int, Path]:
    """Write one segment's routed lines to source-local batch fragments.

    Each fragment retains source line order. The parent merges fragments by
    segment index before workers parse them, preserving serial first-seen
    deduplication and parent-link semantics.
    """
    writers: dict[int, IO[bytes]] = {}
    paths: dict[int, Path] = {}
    try:
        with _open_raw(segment) as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = _parse_typed_line(line, segment=segment)
                if record is None:
                    continue
                normalized = normalize_dynamo_record(record)
                if normalized is None:
                    continue
                batch = session_to_batch.get(normalized.session_id)
                if batch is None:
                    continue
                writer = writers.get(batch)
                if writer is None:
                    fragment = tmpdir / (
                        f"segment_{segment_index:06d}_batch_{batch:05d}"
                        f"{_SHUFFLE_SUFFIX}"
                    )
                    writer = gzip.open(fragment, "wb", compresslevel=1)  # noqa: SIM115
                    writers[batch] = writer
                    paths[batch] = fragment
                writer.write(line if line.endswith(b"\n") else line + b"\n")
    finally:
        close_error: BaseException | None = None
        for writer in writers.values():
            try:
                writer.close()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                close_error = close_error or exc
        if close_error is not None:
            raise close_error
    return paths


def _merge_ordered_shuffle_fragments(
    fragments: list[dict[int, Path]], batch_count: int, tmpdir: Path
) -> list[Path]:
    """Merge segment fragments into source-ordered batch files."""
    batch_files: list[Path] = []
    for batch in range(batch_count):
        output = tmpdir / f"batch_{batch:05d}{_SHUFFLE_SUFFIX}"
        wrote_fragment = False
        try:
            with gzip.open(output, "wb", compresslevel=1) as destination:
                for segment_fragments in fragments:
                    fragment = segment_fragments.get(batch)
                    if fragment is None:
                        continue
                    wrote_fragment = True
                    with gzip.open(fragment, "rb") as source:
                        shutil.copyfileobj(source, destination)
            if wrote_fragment:
                batch_files.append(output)
            else:
                output.unlink(missing_ok=True)
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    return batch_files


def _shuffle_to_batch_files(
    path: str | Path,
    session_to_batch: dict[str, int],
    tmpdir: Path,
    *,
    threads: int,
    batch_count: int,
) -> list[Path]:
    """Route records to source-ordered per-batch temporary files.

    Segment decompression remains parallel, but each producer writes only its
    own fragments. The deterministic merge preserves global source order.
    """
    started = perf_counter()
    segments = discover_segments(Path(path))
    _logger.info(
        f"Dynamo load: parallel shuffle started "
        f"({len(segments):,} segments, {batch_count:,} batches)"
    )
    # A producer can open one fragment writer for every batch. Cap active
    # producers so writer descriptors never exceed the same budget that caps
    # final batch count; the subsequent merge needs only two more descriptors.
    producer_limit = max(1, _handle_budget() // max(1, batch_count))
    workers = max(1, min(threads, len(segments), producer_limit))
    produce = functools.partial(
        _shuffle_produce_segment,
        session_to_batch=session_to_batch,
        tmpdir=tmpdir,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fragments = list(pool.map(produce, range(len(segments)), segments))
    batch_files = _merge_ordered_shuffle_fragments(fragments, batch_count, tmpdir)
    _logger.info(
        f"Dynamo load: parallel shuffle complete in "
        f"{perf_counter() - started:.2f}s ({len(batch_files):,} batch files)"
    )
    return batch_files


# --- Phase 3: fused per-batch read+build in workers ------------------------


def _build_batch_file_to_blob(task: tuple[str, dict[str, Any], int]) -> bytes:
    """Pool worker entry: READ one batch temp file and BUILD it, return a frame.

    The FUSED step: the recorded hashes are parsed here (worker-side), never in
    the parent. Regroups the file's chains into trees and calls the SERIAL
    per-tree loop directly (never ``from_dynamo_trace`` / ``_build_trees_flat``,
    which would re-enter the parallel dispatch and spawn a NESTED pool), so the
    inner build is strictly single-threaded. The shuffle keeps every session-tree
    BATCH-LOCAL, so this worker always sees COMPLETE trees and returns its batch's
    LIST of per-tree single-graph ``ParsedGraph`` blobs (the parent flattens all
    workers' per-tree graphs and merges). ``direct_store=None`` -- the parallel
    path never carries a live write-through store.
    """
    from aiperf.dataset.graph.adapters.dynamo.trace import (
        _build_trees_sequential,
        _collect_chains,
        group_chains_into_trees,
    )

    file_path, build_kwargs, max_depth = task
    chains = _collect_chains(file_path, None, max_depth=max_depth)
    parent_link = {
        sid: chain.parent_session_id
        for sid, chain in chains.items()
        if chain.parent_session_id is not None
    }
    trees = group_chains_into_trees(chains, parent_link)
    tails: list[tuple[int, int, int, int]] = []
    per_tree = _build_trees_sequential(
        trees, direct_store=None, tails_out=tails, **build_kwargs
    )
    return _encode_batch_result(per_tree, tails[0] if tails else (0, 0, 0, 0))


def _build_batch_file_to_payloads(
    task: tuple[str, dict[str, Any], int],
) -> tuple[list[TraceSegmentPayload], tuple[int, int, int, int]]:
    """Pool worker entry: build one batch, emit payloads + dropped-tail counts.

    Returns ``(payloads, (tail_nodes, tail_tokens, trees))``. The counts ride
    back with the payloads because a pool worker has no configured log handler,
    so the rollup can only be emitted by the parent.
    """
    from aiperf.dataset.graph.adapters.dynamo.trace import (
        _build_trees_sequential,
        _collect_chains,
        group_chains_into_trees,
    )
    from aiperf.dataset.graph.segment_trie.store_builder import (
        iter_trace_segment_payloads,
    )

    file_path, build_kwargs, max_depth = task
    chains = _collect_chains(file_path, None, max_depth=max_depth)
    parent_link = {
        sid: chain.parent_session_id
        for sid, chain in chains.items()
        if chain.parent_session_id is not None
    }
    trees = group_chains_into_trees(chains, parent_link)
    tails: list[tuple[int, int, int, int]] = []
    per_tree = _build_trees_sequential(
        trees, direct_store=None, tails_out=tails, **build_kwargs
    )
    return (
        [
            payload
            for graph in per_tree
            for payload in iter_trace_segment_payloads(graph)
        ],
        tails[0] if tails else (0, 0, 0, 0),
    )


def _log_dropped_tail_rollup(
    nodes: int, tokens: int, trees: int, recorded: int
) -> None:
    """Emit the parent-side dropped-tail rollup, or stay silent when there is none.

    Both parallel parents call this after summing their workers' returned
    counts. A corpus whose prompts are all block-aligned drops nothing and must
    log nothing -- a zero line would train the reader to ignore it.
    """
    if not nodes:
        return
    from aiperf.dataset.graph.adapters.dynamo.trace import format_dropped_tail_rollup

    _logger.info(format_dropped_tail_rollup(nodes, tokens, trees, recorded))


def _encode_batch_result(
    per_tree: list[ParsedGraph], tails: tuple[int, int, int, int]
) -> bytes:
    """Encode a batch's per-tree ``ParsedGraph``s to a cross-process list frame.

    Each per-tree ``ParsedGraph`` (one root ``TraceRecord`` + that tree's graph
    and pool) is msgpack-encoded through the existing typed codec; the outer
    frame is ``[pg_blobs, dropped_tail_counts]`` in the batch's tree order.

    ``tails`` is ``(nodes, tokens, trees)``: the worker cannot LOG its dropped
    partial block tails (no handler is configured in a pool worker), so it
    returns them and the parent rolls them up into one line.
    """
    return _FRAME_ENCODER.encode(
        [[encode_parsed_graph_msgpack(pg) for pg in per_tree], list(tails)]
    )


def _decode_batch_result(
    blob: bytes,
) -> tuple[list[ParsedGraph], tuple[int, int, int, int]]:
    """Decode a batch frame into its ``ParsedGraph``s and dropped-tail counts."""
    pg_blobs, tails = _FRAME_DECODER.decode(blob)
    return (
        [decode_parsed_graph_msgpack(pg_bytes) for pg_bytes in pg_blobs],
        (tails[0], tails[1], tails[2], tails[3]),
    )


# --- orchestration ---------------------------------------------------------


def maybe_build_fused_parallel(
    path: str | Path,
    *,
    content_root_seed: int,
    idle_gap_cap_seconds: float | None,
    content_tokenizer: str | None,
    prompt_corpus: str,
    release_replay: bool,
    max_depth: int,
    num_dataset_entries: int | None = None,
    max_context_length: int | None = None,
    max_osl: int | None = None,
    streaming: bool | None = None,
    selection_out: list[SelectionStats] | None = None,
) -> list[ParsedGraph] | None:
    """Fuse read+build across a process pool, or ``None`` to stay serial.

    Runs the Phase-1 grouping scan (cheap, no hash parse) to learn the tree
    count, then returns ``None`` -- caller runs the serial read-then-build, NO
    pool spawn -- when the tree count is at or below
    ``DYNAMO_GRAPH_PARALLEL_THRESHOLD`` or the resolved worker count collapses to
    1 (a single tree, or ``DYNAMO_GRAPH_PARALLEL_WORKERS=1``). Otherwise shuffles
    the raw lines to per-batch temp files and builds every batch on the pool,
    returning the flattened LIST of per-tree single-graph ``ParsedGraph``s (in
    contiguous global tree order) -- byte-identical to
    :func:`~aiperf.dataset.graph.adapters.dynamo.trace._build_trees_sequential`
    over the same capture. The caller folds the base tag and merges them via
    :func:`~aiperf.dataset.graph.merge.merge_parsed_graphs`.

    ``num_dataset_entries`` / ``max_context_length`` drive the SAME filter-then-cap
    tree selection the serial path applies (ai-dynamo/aiperf#1106): the scan
    additionally records per-session peak context (hash-free), trees are screened
    by ``--max-context-length`` and capped at ``--num-dataset-entries`` (arrival-ordered
    order), and ONLY the selected trees are shuffled + built. When both are ``None``
    no selection runs and the output stays byte-identical. ``selection_out``
    receives the :class:`SelectionStats` only when the pool path actually builds
    (a decline hands selection to the serial path, which appends its own).
    """
    started = perf_counter()

    select = num_dataset_entries is not None or max_context_length is not None
    io_threads = _io_threads()
    scan = _scan_grouping(path, threads=io_threads, capture_peak=select)
    if not scan.request_end_sessions:
        # No lowerable records: let the serial path raise the precise
        # EmptyDynamoTraceError (it distinguishes "empty" from "no session
        # identity" via its skip counter, which this scan does not track).
        return None

    from aiperf.dataset.graph.adapters.dynamo.trace import root_of_sessions

    root_of = root_of_sessions(scan.request_end_sessions, scan.parent_link)
    roots = _roots_in_arrival_order(scan, root_of)

    stats: SelectionStats | None = None
    if select:
        roots, root_of, stats = _select_roots_filter_then_cap(
            scan,
            root_of=root_of,
            roots=roots,
            num_dataset_entries=num_dataset_entries,
            max_context_length=max_context_length,
        )
        if not roots:
            # Every tree filtered out: defer to the serial path so it raises the
            # precise EmptyDynamoTraceError (and appends the authoritative stats).
            return None

    if len(roots) <= _dynamo_threshold():
        _logger.info(
            f"Dynamo load: parallel build declined ({len(roots):,} trees <= "
            "threshold); serial path will handle it"
        )
        return None
    workers = _dynamo_workers(item_count=len(roots))
    if workers <= 1:
        _logger.info(
            f"Dynamo load: parallel build declined (resolved worker count="
            f"{workers}); serial path will handle it"
        )
        return None

    if stats is not None:
        # Parent-side finalize point for the fused BUILD path: the scan ran once
        # in this parent process, so log the summary once here (a fused DECLINE
        # returns above and hands logging to the serial fallback).
        log_selection_summary(
            stats,
            source=str(path),
            num_dataset_entries=num_dataset_entries,
            max_context_length=max_context_length,
        )
        if selection_out is not None:
            selection_out.append(stats)
    _logger.info(
        f"Dynamo load: parallel build selected ({len(roots):,} trees, "
        f"{workers} workers; dispatch decision took "
        f"{perf_counter() - started:.2f}s)"
    )
    result = _build_fused_parallel(
        path,
        scan=scan,
        root_of=root_of,
        roots=roots,
        workers=workers,
        io_threads=io_threads,
        content_root_seed=content_root_seed,
        idle_gap_cap_seconds=idle_gap_cap_seconds,
        content_tokenizer=content_tokenizer,
        prompt_corpus=prompt_corpus,
        release_replay=release_replay,
        max_depth=max_depth,
        max_osl=max_osl,
        streaming=streaming,
    )
    _logger.info(
        f"Dynamo load: parallel build complete in "
        f"{perf_counter() - started:.2f}s ({len(result):,} trees)"
    )
    return result


def stream_dynamo_trace_segment_payloads(
    path: str | Path,
    *,
    content_root_seed: int | None,
    idle_gap_cap_seconds: float | None,
    content_tokenizer: str | None,
    prompt_corpus: str,
    release_replay: bool,
    max_depth: int,
    num_dataset_entries: int | None = None,
    max_context_length: int | None = None,
    max_isl: int | None = None,
    max_osl: int | None = None,
    streaming: bool | None = None,
    ignore_trace_delays: bool = False,
) -> Iterator[TraceSegmentPayload]:
    """Yield Dynamo store payloads while worker results remain bounded.

    This is the store-build variant of :func:`maybe_build_fused_parallel`:
    workers lower batches into per-trace ``TraceSegmentPayload`` objects and
    the parent consumes them immediately through the unified-store drain. It
    never returns the corpus-sized ``ParsedGraph`` list to the parent.
    """
    from aiperf.common.environment import Environment
    from aiperf.dataset.graph.adapters.shared.content import (
        resolve_effective_root_seed,
    )

    if max_isl is not None:
        raise DynamoTraceAdapterError(
            "streaming Dynamo graph build does not support max_isl selection"
        )

    started = perf_counter()
    content_root_seed = resolve_effective_root_seed(content_root_seed)
    if ignore_trace_delays and idle_gap_cap_seconds is None:
        idle_gap_cap_seconds = 0.0
    select = num_dataset_entries is not None or max_context_length is not None
    scan = _scan_grouping(path, threads=_io_threads(), capture_peak=select)
    if not scan.request_end_sessions:
        raise DynamoTraceAdapterError(f"{path}: no request_end events found")

    from aiperf.dataset.graph.adapters.dynamo.trace import root_of_sessions

    root_of = root_of_sessions(scan.request_end_sessions, scan.parent_link)
    # Arrival order, NOT sorted root ids: the cap keeps the FIRST N eligible
    # trees, so this is what makes a bounded load a contiguous slice of the
    # recorded TIMELINE instead of of the alphabet (see
    # ``trace.order_trees_by_recorded_start``). Must match the fused path above.
    roots = _roots_in_arrival_order(scan, root_of)
    stats: SelectionStats | None = None
    if select:
        roots, root_of, stats = _select_roots_filter_then_cap(
            scan,
            root_of=root_of,
            roots=roots,
            num_dataset_entries=num_dataset_entries,
            max_context_length=max_context_length,
        )
    workers = _dynamo_workers(item_count=len(roots)) if roots else 0
    if not roots or len(roots) <= _dynamo_threshold() or workers <= 1:
        from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
        from aiperf.dataset.graph.segment_trie.store_builder import (
            iter_trace_segment_payloads,
        )

        try:
            parsed = from_dynamo_trace(
                path,
                content_root_seed=content_root_seed,
                idle_gap_cap_seconds=idle_gap_cap_seconds,
                content_tokenizer=content_tokenizer,
                prompt_corpus=prompt_corpus,
                release_replay=release_replay,
                num_dataset_entries=num_dataset_entries,
                max_context_length=max_context_length,
                max_osl=max_osl,
                streaming=streaming,
                ignore_trace_delays=ignore_trace_delays,
            )
        except ValueError as exc:
            raise GraphParseError(str(exc)) from exc
        payload_iter = iter_trace_segment_payloads(parsed)
        _logger.info("Dynamo load: fallback payload lowering started")
        yield from payload_iter
        _logger.info("Dynamo load: fallback payload lowering complete")
        return

    # AFTER the fallback returns, mirroring ``maybe_build_fused_parallel``: the
    # serial fallback re-selects inside ``from_dynamo_trace``, which logs the
    # summary itself, so logging before the branch double-reported it on every
    # run small enough to decline -- including every ``--num-dataset-entries``
    # below the parallel threshold, the exact case this summary exists for.
    if stats is not None:
        log_selection_summary(
            stats,
            source=str(path),
            num_dataset_entries=num_dataset_entries,
            max_context_length=max_context_length,
        )

    prefetch = Environment.DATASET.DYNAMO_GRAPH_PARALLEL_PREFETCH_MULTIPLIER
    tree_weight: dict[str, int] = defaultdict(int)
    for sid, root in root_of.items():
        tree_weight[root] += scan.session_weight.get(sid, 0)
    weights = [max(1, tree_weight[root]) for root in roots]
    num_batches = max(1, min(len(roots), workers * prefetch, _handle_budget()))
    root_batches = _contiguous_weight_batches(roots, weights, num_batches=num_batches)
    root_to_batch = {
        root: index for index, group in enumerate(root_batches) for root in group
    }
    session_to_batch = {sid: root_to_batch[root_of[sid]] for sid in root_of}
    build_kwargs: dict[str, Any] = {
        "block_size": scan.block_size,
        "content_root_seed": content_root_seed,
        "idle_gap_cap_seconds": idle_gap_cap_seconds,
        "content_tokenizer": content_tokenizer,
        "prompt_corpus": prompt_corpus,
        "release_replay": release_replay,
        "max_osl": max_osl,
        "streaming": streaming,
    }
    tmpdir = Path(tempfile.mkdtemp(prefix="aiperf-dynamo-fused-payloads-"))
    _logger.info(
        f"Dynamo load: streaming payload build selected ({len(roots):,} trees, "
        f"{workers} workers, {num_batches:,} batches)"
    )
    # Workers cannot log their dropped partial block tails (no handler in a
    # pool worker), so they return counts and the parent emits ONE rollup.
    tail_nodes = tail_tokens = tail_trees = tail_recorded = 0
    try:
        batch_files = _shuffle_to_batch_files(
            path,
            session_to_batch,
            tmpdir,
            threads=_io_threads(),
            batch_count=len(root_batches),
        )
        tasks = [
            (str(batch_file), build_kwargs, max_depth) for batch_file in batch_files
        ]
        from aiperf.dataset.graph.adapters.shared.pool import run_pool_streaming

        for payloads in run_pool_streaming(
            _build_batch_file_to_payloads,
            tasks,
            workers=workers,
            root_seed=content_root_seed,
            content_tokenizer=content_tokenizer,
            prompt_corpus=prompt_corpus,
            prefetch_multiplier=prefetch,
            item_timeout_s=Environment.DATASET.DYNAMO_GRAPH_PARALLEL_ITEM_TIMEOUT_SECONDS,
            timeout_hint=_TIMEOUT_HINT,
        ):
            payload_batch, batch_tails = payloads
            tail_nodes += batch_tails[0]
            tail_tokens += batch_tails[1]
            tail_trees += batch_tails[2]
            tail_recorded += batch_tails[3]
            yield from payload_batch
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    _log_dropped_tail_rollup(tail_nodes, tail_tokens, tail_trees, tail_recorded)
    _logger.info(
        f"Dynamo load: streaming payload build finished in "
        f"{perf_counter() - started:.2f}s"
    )


def _roots_in_arrival_order(scan: _GroupingScan, root_of: dict[str, str]) -> list[str]:
    """Tree roots ordered by arrival time, root id breaking ties.

    The parallel twin of
    :func:`~aiperf.dataset.graph.adapters.dynamo.trace.order_trees_by_recorded_start`.
    A tree arrives when its EARLIEST ``request_end`` across all its sessions was
    recorded -- the same instant the serial path derives from its chains' turns,
    which is what keeps the two selections identical. Sorting on
    ``(arrival, root_id)`` rather than sorting by arrival alone makes the tie
    order explicit here, since the input is a ``set`` and has no order to be
    stable with respect to.
    """
    arrival: dict[str, int] = {}
    for session_id, root in root_of.items():
        start = scan.session_start_ms.get(session_id, sys.maxsize)
        if start < arrival.get(root, sys.maxsize):
            arrival[root] = start
    return sorted(
        set(root_of.values()), key=lambda root: (arrival.get(root, sys.maxsize), root)
    )


def _select_roots_filter_then_cap(
    scan: _GroupingScan,
    *,
    root_of: dict[str, str],
    roots: list[str],
    num_dataset_entries: int | None,
    max_context_length: int | None,
) -> tuple[list[str], dict[str, str], SelectionStats]:
    """Filter-then-cap the tree roots by per-tree peak context (parent-side).

    A tree's peak is the max over its sessions' scan-recorded peaks. The roots
    are screened in the order the CALLER supplies, which is arrival order
    (:func:`_roots_in_arrival_order`) on both parallel entry points, so the cap
    keeps the temporally-first N and the selected set matches the serial path's
    :func:`dynamo_tree_peak_context` selection. Returns the selected roots in
    that same order, the ``root_of`` map restricted to the selected trees'
    sessions, and the :class:`SelectionStats`.
    """
    from aiperf.dataset.graph.adapters.shared.selection import filter_then_cap

    tree_peak: dict[str, int] = defaultdict(int)
    for sid, root in root_of.items():
        peak = scan.session_peak.get(sid, 0)
        if peak > tree_peak[root]:
            tree_peak[root] = peak

    selected, stats = filter_then_cap(
        ((root, tree_peak.get(root, 0)) for root in roots),
        num_dataset_entries=num_dataset_entries,
        max_context_length=max_context_length,
    )
    selected_set = set(selected)
    restricted = {sid: root for sid, root in root_of.items() if root in selected_set}
    return selected, restricted, stats


def _build_fused_parallel(
    path: str | Path,
    *,
    scan: _GroupingScan,
    root_of: dict[str, str],
    roots: list[str],
    workers: int,
    io_threads: int,
    content_root_seed: int,
    idle_gap_cap_seconds: float | None,
    content_tokenizer: str | None,
    prompt_corpus: str,
    release_replay: bool,
    max_depth: int,
    max_osl: int | None,
    streaming: bool | None,
) -> list[ParsedGraph]:
    """Shuffle to per-batch temp files, fuse-build on the pool, flatten per-tree.

    ``content_root_seed`` and ``block_size`` are PINNED by the parent and
    threaded to every worker (the seed also seeds ``run_pool_streaming``'s
    parent-built shared-memory corpus), so the fused build is byte-identical to
    the serial loop. Trees are batched CONTIGUOUSLY over the sorted-by-root list
    (weighted by the Phase-1 byte proxy), so batch results arriving in input
    order flatten to the same global tree order. Each worker returns its batch's
    LIST of per-tree ``ParsedGraph`` blobs; this concatenates them across batches
    (arrival order) and hands the flat list back for the caller to merge. The
    temp dir is always removed.
    """
    from aiperf.common.environment import Environment
    from aiperf.dataset.graph.adapters.shared.pool import run_pool_streaming

    started = perf_counter()
    prefetch = Environment.DATASET.DYNAMO_GRAPH_PARALLEL_PREFETCH_MULTIPLIER
    item_timeout_s = Environment.DATASET.DYNAMO_GRAPH_PARALLEL_ITEM_TIMEOUT_SECONDS

    # Iterate ``root_of`` (not ``scan.request_end_sessions``): under selection it
    # is restricted to the kept trees' sessions, so a dropped session never
    # KeyErrors here. Unfiltered, ``root_of`` keys == ``request_end_sessions``,
    # so this is byte-identical to the prior weighting.
    tree_weight: dict[str, int] = defaultdict(int)
    for sid, root in root_of.items():
        tree_weight[root] += scan.session_weight.get(sid, 0)
    weights = [max(1, tree_weight[root]) for root in roots]

    num_batches = max(1, min(len(roots), workers * prefetch, _handle_budget()))
    root_batches = _contiguous_weight_batches(roots, weights, num_batches=num_batches)
    root_to_batch = {
        root: index for index, group in enumerate(root_batches) for root in group
    }
    session_to_batch = {sid: root_to_batch[root_of[sid]] for sid in root_of}
    _logger.info(
        f"Dynamo load: preparing parallel build ({len(roots):,} trees, "
        f"{num_batches:,} batches, {workers} workers)"
    )

    build_kwargs: dict[str, Any] = {
        "block_size": scan.block_size,
        "content_root_seed": content_root_seed,
        "idle_gap_cap_seconds": idle_gap_cap_seconds,
        "content_tokenizer": content_tokenizer,
        "prompt_corpus": prompt_corpus,
        "release_replay": release_replay,
        "max_osl": max_osl,
        "streaming": streaming,
    }

    tmpdir = Path(tempfile.mkdtemp(prefix="aiperf-dynamo-fused-"))
    try:
        batch_files = _shuffle_to_batch_files(
            path,
            session_to_batch,
            tmpdir,
            threads=io_threads,
            batch_count=len(root_batches),
        )
        tasks = [(str(bf), build_kwargs, max_depth) for bf in batch_files]

        per_tree: list[ParsedGraph] = []
        # Workers cannot log their dropped partial block tails (no handler in a
        # pool worker), so they return counts and the parent emits ONE rollup.
        tail_nodes = tail_tokens = tail_trees = tail_recorded = 0
        # Results arrive in INPUT (batch) order -> contiguous global tree order,
        # so the flattened per-tree list matches the serial per-tree loop and the
        # merge is byte-identical (traces are id-sorted, pools content-unioned).
        for blob in run_pool_streaming(
            _build_batch_file_to_blob,
            tasks,
            workers=workers,
            root_seed=content_root_seed,
            content_tokenizer=content_tokenizer,
            prompt_corpus=prompt_corpus,
            prefetch_multiplier=prefetch,
            item_timeout_s=item_timeout_s,
            timeout_hint=_TIMEOUT_HINT,
        ):
            batch_trees, batch_tails = _decode_batch_result(blob)
            per_tree.extend(batch_trees)
            tail_nodes += batch_tails[0]
            tail_tokens += batch_tails[1]
            tail_trees += batch_tails[2]
            tail_recorded += batch_tails[3]
        _log_dropped_tail_rollup(tail_nodes, tail_tokens, tail_trees, tail_recorded)

        _logger.info(
            f"Dynamo load: worker graph builds complete in "
            f"{perf_counter() - started:.2f}s ({len(per_tree):,} trees)"
        )
        return per_tree
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


__all__ = [
    "maybe_build_fused_parallel",
    "stream_dynamo_trace_segment_payloads",
]
