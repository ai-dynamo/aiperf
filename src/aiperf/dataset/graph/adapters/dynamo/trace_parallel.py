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

Three phases, all reusing the weka pool lifecycle
(:func:`~aiperf.dataset.graph.adapters.weka.trace_parallel._run_pool_streaming`
-- forkserver context, parent-built shared-memory corpus, bounded ordered
window, graceful shutdown) with dynamo's own ``DYNAMO_GRAPH_PARALLEL_*`` tuning:

1. **Grouping scan** (parent, :func:`_scan_grouping`): decompress every segment
   ONCE and extract ONLY ``session_id`` + parent links + ``trace_block_size``
   per line via compiled regexes bounded to the prefix BEFORE the giant
   ``input_sequence_hashes`` array (``agent_context`` and the block size both
   precede it in the wire order), so the hash arrays are never scanned. This
   yields the session -> tree-root assignment (the SAME walk
   :func:`~aiperf.dataset.graph.adapters.dynamo.trace.root_of_sessions` runs for
   the serial build), the pinned block size, and a per-session byte-length build
   weight.

2. **Shuffle** (parent, :func:`_shuffle_to_batch_files`): decompress every
   segment a second time and append each raw record line VERBATIM (unparsed) to
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
import queue
import re
import shutil
import tempfile
import threading
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, TypeVar

import msgspec

from aiperf.dataset.graph.adapters.dynamo.trace_reader import (
    DynamoTraceAdapterError,
    discover_segments,
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

_T = TypeVar("_T")

# Batch-result frame codecs: ``[encode_parsed_graph_msgpack(pg) bytes, tags]``.
# The nested typed ``ParsedGraph`` blob keeps the existing cross-process codec
# for the graph + pool; the outer plain frame carries the trace tags alongside.
_FRAME_ENCODER = msgspec.msgpack.Encoder()
_FRAME_DECODER = msgspec.msgpack.Decoder()

# Regexes for the grouping scan. Every field they capture (``session_id``, the
# parent links, ``trace_block_size``) precedes ``input_sequence_hashes`` in the
# wire order, so the caller bounds every search to the prefix before that array
# and the giant hash lists are never scanned. ``"session_id":"`` cannot match
# inside ``"parent_session_id":"`` (the char before ``session_id`` there is ``_``,
# not ``"``), so the anchored quote reliably isolates the real session id.
_SESSION_RE = re.compile(rb'"session_id":"([^"]+)"')
_PARENT_TRAJ_RE = re.compile(rb'"parent_trajectory_id":"([^"]+)"')
_PARENT_SESSION_RE = re.compile(rb'"parent_session_id":"([^"]+)"')
_BLOCK_SIZE_RE = re.compile(rb'"trace_block_size":\s*(\d+)')
_REQUEST_END_MARK = b'"event_type":"request_end"'
_HASHES_KEY = b'"input_sequence_hashes"'

# Hash-free scalar reads for the OPTIONAL peak-context selection scan. Every
# field precedes ``input_sequence_hashes`` in the wire order (``input_length``
# lives inside ``replay`` before its hash array; ``input_tokens`` /
# ``output_tokens`` are ``request`` fields before ``replay``), so the caller's
# prefix bound keeps them hash-free. ``input_length`` appears ONLY in replay, so
# the regex reliably isolates it. These MIRROR
# :func:`~aiperf.dataset.graph.adapters.shared.peak_context.dynamo_tree_peak_context`
# so the parent's scan-based tree selection matches the serial helper.
_INPUT_LENGTH_RE = re.compile(rb'"input_length":\s*(\d+)')
_INPUT_TOKENS_RE = re.compile(rb'"input_tokens":\s*(\d+)')
_OUTPUT_TOKENS_RE = re.compile(rb'"output_tokens":\s*(\d+)')

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
    """Thread count for the parallel decompression rounds (scan + shuffle).

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
class _GroupingScan:
    """Result of the parent's hash-free grouping scan over every segment.

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
    parent_link: dict[str, str] = field(default_factory=dict)
    session_weight: dict[str, int] = field(default_factory=dict)
    session_peak: dict[str, int] = field(default_factory=dict)
    block_sizes: set[int] = field(default_factory=set)
    block_size: int = DEFAULT_VIRTUAL_BLOCK_SIZE


def _open_raw(path: Path) -> IO[bytes]:
    """Open a segment for raw BYTE line iteration (gzip transparently)."""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _line_peak_context(line: bytes, end: int) -> int:
    """Hash-free ``input_length(+output)`` peak for ONE request_end line prefix.

    Mirrors
    :func:`~aiperf.dataset.graph.adapters.shared.peak_context.dynamo_tree_peak_context`
    exactly: ``replay.input_length`` when present, else ``input_tokens`` (a
    recorded 0 is treated as absent -> 1, matching the helper's truthiness
    fallback), else 1; plus ``output_tokens`` or 0. Searches only the prefix
    before ``input_sequence_hashes`` so no hash array is scanned.
    """
    il_match = _INPUT_LENGTH_RE.search(line, 0, end)
    if il_match is not None:
        input_length = int(il_match.group(1))
    else:
        it_match = _INPUT_TOKENS_RE.search(line, 0, end)
        input_length = int(it_match.group(1)) if it_match is not None else 0
        if input_length == 0:
            input_length = 1
    ot_match = _OUTPUT_TOKENS_RE.search(line, 0, end)
    output_tokens = int(ot_match.group(1)) if ot_match is not None else 0
    return input_length + output_tokens


def _scan_one_file(segment: Path, *, capture_peak: bool = False) -> _GroupingScan:
    """Scan ONE segment for grouping fields, NO hash parse (thread worker).

    For each raw line the search is bounded to the bytes BEFORE
    ``"input_sequence_hashes"`` (found with one cheap ``bytes.find``), so the
    multi-kilobyte hash array is never scanned by a regex. A line with no
    ``session_id`` in that prefix has no ``agent_context`` and is ignored -- the
    serial path drops those records the same way. Runs under a thread: gzip
    decompression releases the GIL, so N segments scan concurrently.

    ``capture_peak`` (only when a selection knob is set) additionally records the
    per-session peak context over its ``request_end`` records via
    :func:`_line_peak_context`, so the parent can screen trees by
    ``--max-context-length`` / cap them by ``--num-dataset-entries`` before the
    build. It is OFF by default so the knob-less scan pays no extra regex cost.
    """
    partial = _GroupingScan()
    session_weight: dict[str, int] = defaultdict(int)
    session_peak: dict[str, int] = defaultdict(int)
    with _open_raw(segment) as handle:
        for line in handle:
            cut = line.find(_HASHES_KEY)
            end = len(line) if cut < 0 else cut
            match = _SESSION_RE.search(line, 0, end)
            if match is None:
                continue
            sid = match.group(1).decode()
            session_weight[sid] += len(line)
            if line.find(_REQUEST_END_MARK, 0, end) != -1:
                partial.request_end_sessions.add(sid)
                block_match = _BLOCK_SIZE_RE.search(line, 0, end)
                if block_match is not None:
                    partial.block_sizes.add(int(block_match.group(1)))
                if capture_peak:
                    peak = _line_peak_context(line, end)
                    if peak > session_peak[sid]:
                        session_peak[sid] = peak
            if sid not in partial.parent_link:
                parent_match = _PARENT_TRAJ_RE.search(
                    line, 0, end
                ) or _PARENT_SESSION_RE.search(line, 0, end)
                if parent_match is not None:
                    parent = parent_match.group(1).decode()
                    if parent and parent != sid:
                        partial.parent_link[sid] = parent
    partial.session_weight = dict(session_weight)
    partial.session_peak = dict(session_peak)
    return partial


def _scan_grouping(
    path: str | Path, *, threads: int, capture_peak: bool = False
) -> _GroupingScan:
    """Phase 1: extract session ids + parent links + block size, NO hash parse.

    Scans every segment in parallel across a thread pool (gzip decompression
    releases the GIL) and merges the per-segment partials IN SEGMENT ORDER, so
    the ``parent_link`` "first non-self parent wins" resolution is identical to
    the serial reader's global left-to-right scan. Block sizes are collected only
    from ``request_end`` records carrying a session (matching ``_resolve_block_size``
    over the built chains); a mix fails loud.

    ``capture_peak`` propagates to each :func:`_scan_one_file` so the per-session
    peak context is aggregated (max across segments -- a session's records may
    span files) for the optional filter-then-cap tree selection.
    """
    segments = discover_segments(Path(path))
    workers = max(1, min(threads, len(segments)))
    scan_fn = functools.partial(_scan_one_file, capture_peak=capture_peak)
    if workers == 1:
        partials = [scan_fn(seg) for seg in segments]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            partials = list(pool.map(scan_fn, segments))

    merged = _GroupingScan()
    session_weight: dict[str, int] = defaultdict(int)
    session_peak: dict[str, int] = defaultdict(int)
    block_sizes: set[int] = set()
    # Segment order (ThreadPoolExecutor.map preserves input order) -> first
    # non-self parent from the earliest segment wins, as the serial scan does.
    for partial in partials:
        merged.request_end_sessions |= partial.request_end_sessions
        for sid, parent in partial.parent_link.items():
            if sid not in merged.parent_link:
                merged.parent_link[sid] = parent
        for sid, weight in partial.session_weight.items():
            session_weight[sid] += weight
        for sid, peak in partial.session_peak.items():
            if peak > session_peak[sid]:
                session_peak[sid] = peak
        block_sizes |= partial.block_sizes

    if len(block_sizes) > 1:
        raise DynamoTraceAdapterError(
            f"mixed replay trace_block_size values are not supported: "
            f"{sorted(block_sizes)}"
        )
    merged.session_weight = dict(session_weight)
    merged.session_peak = dict(session_peak)
    merged.block_size = next(iter(block_sizes), 0) or DEFAULT_VIRTUAL_BLOCK_SIZE
    return merged


# --- Phase 2: shuffle raw lines to per-batch temp files --------------------


_SHUFFLE_QUEUE_MAXSIZE = 8192
_SHUFFLE_SENTINEL = object()


def _shuffle_to_batch_files(
    path: str | Path,
    session_to_batch: dict[str, int],
    tmpdir: Path,
    *,
    threads: int,
) -> list[Path]:
    """Phase 2: append each record line VERBATIM to its batch's gzip temp file.

    Producer threads decompress the segments in parallel (gzip releases the GIL)
    and hand each routed line ``(batch, bytes)`` to ONE consumer thread that owns
    every batch writer -- so the open-file-descriptor count is bounded by the
    batch count (not multiplied by the thread count) and no two threads ever
    write the same file. Streaming through a bounded queue keeps resident memory
    flat (never the whole re-sharded corpus). A line's session id (same
    prefix-bounded regex as the scan) selects its batch; lines with no session id
    (schema-less markers, replay-only records) or a session with no request_end
    are dropped -- exactly the records the serial path never lowers. Verbatim
    bytes mean the worker re-reads them through the unchanged reader (envelope
    unwrap, dedup, interning all happen there), so hash parsing stays entirely
    worker-side. Returns the batch temp-file paths in batch-index order
    (contiguous global tree order).
    """
    segments = discover_segments(Path(path))
    line_queue: queue.Queue[Any] = queue.Queue(maxsize=_SHUFFLE_QUEUE_MAXSIZE)
    files: dict[int, Path] = {}

    def _consume() -> None:
        writers: dict[int, IO[bytes]] = {}
        try:
            while True:
                item = line_queue.get()
                if item is _SHUFFLE_SENTINEL:
                    return
                batch, data = item
                writer = writers.get(batch)
                if writer is None:
                    fpath = tmpdir / f"batch_{batch:05d}.jsonl.gz"
                    # Long-lived per-batch writer, closed in the finally below;
                    # a per-line context manager would reopen (and re-header)
                    # the gzip stream on every record.
                    writer = gzip.open(fpath, "wb", compresslevel=1)  # noqa: SIM115
                    writers[batch] = writer
                    files[batch] = fpath
                writer.write(data)
        finally:
            for writer in writers.values():
                writer.close()

    def _produce(segment: Path) -> None:
        with _open_raw(segment) as handle:
            for line in handle:
                cut = line.find(_HASHES_KEY)
                end = len(line) if cut < 0 else cut
                match = _SESSION_RE.search(line, 0, end)
                if match is None:
                    continue
                batch = session_to_batch.get(match.group(1).decode())
                if batch is None:
                    continue
                line_queue.put((batch, line if line.endswith(b"\n") else line + b"\n"))

    consumer = threading.Thread(target=_consume)
    consumer.start()
    try:
        workers = max(1, min(threads, len(segments)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_produce, segments))
    finally:
        line_queue.put(_SHUFFLE_SENTINEL)
        consumer.join()
    return [files[batch] for batch in sorted(files)]


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
    per_tree = _build_trees_sequential(trees, direct_store=None, **build_kwargs)
    return _encode_batch_result(per_tree)


def _encode_batch_result(per_tree: list[ParsedGraph]) -> bytes:
    """Encode a batch's per-tree ``ParsedGraph``s to a cross-process list frame.

    Each per-tree ``ParsedGraph`` (one root ``TraceRecord`` + that tree's graph
    and pool) is msgpack-encoded through the existing typed codec; the outer
    frame is the plain list of those blobs in the batch's tree order.
    """
    return _FRAME_ENCODER.encode([encode_parsed_graph_msgpack(pg) for pg in per_tree])


def _decode_batch_result(blob: bytes) -> list[ParsedGraph]:
    """Decode a per-tree ``ParsedGraph`` list frame back into ``ParsedGraph``s."""
    pg_blobs = _FRAME_DECODER.decode(blob)
    return [decode_parsed_graph_msgpack(pg_bytes) for pg_bytes in pg_blobs]


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
    by ``--max-context-length`` and capped at ``--num-dataset-entries`` (root-sorted
    order), and ONLY the selected trees are shuffled + built. When both are ``None``
    no selection runs and the output stays byte-identical. ``selection_out``
    receives the :class:`SelectionStats` only when the pool path actually builds
    (a decline hands selection to the serial path, which appends its own).
    """
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
    roots = sorted(set(root_of.values()))

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
        return None
    workers = _dynamo_workers(item_count=len(roots))
    if workers <= 1:
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
    return _build_fused_parallel(
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

    A tree's peak is the max over its sessions' scan-recorded peaks; the roots
    are screened in sorted order (deterministic), so the selected set matches the
    serial path's :func:`dynamo_tree_peak_context` selection. Returns the
    selected roots (sorted), the ``root_of`` map restricted to the selected
    trees' sessions, and the :class:`SelectionStats`.
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
) -> list[ParsedGraph]:
    """Shuffle to per-batch temp files, fuse-build on the pool, flatten per-tree.

    ``content_root_seed`` and ``block_size`` are PINNED by the parent and
    threaded to every worker (the seed also seeds ``_run_pool_streaming``'s
    parent-built shared-memory corpus), so the fused build is byte-identical to
    the serial loop. Trees are batched CONTIGUOUSLY over the sorted-by-root list
    (weighted by the Phase-1 byte proxy), so batch results arriving in input
    order flatten to the same global tree order. Each worker returns its batch's
    LIST of per-tree ``ParsedGraph`` blobs; this concatenates them across batches
    (arrival order) and hands the flat list back for the caller to merge. The
    temp dir is always removed.
    """
    from aiperf.common.environment import Environment
    from aiperf.dataset.graph.adapters.weka.trace_parallel import _run_pool_streaming

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

    build_kwargs: dict[str, Any] = {
        "block_size": scan.block_size,
        "content_root_seed": content_root_seed,
        "idle_gap_cap_seconds": idle_gap_cap_seconds,
        "content_tokenizer": content_tokenizer,
        "prompt_corpus": prompt_corpus,
        "release_replay": release_replay,
    }

    tmpdir = Path(tempfile.mkdtemp(prefix="aiperf-dynamo-fused-"))
    try:
        batch_files = _shuffle_to_batch_files(
            path, session_to_batch, tmpdir, threads=io_threads
        )
        tasks = ((str(bf), build_kwargs, max_depth) for bf in batch_files)

        per_tree: list[ParsedGraph] = []
        # Results arrive in INPUT (batch) order -> contiguous global tree order,
        # so the flattened per-tree list matches the serial per-tree loop and the
        # merge is byte-identical (traces are id-sorted, pools content-unioned).
        for blob in _run_pool_streaming(
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
            per_tree.extend(_decode_batch_result(blob))

        return per_tree
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


__all__ = [
    "maybe_build_fused_parallel",
]
