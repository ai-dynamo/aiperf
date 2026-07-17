# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parallel multi-process parse dispatch for the Weka graph adapter.

Every weka route (local directory, HuggingFace corpus, streaming build plane)
funnels work items through :func:`_map_items`: at or below
``WEKA_GRAPH_PARALLEL_THRESHOLD`` (default 8) items parse serially in-process;
above it they stream through a bounded, ordered forkserver pool window. Each
worker parses one item independently and returns a msgpack-encoded
``ParsedGraph`` (or its trie segment payloads) — Pydantic-pickle has
historically broken cross-process transfer of ``ParsedGraph`` instances, so
encoding at the worker boundary side-steps that. The main process decodes and
either merges into a single multi-graph :class:`ParsedGraph` whose traces are
sorted by trace id (byte-deterministic) or streams per-trace payloads into the
unified segment store.
"""

from __future__ import annotations

import contextlib
import itertools
import multiprocessing
import os
import signal
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

from aiperf.common.utils import allow_daemon_children
from aiperf.dataset.graph.adapters.weka.trace_models import (
    EmptyWekaTraceError,
)
from aiperf.dataset.graph.codecs import (
    decode_parsed_graph_msgpack,
    encode_parsed_graph_msgpack,
)
from aiperf.dataset.graph.merge import merge_parsed_graphs
from aiperf.dataset.graph.models import (
    ParsedGraph,
)

_POOL_JOIN_TIMEOUT_S = 10.0


def _loader_pool_context(tokenizer_name: str | None):
    """Return the forkserver (Linux) / spawn (macOS) mp context for the pool.

    The default ``fork`` start method DEADLOCKS here: the parent process has
    already loaded the HF real-content tokenizer and exercised its Rust rayon
    thread pool, so forked workers inherit broken rayon state and hang at ~2%
    CPU while holding the parent's threads and many GB RSS. A forkserver helper
    is a fresh interpreter free of that state; it also preloads the configured
    tokenizer into its heap so workers CoW-share one copy (see
    :mod:`aiperf.dataset._mp_context` / :mod:`aiperf.dataset._tokenizer_preload`).
    """
    from aiperf.dataset._mp_context import get_loader_mp_context

    return get_loader_mp_context(preload_tokenizer=tokenizer_name)


@dataclass(slots=True)
class _WorkerInitArgs:
    """Static args shipped to every pool worker via ``Pool(initializer=...)``.

    ``shm_name``/``corpus_len`` describe the parent-built shared-memory corpus
    block (int32 token ids); the worker attaches it read-only and points its
    cached :class:`CorpusContentSynthesizer` at that array, so the ~600K-token
    coding pool is built ONCE in the parent and CoW-shared across all workers
    instead of rebuilt per worker (the rebuild-per-trace path ballooned RSS into
    tens of GB and abruptly killed pool workers -> ``BrokenProcessPool``).
    """

    tokenizer_name: str | None
    prompt_corpus: str
    root_seed: int | None
    shm_name: str | None
    corpus_len: int


# Per-worker handle to the attached shared-memory corpus; kept alive for the
# worker's lifetime so the mmap is not released between tasks.
_worker_corpus_shm: shared_memory.SharedMemory | None = None


def _install_hard_exit_on_sigterm() -> None:
    """Replace the worker SIGTERM handler with ``os._exit(0)``.

    Pool teardown (``terminate()``) SIGTERMs each worker, but the worker's
    CoW-shared HF tokenizer carries Rust ``rayon`` threads that do not unwind on
    SIGTERM, so the default Python finalizer path wedges. Workers are stateless
    (the parent owns shared memory and collects results), so a hard exit is the
    right behavior. ``signal.signal`` only works on the main thread, so a
    ``ValueError`` (unit test invoking ``_init_worker`` off-thread) is ignored.
    """

    def _hard_exit(_signum, _frame):  # noqa: ANN001
        os._exit(0)

    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGTERM, _hard_exit)


def _init_worker(args: _WorkerInitArgs) -> None:
    """Pool worker init: attach the shared corpus + warm the synthesizer cache.

    Builds (and caches) one :class:`CorpusContentSynthesizer` for the run's
    ``(tokenizer, corpus, seed)`` CONSTRUCTED ON the parent-built shared-memory
    array: passing ``shared_corpus`` into ``get_or_build_synthesizer`` makes the
    generator skip its own corpus build entirely, so the worker never pays the
    ~600K-token pool build only to discard it for the shm attachment. Every
    trace this worker parses then reuses that single CoW-shared pool. Fail-soft:
    any error here leaves the cache unpopulated and the per-trace path builds a
    private synthesizer on demand (correct, just not memory-optimal).
    """
    global _worker_corpus_shm

    _install_hard_exit_on_sigterm()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if args.tokenizer_name is None or args.shm_name is None:
        return

    try:
        from aiperf.dataset.graph.adapters.shared.content import (
            get_or_build_synthesizer,
        )

        shm = shared_memory.SharedMemory(name=args.shm_name)
        _worker_corpus_shm = shm
        corpus = np.ndarray((args.corpus_len,), dtype=np.int32, buffer=shm.buf)

        get_or_build_synthesizer(
            args.tokenizer_name,
            prompt_corpus=args.prompt_corpus,
            root_seed=args.root_seed,
            shared_corpus=corpus,
        )
    except Exception:  # init must never crash the worker
        _worker_corpus_shm = None


def _shutdown_pool(pool, *, timeout_s: float = _POOL_JOIN_TIMEOUT_S) -> None:
    """Drain a ``multiprocessing.Pool`` without the SIGTERM teardown wedge.

    ``with Pool(...)`` exit calls ``terminate()`` which SIGTERMs every worker;
    the CoW-shared HF tokenizer's ``rayon`` threads do not unwind on SIGTERM, so
    ``terminate()``+``join()`` hangs. Graceful path: ``close()`` lets each worker
    drain and exit via the hard-exit handler / normal sentinel; ``join()`` then
    returns promptly. Falls back to ``terminate()`` (bounded by ``timeout_s``)
    if a worker wedges anyway, so a stuck worker never blocks the CLI.
    """
    pool.close()
    done = threading.Event()

    def _wait() -> None:
        try:
            pool.join()
        finally:
            done.set()

    threading.Thread(target=_wait, daemon=True).start()
    if done.wait(timeout=timeout_s):
        return
    pool.terminate()
    done.wait(timeout=timeout_s)


def _build_shared_corpus(
    tokenizer_name: str | None,
    *,
    prompt_corpus: str,
    root_seed: int | None,
) -> tuple[shared_memory.SharedMemory | None, int]:
    """Build the deterministic corpus ONCE in the parent into shared memory.

    Returns ``(shm, corpus_len)``; ``(None, 0)`` when real-content synthesis is
    off (``tokenizer_name is None``) so workers skip corpus attachment. The
    corpus is byte-identical to what each worker would build for the same
    ``(tokenizer, corpus, seed)``, so determinism / parity is preserved.
    """
    if tokenizer_name is None:
        return None, 0

    from aiperf.dataset.graph.adapters.shared.content import (
        get_or_build_synthesizer,
    )

    synth = get_or_build_synthesizer(
        tokenizer_name, prompt_corpus=prompt_corpus, root_seed=root_seed
    )
    corpus_tokens = synth.corpus_tokens()
    corpus_arr = np.ascontiguousarray(corpus_tokens, dtype=np.int32)
    corpus_len = int(corpus_arr.shape[0])
    if corpus_len == 0:
        return None, 0
    shm = shared_memory.SharedMemory(
        create=True, size=corpus_len * np.dtype(np.int32).itemsize
    )
    np.ndarray((corpus_len,), dtype=np.int32, buffer=shm.buf)[:] = corpus_arr
    return shm, corpus_len


def _run_pool_streaming(
    worker_fn: Callable[[Any], Any],
    work_items: Iterable[Any],
    *,
    workers: int,
    root_seed: int | None,
    content_tokenizer: str | None = None,
    prompt_corpus: str | None = None,
    prefetch_multiplier: int | None = None,
    item_timeout_s: float | None = None,
    timeout_hint: str | None = None,
) -> Iterator[Any]:
    """Dispatch ``worker_fn`` across a forkserver ``Pool``, YIELDING each result
    as ``imap``-ordered completion produces it.

    Each yielded value is whatever ``worker_fn`` returns -- a msgpack
    ``ParsedGraph`` blob for the merged consumer, or a pickled
    ``TraceSegmentPayload`` list for the streaming consumer. An eager
    ``list(pool.imap(...))`` would hold EVERY worker's result in memory at once,
    and the caller would then decode ALL of them into a list before merging. For
    a large real corpus (393 weka traces, each with full real-content synthesis
    -- multi-turn message lists + decoded token text) that peak is what blows
    host RAM. This generator instead streams through a bounded ordered
    apply-async window, so the pool feeder cannot consume the full input iterator
    before first result. The consumer can decode -> write-to-mmap -> DROP each
    result before the next arrives, keeping resident memory flat at ~one trace.

    Builds the shared-memory corpus in the parent and opens a raw ``ctx.Pool``
    (NOT ``ProcessPoolExecutor`` -- the latter raises ``BrokenProcessPool`` on
    any worker death, which the rayon-threaded weka workers hit during teardown;
    a raw ``Pool`` + graceful :func:`_shutdown_pool` tolerates it). AIPerf
    services are daemonic and Python refuses to spawn children from a daemon,
    so the daemon flag is cleared around the pool (restored in ``finally``).

    Adapter-neutral tuning: ``prefetch_multiplier`` / ``item_timeout_s`` /
    ``timeout_hint`` default to ``None`` meaning "use the weka
    ``WEKA_GRAPH_PARALLEL_*`` settings + weka error message" (the historical
    behavior every weka caller relies on). The dynamo session-tree build passes
    its own ``DYNAMO_GRAPH_PARALLEL_*`` values and error message so the SAME
    pool lifecycle serves both adapters without a weka-coupled window/timeout.
    """
    from aiperf.common.environment import Environment
    from aiperf.common.tokenizer import BUILTIN_TOKENIZER_NAME

    tokenizer_name = content_tokenizer or BUILTIN_TOKENIZER_NAME
    prompt_corpus = prompt_corpus or "coding"
    if prefetch_multiplier is None:
        prefetch_multiplier = (
            Environment.DATASET.WEKA_GRAPH_PARALLEL_PREFETCH_MULTIPLIER
        )

    shm, corpus_len = _build_shared_corpus(
        tokenizer_name, prompt_corpus=prompt_corpus, root_seed=root_seed
    )
    init_args = _WorkerInitArgs(
        tokenizer_name=tokenizer_name,
        prompt_corpus=prompt_corpus,
        root_seed=root_seed,
        shm_name=shm.name if shm is not None else None,
        corpus_len=corpus_len,
    )

    try:
        with allow_daemon_children():
            ctx = _loader_pool_context(tokenizer_name)
            pool = ctx.Pool(workers, _init_worker, (init_args,))
            try:
                window_size = workers * prefetch_multiplier
                yield from _bounded_ordered_pool_map(
                    pool,
                    worker_fn,
                    work_items,
                    window_size,
                    timeout_s=item_timeout_s,
                    timeout_hint=timeout_hint,
                )
            finally:
                _shutdown_pool(pool)
    finally:
        if shm is not None:
            shm.close()
            shm.unlink()


def _bounded_ordered_pool_map(
    pool: Any,
    worker_fn: Callable[[Any], Any],
    work_items: Iterable[Any],
    window_size: int,
    *,
    timeout_s: float | None = None,
    timeout_hint: str | None = None,
) -> Iterator[Any]:
    """Yield pool results in input order while consuming at most one window ahead.

    Each head-of-window ``AsyncResult`` is waited on with a bounded
    ``get(timeout=...)``: a raw ``multiprocessing.Pool`` repopulates a killed
    worker but never completes its in-flight task, so an unbounded ``get()``
    turns a worker OOM-kill into a silent indefinite CLI hang. On expiry a
    ``RuntimeError`` naming that likely cause is raised instead. Order
    preservation is unchanged.

    ``timeout_s`` defaults to ``None`` meaning "read
    ``WEKA_GRAPH_PARALLEL_ITEM_TIMEOUT_SECONDS``" (the weka behavior);
    ``timeout_hint`` overrides the raised message so the dynamo session-tree
    build can name its own env var / unit ("batch" not "trace") on expiry.
    """
    from aiperf.common.environment import Environment

    if timeout_s is None:
        timeout_s = Environment.DATASET.WEKA_GRAPH_PARALLEL_ITEM_TIMEOUT_SECONDS
    iterator = iter(work_items)
    pending: deque[Any] = deque()
    exhausted = False
    window = max(1, window_size)

    def _fill_window() -> None:
        nonlocal exhausted
        while not exhausted and len(pending) < window:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                return
            pending.append(pool.apply_async(worker_fn, (item,)))

    _fill_window()
    while pending:
        result = pending.popleft()
        try:
            item_result = result.get(timeout=timeout_s)
        except multiprocessing.TimeoutError:
            raise RuntimeError(
                timeout_hint
                or (
                    f"weka graph parse worker produced no result within "
                    f"{timeout_s:g}s for one trace: the worker process was most "
                    "likely killed mid-parse (OOM kill / external SIGKILL) -- a "
                    "raw multiprocessing Pool cannot complete a killed worker's "
                    "in-flight task. Reduce worker count "
                    "(AIPERF_DATASET_WEKA_GRAPH_PARALLEL_WORKERS) to lower peak "
                    "memory, or raise "
                    "AIPERF_DATASET_WEKA_GRAPH_PARALLEL_ITEM_TIMEOUT_SECONDS if a "
                    "single trace legitimately parses slower than the timeout."
                )
            ) from None
        yield item_result
        _fill_window()


def _get_threshold() -> int:
    from aiperf.common.environment import Environment

    return max(0, Environment.DATASET.WEKA_GRAPH_PARALLEL_THRESHOLD)


def _configured_workers(override: int | None) -> int:
    if override is not None:
        return override

    from aiperf.common.environment import Environment

    return Environment.DATASET.WEKA_GRAPH_PARALLEL_WORKERS


def _get_workers(*, item_count: int | None, override: int | None) -> int:
    from aiperf.common.environment import Environment

    configured = _configured_workers(override)
    if configured > 0:
        resolved = configured
    else:
        cpu = os.cpu_count() or 1
        resolved = min(
            max(cpu - 1, 1),
            Environment.DATASET.WEKA_GRAPH_PARALLEL_AUTO_MAX_WORKERS,
        )
    if item_count is not None:
        resolved = min(resolved, item_count)
    return max(1, resolved)


@dataclass(slots=True)
class _WorkItem:
    """One parse unit: a local trace file OR an in-memory HF row.

    Exactly one of ``path`` / ``row`` is set. File items ship only the path
    string (the worker reads the bytes itself — far cheaper to pickle than the
    parsed dict); row items ship the already-loaded dict (HF rows have no
    file to re-read). ``source`` is the human-readable origin label used in
    errors and ``ParsedGraph.source_path``.
    """

    source: str
    """Origin label: a file path or an ``org/name#row`` HF locator."""

    path: str | None
    """Local trace file path; ``None`` for row items."""

    row: dict[str, Any] | None
    """In-memory weka trace dict; ``None`` for file items."""


def file_work_items(files: Iterable[Path]) -> Iterator[_WorkItem]:
    """Wrap local trace files as work items."""
    for p in files:
        yield _WorkItem(source=str(p), path=str(p), row=None)


def row_work_items(rows: Iterable[Any], source_prefix: str) -> Iterator[_WorkItem]:
    """Wrap HF row dicts as work items with stable ``org/name#index`` labels.

    Rows arrive as plain dicts already shallow-copied by :func:`_load_hf_rows`
    (``yield dict(row)``), so they are picklable for the forkserver pool and
    detached from the dataset row view -- this wrapper does not re-copy.
    """
    for index, row in enumerate(rows):
        yield _WorkItem(source=f"{source_prefix}#{index}", path=None, row=row)


def _parse_item(item: _WorkItem, parse_kwargs: dict[str, Any]) -> ParsedGraph:
    """Parse ONE work item through the shared per-trace core.

    This is the only per-item entry for both the serial in-process path and
    the pool workers: file items go through ``_parse_single_file`` (bytes read
    where the parse runs) and row items through ``_parse_trace_dict``. Workers
    deliberately do NOT re-enter the public ``from_weka_trace`` dispatcher —
    it would re-run HF-id and directory detection per item.
    """
    from aiperf.dataset.graph.adapters.weka.trace import (
        _parse_single_file,
        _parse_trace_dict,
    )

    if item.path is not None:
        return _parse_single_file(Path(item.path), **parse_kwargs)
    assert item.row is not None
    return _parse_trace_dict(item.row, source=item.source, **parse_kwargs)


def _parse_item_to_msgpack(task: tuple[_WorkItem, dict[str, Any]]) -> bytes:
    """Pool worker entry: parse one item, return a msgpack ``ParsedGraph`` blob.

    A single picklable ``(item, parse_kwargs)`` argument drops cleanly into
    the pool; msgpack at the worker boundary side-steps the historically
    broken cross-process pickling of ``ParsedGraph`` instances.
    """
    item, parse_kwargs = task
    return encode_parsed_graph_msgpack(_parse_item(item, parse_kwargs))


def _parse_item_to_segment_payloads(
    task: tuple[_WorkItem, dict[str, Any]],
) -> list[Any]:
    """Pool worker entry: parse one item, return its trie segment payloads."""
    from aiperf.dataset.graph.segment_ir.store_builder import (
        iter_trace_segment_payloads,
    )

    item, parse_kwargs = task
    return list(iter_trace_segment_payloads(_parse_item(item, parse_kwargs)))


def _prefetch_items(
    items: Iterable[_WorkItem], limit: int
) -> tuple[list[_WorkItem], Iterator[_WorkItem], bool]:
    """Pull at most ``limit`` items; return ``(prefetched, tail, exhausted)``.

    ``exhausted`` is True only when the source yielded fewer than ``limit``
    items, i.e. ``prefetched`` is the complete input and its length is the
    exact item count (used to cap the worker fan-out).
    """
    iterator = iter(items)
    prefetched: list[_WorkItem] = []
    for _ in range(limit):
        try:
            prefetched.append(next(iterator))
        except StopIteration:
            return prefetched, iterator, True
    return prefetched, iterator, False


def _map_items(
    items: Iterable[_WorkItem],
    *,
    worker_fn: Callable[[tuple[_WorkItem, dict[str, Any]]], Any],
    local_fn: Callable[[_WorkItem], Any],
    decode_fn: Callable[[Any], Any],
    source_label: str,
    item_count: int | None,
    threshold: int | None,
    workers: int | None,
    parse_kwargs: dict[str, Any],
) -> Iterator[Any]:
    """Map work items through the ONE serial-or-pool dispatch, in input order.

    At or below the parallel threshold every item runs through ``local_fn``
    in-process (no pool, no codec round-trip). Above it, items stream through
    :func:`_run_pool_streaming` (bounded ordered window, per-item timeout,
    graceful shutdown, shared-memory corpus) and each pool result passes
    through ``decode_fn``. Only ``threshold + 1`` items are prefetched to pick
    the path, so lazy sources are never fully consumed up front. When the item
    count is known (``item_count`` or an exhausted prefetch) it caps the
    worker fan-out.
    """
    # EmptyWekaTraceError is already a module-top import from trace_models;
    # do not add a lazy import here.
    effective_threshold = max(
        0, threshold if threshold is not None else _get_threshold()
    )
    prefetched, remaining, exhausted = _prefetch_items(items, effective_threshold + 1)
    if not prefetched:
        raise EmptyWekaTraceError(
            f"weka_trace source {source_label!r} yielded zero traces"
        )

    if len(prefetched) <= effective_threshold:
        for item in prefetched:
            yield local_fn(item)
        return

    known_count = (
        item_count
        if item_count is not None
        else (len(prefetched) if exhausted else None)
    )
    effective_workers = _get_workers(item_count=known_count, override=workers)
    tasks = ((item, parse_kwargs) for item in itertools.chain(prefetched, remaining))
    for result in _run_pool_streaming(
        worker_fn,
        tasks,
        workers=effective_workers,
        root_seed=parse_kwargs.get("content_root_seed"),
        content_tokenizer=parse_kwargs.get("content_tokenizer"),
        prompt_corpus=parse_kwargs.get("prompt_corpus"),
    ):
        yield decode_fn(result)


def parse_items(
    items: Iterable[_WorkItem],
    *,
    source_label: str,
    item_count: int | None = None,
    threshold: int | None = None,
    workers: int | None = None,
    parse_kwargs: dict[str, Any] | None = None,
) -> ParsedGraph:
    """Parse work items and merge into ONE multi-graph ``ParsedGraph``.

    The eager-merge consumer of :func:`_map_items`, backing the whole in-memory
    ``ParsedGraph`` that :func:`from_weka_trace` / ``WekaTraceAdapter.parse``
    return (reached via ``parse_graph_workload`` and pinned by the registry/run
    parse-parity tests). The DatasetManager build plane does NOT take this route
    — it drains worker-built segment payloads through
    :func:`iter_item_segment_payloads` instead. Pool results stream straight
    into the merge in input order, so the parent never holds every blob plus
    every decoded graph at once; the post-merge sort by trace id keeps the
    result byte-deterministic across worker counts.
    """
    kwargs = parse_kwargs or {}
    per_item = _map_items(
        items,
        worker_fn=_parse_item_to_msgpack,
        local_fn=lambda item: _parse_item(item, kwargs),
        decode_fn=decode_parsed_graph_msgpack,
        source_label=source_label,
        item_count=item_count,
        threshold=threshold,
        workers=workers,
        parse_kwargs=kwargs,
    )
    return merge_parsed_graphs(per_item)


def iter_item_segment_payloads(
    items: Iterable[_WorkItem],
    *,
    source_label: str,
    item_count: int | None = None,
    threshold: int | None = None,
    workers: int | None = None,
    parse_kwargs: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Yield per-trace trie segment payloads for work items, memory-bounded.

    The streaming consumer of :func:`_map_items` — the build plane serializes
    each trace's envelopes into the unified store and DROPS the payloads
    before the next arrive, so resident memory stays at ~one trace regardless
    of corpus size. Serial and pool paths emit the same payload shape.
    """
    from aiperf.dataset.graph.segment_ir.store_builder import (
        iter_trace_segment_payloads,
    )

    kwargs = parse_kwargs or {}

    def _local(item: _WorkItem) -> list[Any]:
        return list(iter_trace_segment_payloads(_parse_item(item, kwargs)))

    for payloads in _map_items(
        items,
        worker_fn=_parse_item_to_segment_payloads,
        local_fn=_local,
        decode_fn=lambda blobs: blobs,
        source_label=source_label,
        item_count=item_count,
        threshold=threshold,
        workers=workers,
        parse_kwargs=kwargs,
    ):
        yield from payloads


__all__ = [
    "file_work_items",
    "iter_item_segment_payloads",
    "parse_items",
    "row_work_items",
]
