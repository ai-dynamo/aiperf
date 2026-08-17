# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared multi-process pool lifecycle for recorded-trace graph parse dispatch.

Adapter-neutral: any recorded-trace adapter that needs to parse many items
(files / rows / session-tree batches) streams them through
:func:`run_pool_streaming` -- a bounded, ordered forkserver pool window with a
per-item timeout, a graceful shutdown that tolerates rayon-threaded workers, and
a parent-built shared-memory content corpus every worker CoW-attaches.

Each worker parses one item independently and returns an encoded blob (msgpack
``ParsedGraph`` or pickled segment payloads) -- Pydantic-pickle has historically
broken cross-process transfer of ``ParsedGraph`` instances, so encoding at the
worker boundary side-steps that. The consumer decodes -> writes -> DROPS each
result before the next arrives, keeping resident memory flat at ~one item.

Disambiguation -- "pool" names three unrelated things in the graph subsystem.
Here it is a multiprocessing WORKER pool (OS processes). It is NOT
``segment_trie/pool.py`` (the build-plane interned content store:
``SegmentPool`` and its dynamo shims) and NOT ``aiperf.graph.dynamic_pool``
(the worker's runtime cache of captured replies, which owns
``GraphPoolMissingError``).
"""

from __future__ import annotations

import contextlib
import multiprocessing
import multiprocessing.pool  # noqa: F401  -- resolves the _shutdown_pool annotation
import os
import signal
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.utils import allow_daemon_children

_POOL_JOIN_TIMEOUT_S = 10.0
_logger = AIPerfLogger(__name__)


def _has_broken_stdio() -> bool:
    """Return whether a standard stream has an invalid file descriptor."""
    import sys

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            fd = stream.fileno()
            if fd < 0:
                return True
            os.fstat(fd)
        except (OSError, ValueError, AttributeError):
            return True
    return False


def _ensure_valid_stdio_fds() -> None:
    """Redirect invalid standard streams before starting graph pool workers.

    Textual replaces standard streams while the dashboard is active. Those
    stream objects can expose invalid descriptors to multiprocessing's child
    setup, which then fails with ``bad value(s) in fds_to_keep``.
    """
    import sys

    if not _has_broken_stdio():
        return

    devnull = os.open(os.devnull, os.O_RDWR)
    try:
        for fd in (0, 1, 2):
            os.dup2(devnull, fd)
    finally:
        if devnull > 2:
            os.close(devnull)
    sys.stdin = os.fdopen(0, "r", closefd=False)
    sys.stdout = os.fdopen(1, "w", closefd=False)
    sys.stderr = os.fdopen(2, "w", closefd=False)


def _loader_pool_context(
    tokenizer_name: str | None,
) -> multiprocessing.context.BaseContext:
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
class _PoolWorkerInitArgs:
    """Static args shipped to every pool worker via ``Pool(initializer=...)``.

    ``shm_name``/``corpus_len`` describe the parent-built shared-memory corpus
    block (int32 token ids); the worker attaches it read-only and points its
    cached :class:`CorpusContentSynthesizer` at that array, so the ~600K-token
    coding pool is built ONCE in the parent and CoW-shared across all workers
    instead of rebuilt per worker (the rebuild-per-item path ballooned RSS into
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


def _init_worker(args: _PoolWorkerInitArgs) -> None:
    """Pool worker init: attach the shared corpus + warm the synthesizer cache.

    Builds (and caches) one :class:`CorpusContentSynthesizer` for the run's
    ``(tokenizer, corpus, seed)`` CONSTRUCTED ON the parent-built shared-memory
    array: passing ``shared_corpus`` into ``get_or_build_synthesizer`` makes the
    generator skip its own corpus build entirely, so the worker never pays the
    ~600K-token pool build only to discard it for the shm attachment. Every
    item this worker parses then reuses that single CoW-shared pool. Fail-soft:
    any error here leaves the cache unpopulated and the per-item path builds a
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
        # An adopted ndarray may retain the mapping through the synthesizer cache;
        # release cleanup must not leave that view pointing at closed shared memory.
        if _worker_corpus_shm is not None:
            with contextlib.suppress(Exception):
                _worker_corpus_shm.close()
            _worker_corpus_shm = None
        with contextlib.suppress(Exception):
            from aiperf.dataset.graph.adapters.shared.content import (
                CorpusContentSynthesizer,
            )

            CorpusContentSynthesizer.reset_worker_cache()


def _shutdown_pool(
    pool: multiprocessing.pool.Pool, *, timeout_s: float = _POOL_JOIN_TIMEOUT_S
) -> None:
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


def run_pool_streaming(
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
    a large real corpus (hundreds of traces, each with full real-content
    synthesis -- multi-turn message lists + decoded token text) that peak is what
    blows host RAM. This generator instead streams through a bounded ordered
    apply-async window, so the pool feeder cannot consume the full input iterator
    before first result. The consumer can decode -> write-to-mmap -> DROP each
    result before the next arrives, keeping resident memory flat at ~one item.

    Builds the shared-memory corpus in the parent and opens a raw ``ctx.Pool``
    (NOT ``ProcessPoolExecutor`` -- the latter raises ``BrokenProcessPool`` on
    any worker death, which rayon-threaded tokenizer workers hit during
    teardown; a raw ``Pool`` + graceful :func:`_shutdown_pool` tolerates it).
    AIPerf services are daemonic and Python refuses to spawn children from a
    daemon, so the daemon flag is cleared around the pool (restored in
    ``finally``).

    ``prefetch_multiplier`` / ``item_timeout_s`` / ``timeout_hint`` default to
    ``None`` meaning "use the ``DYNAMO_GRAPH_PARALLEL_*`` settings + generic
    error message"; every caller may pass its own window / timeout / message so
    the SAME pool lifecycle serves multiple adapters without coupling.
    """
    from aiperf.common.environment import Environment
    from aiperf.common.tokenizer import BUILTIN_TOKENIZER_NAME

    # SharedMemory starts multiprocessing's resource tracker, which itself
    # spawns a helper process and therefore needs valid stdio descriptors.
    _ensure_valid_stdio_fds()

    tokenizer_name = content_tokenizer or BUILTIN_TOKENIZER_NAME
    prompt_corpus = prompt_corpus or "coding"
    try:
        total_items = len(work_items)  # type: ignore[arg-type]
    except TypeError:
        total_items = None
    _logger.info(
        lambda: "Graph dataset: preparing worker pool "
        f"(workers={workers}, items={total_items or 'unknown'})"
    )
    if prefetch_multiplier is None:
        prefetch_multiplier = (
            Environment.DATASET.DYNAMO_GRAPH_PARALLEL_PREFETCH_MULTIPLIER
        )

    shm, corpus_len = _build_shared_corpus(
        tokenizer_name, prompt_corpus=prompt_corpus, root_seed=root_seed
    )
    _logger.info(
        lambda: f"Graph dataset: shared content corpus ready ({corpus_len:,} tokens)"
    )
    init_args = _PoolWorkerInitArgs(
        tokenizer_name=tokenizer_name,
        prompt_corpus=prompt_corpus,
        root_seed=root_seed,
        shm_name=shm.name if shm is not None else None,
        corpus_len=corpus_len,
    )

    try:
        with allow_daemon_children():
            ctx = _loader_pool_context(tokenizer_name)
            _logger.info("Graph dataset: starting worker processes")
            pool = ctx.Pool(workers, _init_worker, (init_args,))
            try:
                window_size = workers * prefetch_multiplier
                progress_interval = (
                    max(1, total_items // 10) if total_items is not None else 10
                )
                for completed, result in enumerate(
                    _bounded_ordered_pool_map(
                        pool,
                        worker_fn,
                        work_items,
                        window_size,
                        timeout_s=item_timeout_s,
                        timeout_hint=timeout_hint,
                    ),
                    1,
                ):
                    if (
                        completed == 1
                        or completed % progress_interval == 0
                        or (total_items is not None and completed == total_items)
                    ):
                        # ``total_items`` is None when the caller streams an
                        # unsized iterable; separate it only on the int branch
                        # so the numerator and denominator match conventions.
                        total_str = "?" if total_items is None else f"{total_items:,}"
                        _logger.info(
                            f"Graph dataset: worker progress "
                            f"{completed:,}/{total_str} batches"
                        )
                    yield result
            finally:
                _shutdown_pool(pool)
                _logger.info("Graph dataset: worker pool stopped")
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
    ``DYNAMO_GRAPH_PARALLEL_ITEM_TIMEOUT_SECONDS``"; ``timeout_hint`` overrides
    the raised message so each caller can name its own env var / unit.
    """
    from aiperf.common.environment import Environment

    if timeout_s is None:
        timeout_s = Environment.DATASET.DYNAMO_GRAPH_PARALLEL_ITEM_TIMEOUT_SECONDS
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
                    f"graph parse worker produced no result within "
                    f"{timeout_s:g}s for one item: the worker process was most "
                    "likely killed mid-parse (OOM kill / external SIGKILL) -- a "
                    "raw multiprocessing Pool cannot complete a killed worker's "
                    "in-flight task. Reduce worker count to lower peak memory, "
                    "or raise "
                    "AIPERF_DATASET_DYNAMO_GRAPH_PARALLEL_ITEM_TIMEOUT_SECONDS "
                    "if a single item legitimately parses slower than the "
                    "timeout."
                )
            ) from None
        yield item_result
        _fill_window()


__all__ = [
    "run_pool_streaming",
]
