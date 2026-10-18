# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel conversion of trace sessions to conversations.

Uses multiprocessing Pool with shared memory for the token corpus. Each worker
gets its own HashIdRandomGenerator to produce deterministic token sequences per
hash_id regardless of worker count or processing order.

The daemon flag on the current process is temporarily cleared because Python's
multiprocessing refuses to spawn children from daemon processes, and AIPerf
services run as daemons.

This module is the opt-in counterpart to the in-process 3-phase pipeline used
by ``BaseTraceDatasetLoader.convert_to_conversations``. Both paths reseed
``HashIdRandomGenerator`` identically per ``(seed, trace_id, hash_id)`` so the
two paths produce byte-identical output for the exact-tile and
last-block-partial input layouts emitted by Mooncake/Bailian/BurstGPT loaders.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from multiprocessing import shared_memory

import numpy as np

from aiperf.common.exceptions import ConfigurationError
from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset._mp_context import get_loader_mp_context


@dataclass(slots=True)
class _WorkerInitArgs:
    """Arguments passed to each worker process via Pool initargs."""

    shm_name: str
    corpus_len: int
    tokenizer_name: str
    base_seed: int
    block_size: int
    sep_token: int | None
    trace_id: str
    trust_remote_code: bool = False
    revision: str = "main"


@dataclass(slots=True)
class _WorkerState:
    """Per-worker process state, initialized once via _init_worker."""

    tokenizer: Tokenizer
    corpus: np.ndarray
    shm: shared_memory.SharedMemory  # prevent GC from unmapping corpus buffer
    hash_rng: HashIdRandomGenerator
    block_size: int
    sep_token: int | None
    sample_tokens: Callable[..., list[int]]
    block_cache: dict[int, list[int]] = field(default_factory=dict)


# Set once per worker process by _init_worker; read by _process_batch.
_worker_state: _WorkerState | None = None


def _init_worker(args: _WorkerInitArgs) -> None:
    """Initialize worker process with shared corpus and tokenizer.

    Called once per worker when the Pool is created. Attaches to the
    shared-memory corpus, creates a per-worker HashIdRandomGenerator
    (seeded by trace_id for file-level determinism), and loads the
    tokenizer from local cache (offline mode).
    """
    global _worker_state

    from aiperf.dataset.generator.prompt import sample_tokens_from_corpus

    _install_hard_exit_on_sigterm()

    # The main process already downloaded and cached the tokenizer, so force
    # offline mode to skip network requests and alias resolution.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    shm = shared_memory.SharedMemory(name=args.shm_name)

    # Each worker gets its own RNG so reseed_for_hash_id calls are independent.
    hash_rng = HashIdRandomGenerator(args.base_seed, _internal=True)
    hash_rng.set_trace_id(args.trace_id)

    from aiperf.dataset._tokenizer_preload import get_preloaded

    tokenizer = get_preloaded(
        args.tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    if tokenizer is None:
        tokenizer = Tokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            resolve_alias=False,
        )

    _worker_state = _WorkerState(
        tokenizer=tokenizer,
        corpus=np.ndarray((args.corpus_len,), dtype=np.int32, buffer=shm.buf),
        shm=shm,
        hash_rng=hash_rng,
        block_size=args.block_size,
        sep_token=args.sep_token,
        sample_tokens=sample_tokens_from_corpus,
    )


def _process_batch(
    batch: list[tuple[str, list[dict]]],
) -> list[tuple[str, list[tuple]]]:
    """Process a batch of sessions, converting hash_ids to prompts.

    Each trace dict must have ``input_length``, ``output_length``,
    ``timestamp``, ``delay``, and optionally ``hash_ids`` and ``text_input``.
    """
    assert _worker_state is not None
    hash_rng = _worker_state.hash_rng
    corpus = _worker_state.corpus
    block_size = _worker_state.block_size
    sep_token = _worker_state.sep_token
    decode = _worker_state.tokenizer.decode
    sample_tokens = _worker_state.sample_tokens
    block_cache = _worker_state.block_cache

    def get_block_tokens(hash_id: int, size: int) -> list[int]:
        cached = block_cache.get(hash_id)
        if cached is None:
            hash_rng.reseed_for_hash_id(hash_id)
            cached = sample_tokens(corpus, size, hash_rng, sep_token)
            block_cache[hash_id] = cached
        elif len(cached) != size:
            # A hash_id identifies a fixed block of content, so it can only ever
            # have one size. The same id at two sizes means a corrupt trace or a
            # block_size that disagrees with the recorded blocks.
            raise ConfigurationError(
                f"hash_id {hash_id} requested at {size} tokens but was already "
                f"materialized at {len(cached)} tokens. A hash_id must map to a "
                f"single fixed block size; inconsistent sizes indicate a corrupt "
                f"trace or a --isl-block-size that disagrees with the recorded blocks."
            )
        return cached

    results = []
    for session_id, traces in batch:
        turns = []
        for trace in traces:
            if trace.get("text_input"):
                # Literal prompt provided by the trace (no generation needed).
                prompt = trace["text_input"]
            elif trace.get("hash_ids"):
                # Generate prompt from hash_id blocks. All blocks are full-sized
                # except the last. Mirror the serial
                # PromptGenerator._build_token_sequence guard: when hash_ids
                # overshoot input_length the implied final partial block goes
                # non-positive, which serial rejects -- raise the same error
                # here instead of silently emitting a short/empty block.
                hash_ids = trace["hash_ids"]
                input_length = trace["input_length"]
                m = len(hash_ids)
                final_block_size = input_length - (m - 1) * block_size

                if m * block_size > input_length and (
                    final_block_size <= 0 or final_block_size > block_size
                ):
                    raise ConfigurationError(
                        f"Input length: {input_length}, Hash IDs: {hash_ids}, "
                        f"Block size: {block_size} are not compatible. The final "
                        f"hash block size: {final_block_size} must be greater than "
                        f"0 and less than or equal to {block_size}."
                    )

                tokens: list[int] = []
                for i, hid in enumerate(hash_ids):
                    size = final_block_size if i == m - 1 else block_size
                    tokens.extend(get_block_tokens(hid, size))
                prompt = decode(tokens, skip_special_tokens=False)
            else:
                prompt = ""

            turns.append(
                (
                    trace.get("timestamp"),
                    trace.get("delay"),
                    prompt,
                    trace.get("output_length"),
                )
            )
        results.append((session_id, turns))

    return results


def _has_broken_stdio() -> bool:
    """Check if any stdio stream has an invalid file descriptor."""
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
    """Redirect broken stdio to /dev/null before spawning Pool workers.

    Under the Textual terminal UI, child service processes inherit
    Textual-managed sys.stdin/stdout/stderr objects whose fileno() may
    return -1. When Pool workers fork and call util._close_stdin(), the
    invalid FD propagates to _posixsubprocess.fork_exec causing
    "bad value(s) in fds_to_keep". Only redirects when a problem is
    detected so non-dashboard modes keep normal stdio.
    """
    if not _has_broken_stdio():
        return

    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    if devnull > 2:
        os.close(devnull)
    sys.stdin = os.fdopen(0, "r", closefd=False)
    sys.stdout = os.fdopen(1, "w", closefd=False)
    sys.stderr = os.fdopen(2, "w", closefd=False)


def _set_daemon(daemon: bool) -> None:
    """Set the daemon flag on the current process.

    Python's multiprocessing refuses to spawn children from daemon processes,
    and AIPerf services run as daemons. This temporarily clears the flag.
    """
    try:
        mp.current_process().daemon = daemon
    except AssertionError:
        mp.current_process()._config["daemon"] = daemon


def _install_hard_exit_on_sigterm() -> None:
    """Replace the worker's SIGTERM handler with ``os._exit(0)``.

    Backstop for the rayon-thread-pool wedge: ``pool.terminate()`` (used by
    ``Pool.__exit__`` and as the fallback path in ``_shutdown_pool``) sends
    SIGTERM, but Python's default unwind invokes finalizers that block on
    ``rayon`` threads inside the CoW-shared HF tokenizer. Workers are
    stateless — the parent owns shared memory and persists results — so
    ``os._exit(0)`` is the right behavior on SIGTERM: skip finalizers, drop
    in-flight work (which terminate semantically discards anyway), exit
    immediately.

    Called from each pool's ``_init_worker``, after ``shared_memory``
    attachment but before any tokenizer use. ``signal.signal`` only works
    on the main thread of the main interpreter, so we swallow ``ValueError``
    when ``_init_worker`` is invoked directly from a unit test on a
    secondary thread (xdist worker, asyncio loop, etc.) — the handler is a
    backstop for live worker processes, not a correctness requirement.
    """
    import contextlib
    import signal

    def _hard_exit(_signum, _frame):  # noqa: ANN001
        os._exit(0)

    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGTERM, _hard_exit)


# How long to wait for graceful pool shutdown before falling back to SIGKILL.
# Workers exit promptly on the close()+sentinel path, so 10s is generous; the
# fallback only fires if a worker is genuinely wedged.
_POOL_JOIN_TIMEOUT_S: float = 10.0


def _shutdown_pool(
    pool: mp.pool.Pool, *, timeout_s: float = _POOL_JOIN_TIMEOUT_S
) -> None:
    """Drain a ``multiprocessing.Pool`` without the SIGTERM teardown hang.

    The default ``with Pool(...) as pool:`` exit calls ``pool.terminate()``,
    which SIGTERMs every worker. AIPerf trace-loader workers carry a
    CoW-shared HF tokenizer with a Rust ``rayon`` thread pool whose threads
    do not unwind on SIGTERM, so ``terminate()``+``join()`` wedges
    indefinitely after the imap loop ends (the entire CLI hangs ~5 minutes
    until a downstream timeout).

    Graceful path: ``close()`` lets each worker drain its task queue, hit
    the pool's normal sentinel, and exit via ``os._exit``; ``join()`` then
    returns promptly. We still fall back to ``terminate()`` if a worker
    hangs anyway, bounded by ``timeout_s`` so a stuck worker never blocks
    the whole CLI. The terminate fallback runs ``join`` in a thread because
    ``Pool.join`` itself has no timeout argument.
    """
    import threading

    pool.close()

    done = threading.Event()

    def _wait():
        try:
            pool.join()
        finally:
            done.set()

    waiter = threading.Thread(target=_wait, daemon=True)
    waiter.start()
    if done.wait(timeout=timeout_s):
        return

    pool.terminate()
    # Bound the SIGTERM path too — if rayon threads block join even after
    # SIGTERM, we accept the leak rather than hang the CLI. The leaked
    # workers exit with the parent.
    done.wait(timeout=timeout_s)


def parallel_convert(
    sessions: list[tuple[str, list[dict]]],
    *,
    tokenizer_name: str,
    corpus: np.ndarray,
    base_seed: int,
    block_size: int,
    sep_token: int | None,
    trace_id: str,
    trust_remote_code: bool = False,
    revision: str = "main",
    num_workers: int | None = None,
    batch_size: int = 100,
) -> Iterator[Conversation]:
    """Convert trace sessions to conversations using parallel workers.

    Yields Conversation objects one at a time as batches complete, using
    ``pool.imap`` to preserve insertion order while avoiding materializing
    all results in memory at once.

    Args:
        sessions: List of ``(session_id, [trace_dict, ...])`` tuples.
        tokenizer_name: HuggingFace tokenizer name (already cached locally).
        corpus: Tokenized corpus (a sequence of token IDs).
        base_seed: Base seed for HashIdRandomGenerator.
        block_size: Number of tokens per hash block.
        sep_token: Optional separator token prepended to each block.
        trace_id: File-derived trace ID for deterministic per-file seeding.
        num_workers: Number of worker processes. Defaults to ``min(cpu_count, 16)``.
        batch_size: Number of sessions per worker batch.

    Yields:
        Conversation objects in the same order as the input sessions.
    """
    _ensure_valid_stdio_fds()

    corpus_len = len(corpus)
    shm = shared_memory.SharedMemory(
        create=True, size=corpus_len * np.dtype(np.int32).itemsize
    )

    try:
        np.ndarray((corpus_len,), dtype=np.int32, buffer=shm.buf)[:] = corpus

        batches = [
            sessions[i : i + batch_size] for i in range(0, len(sessions), batch_size)
        ]

        workers = num_workers or min(os.cpu_count() or 4, 16)

        was_daemon = mp.current_process().daemon
        try:
            if was_daemon:
                _set_daemon(False)
            init_args = _WorkerInitArgs(
                shm_name=shm.name,
                corpus_len=corpus_len,
                tokenizer_name=tokenizer_name,
                base_seed=base_seed,
                block_size=block_size,
                sep_token=sep_token,
                trace_id=trace_id,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
            pool = get_loader_mp_context(
                preload_tokenizer=tokenizer_name,
                trust_remote_code=trust_remote_code,
                revision=revision,
            ).Pool(workers, _init_worker, (init_args,))
            try:
                # imap preserves submission order (unlike imap_unordered)
                for batch_result in pool.imap(_process_batch, batches):
                    for sid, turns in batch_result:
                        yield Conversation(
                            session_id=sid,
                            turns=[
                                Turn(
                                    timestamp=ts,
                                    delay=delay,
                                    texts=[Text(name="text", contents=[prompt])],
                                    max_tokens=max_tokens,
                                )
                                for ts, delay, prompt, max_tokens in turns
                            ],
                        )
            finally:
                # Avoid the SIGTERM teardown wedge — see ``_shutdown_pool``.
                _shutdown_pool(pool)
        finally:
            if was_daemon:
                _set_daemon(True)
    finally:
        shm.close()
        shm.unlink()
