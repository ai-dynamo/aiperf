# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel decode utilities for batch tokenizer operations.

This module provides functions to decode multiple token sequences in parallel
using ProcessPoolExecutor, bypassing Python's GIL for CPU-bound tokenizer
operations.

The daemon flag on the current process is temporarily cleared because Python's
multiprocessing refuses to spawn children from daemon processes, and AIPerf
services run as daemons.
"""

import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger

if TYPE_CHECKING:
    from aiperf.common.tokenizer import Tokenizer

_logger = AIPerfLogger(__name__)

# Module-level tokenizer for worker processes (initialized once per worker)
_worker_tokenizer: "Tokenizer | None" = None
_worker_tokenizer_key: tuple[str, bool, str] | None = None
_worker_decode_count: int = 0


def _init_worker(
    tokenizer_name: str,
    trust_remote_code: bool = False,
    revision: str = "main",
    log_level: int = logging.WARNING,
) -> None:
    """Initialize tokenizer in worker process.

    Called once per worker by ProcessPoolExecutor. Loads the tokenizer so
    subsequent decode calls reuse it. ``tokenizer_name`` must be a pre-resolved
    model name or local path; aliases are not resolved here (no network).
    ``log_level`` mirrors the parent's effective root log level so worker
    DEBUG output surfaces under --verbose.
    """
    global _worker_tokenizer, _worker_tokenizer_key, _worker_decode_count

    from aiperf.common.logging import setup_subprocess_logging

    setup_subprocess_logging(log_level)
    logger = AIPerfLogger(__name__)
    requested_key = (tokenizer_name, trust_remote_code, revision)

    logger.debug(
        f"parallel_decode worker init: pid={os.getpid()}, ppid={os.getppid()}, "
        f"tokenizer={tokenizer_name!r}, trust_remote_code={trust_remote_code}, "
        f"revision={revision!r}, HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')!r}, "
        f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')!r}, "
        f"HF_HOME={os.environ.get('HF_HOME')!r}, "
        f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')!r}"
    )

    if _worker_tokenizer is not None and _worker_tokenizer_key == requested_key:
        logger.debug(
            f"parallel_decode worker: reusing cached tokenizer for {tokenizer_name!r}"
        )
        return
    if _worker_tokenizer is not None:
        logger.debug(
            f"parallel_decode worker: key changed "
            f"({_worker_tokenizer_key} -> {requested_key}), reloading"
        )

    # Worker-scoped offline mode: the main process already cached the
    # tokenizer; this env mutation doesn't escape this child.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from aiperf.common.tokenizer import Tokenizer

    start = time.perf_counter()
    try:
        _worker_tokenizer = Tokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            resolve_alias=False,
        )
    except Exception:
        logger.exception(
            f"parallel_decode worker FAILED to load tokenizer {tokenizer_name!r} "
            f"after {time.perf_counter() - start:.2f}s"
        )
        raise
    _worker_tokenizer_key = requested_key
    _worker_decode_count = 0
    logger.debug(
        f"parallel_decode worker: loaded tokenizer {tokenizer_name!r} in "
        f"{time.perf_counter() - start:.2f}s"
    )


def _decode_tokens(token_ids: list[int]) -> str:
    """Decode tokens using worker's tokenizer.

    Args:
        token_ids: List of token IDs to decode.

    Returns:
        Decoded string.

    Raises:
        RuntimeError: If worker tokenizer is not initialized.
    """
    global _worker_decode_count
    if _worker_tokenizer is None:
        raise RuntimeError("Worker tokenizer not initialized")
    result = _worker_tokenizer.decode(token_ids)
    _worker_decode_count += 1
    # Log only the first decode per worker to confirm the path is working
    # without per-call noise in DEBUG.
    if _worker_decode_count == 1:
        _logger.debug(
            f"parallel_decode worker pid={os.getpid()}: first decode succeeded "
            f"({len(token_ids)} tokens in, {len(result)} chars out)"
        )
    return result


def parallel_decode(
    token_sequences: list[list[int]],
    tokenizer_name: str,
    *,
    max_workers: int | None = None,
    chunksize: int = 50,
    trust_remote_code: bool = False,
    revision: str = "main",
) -> list[str]:
    """Decode multiple token sequences in parallel using ProcessPoolExecutor.

    This function is optimized for batch decoding of many token sequences.
    For small batches (< 10 sequences), it falls back to sequential decoding
    to avoid process spawn overhead.

    Args:
        token_sequences: List of token ID lists to decode.
        tokenizer_name: Pre-resolved model name or local path (alias resolution
            is skipped; callers must resolve aliases beforehand).
        max_workers: Number of worker processes. Defaults to min(cpu_count, 8).
        chunksize: Number of items per worker batch for map().
        trust_remote_code: Whether to trust remote code when loading.
        revision: The specific model version to use.

    Returns:
        List of decoded strings in the same order as input.
    """
    if not token_sequences:
        _logger.debug("parallel_decode: no token sequences, returning empty list")
        return []

    # For small batches, sequential is faster (avoid process overhead)
    if len(token_sequences) < 10:
        return _sequential_decode(
            token_sequences, tokenizer_name, trust_remote_code, revision
        )

    num_workers = max_workers or min(mp.cpu_count() or 4, 8)
    log_level = logging.getLogger().getEffectiveLevel()
    _logger.debug(
        f"parallel_decode: spawning {num_workers} worker(s) for "
        f"{len(token_sequences)} sequence(s), tokenizer={tokenizer_name!r}, "
        f"chunksize={chunksize}, propagating log_level={log_level}"
    )

    # Temporarily clear the daemon flag so ProcessPoolExecutor can spawn workers.
    # Python's multiprocessing refuses to spawn children from daemon processes,
    # and AIPerf services run as daemons.
    #
    # Alternatives considered:
    # - billiard: bypasses the daemon restriction natively, but crashes with
    #   BrokenProcessPool on macOS due to terminal FD inheritance issues.
    # - loky: robust reusable executor, but still requires the same daemon flag
    #   hack, so no advantage over stdlib.
    was_daemon = mp.current_process().daemon
    start = time.perf_counter()
    try:
        if was_daemon:
            _set_daemon(False)
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(tokenizer_name, trust_remote_code, revision, log_level),
        ) as executor:
            results = list(
                executor.map(_decode_tokens, token_sequences, chunksize=chunksize)
            )
    finally:
        if was_daemon:
            _set_daemon(True)
    _logger.debug(
        f"parallel_decode: completed {len(results)} decode(s) in "
        f"{time.perf_counter() - start:.2f}s"
    )

    return results


def _sequential_decode(
    token_sequences: list[list[int]],
    tokenizer_name: str,
    trust_remote_code: bool,
    revision: str,
) -> list[str]:
    """Decode in-process for small batches where the worker spawn overhead
    outweighs parallelism. Mirrors ``parallel_decode`` semantics for n<10."""
    _logger.debug(
        f"parallel_decode: {len(token_sequences)} sequence(s) < 10 threshold, "
        f"using sequential path for tokenizer {tokenizer_name!r}"
    )
    from aiperf.common.tokenizer import Tokenizer

    tokenizer = Tokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
        revision=revision,
        resolve_alias=False,
    )
    return [tokenizer.decode(tokens) for tokens in token_sequences]


def _set_daemon(daemon: bool) -> None:
    """Set the daemon flag on the current process."""
    try:
        mp.current_process().daemon = daemon
    except AssertionError:
        # Fallback to using the internal _config dictionary if assertions are enabled
        mp.current_process()._config["daemon"] = daemon
