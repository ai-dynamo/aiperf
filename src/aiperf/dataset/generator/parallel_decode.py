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

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.common.tokenizer import Tokenizer

# Module-level tokenizer for worker processes (initialized once per worker)
_worker_tokenizer: "Tokenizer | None" = None
_worker_tokenizer_name: str | None = None


def _init_worker(tokenizer_name: str) -> None:
    """Initialize tokenizer in worker process.

    This function is called once per worker process when the ProcessPoolExecutor
    starts. It loads the tokenizer so subsequent decode calls don't need to reload it.

    Args:
        tokenizer_name: Name or path of the pretrained tokenizer to load.
    """
    global _worker_tokenizer, _worker_tokenizer_name
    if _worker_tokenizer is None or _worker_tokenizer_name != tokenizer_name:
        # The main process already downloaded and cached the tokenizer, so force
        # offline mode to skip network requests and alias resolution.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from aiperf.common.tokenizer import Tokenizer

        _worker_tokenizer = Tokenizer.from_pretrained(
            tokenizer_name, resolve_alias=False
        )
        _worker_tokenizer_name = tokenizer_name


def _decode_tokens(token_ids: list[int]) -> str:
    """Decode tokens using worker's tokenizer.

    Args:
        token_ids: List of token IDs to decode.

    Returns:
        Decoded string.

    Raises:
        RuntimeError: If worker tokenizer is not initialized.
    """
    if _worker_tokenizer is None:
        raise RuntimeError("Worker tokenizer not initialized")
    return _worker_tokenizer.decode(token_ids, skip_special_tokens=False)


def parallel_decode(
    token_sequences: list[list[int]],
    tokenizer_name: str,
    max_workers: int | None = None,
    chunksize: int = 50,
) -> list[str]:
    """Decode multiple token sequences in parallel using ProcessPoolExecutor.

    This function is optimized for batch decoding of many token sequences.
    For small batches (< 10 sequences), it falls back to sequential decoding
    to avoid process spawn overhead.

    Args:
        token_sequences: List of token ID lists to decode.
        tokenizer_name: Name or path of the pretrained tokenizer to use in workers.
        max_workers: Number of worker processes. Defaults to min(cpu_count, 8).
        chunksize: Number of items per worker batch for map().

    Returns:
        List of decoded strings in the same order as input.
    """
    if not token_sequences:
        return []

    # For small batches, sequential is faster (avoid process overhead)
    if len(token_sequences) < 10:
        from aiperf.common.tokenizer import Tokenizer

        tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        return [
            tokenizer.decode(tokens, skip_special_tokens=False)
            for tokens in token_sequences
        ]

    num_workers = max_workers or min(mp.cpu_count() or 4, 8)

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
    try:
        if was_daemon:
            _set_daemon(False)
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(tokenizer_name,),
        ) as executor:
            results = list(
                executor.map(_decode_tokens, token_sequences, chunksize=chunksize)
            )
    finally:
        if was_daemon:
            _set_daemon(True)

    return results


def _set_daemon(daemon: bool) -> None:
    """Set the daemon flag on the current process."""
    try:
        mp.current_process().daemon = daemon
    except AssertionError:
        # Fallback to using the internal _config dictionary if assertions are enabled
        mp.current_process()._config["daemon"] = daemon
