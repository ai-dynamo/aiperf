# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated multiprocessing context for trace-loader worker pools.

Trace-loader pools (:mod:`aiperf.dataset.loader.weka_parallel_convert`,
:mod:`aiperf.dataset.loader.parallel_convert`, and
:mod:`aiperf.dataset.generator.parallel_decode`) fork worker processes after
the parent has loaded HF tokenizers and exercised their Rust thread pool.
Under the default ``fork`` start method, that inherits broken rayon state and
``transformers`` whose offline-mode flag was cached at parent-import time —
the combination deadlocks the workers.

Forking from a long-lived ``forkserver`` helper instead bypasses parent state
entirely: the helper is a fresh Python interpreter that imports only the
modules in ``_LOADER_PRELOAD``. The helper additionally instantiates the
benchmark's tokenizer (driven by env vars set in
:func:`get_loader_mp_context`) so every worker fork CoW-shares the in-memory
copy instead of re-loading from disk.

This context is intentionally *separate* from any future service-spawning
context — its sole consumer is trace-loader worker pools, so its preload
list and lifecycle are scoped to that use case.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import platform

_LOADER_PRELOAD = [
    # Module imports happen once in the forkserver helper so workers don't
    # pay the transformers/HF import cost on every spawn. Order matters
    # only loosely — the tokenizer-preload module is last so its instance
    # creation finds Tokenizer already imported.
    "aiperf.common.tokenizer",
    "aiperf.common.hash_id_random_generator",
    "numpy",
    # Side-effecting: instantiates the tokenizer named by
    # AIPERF_LOADER_PRELOAD_TOKENIZER into the helper's heap.
    "aiperf.dataset._tokenizer_preload",
]

_ENV_PRELOAD_NAME = "AIPERF_LOADER_PRELOAD_TOKENIZER"
_ENV_PRELOAD_TRUST = "AIPERF_LOADER_PRELOAD_TRUST_REMOTE_CODE"
_ENV_PRELOAD_REVISION = "AIPERF_LOADER_PRELOAD_REVISION"

_loader_ctx: multiprocessing.context.BaseContext | None = None


def get_loader_mp_context(
    *,
    preload_tokenizer: str | None = None,
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> multiprocessing.context.BaseContext:
    """Return the trace-loader-specific multiprocessing context.

    On Linux this is a ``forkserver`` context whose helper is started eagerly
    with stdio redirected to ``/dev/null`` and (optionally) the named
    tokenizer pre-instantiated in its heap so workers CoW-share it. On
    macOS this is a ``spawn`` context (no helper; each worker is a fresh
    interpreter, and ``preload_tokenizer`` is a no-op).

    The context is built once and cached; later calls with a different
    ``preload_tokenizer`` reuse the original helper. Callers are expected
    to share a single tokenizer per process lifetime (the typical AIPerf
    flow). Workers receiving a different name fall back to on-demand load.
    """
    global _loader_ctx
    if _loader_ctx is not None:
        return _loader_ctx

    # Env must be set BEFORE the forkserver helper is spawned: it reads
    # these at module-import time and instantiates the tokenizer once in
    # its own heap, where every forked worker CoW-shares it.
    if preload_tokenizer:
        os.environ[_ENV_PRELOAD_NAME] = preload_tokenizer
        os.environ[_ENV_PRELOAD_TRUST] = "true" if trust_remote_code else "false"
        os.environ[_ENV_PRELOAD_REVISION] = revision or "main"

    method = "forkserver" if platform.system() == "Linux" else "spawn"
    ctx = multiprocessing.get_context(method)
    if method == "forkserver":
        ctx.set_forkserver_preload(_LOADER_PRELOAD)
        _eagerly_start_forkserver()
    _loader_ctx = ctx
    return _loader_ctx


def _eagerly_start_forkserver() -> None:
    """Boot the forkserver helper with stdio pointing at ``/dev/null``.

    Must run before any fork through the context so the helper inherits
    ``/dev/null`` rather than the parent's possibly-captured stdio (pytest,
    Textual dashboard, etc.). If the helper is already running, we're too
    late to redirect — bail out.
    """
    from multiprocessing import forkserver as _fs

    if getattr(_fs, "_forkserver", None) and getattr(
        _fs._forkserver, "_forkserver_pid", None
    ):
        return

    devnull_fd = os.open(os.devnull, os.O_RDWR)
    saved = [os.dup(fd) for fd in (0, 1, 2)]
    try:
        for fd in (0, 1, 2):
            os.dup2(devnull_fd, fd)
        with contextlib.suppress(Exception):
            _fs.ensure_running()
    finally:
        for fd, original in zip((0, 1, 2), saved, strict=False):
            os.dup2(original, fd)
            os.close(original)
        os.close(devnull_fd)
