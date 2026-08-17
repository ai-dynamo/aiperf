# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated multiprocessing context for trace-loader worker pools.

Trace-loader pools (:mod:`aiperf.dataset.loader.weka_parallel_convert`,
:mod:`aiperf.dataset.loader.parallel_convert`, and the graph adapter pool in
:mod:`aiperf.dataset.graph.adapters.shared.pool`) fork worker processes after
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

from aiperf.common.constants import IS_LINUX

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
_loader_ctx_key: tuple[str, bool, str] | None = None


def _preload_key(
    preload_tokenizer: str | None,
    *,
    trust_remote_code: bool | None,
    revision: str | None,
) -> tuple[str, bool, str] | None:
    """Stable identity for the tokenizer the forkserver helper should CoW-share.

    ``None`` trust/revision mean "not specified by this caller", so the identity
    falls back to whatever ``configure_loader_tokenizer_env`` published (the run
    config) rather than to the parameter defaults -- otherwise a name-only call
    would key on ``(name, False, "main")`` and spuriously conflict with the run's
    own trust/revision.
    """
    if not preload_tokenizer:
        return None
    trust = (
        trust_remote_code
        if trust_remote_code is not None
        else os.environ.get(_ENV_PRELOAD_TRUST, "false") == "true"
    )
    rev = revision or os.environ.get(_ENV_PRELOAD_REVISION) or "main"
    return (preload_tokenizer, trust, rev)


def configure_loader_tokenizer_env(
    *,
    trust_remote_code: bool,
    revision: str | None,
) -> None:
    """Publish the run's tokenizer ``trust_remote_code``/``revision`` for the preload.

    The graph adapter pool threads only the tokenizer NAME into
    :func:`get_loader_mp_context`; trust/revision travel via env because the
    forkserver helper snapshots the env once at spawn and every worker (preload
    hit or on-demand fallback) reads the SAME inherited triple. Must run in the
    parent before the first loader pool is opened -- the graph parse seam calls
    it at graph-configure time, before any parse.
    """
    os.environ[_ENV_PRELOAD_TRUST] = "true" if trust_remote_code else "false"
    os.environ[_ENV_PRELOAD_REVISION] = revision or "main"


def get_loader_mp_context(
    *,
    preload_tokenizer: str | None = None,
    trust_remote_code: bool | None = None,
    revision: str | None = None,
) -> multiprocessing.context.BaseContext:
    """Return the trace-loader-specific multiprocessing context.

    On Linux this is a ``forkserver`` context whose helper is started eagerly
    with stdio redirected to ``/dev/null`` and (optionally) the named
    tokenizer pre-instantiated in its heap so workers CoW-share it. On
    macOS this is a ``spawn`` context (no helper; each worker is a fresh
    interpreter, and ``preload_tokenizer`` is a no-op).

    The context is built once and cached under the first non-``None``
    tokenizer identity. Later calls with the same identity (or with no
    preload request) reuse it. A later call with a *different* tokenizer /
    trust / revision raises :class:`ValueError`: the forkserver helper is
    process-global and cannot swap its CoW-preloaded tokenizer mid-process.
    """
    global _loader_ctx, _loader_ctx_key
    key = _preload_key(
        preload_tokenizer,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    if _loader_ctx is not None:
        if key is not None and _loader_ctx_key is not None and key != _loader_ctx_key:
            raise ValueError(
                "loader mp context already preloaded for "
                f"tokenizer={_loader_ctx_key[0]!r} trust_remote_code="
                f"{_loader_ctx_key[1]!r} revision={_loader_ctx_key[2]!r}; "
                f"cannot switch to tokenizer={key[0]!r} trust_remote_code="
                f"{key[1]!r} revision={key[2]!r}. Use one tokenizer per process."
            )
        return _loader_ctx

    # Env must be set BEFORE the forkserver helper is spawned: it reads
    # these at module-import time and instantiates the tokenizer once in
    # its own heap, where every forked worker CoW-shares it.
    # Only an EXPLICIT argument overrides what ``configure_loader_tokenizer_env``
    # already published: production calls this name-only (via
    # ``_loader_pool_context``), and writing the parameter defaults would clobber
    # the run config's trust/revision with false/main.
    if preload_tokenizer:
        os.environ[_ENV_PRELOAD_NAME] = preload_tokenizer
        if trust_remote_code is not None:
            os.environ[_ENV_PRELOAD_TRUST] = "true" if trust_remote_code else "false"
        else:
            os.environ.setdefault(_ENV_PRELOAD_TRUST, "false")
        if revision is not None:
            os.environ[_ENV_PRELOAD_REVISION] = revision
        else:
            os.environ.setdefault(_ENV_PRELOAD_REVISION, "main")

    method = "forkserver" if IS_LINUX else "spawn"
    ctx = multiprocessing.get_context(method)
    if method == "forkserver":
        ctx.set_forkserver_preload(_LOADER_PRELOAD)
        _eagerly_start_forkserver()
    _loader_ctx = ctx
    _loader_ctx_key = key
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
