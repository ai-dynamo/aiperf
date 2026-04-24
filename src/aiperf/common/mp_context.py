# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocessing context setup for AIPerf subprocess spawning.

Centralizes the forkserver/spawn context selection and the one-time
forkserver-helper startup dance. Imported by ``subprocess_manager`` and
a few lazy call sites in ``logging``/``error_queue``.
"""

from __future__ import annotations

import multiprocessing
import os
import platform

_SPAWN_TIMEOUT = 60.0
"""Safety-net timeout for process.start(). Normal spawns complete in
milliseconds; this guards against extreme system conditions (memory
pressure, exhausted forkserver) blocking the event loop indefinitely."""

_FORKSERVER_PRELOAD = [
    # -- aiperf core (shared by all services) --
    "aiperf.common.bootstrap",
    "aiperf.config",
    "aiperf.common.environment",
    "aiperf.common.logging",
    "aiperf.common.enums",
    "aiperf.common.hooks",
    "aiperf.common.messages",
    "aiperf.common.models",
    "aiperf.common.control_structs",
    "aiperf.common.types",
    "aiperf.plugin",
    "aiperf.plugin.enums",
    "aiperf.common.base_service",
    "aiperf.common.base_component_service",
    "aiperf.common.mixins",
    # -- Worker (replicable: num_workers instances) --
    "aiperf.workers.worker",
    "aiperf.workers.inference_client",
    "aiperf.workers.session_manager",
    "aiperf.credit",
    "aiperf.credit.issuer",
    "aiperf.transports",
    "aiperf.transports.aiohttp_client",
    # -- RecordProcessor (replicable: num_record_processors instances) --
    "aiperf.records.record_processor_service",
    "aiperf.metrics",
    "aiperf.post_processors",
    # Imports + instantiates HF tokenizers into the forkserver helper's
    # anon heap when AIPERF_PRELOAD_TOKENIZERS is set, so every RP child
    # CoW-shares them. No-op when the env var is empty (K8s mode).
    "aiperf.records._tokenizer_preload",
    # -- heavy third-party deps --
    "pydantic",
    "numpy",
    "zmq",
    "uvloop",
    "orjson",
    "msgspec",
    "rich.console",
    "rich.logging",
    "aiohttp",
    "aiofiles",
    "psutil",
]

# Module-level singleton: multiprocessing contexts are expensive to create
# and must be shared so all subprocess spawns use the same forkserver.
_mp_context: multiprocessing.context.BaseContext | None = None


def get_mp_context() -> multiprocessing.context.BaseContext:
    """Return the forkserver (Linux) or spawn (macOS) multiprocessing context.

    Lazily created on first call to avoid side-effects at import time
    (e.g. during pytest-xdist worker collection).
    """
    global _mp_context
    if _mp_context is None:
        method = "forkserver" if platform.system() == "Linux" else "spawn"
        _mp_context = multiprocessing.get_context(method)
        if platform.system() == "Linux":
            _mp_context.set_forkserver_preload(_FORKSERVER_PRELOAD)
            # The forkserver is a long-lived helper process that inherits
            # the parent's stdin/stdout/stderr at spawn time. If the
            # parent aiperf process runs under captured pipes (pytest
            # subprocess, CI harnesses), the forkserver keeps those pipe
            # fds open even after the parent exits, preventing
            # process.communicate() from seeing EOF. Force-start the
            # forkserver here with stdio redirected to /dev/null so it
            # never holds the pipes.
            _eagerly_start_forkserver(_mp_context)
    return _mp_context


def _eagerly_start_forkserver(
    ctx: multiprocessing.context.BaseContext,
) -> None:
    """Start the forkserver helper with stdio pointing at /dev/null.

    Must run before any fork/spawn happens through ``ctx`` so the helper
    inherits /dev/null rather than the parent's captured pipes.
    """
    import contextlib
    from multiprocessing import forkserver as _fs

    # If forkserver is already running, we're too late to redirect its
    # stdio — it has already inherited whatever the parent had at
    # startup. Nothing we can do safely here without a larger refactor.
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
            ctx._prepare_data = getattr(ctx, "_prepare_data", None)
            _fs.ensure_running()
    finally:
        for fd, original in zip((0, 1, 2), saved, strict=False):
            os.dup2(original, fd)
            os.close(original)
        os.close(devnull_fd)
