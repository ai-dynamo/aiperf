# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process and UI bootstrap helpers for aiperf.cli_runner.

Multiprocessing start method, Dashboard log-queue creation, macOS terminal
FD protection, and tokenizer env-var preload all run before any service is
spawned. Kept together because they share fragile ordering constraints —
notably the forkserver snapshots env once on first spawn, so the tokenizer
preload env vars must be set before the first queue is created.
"""

from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing as _mp

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkRun


def _configure_multiprocessing_start_method(using_dashboard: bool) -> None:
    """Pick a multiprocessing start method compatible with the current UI.

    NOTE: On macOS, when using the Textual UI with multiprocessing, terminal
    corruption (ASCII garbage, freezing) can occur when mouse events interfere
    with child processes. We apply multiple layers of protection:
      1. Set spawn method early (before any multiprocessing operations)
      2. Create log_queue before any UI initialization
      3. Set FD_CLOEXEC on terminal file descriptors
      4. Close terminal FDs in child processes (done in bootstrap.py)
    Env override takes precedence for all platforms.
    """
    import multiprocessing
    import platform

    from aiperf.common.environment import Environment

    configured_start_method = getattr(
        Environment.SERVICE, "MULTIPROCESSING_START_METHOD", None
    )
    if configured_start_method:
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method(configured_start_method, force=True)
        return

    if platform.system() == "Darwin" and using_dashboard:
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method("spawn", force=True)


def _setup_ui_queues(
    using_dashboard: bool, run: BenchmarkRun, logger: AIPerfLogger
) -> _mp.Queue | None:
    """Create the Dashboard log queue when needed.

    Returns the log_queue (or ``None`` when no Dashboard UI is active). When
    Dashboard UI is running on macOS, FD_CLOEXEC is set on terminal
    descriptors to prevent child processes corrupting the parent terminal.
    """
    import platform

    if not using_dashboard:
        from aiperf.common.logging import setup_rich_logging

        setup_rich_logging(run)
        return None

    from aiperf.common.logging import get_global_log_queue

    log_queue = get_global_log_queue()

    if platform.system() == "Darwin":
        _set_fd_cloexec_on_terminal(logger)
    return log_queue


def _set_fd_cloexec_on_terminal(logger: AIPerfLogger) -> None:
    """Mark stdio as close-on-exec (macOS terminal-corruption mitigation)."""
    import fcntl

    try:
        for fd in [sys.stdin.fileno(), sys.stdout.fileno(), sys.stderr.fileno()]:
            flags = fcntl.fcntl(fd, fcntl.F_GETFD)
            fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
        logger.debug("Set FD_CLOEXEC on terminal file descriptors for macOS")
    except (OSError, ValueError, AttributeError) as e:
        # Non-fatal if this fails, other layers will protect
        logger.debug(f"Could not set FD_CLOEXEC on terminal descriptors: {e}")


def _configure_tokenizer_preload(run: BenchmarkRun) -> None:
    """Surface tokenizer identities into env so the forkserver preload sees them.

    Read by :mod:`aiperf.records._tokenizer_preload` at forkserver-helper
    startup. Must be called before the first subprocess spawn (and
    therefore before queue creation in :func:`_setup_ui_queues`), since
    Python's forkserver starts on demand and snapshots the env once.

    Name selection mirrors :class:`~aiperf.records.inference_result_parser.InferenceResultParser`:
    an explicit ``tokenizer.name`` in config overrides per-model defaults
    for every model. Without it, each model name is used as its own
    tokenizer name.

    Uses raw (unresolved) names because the resolver chain hasn't run yet
    when this is called. In the common case of canonical HF IDs (e.g.
    ``Qwen/Qwen3-0.6B``) the raw name is the correct tokenizer name and
    CoW sharing works; aliased names (e.g. ``gpt2``) miss the preload
    cache and fall through to per-RP on-demand loading — same as without
    this feature.
    """
    import os

    cfg = run.cfg
    tokenizer_cfg = cfg.tokenizer
    if tokenizer_cfg is not None and tokenizer_cfg.name:
        names = [tokenizer_cfg.name]
    else:
        names = cfg.get_model_names()
    if not names:
        return
    os.environ.setdefault("AIPERF_PRELOAD_TOKENIZERS", ",".join(names))
    if tokenizer_cfg is not None:
        os.environ.setdefault(
            "AIPERF_PRELOAD_TOKENIZER_TRUST_REMOTE_CODE",
            "true" if tokenizer_cfg.trust_remote_code else "false",
        )
        os.environ.setdefault(
            "AIPERF_PRELOAD_TOKENIZER_REVISION",
            tokenizer_cfg.revision or "main",
        )
