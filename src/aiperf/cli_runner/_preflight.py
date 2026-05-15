# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preflight checks for aiperf.cli_runner.

These run before any service bootstrap so misconfiguration surfaces as a
clean ``ConfigurationError`` instead of a stack trace from deep inside the
controller. The three checks are: artifact-dir creatable+writable, file
descriptor soft limit raised (and hard limit large enough), and the target
endpoint reachable.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.config import BenchmarkPlan


def _preflight_artifact_dir(plan: BenchmarkPlan) -> None:
    """Validate that the artifact directory is creatable and writable.

    Why: ``setup_rich_logging`` calls ``log_folder.mkdir(parents=True)`` deep
    inside the controller bootstrap; without this preflight, an existing-file
    artifact path or a read-only parent surfaces as a stack-trace-laden
    ``NotADirectoryError``/``PermissionError`` instead of a clean config error.
    Surfacing it here lets ``profile.py`` render a single actionable panel via
    ``exit_on_error(quiet_for=(ConfigurationError,))``.
    """
    from aiperf.config.loader.errors import ConfigurationError

    artifact_dir: Path = plan.configs[0].artifacts.dir
    if artifact_dir.exists() and not artifact_dir.is_dir():
        raise ConfigurationError(
            f"artifact_dir '{artifact_dir}' exists but is not a directory. "
            f"Remove the file or pick a different --artifact-dir."
        )

    parent = artifact_dir if artifact_dir.exists() else artifact_dir.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if parent.exists() and not os.access(parent, os.W_OK):
        raise ConfigurationError(
            f"artifact_dir '{artifact_dir}' is not writable "
            f"(no write permission on existing parent '{parent}'). "
            f"Pick a different --artifact-dir or fix permissions."
        )


def _preflight_fd_limit() -> None:
    """Raise RLIMIT_NOFILE soft limit and bail early if hard limit is too low.

    Why: aiperf's multiprocess service mesh (ZMQ inproc/IPC + per-worker HTTP
    pools) needs hundreds of file descriptors. With the default soft limit of
    1024 on most distros it usually fits, but bumping to 8192 leaves headroom
    for higher concurrency. When the hard limit is below the working floor,
    the ZMQ ipc_listener SIGABRTs mid-startup (`Too many open files
    src/ipc_listener.cpp:297`) — surface a clean error here instead.
    """
    try:
        import resource
    except ImportError:
        return  # Windows / unsupported platform; nothing to do.

    from aiperf.config.loader.errors import ConfigurationError

    target_soft = 8192
    min_required = 256
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < min_required:
        raise ConfigurationError(
            f"file descriptor hard limit too low: {hard} (need at least "
            f"{min_required}). Raise it via `ulimit -n 4096` (or larger) "
            f"before running aiperf."
        )
    if soft >= target_soft or soft == resource.RLIM_INFINITY:
        return
    new_soft = target_soft if hard == resource.RLIM_INFINITY else min(target_soft, hard)
    if new_soft <= soft:
        return
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))


def _preflight_endpoint_ready(plan: BenchmarkPlan) -> None:
    """Block until the target endpoint is ready (see ready_checker).

    Runs before any service bootstrap so a slow/down server fails fast with
    a clear error instead of timing out inside the system controller. Uses
    the endpoint config of the first run in the plan — multi-run sweeps are
    assumed to share an endpoint.
    """
    import asyncio
    import logging

    cfg = plan.configs[0].endpoint
    if cfg.wait_for_model_timeout <= 0:
        return

    # Preflight runs before rich logging is installed; install a minimal
    # stderr handler so probe lines are visible.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    from aiperf.common.readiness_probe import wait_for_endpoint

    headers = dict(cfg.headers or {})
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    asyncio.run(
        wait_for_endpoint(
            urls=list(cfg.urls),
            model_names=plan.configs[0].get_model_names(),
            mode=cfg.wait_for_model_mode,
            endpoint_type=str(cfg.type),
            custom_endpoint=cfg.path,
            timeout_s=cfg.wait_for_model_timeout,
            interval_s=cfg.wait_for_model_interval,
            headers=headers,
        )
    )
