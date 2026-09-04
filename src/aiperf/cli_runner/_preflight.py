# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preflight checks for aiperf.cli_runner.

These run before any service bootstrap so misconfiguration surfaces as a
clean ``ConfigurationError`` instead of a stack trace from deep inside the
controller. The checks are: artifact-dir creatable+writable, accuracy
benchmark/grader optional dependencies present, file descriptor soft limit
raised (and hard limit large enough), and the target endpoint reachable.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.endpoint_auth import auth_headers_for_endpoint

if TYPE_CHECKING:
    from aiperf.config import BenchmarkPlan

# Re-export for callers still importing the private name; prefer
# ``auth_headers_for_endpoint`` from ``aiperf.common.endpoint_auth``.
_readiness_auth_headers = auth_headers_for_endpoint


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


def _preflight_accuracy_deps(plan: BenchmarkPlan) -> None:
    """Fail fast if a selected accuracy benchmark's or grader's optional
    dependency (lighteval / deepeval, the ``[accuracy]`` extra) is missing.

    Why: both the benchmark loader (dataset-manager service) and the grader
    (record-processor daemon) raise at instantiation when their optional
    package is absent. The grader crash isn't propagated to the controller, so
    the user sees a raw multiprocessing traceback and the run hangs waiting for
    records that never arrive; the loader crash surfaces later and less
    cleanly. Checking here — in the main process, before any service spawns —
    turns both into a single clean ``ConfigurationError`` panel with a non-zero
    exit and no hang.
    """
    from aiperf.config.loader.errors import ConfigurationError
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType
    from aiperf.plugin.types import TypeNotFoundError

    checked: set[tuple[str, str]] = set()
    for config in plan.configs:
        acc_cfg = getattr(config, "accuracy", None)
        if acc_cfg is None or not acc_cfg.enabled:
            continue

        # Keep every preflight failure on the ConfigurationError path: plugin
        # lookups raise TypeNotFoundError/KeyError/ValueError for an unknown or
        # malformed benchmark/grader name, ImportError for a broken external
        # plugin module, and AttributeError when the module imports but the
        # configured class is missing (PluginEntry.load); check_available raises
        # RuntimeError for a missing optional dependency. Any of these would
        # otherwise leak a raw traceback.
        try:
            meta = plugins.get_metadata(
                PluginType.ACCURACY_BENCHMARK, acc_cfg.benchmark
            )
            grader_name = acc_cfg.grader or meta.get(
                "default_grader", "multiple_choice"
            )

            key = (str(acc_cfg.benchmark), grader_name)
            if key in checked:
                continue
            checked.add(key)

            # ``check_available`` is an optional hook on both the benchmark
            # loader and grader: the plugin contracts don't require it, so a
            # custom plugin need not define it. Built-in graders inherit a
            # no-op default from ``BaseGrader``; the deepeval-gated benchmark
            # loaders define it. Treat absence as "no optional deps to verify".
            benchmark_cls = plugins.get_class(
                PluginType.ACCURACY_BENCHMARK, acc_cfg.benchmark
            )
            grader_cls = plugins.get_class(PluginType.ACCURACY_GRADER, grader_name)
            for cls in (benchmark_cls, grader_cls):
                check = getattr(cls, "check_available", None)
                if callable(check):
                    check()
        except (
            TypeNotFoundError,
            KeyError,
            ValueError,
            ImportError,
            AttributeError,
            RuntimeError,
        ) as exc:
            raise ConfigurationError(str(exc)) from exc


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

    headers = _readiness_auth_headers(cfg)

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


def _public_dataset_loaders(plan: BenchmarkPlan):
    """Yield ``(LoaderClass, loader_kwargs)`` for each public dataset in the plan.

    Reads plugin metadata directly rather than going through the composer,
    which needs a live run. Only the fields the preflight hooks act on are
    forwarded, and the yielded pairs are deduplicated by those fields rather
    than by selector name: a category sweep names 11 distinct selectors that
    all resolve the same underlying config, and keying on the selector would
    make preflight issue 11 identical auth probes before the run starts.
    """
    from aiperf.config.dataset import PublicDataset
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    seen: set[tuple[type, str | None]] = set()
    for cfg in plan.configs:
        dataset = cfg.get_default_dataset()
        if not isinstance(dataset, PublicDataset):
            continue

        LoaderClass = plugins.get_class(
            PluginType.PUBLIC_DATASET_LOADER, dataset.dataset
        )
        metadata = plugins.get_public_dataset_loader_metadata(dataset.dataset)
        hf_subset = dataset.hf_subset or metadata.hf_subset

        key = (LoaderClass, hf_subset)
        if key in seen:
            continue
        seen.add(key)

        yield LoaderClass, ({"hf_subset": hf_subset} if hf_subset else {})


def _call_optional_hook(
    loader_class: type, name: str, kwargs: dict[str, object]
) -> None:
    """Invoke a preflight hook when the loader defines one.

    ``PublicDatasetLoaderProtocol`` does not require these hooks, so a loader
    registered by an external plugin need not implement them. Absence means
    "nothing to prepare", matching how ``_preflight_accuracy_deps`` treats the
    optional ``check_available`` hook.
    """
    hook = getattr(loader_class, name, None)
    if callable(hook):
        hook(**kwargs)


def _preflight_dataset_access(plan: BenchmarkPlan) -> None:
    """Verify gated public datasets are reachable before anything expensive.

    Why: access to a gated dataset is granted per user through a browser, so it
    cannot be fixed mid-run. Probing first turns a multi-GB download that ends
    in a 403 -- or a wait on a model that was never going to be used -- into an
    immediate, actionable message.
    """
    for LoaderClass, kwargs in _public_dataset_loaders(plan):
        _call_optional_hook(LoaderClass, "preflight_access", kwargs)


def _preflight_dataset_materialize(plan: BenchmarkPlan) -> None:
    """Fetch and cache public datasets that need resolution before services start.

    Why: ``DatasetManager`` performs dataset setup while ``TimingManager``
    blocks on the profiling handshake, so a large download there trips
    ``AIPERF_DATASET_CONFIGURATION_TIMEOUT`` (and raising that limit would
    weaken hang detection for every run, since ``PROFILE_CONFIGURE_TIMEOUT`` is
    constrained to be at least as large). Runs after the endpoint probe so an
    unreachable server fails in seconds rather than after the download.
    """
    import logging

    # Same reason as _preflight_endpoint_ready: rich logging isn't installed
    # yet, and a multi-GB fetch with no output looks like a hang. That check
    # returns early when endpoint waiting is disabled, so don't rely on it
    # having installed the handler.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    for LoaderClass, kwargs in _public_dataset_loaders(plan):
        _call_optional_hook(LoaderClass, "preflight_materialize", kwargs)
