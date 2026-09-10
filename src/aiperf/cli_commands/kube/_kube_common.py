# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers used across the `aiperf kube` subcommands.

These helpers do not depend on AIPerfJob CR shape; they resolve sweep child
names and worker counts, generate a DNS-safe benchmark name, and print the
memory estimate panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.deployment import DeploymentConfig
    from aiperf.config.kube import KubeOptions


def pin_default_direct_image(deployment: DeploymentConfig) -> None:
    """Replace the operator-less image defaults with immutable release values."""
    from aiperf import __version__
    from aiperf.config.deployment import DeploymentConfig
    from aiperf.kubernetes.enums import ImagePullPolicy

    default_image = DeploymentConfig.model_fields["image"].default
    if "image" in deployment.model_fields_set or deployment.image != default_image:
        return

    deployment.image = f"nvcr.io/nvidia/aiperf:{__version__}"
    if deployment.image_pull_policy is None:
        deployment.image_pull_policy = ImagePullPolicy.IF_NOT_PRESENT


def resolve_child_name(
    parent: str,
    variation: int | None = None,
    trial: int | None = None,
) -> str | None:
    """Resolve parent + (variation, trial) selectors to a child AIPerfJob name.

    Mirrors :func:`aiperf.sweep_controller._naming.build_child_name` exactly so
    CLI selectors line up with what the operator actually creates. The format
    is ``<parent>-v<idx:02d>[-t<trial:01d>]``.

    Args:
        parent: AIPerfSweep name (e.g. ``"my-sweep"``).
        variation: Variation index (0..199). ``None`` -> caller picks a
            fallback such as ``AIPerfSweep.status.currentChildRef.name``.
        trial: Trial index (0..9) when the sweep includes ``multi_run.trials``
            or convergence. ``None`` omits the ``-tN`` suffix.

    Returns:
        Child AIPerfJob name, or ``None`` when ``variation`` is ``None``.

    Raises:
        ValueError: If the selector cannot map to the operator's child-name
            cardinality budget.

    Examples:
        >>> resolve_child_name("my-sweep")
        None
        >>> resolve_child_name("my-sweep", variation=7)
        'my-sweep-v07'
        >>> resolve_child_name("my-sweep", variation=5, trial=0)
        'my-sweep-v05-t0'
    """
    if trial is not None and variation is None:
        raise ValueError(
            "Invalid sweep child selector: trial requires variation. "
            "Pass --variation with --trial."
        )
    if variation is None:
        return None
    if not 0 <= variation <= 199:
        raise ValueError(
            f"Invalid sweep child selector: variation {variation} is outside "
            "the supported range 0..199."
        )
    if trial is not None and not 0 <= trial <= 9:
        raise ValueError(
            f"Invalid sweep child selector: trial {trial} is outside the "
            "supported range 0..9."
        )
    suffix = f"-t{trial:01d}" if trial is not None else ""
    return f"{parent}-v{variation:02d}{suffix}"


def resolve_child_target(
    job_id: str | None,
    *,
    variation: int | None = None,
    trial: int | None = None,
    command: str,
) -> str | None:
    """Apply sweep child selectors to ``job_id``, refusing an implicit parent.

    ``--variation`` / ``--trial`` only mean something relative to a named
    AIPerfSweep. When ``job_id`` is ``None`` the caller falls back to the
    last-deployed benchmark, and silently dropping the selectors would target
    the parent sweep instead of the requested child - which for ``kube cancel``
    means cancelling every variation. Hard-fail instead of guessing.

    Args:
        job_id: Explicit AIPerfJob/AIPerfSweep name, or ``None`` to use the
            last deployed benchmark.
        variation: Variation index selector (0..199).
        trial: Trial index selector (0..9); requires ``variation``.
        command: Command path used in the error message (e.g. ``"kube cancel"``).

    Returns:
        The child name when a selector applies, otherwise ``job_id`` unchanged.

    Raises:
        ValueError: If a selector is passed without an explicit ``job_id``, or
            if the selector is out of range.
    """
    if job_id is None:
        if variation is None and trial is None:
            return None
        raise ValueError(
            "--variation/--trial require an explicit job_id: "
            f"use `aiperf {command} <sweep-name> -v N`. Without a name the "
            "last deployed benchmark is used, which would target the parent "
            "sweep instead of the child variation."
        )
    child = resolve_child_name(job_id, variation=variation, trial=trial)
    return child if child is not None else job_id


def generate_benchmark_name(config: Any, *, suffix: str = "") -> str:
    """Generate a short DNS-safe benchmark name from `config`.

    Used by both `aiperf kube profile` and `aiperf kube sweep`.

    Args:
        config: AIPerfConfig instance.
        suffix: Optional suffix appended after a hyphen (e.g. ``"sweep"``).

    Returns:
        A short hyphenated name like ``"qwen3-openai-throughput"`` or
        ``"qwen3-openai-throughput-sweep"`` when a suffix is provided.
    """
    import re

    model_name = config.benchmark.get_model_names()[0].split("/")[-1].lower()
    endpoint_type = str(config.benchmark.endpoint.type)
    first_phase = config.benchmark.phases[0]
    phase_type = str(first_phase.type)
    parts = [model_name, endpoint_type, phase_type]
    if suffix:
        parts.append(suffix)
    raw = "-".join(parts)
    return re.sub(r"[^a-z0-9-]", "-", raw)[:40].strip("-")


def resolve_total_workers(
    kube_options: KubeOptions,
    *,
    concurrency: int,
    connections_per_worker: int,
    configured_workers: int | None = None,
    workers_per_pod: int | None = None,
) -> int:
    """Resolve direct-mode worker count without materializing the CLI default.

    An explicit ``--total-workers`` owns the direct deployment fan-out. When
    omitted, ``benchmark.runtime.workers`` from YAML is the canonical total;
    only an absent total falls back to the concurrency-per-connection ratio.

    Both authored forms pass through untouched, so a total that cannot fill
    uniform pods still fails loudly in ``apply_worker_config``. The derived
    ratio is rounded up to a whole number of pods instead: nobody typed it, and
    failing on it would report a worker count the user never chose.

    ``connections_per_worker`` arrives straight off an unvalidated CR spec dict
    here, so the ratio goes through ``workers_for_concurrency`` rather than
    dividing inline.
    """
    if "total_workers" in kube_options.model_fields_set:
        return kube_options.total_workers
    if isinstance(configured_workers, int) and not isinstance(configured_workers, bool):
        return configured_workers

    from aiperf.common.environment import Environment
    from aiperf.kubernetes.spec_converter import (
        round_workers_to_pod_multiple,
        workers_for_concurrency,
    )

    return round_workers_to_pod_multiple(
        workers_for_concurrency(concurrency, connections_per_worker),
        workers_per_pod or Environment.WORKER.DEFAULT_WORKERS_PER_POD,
    )


def print_memory_estimate(
    config: Any,
    kube_options: KubeOptions,
    spec: dict,
    *,
    label_prefix: str = "",
) -> None:
    """Display the memory estimate on stderr for the planned benchmark.

    Keeping the estimate off stdout preserves machine-readable dry-run output.

    Args:
        config: Resolved `AIPerfConfig`.
        kube_options: Composite kube CLI options (workers count, etc.).
        spec: Submitted CRD spec dict; used to read ``connectionsPerWorker``.
        label_prefix: Optional stderr prefix printed before the estimate (e.g.
            ``"Sweep template: "``); empty by default.
    """
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.memory_estimator import estimate_memory, format_estimate

    phase_concurrency = max(
        (getattr(phase, "concurrency", 1) or 1 for phase in config.benchmark.phases),
        default=1,
    )
    total_workers = resolve_total_workers(
        kube_options,
        concurrency=phase_concurrency,
        connections_per_worker=spec.get("connectionsPerWorker", 100),
        configured_workers=config.benchmark.runtime.workers,
        workers_per_pod=config.benchmark.runtime.workers_per_pod,
    )
    try:
        mem_est = estimate_memory(
            config,
            total_workers=total_workers,
            workers_per_pod=config.benchmark.runtime.workers_per_pod,
            connections_per_worker=spec.get("connectionsPerWorker", 100),
        )
    except ValueError as exc:
        kube_console.stderr_console.print(
            f"Memory estimation skipped: {exc}", highlight=False
        )
        return
    rendered = format_estimate(mem_est)
    if label_prefix:
        kube_console.stderr_console.print(f"{label_prefix}", highlight=False)
    kube_console.stderr_console.print(rendered, highlight=False)
