# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers used by `aiperf kube profile` and `aiperf kube sweep`.

These helpers do not depend on AIPerfJob CR shape; they generate a DNS-safe
benchmark name and print the memory estimate panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.kube import KubeOptions


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
) -> int:
    """Resolve direct-mode worker count without materializing the CLI default.

    An explicit ``--total-workers`` owns the direct deployment fan-out. When
    omitted, ``benchmark.runtime.workers`` from YAML is the canonical total;
    only an absent total falls back to the concurrency-per-connection ratio.
    """
    if "total_workers" in kube_options.model_fields_set:
        return kube_options.total_workers
    if isinstance(configured_workers, int) and not isinstance(configured_workers, bool):
        return configured_workers

    import math

    return max(1, math.ceil(concurrency / connections_per_worker))


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
    )
    mem_est = estimate_memory(
        config,
        total_workers=total_workers,
        workers_per_pod=config.benchmark.runtime.workers_per_pod,
        connections_per_worker=spec.get("connectionsPerWorker", 100),
    )
    rendered = format_estimate(mem_est)
    if label_prefix:
        kube_console.stderr_console.print(f"{label_prefix}", highlight=False)
    kube_console.stderr_console.print(rendered, highlight=False)
