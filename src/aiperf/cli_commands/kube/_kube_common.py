# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers used by `aiperf kube profile` and `aiperf kube sweep`.

These helpers do not depend on AIPerfJob CR shape; they generate a DNS-safe
benchmark name and print the memory estimate panel. The
``UserConfig + ServiceConfig + config_file -> AIPerfConfig`` resolution lives
in ``aiperf.config.v1._resolver`` so non-kube commands (``aiperf profile``)
can share the same YAML+CLI merge semantics; we re-export it here for
backwards compatibility with existing kube-side imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Re-exported for back-compat with kube-side callers that imported from here.
from aiperf.config.v1._resolver import (
    build_v1_overrides as _build_v1_overrides,  # noqa: F401
)
from aiperf.config.v1._resolver import (
    deep_merge as _deep_merge,  # noqa: F401
)
from aiperf.config.v1._resolver import (
    resolve_config,  # noqa: F401
)

if TYPE_CHECKING:
    from aiperf.config.kube import KubeOptions


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
    return re.sub(r"[^a-z0-9-]", "-", raw).strip("-")[:40]


def print_memory_estimate(
    config: Any,
    kube_options: KubeOptions,
    spec: dict,
    *,
    label_prefix: str = "",
) -> None:
    """Compute and display the memory estimate panel for the planned benchmark.

    Args:
        config: Resolved `AIPerfConfig`.
        kube_options: Composite kube CLI options (workers count, etc.).
        spec: Submitted CRD spec dict; used to read ``connectionsPerWorker``.
        label_prefix: Optional prefix printed before the estimate (e.g.
            ``"Sweep template: "``); empty by default.
    """
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.memory_estimator import estimate_memory, format_estimate

    mem_est = estimate_memory(
        config,
        total_workers=kube_options.workers,
        workers_per_pod=config.benchmark.runtime.workers_per_pod,
        connections_per_worker=spec.get("connectionsPerWorker", 100),
    )
    rendered = format_estimate(mem_est)
    if label_prefix:
        kube_console.console.print(f"{label_prefix}", highlight=False)
    kube_console.console.print(rendered, highlight=False)
