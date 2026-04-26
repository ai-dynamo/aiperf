# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers used by `aiperf kube profile` and `aiperf kube sweep`.

These helpers do not depend on AIPerfJob CR shape; they are concerned with
turning a `UserConfig` + `ServiceConfig` / config-file pair into an
`AIPerfConfig`, generating a DNS-safe benchmark name, and printing the memory
estimate panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.config import AIPerfConfig
    from aiperf.config.kube import KubeOptions
    from aiperf.config.v1 import ServiceConfig, UserConfig


def resolve_config(
    user_config: UserConfig,
    service_config: ServiceConfig,
    config_file: Path | None,
) -> AIPerfConfig:
    """Return an `AIPerfConfig` from a plain YAML config file or CLI flags.

    Args:
        user_config: Parsed v1 ``UserConfig`` carrying flag-form benchmark options.
        service_config: Parsed v1 ``ServiceConfig`` carrying service-level
            options (UI, log level, ZMQ, etc.).
        config_file: Optional path to a YAML config file; when provided, takes
            precedence over CLI flags. The YAML config is always interpreted as
            the v2 ``AIPerfConfig`` shape (no v1 round-trip).

    Returns:
        Fully resolved `AIPerfConfig` ready for downstream use.
    """
    if config_file is not None:
        from aiperf.config.loader import load_config

        return load_config(config_file)
    from aiperf.config.v1.converter import convert_user_to_aiperf

    return convert_user_to_aiperf(user_config, service_config)


def generate_benchmark_name(config: AIPerfConfig, *, suffix: str = "") -> str:
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

    model_name = config.get_model_names()[0].split("/")[-1].lower()
    endpoint_type = str(config.endpoint.type)
    first_phase = config.phases[0]
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
        workers_per_pod=config.runtime.workers_per_pod,
        connections_per_worker=spec.get("connectionsPerWorker", 100),
    )
    rendered = format_estimate(mem_est)
    if label_prefix:
        kube_console.console.print(f"{label_prefix}", highlight=False)
    kube_console.console.print(rendered, highlight=False)
