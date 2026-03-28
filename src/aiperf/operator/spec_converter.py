# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert AIPerfJob CRD spec to AIPerfConfig and DeploymentConfig.

The CRD spec is nested: AIPerfConfig fields (models, endpoint, datasets, phases, ...)
live under ``spec.benchmark``, while DeploymentConfig fields (image, podTemplate,
scheduling, ...) live directly under ``spec``. This module reads from each location
and builds the appropriate models.
"""

from __future__ import annotations

import contextlib
import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.benchmark import BenchmarkRun

from aiperf.common.enums import AIPerfLogLevel, CommunicationType
from aiperf.common.environment import Environment
from aiperf.config import AIPerfConfig
from aiperf.config.deployment import DeploymentConfig
from aiperf.config.loader import expand_config_dict
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.plugin.enums import ServiceRunType, UIType

# Default connections per worker for auto-scaling calculation.
# Must match DeploymentConfig.connections_per_worker default.
DEFAULT_CONNECTIONS_PER_WORKER = 100

# AIPerfConfig field names — all keys that belong under spec.benchmark.
# Used for validation to detect unknown benchmark fields.
CONFIG_FIELDS: frozenset[str] = frozenset(AIPerfConfig.model_fields.keys())


@dataclass(slots=True)
class AIPerfJobSpecConverter:
    """Converts AIPerfJob CRD spec to AIPerfConfig and DeploymentConfig.

    The CRD spec is nested: AIPerfConfig fields live under ``spec.benchmark``
    and deployment/operator fields live directly under ``spec``.

    Example:
        >>> converter = AIPerfJobSpecConverter(spec, "my-job", "default")
        >>> config = converter.to_aiperf_config()
        >>> dc = converter.to_deployment_config()
    """

    spec: dict[str, Any]
    name: str
    namespace: str
    job_id: str | None = field(default=None)

    def __post_init__(self) -> None:
        """Set job_id to name if not explicitly provided."""
        if self.job_id is None:
            self.job_id = self.name

    def _get_config_dict(self) -> dict[str, Any]:
        """Extract AIPerfConfig fields from spec.benchmark."""
        benchmark = self.spec.get("benchmark") or {}
        return copy.deepcopy(benchmark)

    def to_aiperf_config(self) -> AIPerfConfig:
        """Convert AIPerfJob spec to AIPerfConfig.

        Reads AIPerfConfig fields from spec.benchmark, applies env var and
        Jinja2 expansion (mirroring the CLI file-load pipeline), then merges
        in Kubernetes runtime settings.

        Returns:
            AIPerfConfig populated from the AIPerfJob spec.
        """
        config_dict = self._get_config_dict()

        config_dict = expand_config_dict(config_dict)
        apply_k8s_runtime_config(config_dict, self.job_id or self.name, self.namespace)
        return AIPerfConfig.model_validate(config_dict)

    def to_deployment_config(self) -> DeploymentConfig:
        """Convert CRD spec to DeploymentConfig.

        Extracts deployment-related fields (image, imagePullPolicy, podTemplate,
        scheduling, etc.) from the top-level CRD spec using camelCase keys.

        Returns:
            DeploymentConfig with all deployment-related settings.
        """
        deployment_dict: dict[str, Any] = {}
        for key in (
            "image",
            "imagePullPolicy",
            "resourceMode",
            "connectionsPerWorker",
            "timeoutSeconds",
            "ttlSecondsAfterFinished",
            "resultsTtlDays",
            "keepFailedPods",
            "cancel",
            "podTemplate",
            "scheduling",
        ):
            if key in self.spec:
                deployment_dict[key] = self.spec[key]

        return DeploymentConfig.model_validate(deployment_dict)

    def calculate_workers(self, dc: DeploymentConfig | None = None) -> int:
        """Calculate optimal worker count based on concurrency.

        Uses an explicit runtime.workers override when provided. Otherwise,
        workers = ceil(concurrency / connections_per_worker).

        Args:
            dc: Optional DeploymentConfig to read connections_per_worker from.
                If None, reads connectionsPerWorker from the raw spec.

        Returns:
            Number of worker pods needed.
        """
        config_dict = self._get_config_dict()
        # Expand so Jinja2/env-var concurrency values resolve to integers.
        # Suppress errors: if expansion fails, _int() below falls back to 1.
        with contextlib.suppress(Exception):
            config_dict = expand_config_dict(config_dict)

        runtime = config_dict.get("runtime", {})
        phases = config_dict.get("phases", {})

        def _int(v: object, default: int = 1) -> int:
            try:
                return int(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        explicit_workers = _int(runtime.get("workers"), 0)
        if explicit_workers >= 1:
            return explicit_workers

        # Find max concurrency across all phases.
        # phases can be a single config (has "type") or named phases (dict of dicts).
        if "type" in phases:
            concurrency = _int(phases.get("concurrency", 1))
        else:
            concurrency = max(
                (
                    _int(phase.get("concurrency", 1))
                    for phase in phases.values()
                    if isinstance(phase, dict)
                ),
                default=1,
            )

        if dc is not None:
            connections_per_worker = dc.connections_per_worker
        else:
            connections_per_worker = self.spec.get(
                "connectionsPerWorker", DEFAULT_CONNECTIONS_PER_WORKER
            )

        return max(1, math.ceil(concurrency / connections_per_worker))


def build_benchmark_run(
    run_config: dict[str, Any],
    run_id: str,
    namespace: str,
) -> BenchmarkRun:
    """Build a BenchmarkRun from a config dict for a single K8s run.

    Args:
        run_config: AIPerfConfig dict (already has k8s runtime config applied).
        run_id: DNS-safe run identifier (used as benchmark_id and for DNS).
        namespace: Kubernetes namespace (for DNS name generation).

    Returns:
        A BenchmarkRun ready for serialization into a ConfigMap.
    """
    from pathlib import Path

    from aiperf.config.benchmark import BenchmarkRun
    from aiperf.config.config import BenchmarkConfig

    run_config.pop("multi_run", None)
    run_config.pop("sweep", None)
    apply_k8s_runtime_config(run_config, run_id, namespace)
    cfg = BenchmarkConfig.model_validate(run_config)

    return BenchmarkRun(
        benchmark_id=run_id,
        cfg=cfg,
        trial=0,
        artifact_dir=Path(run_config.get("artifacts", {}).get("dir", "/results")),
        label="",
        variation=None,
    )


def apply_worker_config(config: AIPerfConfig, total_workers: int) -> int:
    """Apply worker scaling to the config.

    Calculates the number of pods and workers per pod, then sets
    workers_per_pod, total workers, and record processors on the config.

    Args:
        config: AIPerfConfig to modify in-place.
        total_workers: Total workers from calculate_workers().

    Returns:
        Number of worker pods needed.
    """
    default_workers_per_pod = (
        config.runtime.workers_per_pod or Environment.WORKER.DEFAULT_WORKERS_PER_POD
    )

    if total_workers <= default_workers_per_pod:
        workers_per_pod = total_workers
        num_pods = 1
    else:
        workers_per_pod = default_workers_per_pod
        num_pods = math.ceil(total_workers / workers_per_pod)

    config.runtime.workers_per_pod = workers_per_pod
    config.runtime.workers = num_pods * workers_per_pod

    # Respect user-provided record_processors_per_pod when set in the spec.
    if config.runtime.record_processors_per_pod is None:
        rp_per_pod = max(
            1, workers_per_pod // K8sEnvironment.RECORD_PROCESSOR_SCALE_FACTOR
        )
        config.runtime.record_processors_per_pod = rp_per_pod
    rp_per_pod = config.runtime.record_processors_per_pod
    config.runtime.record_processors = rp_per_pod * num_pods

    return num_pods


def apply_k8s_runtime_config(
    config_dict: dict[str, Any], job_id: str, namespace: str
) -> None:
    """Apply Kubernetes runtime settings to a config dict in-place.

    Sets up dual-bind ZMQ, API service, dataset URL, and K8s service run type.

    Args:
        config_dict: AIPerfConfig dict to modify in-place.
        job_id: Job identifier for DNS name generation.
        namespace: Kubernetes namespace for DNS resolution.
    """
    config_dict.setdefault("artifacts", {})
    config_dict["artifacts"]["dir"] = "/results"

    api_port = K8sEnvironment.PORTS.API_SERVICE
    jobset_name = f"aiperf-{job_id}"
    controller_dns = (
        f"{jobset_name}-controller-0-0.{jobset_name}.{namespace}.svc.cluster.local"
    )
    dataset_api_base_url = f"http://{controller_dns}:{api_port}/api/dataset"

    config_dict.setdefault("runtime", {})
    config_dict["runtime"].update(
        {
            "service_run_type": ServiceRunType.KUBERNETES,
            "ui": UIType.SIMPLE,
            "api_port": api_port,
            "api_host": "0.0.0.0",
            "dataset_api_base_url": dataset_api_base_url,
            "communication": {
                "type": CommunicationType.DUAL,
                "ipc_path": K8sEnvironment.ZMQ.IPC_PATH,
                "tcp_host": "0.0.0.0",
            },
        }
    )

    config_dict.setdefault("logging", {})
    config_dict["logging"].setdefault("level", AIPerfLogLevel.INFO)


def extract_benchmark_config(spec: dict[str, Any]) -> AIPerfConfig:
    """Extract an AIPerfConfig from an AIPerfJob CRD spec dict.

    Reads AIPerfConfig fields from ``spec.benchmark``, leaving deployment
    fields (image, podTemplate, etc.) at the top level. Does NOT apply
    Kubernetes runtime config (ZMQ, API service URLs), so the result is
    suitable for name generation and CLI validation without polluting it
    with placeholder host names.

    Args:
        spec: AIPerfJob spec dict (from CR's ``spec`` key).

    Returns:
        Validated AIPerfConfig populated from spec.benchmark.
    """
    benchmark = spec.get("benchmark") or {}
    config_dict = copy.deepcopy(benchmark)
    config_dict = expand_config_dict(config_dict)
    return AIPerfConfig.model_validate(config_dict)
