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
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

from aiperf.common.enums import AIPerfLogLevel
from aiperf.common.environment import Environment
from aiperf.config import AIPerfConfig
from aiperf.config.config import BenchmarkConfig
from aiperf.config.deployment import DeploymentConfig
from aiperf.config.loader import expand_config_dict
from aiperf.kubernetes.environment import K8sEnvironment

logger = logging.getLogger(__name__)

# Default connections per worker for auto-scaling calculation.
# Must match DeploymentConfig.connections_per_worker default.
DEFAULT_CONNECTIONS_PER_WORKER = 100

# BenchmarkConfig field names — all keys that belong under spec.benchmark.
# Used for validation to detect unknown benchmark fields. Includes camelCase
# aliases (via BaseConfig's alias_generator) and shorthand aliases (model,
# dataset, warmup, profiling) that normalize to canonical forms at parse time.
CONFIG_FIELDS: frozenset[str] = (
    frozenset(BenchmarkConfig.model_fields.keys())
    | frozenset(
        f.alias for f in BenchmarkConfig.model_fields.values() if f.alias is not None
    )
    | {"model", "dataset", "warmup", "profiling"}
)


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
    """Raw AIPerfJob CRD spec dictionary."""

    name: str
    """Name of the AIPerfJob resource."""

    namespace: str
    """Kubernetes namespace for the job."""

    job_id: str | None = field(default=None)
    """Optional job identifier; defaults to name if not provided."""

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
        apply_k8s_runtime_config(
            config_dict, self.job_id or self.name, self.namespace, use_aliases=True
        )
        # AIPerfJob.spec.benchmark may carry envelope-level keys (variables,
        # random_seed) for Jinja templating convenience — lift them out of the
        # body before validating, since BenchmarkConfig forbids them.
        envelope: dict[str, Any] = {}
        for key in ("variables", "random_seed", "randomSeed"):
            if key in config_dict:
                envelope_key = "random_seed" if key == "randomSeed" else key
                envelope[envelope_key] = config_dict.pop(key)
        return AIPerfConfig.model_validate({"benchmark": config_dict, **envelope})

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

        # Seed shareProcessNamespace from AIPERF_K8S_SHARE_PROCESS_NAMESPACE
        # when the CR does not set it explicitly. Chaos fixtures flip the env
        # var on to unlock cross-container kubectl exec kills.
        if K8sEnvironment.SHARE_PROCESS_NAMESPACE:
            pod_template = deployment_dict.setdefault("podTemplate", {})
            pod_template.setdefault("shareProcessNamespace", True)

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
        phases = config_dict.get("phases", [])

        def _int(v: object, default: int = 1) -> int:
            try:
                return int(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        explicit_workers = _int(runtime.get("workers"), 0)
        if explicit_workers >= 1:
            return explicit_workers

        # Find max concurrency across all phases.
        # phases is a list of named phase configs (each a dict with "name" and "type").
        if isinstance(phases, dict) and "type" in phases:
            # legacy single-config dict shorthand still understood by normalizer
            concurrency = _int(phases.get("concurrency", 1))
        else:
            phase_iter = phases if isinstance(phases, list) else []
            concurrency = max(
                (
                    _int(phase.get("concurrency", 1))
                    for phase in phase_iter
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
        run_config: AIPerfConfig envelope dict (already has k8s runtime config applied).
        run_id: DNS-safe run identifier (used as benchmark_id and for DNS).
        namespace: Kubernetes namespace (for DNS name generation).

    Returns:
        A BenchmarkRun ready for serialization into a ConfigMap.
    """
    from pathlib import Path

    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.resolution.plan import BenchmarkRun

    for unsupported_key in ("sweep", "multi_run"):
        if unsupported_key in run_config:
            logger.warning(
                "AIPerfJob includes '%s' config; Kubernetes does not currently "
                "orchestrate parameter sweeps. Running base config as a single "
                "benchmark. To sweep, use the local `aiperf profile` CLI.",
                unsupported_key,
            )
            run_config.pop(unsupported_key, None)

    # Envelope shape: body lives under run_config["benchmark"]; envelope-only
    # keys (sweep/multi_run/variables/random_seed) and `benchmark` itself are
    # not valid BenchmarkConfig inputs.
    body_dict = run_config.get("benchmark", run_config)
    apply_k8s_runtime_config(body_dict, run_id, namespace)
    cfg = BenchmarkConfig.model_validate(body_dict)

    return BenchmarkRun(
        benchmark_id=run_id,
        cfg=cfg,
        trial=0,
        artifact_dir=Path(body_dict.get("artifacts", {}).get("dir", "/results")),
        label="",
        variation=None,
        random_seed=run_config.get("random_seed"),
        variables=dict(run_config.get("variables") or {}),
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
    runtime = config.benchmark.runtime
    default_workers_per_pod = (
        runtime.workers_per_pod or Environment.WORKER.DEFAULT_WORKERS_PER_POD
    )

    if total_workers <= default_workers_per_pod:
        workers_per_pod = total_workers
        num_pods = 1
    else:
        workers_per_pod = default_workers_per_pod
        num_pods = math.ceil(total_workers / workers_per_pod)

    runtime.workers_per_pod = workers_per_pod
    runtime.workers = num_pods * workers_per_pod

    # Respect user-provided record_processors_per_pod when set in the spec.
    if runtime.record_processors_per_pod is None:
        rp_per_pod = max(
            1, workers_per_pod // K8sEnvironment.RECORD_PROCESSOR_SCALE_FACTOR
        )
        runtime.record_processors_per_pod = rp_per_pod
    rp_per_pod = runtime.record_processors_per_pod
    runtime.record_processors = rp_per_pod * num_pods

    return num_pods


def apply_k8s_runtime_config(
    config_dict: dict[str, Any],
    job_id: str,
    namespace: str,
    *,
    use_aliases: bool = False,
) -> None:
    """Apply Kubernetes runtime settings for the native Rust-runner execution path.

    Unlike the retired Python service mesh, the native ``aiperf`` path has
    no ZMQ dual-bind, no ``service_run_type``, no per-service API host/port, and no
    HTTP dataset service: one runner (or, for cellular runs, a controller plus its
    cell pods) owns the whole run and resolves its dataset in-process.
    ``rust_wire.dump_benchmark_run`` keeps only ``{workers, workers_max,
    workers_min, cells}`` from ``runtime`` when it lowers the config to the
    protocol-v2 envelope, so injecting the mesh runtime here is at best dropped and
    at worst fatal (e.g. ``ServiceRunType.KUBERNETES`` does not exist on the native
    branch). This therefore only pins the artifact directory the results volume is
    mounted at and a default log level; the cell count and worker count flow through
    from ``spec.benchmark.runtime`` untouched.

    Args:
        config_dict: AIPerfConfig dict to modify in-place.
        job_id: Job identifier (unused on the native path; kept for signature
            parity with the mesh callers and future controller-DNS needs).
        namespace: Kubernetes namespace (unused on the native path; see ``job_id``).
        use_aliases: Whether ``config_dict`` uses camelCase aliases (kept for
            signature parity; the native runtime keys written here are identical in
            either form).
    """
    del job_id, namespace, use_aliases  # native path needs no controller DNS wiring

    config_dict.setdefault("artifacts", {})
    config_dict["artifacts"]["dir"] = "/results"

    config_dict.setdefault("logging", {})
    config_dict["logging"].setdefault("level", AIPerfLogLevel.INFO)


def extract_benchmark_config(spec: dict[str, Any]) -> AIPerfConfig:
    """Extract an AIPerfConfig from an AIPerfJob/AIPerfSweep CRD spec dict.

    Reads AIPerfConfig body fields from ``spec.benchmark`` and envelope
    fields (``variables``, ``random_seed``, ``schemaVersion``/``schema_version``,
    ``multi_run``, ``sweep``) from the top level of ``spec``, then runs the
    Jinja2/env-var expansion pipeline against the assembled envelope so
    ``{{ ... }}`` expressions inside ``spec.benchmark`` can resolve against
    ``spec.variables``. Deployment fields (image, podTemplate, etc.) stay
    at the spec top level and are NOT carried into the returned config.

    Does NOT apply Kubernetes runtime config (ZMQ, API service URLs), so
    the result is suitable for name generation and CLI validation without
    polluting it with placeholder host names.

    Args:
        spec: AIPerfJob spec dict (from CR's ``spec`` key).

    Returns:
        Validated AIPerfConfig populated from spec.benchmark plus any
        envelope-level fields (variables, random_seed, multi_run, sweep,
        schema_version) discovered on the spec.
    """
    benchmark = spec.get("benchmark") or {}
    envelope: dict[str, Any] = {"benchmark": copy.deepcopy(benchmark)}
    # Envelope fields recognised by AIPerfConfig. Both camelCase (CRD wire
    # form) and snake_case (local YAML form) aliases are accepted because
    # AIPerfConfig's BaseConfig alias_generator handles either.
    for key in (
        "variables",
        "random_seed",
        "randomSeed",
        "schema_version",
        "schemaVersion",
        "multi_run",
        "multiRun",
        "sweep",
    ):
        if key in spec and spec[key] is not None:
            envelope[key] = copy.deepcopy(spec[key])
    expanded = expand_config_dict(envelope)
    return AIPerfConfig.model_validate(expanded)
