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

from pydantic.alias_generators import to_camel

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.kubernetes.crd_models import AIPerfJobSpec, AIPerfSweepSpec

from aiperf.common.enums import AIPerfLogLevel, CommunicationType
from aiperf.common.environment import Environment
from aiperf.config import AIPerfConfig
from aiperf.config.config import BenchmarkConfig
from aiperf.config.deployment import DeploymentConfig
from aiperf.config.loader import expand_config_dict, load_config_from_mapping
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.plugin.enums import ServiceRunType, UIType

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


def build_config_envelope(spec: dict[str, Any]) -> dict[str, Any]:
    """Project the Config-v2 envelope fields from a raw Kubernetes spec.

    Projection follows ``AIPerfConfig.model_fields`` so new envelope fields
    automatically cross the Kubernetes adapter instead of requiring another
    hand-maintained allowlist.

    Args:
        spec: AIPerfJob or AIPerfSweep ``spec`` mapping.

    Returns:
        Deep-copied Config-v2 envelope fields, normalized to field names.
    """
    envelope: dict[str, Any] = {}
    for name, model_field in AIPerfConfig.model_fields.items():
        for key in (name, model_field.alias):
            if key is not None and key in spec:
                envelope[name] = copy.deepcopy(spec[key])
                break

    benchmark = envelope.get("benchmark")
    if isinstance(benchmark, dict):
        for key in ("variables", "random_seed", "randomSeed"):
            if key not in benchmark:
                continue
            envelope_name = "random_seed" if key == "randomSeed" else key
            value = benchmark.pop(key)
            if value is not None:
                envelope[envelope_name] = value
    return envelope


def restore_jinja_templates(rendered: Any, raw: Any) -> Any:
    """Restore authored Jinja leaves onto a canonical rendered config value.

    ``AIPerfConfig.model_dump`` supplies origin/main's canonical aliases and
    intent-preserving field set. Its retained ``_raw_envelope`` supplies the
    pre-Jinja leaves that a Kubernetes sweep must render again per variation.
    Overlay only template strings so static values continue to come from the
    validated model rather than bypassing normalization or secret redaction.

    Args:
        rendered: Canonical model dump to update in place.
        raw: Post-environment, pre-Jinja config value.

    Returns:
        The canonical value with authored Jinja template leaves restored.
    """
    if isinstance(raw, str) and "{{" in raw and "}}" in raw:
        return raw
    if isinstance(rendered, dict) and isinstance(raw, dict):
        for raw_key, raw_value in raw.items():
            key = raw_key if raw_key in rendered else to_camel(raw_key)
            if key in rendered:
                rendered[key] = restore_jinja_templates(rendered[key], raw_value)
            elif isinstance(raw_value, str) and "{{" in raw_value and "}}" in raw_value:
                rendered[key] = raw_value
        return rendered
    if isinstance(rendered, list) and isinstance(raw, list):
        for index, raw_value in enumerate(raw[: len(rendered)]):
            rendered[index] = restore_jinja_templates(rendered[index], raw_value)
    return rendered


def prepare_workload_spec(
    spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render Config-v2 fields for CR validation and retain sweep templates.

    The canonical mapping loader performs the same environment substitution,
    Jinja rendering, normalization, and validation used by local config files.
    Kubernetes-only fields are then merged back for validation by the workload
    CR model. The retained pre-Jinja envelope lets the canonical plan builder
    render templates again after applying each sweep variation.

    Args:
        spec: Raw AIPerfJob or AIPerfSweep ``spec`` mapping.

    Returns:
        A rendered workload spec and its post-environment, pre-Jinja envelope.
    """
    config = load_config_from_mapping(build_config_envelope(spec))
    envelope_keys = set(AIPerfConfig.model_fields)
    envelope_keys.update(
        model_field.alias
        for model_field in AIPerfConfig.model_fields.values()
        if model_field.alias is not None
    )
    deployment_fields = {
        key: copy.deepcopy(value)
        for key, value in spec.items()
        if key not in envelope_keys
    }
    rendered_envelope = config.model_dump(
        mode="python",
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
        context={"include_secrets": True},
    )
    raw_envelope = config._raw_envelope
    assert raw_envelope is not None
    return {**deployment_fields, **rendered_envelope}, copy.deepcopy(raw_envelope)


def validate_sweep_spec(spec: dict[str, Any]) -> AIPerfSweepSpec:
    """Validate an AIPerfSweep spec while retaining its sweep templates."""
    from aiperf.kubernetes.crd_models import AIPerfSweepSpec

    prepared_spec, raw_envelope = prepare_workload_spec(spec)
    validated = AIPerfSweepSpec.model_validate(prepared_spec)
    validated._raw_envelope = raw_envelope
    return validated


def validate_job_spec(spec: dict[str, Any]) -> AIPerfJobSpec:
    """Validate an AIPerfJob spec after rendering its Config-v2 envelope."""
    from aiperf.kubernetes.crd_models import AIPerfJobSpec

    prepared_spec, raw_envelope = prepare_workload_spec(spec)
    validated = AIPerfJobSpec.model_validate(prepared_spec)
    validated._raw_envelope = raw_envelope
    return validated


@dataclass(frozen=True, slots=True)
class KeyExportNames:
    """Authoritative summary-export filenames for one benchmark run."""

    json_name: str
    """Summary JSON filename."""

    csv_name: str
    """Summary CSV filename."""

    jsonl_name: str
    """Per-record JSONL filename."""

    server_metrics_json_name: str
    """Server-metrics JSON filename."""

    @property
    def names(self) -> frozenset[str]:
        """Return both authoritative filenames."""
        return frozenset({self.json_name, self.csv_name})


DEFAULT_KEY_EXPORT_NAMES = KeyExportNames(
    json_name="profile_export_aiperf.json",
    csv_name="profile_export_aiperf.csv",
    jsonl_name="profile_export.jsonl",
    server_metrics_json_name="server_metrics_export.json",
)


def extract_deployment_config(spec: dict[str, Any]) -> DeploymentConfig:
    """Project every deployment field from a workload spec.

    The projection follows :class:`DeploymentConfig` rather than a manual
    allowlist so new typed deployment fields automatically cross the operator,
    direct-profile, and raw-manifest adapters together.

    Args:
        spec: AIPerfJob or AIPerfSweep ``spec`` mapping.

    Returns:
        Validated Kubernetes deployment configuration.
    """
    deployment_dict: dict[str, Any] = {}
    for name, model_field in DeploymentConfig.model_fields.items():
        for key in (model_field.alias, name):
            if key is not None and key in spec:
                deployment_dict[key] = copy.deepcopy(spec[key])
                break

    if K8sEnvironment.SHARE_PROCESS_NAMESPACE:
        pod_key = "podTemplate" if "podTemplate" in deployment_dict else "pod_template"
        pod_template = deployment_dict.setdefault(pod_key, {})
        share_key = (
            "shareProcessNamespace"
            if pod_key == "podTemplate"
            else "share_process_namespace"
        )
        pod_template.setdefault(share_key, True)

    return DeploymentConfig.model_validate(deployment_dict)


def resolve_artifacts_prefix(spec: dict[str, Any] | None) -> str | None:
    """Resolve ``spec.benchmark.artifacts.prefix`` through the config loader."""
    if not isinstance(spec, dict):
        return None
    benchmark = spec.get("benchmark")
    if not isinstance(benchmark, dict):
        return None
    artifacts = benchmark.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    prefix = artifacts.get("prefix")
    if not isinstance(prefix, str) or not prefix:
        return None
    if "{{" not in prefix and "${" not in prefix:
        return prefix

    try:
        expanded = expand_config_dict(build_config_envelope(spec))
    except Exception:  # noqa: BLE001 - unresolved templates retain default export names
        logger.debug(
            "Could not resolve artifacts.prefix %r; using default export names",
            prefix,
        )
        return None
    resolved = (expanded.get("benchmark") or {}).get("artifacts", {}).get("prefix")
    return resolved if isinstance(resolved, str) and resolved else None


def key_export_names(spec: dict[str, Any] | None) -> KeyExportNames:
    """Derive the summary-export filenames selected by ``artifacts.prefix``."""
    prefix = resolve_artifacts_prefix(spec)
    if prefix is None:
        return DEFAULT_KEY_EXPORT_NAMES

    from aiperf.config.artifacts import ArtifactsConfig

    try:
        artifacts = ArtifactsConfig(prefix=prefix)
    except Exception:  # noqa: BLE001 - invalid prefixes cannot rename real exports
        logger.warning(
            "Ignoring unusable artifacts.prefix %r; using %s",
            prefix,
            DEFAULT_KEY_EXPORT_NAMES.json_name,
        )
        return DEFAULT_KEY_EXPORT_NAMES
    return KeyExportNames(
        json_name=artifacts.profile_export_json_file.name,
        csv_name=artifacts.profile_export_csv_file.name,
        jsonl_name=artifacts.profile_export_jsonl_file.name,
        server_metrics_json_name=artifacts.server_metrics_export_json_file.name,
    )


def key_export_names_from_body(body: dict[str, Any] | None) -> KeyExportNames:
    """Derive summary-export filenames from a full Kubernetes CR body."""
    return key_export_names((body or {}).get("spec"))


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

    def _expanded_envelope(self) -> dict[str, Any]:
        """Return the complete config envelope after env and Jinja expansion."""
        return expand_config_dict(build_config_envelope(self.spec))

    def to_aiperf_config(self) -> AIPerfConfig:
        """Convert AIPerfJob spec to AIPerfConfig.

        Reads AIPerfConfig fields from spec.benchmark, applies env var and
        Jinja2 expansion (mirroring the CLI file-load pipeline), then merges
        in Kubernetes runtime settings.

        Returns:
            AIPerfConfig populated from the AIPerfJob spec.
        """
        envelope = self._expanded_envelope()
        config_dict = envelope.setdefault("benchmark", {})
        apply_k8s_runtime_config(
            config_dict, self.job_id or self.name, self.namespace, use_aliases=True
        )
        return AIPerfConfig.model_validate(envelope)

    def to_deployment_config(self) -> DeploymentConfig:
        """Convert CRD spec to DeploymentConfig.

        Extracts deployment-related fields (image, imagePullPolicy, podTemplate,
        scheduling, etc.) from the top-level CRD spec using camelCase keys.

        Returns:
            DeploymentConfig with all deployment-related settings.
        """
        return extract_deployment_config(self.spec)

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
            config_dict = self._expanded_envelope().get("benchmark", config_dict)

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

    sweep = run_config.get("sweep")
    multi_run = run_config.get("multi_run")
    if sweep is not None:
        raise ValueError(
            "A single Kubernetes benchmark run cannot include `sweep`; "
            "submit an AIPerfSweep instead."
        )
    if isinstance(multi_run, dict):
        num_runs = multi_run.get(
            "num_runs",
            multi_run.get("numRuns", multi_run.get("trials", 1)),
        )
        if int(num_runs) > 1 or multi_run.get("convergence") is not None:
            raise ValueError(
                "A single Kubernetes benchmark run cannot include multi-run "
                "or convergence orchestration; submit an AIPerfSweep instead."
            )
    elif multi_run is not None:
        raise ValueError(
            "A single Kubernetes benchmark run received an invalid multi-run "
            "configuration; submit an AIPerfSweep instead."
        )

    # Envelope shape: body lives under run_config["benchmark"]; envelope-only
    # keys (sweep/multi_run/variables/random_seed) and `benchmark` itself are
    # not valid BenchmarkConfig inputs.
    body_source = run_config.get("benchmark", run_config)
    body_dict = copy.deepcopy(body_source)
    if body_source is run_config:
        body_dict.pop("sweep", None)
        body_dict.pop("multi_run", None)
    apply_k8s_runtime_config(body_dict, run_id, namespace)
    cfg = BenchmarkConfig.model_validate(body_dict)

    return BenchmarkRun(
        benchmark_id=run_id,
        cfg=cfg,
        trial=0,
        artifact_dir=Path(body_dict.get("artifacts", {}).get("dir", "/results")),
        label="",
        cli_command=None,
        variation=None,
        random_seed=run_config.get("random_seed"),
        variables=dict(run_config.get("variables") or {}),
        plot=run_config.get("plot"),
    )


def apply_worker_config(config: AIPerfConfig, total_workers: int) -> int:
    """Apply worker scaling to the config.

    Calculates the exact uniform worker-pod topology, then sets workers per
    pod, total workers, and record processors on the config. A JobSet
    replicated job cannot express a partial final pod, so a non-divisible
    total uses one worker pod to preserve the requested total exactly.

    Args:
        config: AIPerfConfig to modify in-place.
        total_workers: Total workers from calculate_workers().

    Returns:
        Number of worker pods needed.

    Raises:
        ValueError: An authored total record-processor count cannot be evenly
            represented by identical Kubernetes worker pods.
    """
    runtime = config.benchmark.runtime
    configured_record_processors = runtime.record_processors
    configured_record_processors_per_pod = runtime.record_processors_per_pod
    default_workers_per_pod = (
        runtime.workers_per_pod or Environment.WORKER.DEFAULT_WORKERS_PER_POD
    )

    if total_workers % default_workers_per_pod:
        workers_per_pod = total_workers
        num_pods = 1
    else:
        workers_per_pod = default_workers_per_pod
        num_pods = total_workers // workers_per_pod

    runtime.workers_per_pod = workers_per_pod
    runtime.workers = num_pods * workers_per_pod

    if configured_record_processors_per_pod is not None:
        rp_per_pod = configured_record_processors_per_pod
        if (
            configured_record_processors is not None
            and configured_record_processors != rp_per_pod * num_pods
        ):
            raise ValueError(
                "runtime.record_processors must equal "
                "runtime.record_processors_per_pod multiplied by the Kubernetes "
                f"worker pod count ({rp_per_pod} * {num_pods})"
            )
    elif configured_record_processors is not None:
        if configured_record_processors % num_pods:
            raise ValueError(
                "runtime.record_processors must be divisible by the Kubernetes "
                f"worker pod count ({num_pods}); set recordProcessorsPerPod "
                "explicitly or choose a divisible total"
            )
        rp_per_pod = configured_record_processors // num_pods
        runtime.record_processors_per_pod = rp_per_pod
    else:
        rp_per_pod = max(
            1, workers_per_pod // K8sEnvironment.RECORD_PROCESSOR_SCALE_FACTOR
        )
        runtime.record_processors_per_pod = rp_per_pod
    runtime.record_processors = rp_per_pod * num_pods

    return num_pods


def apply_k8s_runtime_config(
    config_dict: dict[str, Any],
    job_id: str,
    namespace: str,
    *,
    use_aliases: bool = False,
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

    if use_aliases:
        runtime_config: dict[str, Any] = {
            "serviceRunType": ServiceRunType.KUBERNETES,
            "ui": UIType.SIMPLE,
            "apiPort": api_port,
            "apiHost": "0.0.0.0",
            "datasetApiBaseUrl": dataset_api_base_url,
            "communication": {
                "type": CommunicationType.DUAL,
                "ipcPath": K8sEnvironment.ZMQ.IPC_PATH,
                "tcpHost": "0.0.0.0",
            },
        }
    else:
        runtime_config = {
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

    config_dict.setdefault("runtime", {})
    config_dict["runtime"].update(runtime_config)

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
    expanded = expand_config_dict(build_config_envelope(spec))
    return AIPerfConfig.model_validate(expanded)
