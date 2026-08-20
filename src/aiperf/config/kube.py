# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes deployment configuration."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Any

import orjson

if TYPE_CHECKING:
    from aiperf.config import AIPerfConfig
    from aiperf.config.deployment import (
        DeploymentConfig,
        PodTemplateConfig,
        SchedulingConfig,
    )

from cyclopts import Group
from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator

from aiperf.config.cli_parameter import CLIParameter
from aiperf.kubernetes.enums import ImagePullPolicy


class _KubeGroups:
    """Groups for Kubernetes CLI options."""

    KUBERNETES = Group.create_ordered("Kubernetes")
    K8S_NODE_PLACEMENT = Group.create_ordered("Kubernetes Node Placement")
    K8S_SCHEDULING = Group.create_ordered("Kubernetes Scheduling")
    K8S_SECRETS = Group.create_ordered("Kubernetes Secrets")
    K8S_METADATA = Group.create_ordered("Kubernetes Metadata")


def _parse_json_cli_value(raw: str, field_name: str, expected: str) -> Any:
    """Parse a JSON CLI token and include the field name in failures."""
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid {expected}") from exc


def _coerce_node_selector_entries(value: Any) -> list[str | dict[str, str]]:
    """Coerce CLI/Python node-selector input into mergeable entries."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [_string_map(value, "node_selector")]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{"):
            parsed = _parse_json_cli_value(
                stripped,
                "node_selector",
                "JSON object or key=value",
            )
            if not isinstance(parsed, dict):
                raise ValueError("node_selector JSON must be an object")
            return [_string_map(parsed, "node_selector")]
        if "=" in stripped:
            return [stripped]
        raise ValueError("node_selector must be a JSON object or key=value")
    if isinstance(value, list):
        entries: list[str | dict[str, str]] = []
        for item in value:
            entries.extend(_coerce_node_selector_entries(item))
        return entries
    raise ValueError("node_selector must be a JSON object or key=value")


def _merge_node_selector_entries(value: Any) -> dict[str, str]:
    """Merge parsed node-selector entries into the Kubernetes map shape."""
    selector: dict[str, str] = {}
    for item in _coerce_node_selector_entries(value):
        if isinstance(item, dict):
            selector.update(item)
            continue
        key, label_value = item.split("=", 1)
        if not key:
            raise ValueError("node_selector key=value entries require a key")
        selector[key] = label_value
    return selector


def _coerce_tolerations(value: Any) -> list[dict[str, Any]]:
    """Coerce CLI/Python tolerations input into Kubernetes list shape."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [_string_keyed_map(value, "tolerations")]
    if isinstance(value, str):
        parsed = _parse_json_cli_value(
            value.strip(), "tolerations", "JSON object or array"
        )
        return _coerce_tolerations(parsed)
    if isinstance(value, list):
        tolerations: list[dict[str, Any]] = []
        for item in value:
            tolerations.extend(_coerce_tolerations(item))
        return tolerations
    raise ValueError("tolerations must be a JSON object or array")


def _string_map(value: dict[Any, Any], field_name: str) -> dict[str, str]:
    """Validate a JSON object whose keys and values must be strings."""
    parsed: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{field_name} keys and values must be strings")
        parsed[key] = item
    return parsed


def _string_keyed_map(value: dict[Any, Any], field_name: str) -> dict[str, Any]:
    """Validate a JSON object whose keys must be strings."""
    parsed: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        parsed[key] = item
    return parsed


class SecretMountConfig(BaseModel):
    """Configuration for mounting a Kubernetes secret as a volume."""

    name: str = Field(description="Secret name in Kubernetes")
    mount_path: str = Field(description="Path to mount the secret")
    sub_path: str | None = Field(
        default=None, description="Specific key to mount (optional)"
    )


class KubeManageOptions(BaseModel):
    """Common options for Kubernetes job management commands.

    This config contains the kubeconfig and namespace options shared by
    management commands (status, logs, delete, attach, results, cancel, preflight).

    Example CLI usage:
        aiperf kube status --kubeconfig ~/.kube/prod-config --namespace benchmarks
        aiperf kube logs abc123 --namespace aiperf-bench
    """

    kubeconfig: Annotated[
        str | None,
        Field(
            description="Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env)"
        ),
        CLIParameter(name="--kubeconfig", group=_KubeGroups.KUBERNETES),
    ] = None

    kube_context: Annotated[
        str | None,
        Field(
            description="Kubernetes context to use (defaults to current context in kubeconfig)"
        ),
        CLIParameter(name="--kube-context", group=_KubeGroups.KUBERNETES),
    ] = None

    namespace: Annotated[
        str | None,
        Field(description="Kubernetes namespace (default: aiperf-benchmarks)"),
        CLIParameter(name=["-n", "--namespace"], group=_KubeGroups.KUBERNETES),
    ] = None


class KubeOptions(KubeManageOptions):
    """Kubernetes-specific deployment options.

    This config contains the Kubernetes deployment settings (not benchmark config).
    Inherits kubeconfig and namespace from KubeManageOptions.
    Use with AIPerfConfig for the complete deployment specification.

    Example YAML:
        ```yaml
        image: aiperf:latest
        namespace: benchmarks
        total_workers: 10
        ttl_seconds: 300
        node_selector:
          gpu: "true"
        tolerations:
          - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
        ```
    """

    # Optional: Human-readable name
    name: Annotated[
        str | None,
        Field(
            default=None,
            description="Human-readable name for the benchmark job (DNS label, max 40 chars)",
        ),
        CLIParameter(name="--name", group=_KubeGroups.KUBERNETES),
    ] = None

    # Optional CLI override: raw workload YAML may already declare the image.
    image: Annotated[
        str | None,
        Field(
            default=None,
            description="AIPerf container image to use for Kubernetes deployment",
            min_length=1,
        ),
        CLIParameter(name="--image", group=_KubeGroups.KUBERNETES),
    ] = None

    image_pull_policy: Annotated[
        ImagePullPolicy | None,
        Field(
            default=None,
            description="Image pull policy (Always, IfNotPresent, Never). "
            "Use 'Never' for minikube (or local clusters) with locally loaded images.",
        ),
        CLIParameter(name="--image-pull-policy", group=_KubeGroups.KUBERNETES),
    ] = None

    total_workers: Annotated[
        int,
        Field(
            gt=0,
            description="Total number of workers. Automatically distributed across pods "
            "based on --workers-per-pod (default 10). E.g., --total-workers 50 = 5 pods × 10 workers.",
        ),
        CLIParameter(name="--total-workers", group=_KubeGroups.KUBERNETES),
    ] = 10

    ttl_seconds: Annotated[
        int | None,
        Field(
            ge=0,
            description="Seconds to keep pods after completion (None to disable TTL)",
        ),
        CLIParameter(name="--ttl-seconds", group=_KubeGroups.KUBERNETES),
    ] = 300

    # Node placement
    node_selector: Annotated[
        dict[str, str],
        BeforeValidator(_merge_node_selector_entries),
        Field(description="Node selector labels (JSON object or repeated key=value)"),
        CLIParameter(parse=False),
    ] = {}

    node_selector_cli: Annotated[
        list[str | dict[str, str]],
        BeforeValidator(_coerce_node_selector_entries),
        Field(
            exclude=True,
            description="CLI-only node selector labels (JSON object or repeated key=value)",
        ),
        CLIParameter(
            name="--node-selector",
            group=_KubeGroups.K8S_NODE_PLACEMENT,
            accepts_keys=False,
            n_tokens=-1,
        ),
    ] = []

    tolerations: Annotated[
        list[dict[str, Any]],
        BeforeValidator(_coerce_tolerations),
        Field(
            description="Pod tolerations as JSON object/array or repeated JSON values"
        ),
        CLIParameter(
            name="--tolerations",
            group=_KubeGroups.K8S_NODE_PLACEMENT,
            n_tokens=-1,
        ),
    ] = []

    # Scheduling / Kueue
    queue_name: Annotated[
        str | None,
        Field(
            default=None,
            description="Kueue LocalQueue name for gang-scheduling. When set, the JobSet "
            "is submitted to Kueue for quota-managed admission.",
        ),
        CLIParameter(name="--queue-name", group=_KubeGroups.K8S_SCHEDULING),
    ] = None

    priority_class: Annotated[
        str | None,
        Field(
            default=None,
            description="Kueue WorkloadPriorityClass name for scheduling priority",
        ),
        CLIParameter(name="--priority-class", group=_KubeGroups.K8S_SCHEDULING),
    ] = None

    # Metadata
    annotations: Annotated[
        dict[str, str],
        Field(description="Additional pod annotations"),
        CLIParameter(name="--annotations", group=_KubeGroups.K8S_METADATA),
    ] = {}

    labels: Annotated[
        dict[str, str],
        Field(description="Additional pod labels"),
        CLIParameter(name="--labels", group=_KubeGroups.K8S_METADATA),
    ] = {}

    # Secrets and credentials
    image_pull_secrets: Annotated[
        list[str],
        Field(description="Image pull secret names"),
        CLIParameter(name="--image-pull-secrets", group=_KubeGroups.K8S_SECRETS),
    ] = []

    env_vars: Annotated[
        dict[str, str],
        Field(description="Extra environment variables (key: value)"),
        CLIParameter(name="--env-vars", group=_KubeGroups.K8S_SECRETS),
    ] = {}

    env_from_secrets: Annotated[
        dict[str, str],
        Field(
            description="Environment variables from secrets (ENV_NAME: secret_name/key)"
        ),
        CLIParameter(name="--env-from-secrets", group=_KubeGroups.K8S_SECRETS),
    ] = {}

    secret_mounts: Annotated[
        list[SecretMountConfig],
        Field(description="Secret volume mounts"),
        CLIParameter(name="--secret-mounts", group=_KubeGroups.K8S_SECRETS),
    ] = []

    service_account: Annotated[
        str | None,
        Field(description="Service account name for pods"),
        CLIParameter(name="--service-account", group=_KubeGroups.K8S_SECRETS),
    ] = None

    def to_crd_spec(self, config: AIPerfConfig) -> dict[str, Any]:
        """Build a nested CRD spec dict from CLI options + AIPerfConfig.

        Places the complete AIPerfConfig envelope alongside deployment fields.
        Explicit values equal to their defaults remain present so downstream
        scenario validation can distinguish authored intent from defaults.

        Args:
            config: The validated AIPerfConfig from CLI flags.

        Returns:
            Nested CRD spec dict: {benchmark: {...}, image: ..., ...}
        """

        envelope = config.model_dump(
            mode="json",
            by_alias=True,
            exclude_unset=True,
            exclude_none=True,
        )
        self.apply_total_workers_override(envelope)

        dc = self.to_deployment_config()
        from aiperf.common.endpoint_credentials import (
            validate_kubernetes_credential_transport,
        )

        validate_kubernetes_credential_transport(
            config.benchmark.endpoint, dc.pod_template.env
        )
        dc_dict = dc.model_dump(
            mode="json", by_alias=True, exclude_unset=True, exclude_none=True
        )

        # Compute connections_per_worker from --total-workers (only when
        # explicitly set by the user, not the default). When the CR YAML
        # already has connectionsPerWorker, don't override it.
        if "total_workers" in self.model_fields_set and self.total_workers > 0:
            concurrency = max(
                (
                    getattr(phase, "concurrency", 1) or 1
                    for phase in config.benchmark.phases
                ),
                default=1,
            )
            dc_dict["connectionsPerWorker"] = max(
                1, math.ceil(concurrency / self.total_workers)
            )

        return {**envelope, **dc_dict}

    def apply_total_workers_override(self, envelope: dict[str, Any]) -> None:
        """Stamp an explicit Kubernetes worker total onto the benchmark runtime.

        ``connectionsPerWorker`` remains useful as the automatic sizing ratio,
        but it cannot represent an exact worker count when concurrency is not
        evenly divisible. ``runtime.workers`` is the canonical Config-v2 field
        consumed by both the operator and direct deployment paths.

        Args:
            envelope: Mutable AIPerfConfig wire-shape mapping.
        """
        if "total_workers" not in self.model_fields_set:
            return
        benchmark = envelope.setdefault("benchmark", {})
        if not isinstance(benchmark, dict):
            raise ValueError("Kubernetes worker override requires benchmark mapping")
        runtime = benchmark.setdefault("runtime", {})
        if not isinstance(runtime, dict):
            raise ValueError("Kubernetes worker override requires runtime mapping")
        runtime["workers"] = self.total_workers

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        """Validate name is a valid DNS label (max 40 chars)."""
        if v is not None:
            # Deferred: aiperf.kubernetes.resources imports aiperf.config at module
            # scope, so a top-level import here would be a config -> kubernetes ->
            # config cycle.
            from aiperf.kubernetes.resources import validate_dns_label

            validate_dns_label(v, "name", max_length=40)
        return v

    @model_validator(mode="after")
    def normalize_placement_options(self) -> KubeOptions:
        """Normalize CLI-friendly placement values to Kubernetes deployment shapes."""
        node_selector_explicit = bool(
            self.model_fields_set & {"node_selector", "node_selector_cli"}
        )
        tolerations_explicit = "tolerations" in self.model_fields_set
        object.__setattr__(
            self,
            "node_selector",
            _merge_node_selector_entries([self.node_selector, *self.node_selector_cli]),
        )
        object.__setattr__(self, "tolerations", _coerce_tolerations(self.tolerations))
        if not node_selector_explicit:
            self.__pydantic_fields_set__.discard("node_selector")
        if not tolerations_explicit:
            self.__pydantic_fields_set__.discard("tolerations")
        return self

    @model_validator(mode="after")
    def validate_env_from_secrets_format(self) -> KubeOptions:
        """Validate that env_from_secrets values use 'secret_name/key' format."""
        invalid = [k for k, v in self.env_from_secrets.items() if "/" not in v]
        if invalid:
            raise ValueError(
                f"env_from_secrets values must use 'secret_name/key' format. "
                f"Missing '/' in entries: {', '.join(sorted(invalid))}"
            )
        return self

    def _to_pod_template_config(self) -> PodTemplateConfig | None:
        """Build only the pod-template fields explicitly authored by the user."""
        from aiperf.config.deployment import PodTemplateConfig

        env: list[dict[str, Any]] = [
            {"name": name, "value": value} for name, value in self.env_vars.items()
        ]
        env.extend(
            {
                "name": env_name,
                "valueFrom": {
                    "secretKeyRef": {
                        "name": secret_ref.split("/", 1)[0],
                        "key": secret_ref.split("/", 1)[1],
                    },
                },
            }
            for env_name, secret_ref in self.env_from_secrets.items()
        )
        volumes = [
            {"name": f"secret-{mount.name}", "secret": {"secretName": mount.name}}
            for mount in self.secret_mounts
        ]
        volume_mounts = [
            {
                "name": f"secret-{mount.name}",
                "mountPath": mount.mount_path,
                "readOnly": True,
                **({"subPath": mount.sub_path} if mount.sub_path else {}),
            }
            for mount in self.secret_mounts
        ]

        fields_set = self.model_fields_set
        pod_kwargs: dict[str, Any] = {}
        authored_fields = (
            ({"env_vars", "env_from_secrets"}, "env", env),
            ({"secret_mounts"}, "volumes", volumes),
            ({"secret_mounts"}, "volume_mounts", volume_mounts),
            (
                {"node_selector", "node_selector_cli"},
                "node_selector",
                self.node_selector,
            ),
            ({"tolerations"}, "tolerations", self.tolerations),
            ({"annotations"}, "annotations", self.annotations),
            ({"labels"}, "labels", self.labels),
            (
                {"image_pull_secrets"},
                "image_pull_secrets",
                [{"name": secret} for secret in self.image_pull_secrets],
            ),
            ({"service_account"}, "service_account_name", self.service_account),
        )
        for source_fields, target, value in authored_fields:
            if fields_set & source_fields:
                pod_kwargs[target] = value
        return PodTemplateConfig(**pod_kwargs) if pod_kwargs else None

    def _to_scheduling_config(self) -> SchedulingConfig | None:
        """Build scheduling only when at least one scheduling flag was authored."""
        from aiperf.config.deployment import SchedulingConfig

        scheduling_kwargs: dict[str, Any] = {}
        for source, target in (
            ("queue_name", "queue_name"),
            ("priority_class", "priority_class"),
        ):
            if source in self.model_fields_set:
                scheduling_kwargs[target] = getattr(self, source)
        return SchedulingConfig(**scheduling_kwargs) if scheduling_kwargs else None

    def to_deployment_config(self) -> DeploymentConfig:
        """Convert CLI KubeOptions to a DeploymentConfig.

        Translates flat CLI fields and dict-based env/secret formats into
        K8s-native formats used by DeploymentConfig/PodTemplateConfig.

        Returns:
            DeploymentConfig with all deployment-related settings.
        """
        from aiperf.config.deployment import DeploymentConfig

        fields_set = self.model_fields_set
        deployment_kwargs: dict[str, Any] = {}
        if "image" in fields_set and self.image is not None:
            deployment_kwargs["image"] = self.image
        for source, target in (
            ("image_pull_policy", "image_pull_policy"),
            ("ttl_seconds", "ttl_seconds_after_finished"),
        ):
            if source in fields_set:
                deployment_kwargs[target] = getattr(self, source)
        if pod_template := self._to_pod_template_config():
            deployment_kwargs["pod_template"] = pod_template
        if scheduling := self._to_scheduling_config():
            deployment_kwargs["scheduling"] = scheduling

        return DeploymentConfig(**deployment_kwargs)
