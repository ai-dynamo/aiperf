# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes deployment configuration."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from aiperf.config import AIPerfConfig
    from aiperf.config.deployment import DeploymentConfig

from cyclopts import Group
from pydantic import BaseModel, Field, field_validator, model_validator

from aiperf.config.cli_parameter import CLIParameter
from aiperf.kubernetes.enums import ImagePullPolicy


class _KubeGroups:
    """Groups for Kubernetes CLI options."""

    KUBERNETES = Group.create_ordered("Kubernetes")
    K8S_NODE_PLACEMENT = Group.create_ordered("Kubernetes Node Placement")
    K8S_SCHEDULING = Group.create_ordered("Kubernetes Scheduling")
    K8S_SECRETS = Group.create_ordered("Kubernetes Secrets")
    K8S_METADATA = Group.create_ordered("Kubernetes Metadata")


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
        CLIParameter(name="--namespace", group=_KubeGroups.KUBERNETES),
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
        workers: 10
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

    # Required: Container image
    image: Annotated[
        str,
        Field(
            description="AIPerf container image to use for Kubernetes deployment",
            min_length=1,
        ),
        CLIParameter(name="--image", group=_KubeGroups.KUBERNETES),
    ]

    image_pull_policy: Annotated[
        ImagePullPolicy | None,
        Field(
            default=None,
            description="Image pull policy (Always, IfNotPresent, Never). "
            "Use 'Never' for minikube (or local clusters) with locally loaded images.",
        ),
        CLIParameter(name="--image-pull-policy", group=_KubeGroups.KUBERNETES),
    ] = None

    workers: Annotated[
        int,
        Field(
            gt=0,
            description="Total number of workers. Automatically distributed across pods "
            "based on --workers-per-pod (default 10). E.g., --workers-max 50 = 5 pods × 10 workers.",
        ),
        CLIParameter(name="--workers-max", group=_KubeGroups.KUBERNETES),
    ] = 10

    ttl_seconds: Annotated[
        int | None,
        Field(
            description="Seconds to keep pods after completion (None to disable TTL)"
        ),
        CLIParameter(name="--ttl-seconds", group=_KubeGroups.KUBERNETES),
    ] = 300

    # Node placement
    node_selector: Annotated[
        dict[str, str],
        Field(description="Node selector labels (e.g., {'gpu': 'true'})"),
        CLIParameter(name="--node-selector", group=_KubeGroups.K8S_NODE_PLACEMENT),
    ] = {}

    tolerations: Annotated[
        list[dict[str, Any]],
        Field(description="Pod tolerations for scheduling on tainted nodes"),
        CLIParameter(name="--tolerations", group=_KubeGroups.K8S_NODE_PLACEMENT),
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

        Places AIPerfConfig fields under the ``benchmark`` key and deployment
        fields (image, podTemplate, scheduling) at the top level of spec.

        Args:
            config: The validated AIPerfConfig from CLI flags.

        Returns:
            Nested CRD spec dict: {benchmark: {...}, image: ..., ...}
        """

        # AIPerfConfig fields go under benchmark key (camelCase for CRD)
        benchmark = config.model_dump(mode="json", by_alias=True, exclude_defaults=True)

        # Deployment fields at spec level in camelCase via DeploymentConfig serialization
        dc = self.to_deployment_config()
        dc_dict = dc.model_dump(mode="json", by_alias=True, exclude_defaults=True)

        # Compute connections_per_worker from --workers-max flag (only when
        # explicitly set by the user, not the default). When the CR YAML
        # already has connectionsPerWorker, don't override it.
        if "workers" in self.model_fields_set and self.workers > 0:
            concurrency = max(
                (
                    getattr(phase, "concurrency", 1) or 1
                    for phase in config.phases.values()
                ),
                default=1,
            )
            dc_dict["connectionsPerWorker"] = max(
                1, math.ceil(concurrency / self.workers)
            )

        return {"benchmark": benchmark, **dc_dict}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        """Validate name is a valid DNS label (max 40 chars)."""
        if v is not None:
            from aiperf.kubernetes.resources import validate_dns_label

            validate_dns_label(v, "name", max_length=40)
        return v

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

    def to_deployment_config(self) -> DeploymentConfig:
        """Convert CLI KubeOptions to a DeploymentConfig.

        Translates flat CLI fields and dict-based env/secret formats into
        K8s-native formats used by DeploymentConfig/PodTemplateConfig.

        Returns:
            DeploymentConfig with all deployment-related settings.
        """
        from aiperf.config.deployment import (
            DeploymentConfig,
            PodTemplateConfig,
            SchedulingConfig,
        )

        # Build K8s-native env list from dict formats
        env: list[dict[str, Any]] = [
            {"name": name, "value": value} for name, value in self.env_vars.items()
        ]
        for env_name, secret_ref in self.env_from_secrets.items():
            parts = secret_ref.split("/", 1)
            secret_name = parts[0]
            secret_key = parts[1] if len(parts) > 1 else env_name
            env.append(
                {
                    "name": env_name,
                    "valueFrom": {
                        "secretKeyRef": {"name": secret_name, "key": secret_key},
                    },
                }
            )

        # Build K8s-native volumes and volume_mounts from SecretMountConfig
        volumes: list[dict[str, Any]] = [
            {"name": f"secret-{s.name}", "secret": {"secretName": s.name}}
            for s in self.secret_mounts
        ]
        volume_mounts: list[dict[str, Any]] = [
            {
                "name": f"secret-{s.name}",
                "mountPath": s.mount_path,
                "readOnly": True,
                **({"subPath": s.sub_path} if s.sub_path else {}),
            }
            for s in self.secret_mounts
        ]

        pod_template = PodTemplateConfig(
            env=env,
            volumes=volumes,
            volume_mounts=volume_mounts,
            node_selector=self.node_selector,
            tolerations=self.tolerations,
            annotations=self.annotations,
            labels=self.labels,
            image_pull_secrets=self.image_pull_secrets,
            service_account_name=self.service_account,
        )

        scheduling = SchedulingConfig(
            queue_name=self.queue_name,
            priority_class=self.priority_class,
        )

        return DeploymentConfig(
            image=self.image,
            image_pull_policy=self.image_pull_policy,
            ttl_seconds_after_finished=self.ttl_seconds,
            pod_template=pod_template,
            scheduling=scheduling,
        )
