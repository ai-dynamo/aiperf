# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes resource generation for AIPerf deployments.

This module provides utilities for generating ConfigMaps, Services, RBAC
resources, and orchestrating the complete deployment.
"""

import re
import uuid
from typing import Any, ClassVar

from pydantic import Field, field_validator

from aiperf.common.models import AIPerfBaseModel
from aiperf.config import AIPerfConfig
from aiperf.config.deployment import DeploymentConfig
from aiperf.kubernetes.constants import (
    AIPERF_GROUP,
    DEFAULT_BENCHMARK_NAMESPACE,
    Annotations,
    Labels,
)
from aiperf.kubernetes.jobset import JobSetSpec

# Kubernetes ConfigMap size limit is 1 MiB (1,048,576 bytes)
CONFIGMAP_MAX_SIZE_BYTES = 1_048_576

# RFC 1123 DNS label pattern (used for Kubernetes resource names)
# Must consist of lowercase alphanumeric characters or '-'
# Must start and end with an alphanumeric character
# Maximum 63 characters (we use shorter for job_id since it's prefixed)
DNS_LABEL_PATTERN = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
DNS_LABEL_MAX_LENGTH = 63


def validate_dns_label(
    value: str, field_name: str, max_length: int = DNS_LABEL_MAX_LENGTH
) -> str:
    """Validate that a string is a valid DNS label name.

    Args:
        value: The string to validate.
        field_name: Name of the field for error messages.
        max_length: Maximum allowed length.

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is not a valid DNS label.
    """
    if not value:
        raise ValueError(f"{field_name} cannot be empty")

    if len(value) > max_length:
        raise ValueError(
            f"{field_name} '{value}' exceeds maximum length of {max_length} characters"
        )

    if not DNS_LABEL_PATTERN.match(value):
        raise ValueError(
            f"{field_name} '{value}' must be a valid DNS label: "
            "lowercase alphanumeric characters or '-', must start and end with "
            "an alphanumeric character"
        )

    return value


class ConfigMapSpec(AIPerfBaseModel):
    """Specification for a Kubernetes ConfigMap."""

    name: str = Field(description="ConfigMap name")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    data: dict[str, str] = Field(default_factory=dict, description="ConfigMap data")
    labels: dict[str, str] = Field(default_factory=dict, description="Labels")

    def get_data_size_bytes(self) -> int:
        """Calculate the total size of the ConfigMap data in bytes."""
        return sum(len(k.encode()) + len(v.encode()) for k, v in self.data.items())

    def validate_size(self) -> None:
        """Validate that the ConfigMap data doesn't exceed the Kubernetes limit.

        Raises:
            ValueError: If the ConfigMap data exceeds the 1 MiB limit.
        """
        size = self.get_data_size_bytes()
        if size > CONFIGMAP_MAX_SIZE_BYTES:
            size_kb = size / 1024
            max_kb = CONFIGMAP_MAX_SIZE_BYTES / 1024
            raise ValueError(
                f"ConfigMap '{self.name}' data size ({size_kb:.1f} KB) exceeds "
                f"Kubernetes limit of {max_kb:.0f} KB. "
                "Consider reducing the configuration size."
            )

    def to_k8s_manifest(self) -> dict[str, Any]:
        """Generate the ConfigMap Kubernetes manifest."""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": self.labels,
            },
            "data": self.data,
        }

    @classmethod
    def from_benchmark_run(
        cls,
        name: str,
        namespace: str,
        run: Any,
        job_id: str,
    ) -> "ConfigMapSpec":
        """Create a ConfigMapSpec from a BenchmarkRun.

        Serializes the BenchmarkRun as ``run_config.json`` so the pod
        can bootstrap with ``--benchmark-run``.

        Args:
            name: ConfigMap name.
            namespace: Kubernetes namespace.
            run: BenchmarkRun instance.
            job_id: Unique benchmark job ID.

        Returns:
            ConfigMapSpec instance.
        """
        import orjson

        run_json = orjson.dumps(
            run.model_dump(mode="json", exclude_none=True),
            option=orjson.OPT_INDENT_2,
        ).decode()
        spec = cls(
            name=name,
            namespace=namespace,
            data={"run_config.json": run_json},
            labels={
                Labels.APP_KEY: Labels.APP_VALUE,
                Labels.JOB_ID: job_id,
            },
        )
        spec.validate_size()
        return spec


class NamespaceSpec(AIPerfBaseModel):
    """Specification for a Kubernetes Namespace."""

    name: str = Field(description="Namespace name")
    labels: dict[str, str] = Field(default_factory=dict, description="Labels")

    def to_k8s_manifest(self) -> dict[str, Any]:
        """Generate the Namespace Kubernetes manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.name,
                "labels": self.labels,
            },
        }


class RBACSpec(AIPerfBaseModel):
    """Specification for RBAC resources (Role + RoleBinding)."""

    name: str = Field(description="RBAC resource name prefix")
    namespace: str = Field(description="Kubernetes namespace")
    job_id: str = Field(description="Job ID for labeling")
    service_account: str = Field(default="default", description="Service account name")

    # RBAC rules required by AIPerf pods, organized by resource type.
    _RULES: ClassVar[list[dict[str, Any]]] = [
        # AIPerfJob CRs: Controller pod reads config and patches completion status
        {
            "apiGroups": [AIPERF_GROUP],
            "resources": ["aiperfjobs", "aiperfjobs/status"],
            "verbs": ["get", "list", "watch", "patch", "update"],
        },
        # ConfigMaps: Full CRUD for config storage
        {
            "apiGroups": [""],
            "resources": ["configmaps"],
            "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
        },
        # Pods and logs: Read-only for monitoring and debugging
        {
            "apiGroups": [""],
            "resources": ["pods", "pods/log"],
            "verbs": ["get", "list", "watch"],
        },
        # Services and endpoints: Read + create/delete for networking
        {
            "apiGroups": [""],
            "resources": ["services", "endpoints"],
            "verbs": ["get", "list", "watch", "create", "delete"],
        },
        # Events: Read + create for operator diagnostics
        {
            "apiGroups": [""],
            "resources": ["events"],
            "verbs": ["get", "list", "watch", "create", "patch"],
        },
        # Jobs: Read-only for status monitoring
        {
            "apiGroups": ["batch"],
            "resources": ["jobs"],
            "verbs": ["get", "list", "watch"],
        },
        # JobSets: Full lifecycle management for jobsets, read-only for status
        {
            "apiGroups": ["jobset.x-k8s.io"],
            "resources": ["jobsets"],
            "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
        },
        {
            "apiGroups": ["jobset.x-k8s.io"],
            "resources": ["jobsets/status"],
            "verbs": ["get", "list", "watch"],
        },
    ]

    @property
    def _labels(self) -> dict[str, str]:
        return {
            Labels.APP_KEY: Labels.APP_VALUE,
            Labels.JOB_ID: self.job_id,
        }

    def to_role_manifest(self) -> dict[str, Any]:
        """Generate the Role Kubernetes manifest."""
        return {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {
                "name": f"{self.name}-role",
                "namespace": self.namespace,
                "labels": self._labels,
            },
            "rules": self._RULES,
        }

    def to_role_binding_manifest(self) -> dict[str, Any]:
        """Generate the RoleBinding Kubernetes manifest."""
        return {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {
                "name": f"{self.name}-binding",
                "namespace": self.namespace,
                "labels": self._labels,
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": self.service_account,
                    "namespace": self.namespace,
                }
            ],
            "roleRef": {
                "kind": "Role",
                "name": f"{self.name}-role",
                "apiGroup": "rbac.authorization.k8s.io",
            },
        }


class KubernetesDeployment(AIPerfBaseModel):
    """Complete Kubernetes deployment specification for an AIPerf benchmark.

    This class orchestrates the generation of all necessary Kubernetes resources
    for deploying AIPerf in a distributed configuration.
    """

    job_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:8],
        description="Unique benchmark job ID",
    )
    namespace: str | None = Field(
        default=None,
        description="Kubernetes namespace (auto-generated if None)",
    )
    worker_replicas: int = Field(default=1, description="Number of worker pods")
    workers_per_pod: int | None = Field(
        default=None,
        description="Actual workers per pod for resource calculation",
    )
    config: AIPerfConfig = Field(description="AIPerf configuration")
    run: Any = Field(
        default=None,
        description="BenchmarkRun instance for this deployment. "
        "Auto-built from config if not provided.",
    )
    deployment: DeploymentConfig = Field(
        description="Kubernetes deployment configuration"
    )

    # Optional metadata for job discovery
    name: str | None = Field(
        default=None, description="Human-readable name for the benchmark job"
    )
    model_names: list[str] = Field(
        default_factory=list, description="Model names being benchmarked"
    )
    endpoint_url: str | None = Field(
        default=None, description="Target LLM endpoint URL"
    )

    def model_post_init(self, __context: Any) -> None:
        """Auto-build a BenchmarkRun from config if not provided."""
        if self.run is None:
            from pathlib import Path

            from aiperf.config.benchmark import BenchmarkRun

            self.run = BenchmarkRun(
                benchmark_id=self.job_id,
                cfg=self.config,
                artifact_dir=Path("/results"),
            )

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, v: str) -> str:
        """Validate job_id is a valid DNS label."""
        # Pod names: "aiperf-{job_id}-controller-0-0-xxxxx" (28 + job_id)
        # Must fit in 63-char DNS label, so job_id <= 35
        return validate_dns_label(v, "job_id", max_length=35)

    @property
    def effective_namespace(self) -> str:
        """Get the effective namespace (default shared namespace if not specified)."""
        return self.namespace or DEFAULT_BENCHMARK_NAMESPACE

    @property
    def auto_namespace(self) -> bool:
        """Check if namespace is auto-generated."""
        return self.namespace is None

    @property
    def jobset_name(self) -> str:
        """Get the JobSet name."""
        return f"aiperf-{self.job_id}"

    @property
    def configmap_name(self) -> str:
        """Get the ConfigMap name."""
        return f"{self.jobset_name}-config"

    def get_namespace_spec(self) -> NamespaceSpec | None:
        """Get the Namespace spec (only if using the default shared namespace)."""
        if not self.auto_namespace:
            return None
        return NamespaceSpec(
            name=self.effective_namespace,
            labels={
                Labels.APP_KEY: Labels.APP_VALUE,
                "aiperf.nvidia.com/auto-generated": "true",
            },
        )

    def get_configmap_spec(self) -> ConfigMapSpec:
        """Get the ConfigMap spec from the stored BenchmarkRun."""
        spec = ConfigMapSpec.from_benchmark_run(
            name=self.configmap_name,
            namespace=self.effective_namespace,
            run=self.run,
            job_id=self.job_id,
        )
        if self.name:
            spec.labels[Labels.NAME] = self.name
        return spec

    def get_rbac_spec(self) -> RBACSpec:
        """Get the RBAC spec."""
        return RBACSpec(
            name=self.jobset_name,
            namespace=self.effective_namespace,
            job_id=self.job_id,
        )

    def get_jobset_spec(self) -> JobSetSpec:
        """Get the JobSet spec."""
        extra_annotations: dict[str, str] = {}
        if self.model_names:
            extra_annotations[Annotations.MODEL] = ", ".join(self.model_names)
        if self.endpoint_url:
            extra_annotations[Annotations.ENDPOINT] = self.endpoint_url

        return JobSetSpec(
            name=self.jobset_name,
            namespace=self.effective_namespace,
            job_id=self.job_id,
            image=self.deployment.image,
            image_pull_policy=self.deployment.image_pull_policy,
            resource_mode=self.deployment.resource_mode,
            worker_replicas=self.worker_replicas,
            workers_per_pod=self.workers_per_pod or self.config.runtime.workers_per_pod,
            record_processors_per_pod=self.config.runtime.record_processors_per_pod,
            ttl_seconds=self.deployment.ttl_seconds_after_finished,
            pod_template=self.deployment.pod_template,
            scheduling=self.deployment.scheduling,
            gpu_telemetry_enabled=self.run.cfg.gpu_telemetry.enabled,
            server_metrics_enabled=self.run.cfg.server_metrics.enabled,
            name_label=self.name,
            extra_annotations=extra_annotations,
        )

    def get_all_manifests(self) -> list[dict[str, Any]]:
        """Get all Kubernetes manifests for the deployment.

        Returns:
            List of Kubernetes resource manifests in creation order.
        """
        manifests = []

        # 1. Namespace (if auto-generated)
        ns_spec = self.get_namespace_spec()
        if ns_spec:
            manifests.append(ns_spec.to_k8s_manifest())

        # 2. RBAC resources
        rbac_spec = self.get_rbac_spec()
        manifests.append(rbac_spec.to_role_manifest())
        manifests.append(rbac_spec.to_role_binding_manifest())

        # 3. ConfigMap (with size validation)
        configmap_spec = self.get_configmap_spec()
        configmap_spec.validate_size()
        manifests.append(configmap_spec.to_k8s_manifest())

        # 4. JobSet
        manifests.append(self.get_jobset_spec().to_k8s_manifest())

        return manifests
