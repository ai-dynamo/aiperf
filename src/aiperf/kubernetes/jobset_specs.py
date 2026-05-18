# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable JobSet container and replicated-job specification models.

These Pydantic models render JobSet v1alpha2 spec fragments for use by the
higher-level :class:`aiperf.kubernetes.jobset.AIPerfJobSetSpec`.
"""

from typing import Any

from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from aiperf.common.models import AIPerfBaseModel
from aiperf.config.deployment import PodTemplateConfig
from aiperf.kubernetes.constants import AIPerfLabels
from aiperf.kubernetes.enums import ImagePullPolicy, RestartPolicy


class AIPerfContainerSpec(AIPerfBaseModel):
    """Specification for a container within a pod."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    name: str = Field(description="Container name")
    image: str = Field(description="Container image")
    image_pull_policy: ImagePullPolicy | None = Field(
        default=None,
        description="Image pull policy (Always, Never, IfNotPresent). "
        "Defaults to Always for :latest tags, IfNotPresent otherwise.",
    )
    command: list[str] = Field(default_factory=list, description="Command to run")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: list[dict[str, Any]] = Field(
        default_factory=list, description="Environment variables"
    )
    resources: dict[str, dict[str, str]] | None = Field(
        default=None, description="Resource requests and limits"
    )
    volume_mounts: list[dict[str, Any]] = Field(
        default_factory=list, description="Volume mounts"
    )
    ports: list[dict[str, Any]] = Field(
        default_factory=list, description="Container ports"
    )
    startup_probe: dict[str, Any] | None = Field(
        default=None, description="Startup probe configuration"
    )
    liveness_probe: dict[str, Any] | None = Field(
        default=None, description="Liveness probe configuration"
    )
    readiness_probe: dict[str, Any] | None = Field(
        default=None, description="Readiness probe configuration"
    )
    security_context: dict[str, Any] | None = Field(
        default=None, description="Container security context"
    )

    def to_k8s_spec(self) -> dict[str, Any]:
        """Convert to Kubernetes container spec."""
        return self.model_dump(
            by_alias=True, exclude_unset=True, exclude_none=True, mode="json"
        )


# Pod-level security context applied to every replicated job.
_POD_SECURITY_CONTEXT: dict[str, Any] = {
    "runAsNonRoot": True,
    "runAsUser": 1000,
    "runAsGroup": 1000,
    "fsGroup": 1000,
    "seccompProfile": {"type": "RuntimeDefault"},
}

# (attr_name, camelCase pod-spec key) — truthy values copy straight through.
# Kept as a table so _build_pod_spec stays simple as new fields are added.
_POD_TEMPLATE_PASSTHROUGH: tuple[tuple[str, str], ...] = (
    ("node_selector", "nodeSelector"),
    ("tolerations", "tolerations"),
    ("affinity", "affinity"),
    ("service_account_name", "serviceAccountName"),
    ("priority_class_name", "priorityClassName"),
    ("runtime_class_name", "runtimeClassName"),
    ("scheduler_name", "schedulerName"),
    ("topology_spread_constraints", "topologySpreadConstraints"),
    ("host_aliases", "hostAliases"),
    ("dns_policy", "dnsPolicy"),
    ("dns_config", "dnsConfig"),
    ("init_containers", "initContainers"),
)


class AIPerfReplicatedJobSpec(AIPerfBaseModel):
    """Specification for a replicated job within a JobSet."""

    name: str = Field(description="Replicated job name")
    replicas: int = Field(default=1, description="Number of replicas")
    containers: list[AIPerfContainerSpec] = Field(
        default_factory=list, description="Containers in the pod"
    )
    volumes: list[dict[str, Any]] = Field(
        default_factory=list, description="Pod volumes"
    )
    restart_policy: RestartPolicy = Field(
        default=RestartPolicy.ON_FAILURE, description="Pod restart policy"
    )
    backoff_limit: int = Field(default=0, description="Job backoff limit for retries")
    job_ttl_seconds: int | None = Field(
        default=None,
        description="TTL for the Job after completion. 0 = delete immediately.",
    )
    pod_template: PodTemplateConfig | None = Field(
        default=None, description="Pod template configuration"
    )
    job_id: str | None = Field(default=None, description="Job ID for pod labeling")
    extra_annotations: dict[str, str] = Field(
        default_factory=dict,
        description="Additional annotations to add to the pod template",
    )

    def _build_pod_spec(self) -> dict[str, Any]:
        """Assemble the pod spec (containers, volumes, scheduling overrides)."""
        pod_spec: dict[str, Any] = {
            "restartPolicy": str(self.restart_policy),
            "containers": [c.to_k8s_spec() for c in self.containers],
            "volumes": self.volumes,
            "securityContext": _POD_SECURITY_CONTEXT,
        }
        tmpl = self.pod_template
        if tmpl is None:
            return pod_spec
        for attr, key in _POD_TEMPLATE_PASSTHROUGH:
            value = getattr(tmpl, attr)
            if value:
                pod_spec[key] = value
        if tmpl.image_pull_secrets:
            pod_spec["imagePullSecrets"] = list(tmpl.image_pull_secrets)
        if tmpl.share_process_namespace:
            pod_spec["shareProcessNamespace"] = True
        if tmpl.termination_grace_period_seconds is not None:
            pod_spec["terminationGracePeriodSeconds"] = (
                tmpl.termination_grace_period_seconds
            )
        if tmpl.pod_security_context:
            # Pod-level securityContext; container-level is merged in per-container
            # via build_security_context().
            pod_spec["securityContext"] = {
                **_POD_SECURITY_CONTEXT,
                **tmpl.pod_security_context,
            }
        if tmpl.extra_pod_spec:
            pod_spec.update(tmpl.extra_pod_spec)
        return pod_spec

    def _build_pod_annotations(self) -> dict[str, str]:
        """Merge annotations from the pod template and extras."""
        annotations: dict[str, str] = {}
        if self.pod_template and self.pod_template.annotations:
            annotations.update(self.pod_template.annotations)
        if self.extra_annotations:
            annotations.update(self.extra_annotations)
        return annotations

    def _build_pod_labels(self) -> dict[str, str]:
        """Build pod labels with reserved AIPerf labels kept authoritative."""
        pod_labels: dict[str, str] = {}
        if self.pod_template and self.pod_template.labels:
            pod_labels.update(self.pod_template.labels)
        pod_labels[AIPerfLabels.APP_KEY] = AIPerfLabels.APP_VALUE
        if self.job_id:
            pod_labels[AIPerfLabels.JOB_ID] = self.job_id
        return pod_labels

    def to_k8s_spec(self) -> dict[str, Any]:
        """Convert to Kubernetes replicatedJob spec."""
        pod_metadata: dict[str, Any] = {"labels": self._build_pod_labels()}
        annotations = self._build_pod_annotations()
        if annotations:
            pod_metadata["annotations"] = annotations

        pod_template: dict[str, Any] = {
            "spec": self._build_pod_spec(),
            "metadata": pod_metadata,
        }

        job_spec: dict[str, Any] = {
            "parallelism": 1,
            "completions": 1,
            "completionMode": "Indexed",
            "backoffLimit": self.backoff_limit,
            "template": pod_template,
        }
        if self.job_ttl_seconds is not None:
            job_spec["ttlSecondsAfterFinished"] = self.job_ttl_seconds

        return {
            "name": self.name,
            "replicas": self.replicas,
            "template": {"spec": job_spec},
        }
