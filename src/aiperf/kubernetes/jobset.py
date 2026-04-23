# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JobSet specification generation for Kubernetes deployments.

This module generates JobSet YAML for deploying AIPerf as a distributed
benchmark across multiple pods. All resource and port settings are configurable
via environment variables through K8sEnvironment.
"""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from aiperf.common.models import AIPerfBaseModel
from aiperf.config.deployment import PodTemplateConfig, SchedulingConfig
from aiperf.kubernetes.constants import AIPerfLabels, KueueLabels
from aiperf.kubernetes.cr_refs import JOBSET_API_VERSION
from aiperf.kubernetes.enums import ImagePullPolicy
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset_helpers import build_shared_volumes
from aiperf.kubernetes.jobset_specs import AIPerfContainerSpec, AIPerfReplicatedJobSpec
from aiperf.kubernetes.jobset_urls import (
    JOBSET_FALLBACK_VERSION,
    JOBSET_GITHUB_REPO,
    get_jobset_install_hint,
    get_jobset_manifest_url,
    get_latest_jobset_version,
)

if TYPE_CHECKING:
    from aiperf.kubernetes.jobset_builder import _JobSetManifestBuilder

__all__ = [
    "JOBSET_FALLBACK_VERSION",
    "JOBSET_GITHUB_REPO",
    "AIPerfContainerSpec",
    "AIPerfJobSetSpec",
    "AIPerfReplicatedJobSpec",
    "controller_dns_name",
    "get_jobset_install_hint",
    "get_jobset_manifest_url",
    "get_latest_jobset_version",
]


def controller_dns_name(jobset_name: str, namespace: str) -> str:
    """Build the controller pod DNS hostname for a JobSet.

    JobSet with enableDNSHostnames creates a headless service with the same name
    as the JobSet, and pods get DNS names like:
    {jobset-name}-{job-name}-{job-index}-{pod-index}.{jobset-name}.{namespace}.svc.cluster.local

    Since we have exactly 1 controller replica with 1 pod, indices are always 0-0.

    Args:
        jobset_name: The JobSet resource name.
        namespace: Kubernetes namespace.

    Returns:
        Fully qualified DNS hostname for the controller pod.
    """
    return f"{jobset_name}-controller-0-0.{jobset_name}.{namespace}.svc.cluster.local"


class AIPerfJobSetSpec(AIPerfBaseModel):
    """Specification for a complete JobSet deployment.

    Resource settings, ports, and health probe configuration are loaded from
    K8sEnvironment and can be customized via AIPERF_K8S_* environment variables.
    """

    name: str = Field(description="JobSet name")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    job_id: str = Field(description="Unique benchmark job ID")
    image: str = Field(description="AIPerf container image")
    image_pull_policy: ImagePullPolicy | None = Field(
        default=None,
        description="Image pull policy for all containers (Always, Never, IfNotPresent). "
        "Set to 'Never' for local development with minikube.",
    )
    resource_mode: Literal["guaranteed", "burstable", "none"] = Field(
        default="guaranteed",
        description="CPU/memory resource mode for controller and worker pods. "
        "'guaranteed' emits requests==limits. "
        "'burstable' emits requests only (no limits). "
        "'none' omits the resources block.",
    )
    worker_replicas: int = Field(default=1, description="Number of worker pods")
    workers_per_pod: int | None = Field(
        default=None,
        description="Actual workers per pod (used for resource calculation). "
        "Defaults to Environment.WORKER.DEFAULT_WORKERS_PER_POD if not set.",
    )
    record_processors_per_pod: int | None = Field(
        default=None,
        description="Actual record processors per worker pod. "
        "Defaults to a Kubernetes scale factor derived from workers_per_pod.",
    )
    ttl_seconds: int | None = Field(
        default=None, description="TTL after finished (uses K8sEnvironment default)"
    )
    keep_failed_pods: bool = Field(
        default=False,
        description="Preserve failed JobSet pod attempts for debugging.",
    )

    # Pod template
    pod_template: PodTemplateConfig = Field(
        default_factory=PodTemplateConfig, description="Pod template configuration"
    )

    # Scheduling
    scheduling: SchedulingConfig = Field(
        default_factory=SchedulingConfig, description="Kueue scheduling configuration"
    )
    gpu_telemetry_enabled: bool = Field(
        default=True,
        description="Whether to include the GPU telemetry manager container.",
    )
    server_metrics_enabled: bool = Field(
        default=True,
        description="Whether to include the server metrics manager container.",
    )

    # Optional metadata for discovery
    name_label: str | None = Field(
        default=None, description="Human-readable name label for the JobSet"
    )
    extra_annotations: dict[str, str] = Field(
        default_factory=dict,
        description="Additional annotations for the JobSet metadata",
    )

    def _build_manifest_labels(self) -> dict[str, str]:
        """Build top-level JobSet labels (AIPerf, name, Kueue scheduling)."""
        labels: dict[str, str] = {
            AIPerfLabels.APP_KEY: AIPerfLabels.APP_VALUE,
            AIPerfLabels.JOB_ID: self.job_id,
        }
        if self.name_label:
            labels[AIPerfLabels.NAME] = self.name_label
        if self.scheduling.queue_name:
            labels[KueueLabels.QUEUE_NAME] = self.scheduling.queue_name
        if self.scheduling.priority_class:
            labels[KueueLabels.PRIORITY_CLASS] = self.scheduling.priority_class
        return labels

    def _resolve_manifest_ttl(self) -> int | None:
        """Resolve the top-level JobSet ttlSecondsAfterFinished value, if any."""
        if self.keep_failed_pods:
            return None
        if self.ttl_seconds is not None:
            return self.ttl_seconds
        return K8sEnvironment.JOBSET.TTL_SECONDS_AFTER_FINISHED

    # ------------------------------------------------------------------
    # Thin delegating wrappers for the internal builder. Kept so tests and
    # callers can keep poking at ``_create_*``/``_get_*`` private helpers
    # without reaching into ``jobset_builder`` directly. The implementations
    # live in :mod:`aiperf.kubernetes.jobset_helpers` and
    # :mod:`aiperf.kubernetes.jobset_builder`.
    # ------------------------------------------------------------------

    def _builder(self) -> "_JobSetManifestBuilder":
        from aiperf.kubernetes.jobset_builder import _JobSetManifestBuilder

        return _JobSetManifestBuilder(self)

    def _create_security_context(self) -> dict[str, Any]:
        from aiperf.kubernetes.jobset_helpers import build_security_context

        return build_security_context(self.pod_template)

    def _create_health_probe(self, port: int, path: str = "/healthz") -> dict[str, Any]:
        from aiperf.kubernetes.jobset_helpers import build_health_probe

        return build_health_probe(port, path)

    def _create_startup_probe(
        self, port: int, path: str = "/healthz"
    ) -> dict[str, Any]:
        from aiperf.kubernetes.jobset_helpers import build_startup_probe

        return build_startup_probe(port, path)

    def _create_env_vars(
        self,
        controller_host: str | None = None,
        include_pod_index: bool = True,
    ) -> list[dict[str, Any]]:
        from aiperf.kubernetes.jobset_helpers import build_env_vars

        return build_env_vars(
            job_id=self.job_id,
            namespace=self.namespace,
            pod_template=self.pod_template,
            controller_host=controller_host,
            include_pod_index=include_pod_index,
        )

    def _get_volume_mounts(self) -> list[dict[str, Any]]:
        from aiperf.kubernetes.jobset_helpers import build_volume_mounts

        return build_volume_mounts(self.pod_template)

    def _create_container(self, *args: Any, **kwargs: Any) -> AIPerfContainerSpec:
        return self._builder()._create_container(*args, **kwargs)

    def _split_worker_pod_resources(
        self,
        worker_count: int,
        record_processor_count: int,
    ) -> list[dict[str, dict[str, str]] | None]:
        return self._builder()._split_worker_pod_resources(
            worker_count, record_processor_count
        )

    def to_k8s_manifest(self) -> dict[str, Any]:
        """Generate the complete JobSet Kubernetes manifest."""
        builder = self._builder()
        controller_dns = controller_dns_name(self.name, self.namespace)
        volumes = build_shared_volumes(self.name, self.pod_template)

        controller_job = builder.build_controller_replicated_job(volumes)
        worker_job = builder.build_worker_replicated_job(volumes, controller_dns)

        metadata: dict[str, Any] = {
            "name": self.name,
            "namespace": self.namespace,
            "labels": self._build_manifest_labels(),
        }
        if self.extra_annotations:
            metadata["annotations"] = self.extra_annotations

        manifest: dict[str, Any] = {
            "apiVersion": JOBSET_API_VERSION,
            "kind": "JobSet",
            "metadata": metadata,
            "spec": {
                # Enable DNS hostnames for pod-to-pod communication
                # This creates a headless service with the same name as the JobSet,
                # allowing pods to have DNS names like:
                # {jobset-name}-{job-name}-{job-index}-{pod-index}.{jobset-name}.{namespace}.svc.cluster.local
                "network": {
                    "enableDNSHostnames": True,
                },
                "successPolicy": {
                    "operator": "All",
                    "targetReplicatedJobs": ["controller"],
                },
                "replicatedJobs": [
                    controller_job.to_k8s_spec(),
                    worker_job.to_k8s_spec(),
                ],
            },
        }

        # Kueue requires JobSets to start suspended; it unsuspends after admission
        if self.scheduling.queue_name:
            manifest["spec"]["suspend"] = True

        ttl = self._resolve_manifest_ttl()
        if ttl is not None:
            manifest["spec"]["ttlSecondsAfterFinished"] = ttl

        return manifest
