# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified Kubernetes deployment configuration models.

These models provide a single source of truth for all Kubernetes deployment
concerns (pod templates, scheduling, images) with camelCase aliases for
CRD round-tripping.
"""

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field

from aiperf.common.finite import FiniteFloat
from aiperf.config.base import BaseConfig
from aiperf.kubernetes.enums import ImagePullPolicy


class SchedulingConfig(BaseConfig):
    """Kueue gang-scheduling configuration."""

    model_config = ConfigDict(extra="forbid")

    queue_name: str | None = Field(
        default=None,
        description="Kueue LocalQueue name for gang-scheduling",
    )
    priority_class: str | None = Field(
        default=None,
        description="Kueue WorkloadPriorityClass name (for queue admission ordering). "
        "Distinct from podTemplate.priorityClassName, which is the native K8s "
        "PriorityClass used by the default scheduler for preemption.",
    )


class PodTemplateConfig(BaseConfig):
    """Kubernetes pod template configuration in K8s-native formats."""

    model_config = ConfigDict(extra="forbid")

    env: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Environment variables in K8s EnvVar format",
    )
    volumes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Pod volumes in K8s Volume format",
    )
    volume_mounts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Volume mounts in K8s VolumeMount format",
    )
    node_selector: dict[str, str] = Field(
        default_factory=dict,
        description="Node selector labels",
    )
    tolerations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Pod tolerations for scheduling on tainted nodes",
    )
    affinity: dict[str, Any] = Field(
        default_factory=dict,
        description="Pod affinity/anti-affinity rules in K8s Affinity format "
        "(nodeAffinity, podAffinity, podAntiAffinity). Use to co-locate or "
        "separate bench pods from other workloads (e.g. keep benchmark pods "
        "off inference nodes via podAntiAffinity topologyKey=kubernetes.io/hostname).",
    )
    annotations: dict[str, str] = Field(
        default_factory=dict,
        description="Additional pod annotations",
    )
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Additional pod labels",
    )
    image_pull_secrets: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Image pull secrets, K8s LocalObjectReference shape: "
            "`[{name: secretName}, ...]`."
        ),
    )
    service_account_name: str | None = Field(
        default=None,
        description="Service account name for pods",
    )
    container_security_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Container securityContext overrides (merged into each container spec)",
    )
    share_process_namespace: bool = Field(
        default=False,
        description="When true, all containers in the pod share a single PID namespace. "
        "Enables kubectl exec cross-container kills for chaos tests. Keep false in production.",
    )
    priority_class_name: str | None = Field(
        default=None,
        description="Native K8s PriorityClass name for the pod. Distinct from "
        "scheduling.priorityClass, which is the Kueue WorkloadPriorityClass. "
        "Use this to preempt lower-priority pods via the default scheduler.",
    )
    runtime_class_name: str | None = Field(
        default=None,
        description="K8s RuntimeClass name (e.g. 'nvidia' for GPU runtime, "
        "'kata' for sandboxed runtime).",
    )
    scheduler_name: str | None = Field(
        default=None,
        description="Name of the scheduler to dispatch the pod to (e.g. a custom "
        "GPU topology scheduler). Omit to use the default scheduler.",
    )
    topology_spread_constraints: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Pod TopologySpreadConstraints in K8s format. Useful to spread "
        "worker pods evenly across zones/nodes independent of affinity rules.",
    )
    host_aliases: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Entries appended to the pod's /etc/hosts (K8s HostAlias format: "
        "{ip, hostnames: [...]}). Useful when benchmarking endpoints that aren't in "
        "cluster DNS.",
    )
    dns_policy: Annotated[
        Literal["ClusterFirst", "ClusterFirstWithHostNet", "Default", "None"] | None,
        Field(
            default=None,
            description="Pod DNS policy. Defaults to 'ClusterFirst' (K8s default); "
            "set 'None' to supply dns_config entirely.",
        ),
    ] = None
    dns_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Pod DNS config (K8s PodDNSConfig format: nameservers, searches, "
        "options). Typically paired with dns_policy='None'.",
    )
    termination_grace_period_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Seconds the kubelet waits for the pod to terminate gracefully "
        "before SIGKILL. Defaults to 30 in K8s; raise for long-running benchmarks "
        "that need extra time to flush artifacts.",
    )
    pod_security_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Pod-level securityContext (PodSecurityContext format: fsGroup, "
        "runAsUser, runAsNonRoot, supplementalGroups, sysctls, etc.). Distinct from "
        "container_security_context which applies per-container.",
    )
    init_containers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="InitContainers that run to completion before the main containers "
        "start. Full K8s Container format. Useful for sysctl tweaks (e.g. bumping "
        "ip_local_port_range), model pre-fetch, or permission fixups.",
    )
    extra_pod_spec: dict[str, Any] = Field(
        default_factory=dict,
        description="Escape hatch: raw PodSpec keys merged into the rendered pod spec "
        "last (keys here override typed fields above). Use for K8s PodSpec fields not "
        "yet modeled here (e.g. schedulingGates, resourceClaims, overhead). Typed "
        "fields are preferred when available because preflight checks, env merging, "
        "and securityContext merging only apply to typed fields.",
    )


class DeploymentConfig(BaseConfig):
    """Complete Kubernetes deployment configuration.

    Unifies image settings, pod template, and scheduling into a single model.
    """

    model_config = ConfigDict(extra="forbid")

    image: str = Field(
        default="nvcr.io/nvidia/aiperf:latest",
        description="Container image for AIPerf",
    )
    image_pull_policy: ImagePullPolicy | None = Field(
        default=None,
        description="Image pull policy (Always, Never, IfNotPresent)",
    )
    resource_mode: Literal["guaranteed", "burstable", "none"] = Field(
        default="burstable",
        description="CPU/memory resource mode for controller and worker pods. "
        "'burstable' (default) sets requests only, no limits (Burstable QoS) "
        "so the controller can grow beyond the request during aggregation "
        "without being OOM-killed. "
        "'guaranteed' applies requests==limits (Guaranteed QoS). "
        "'none' omits CPU/memory requests and limits as an escape hatch.",
    )
    connections_per_worker: int = Field(
        default=100,
        ge=1,
        description="Maximum concurrent connections each worker handles. "
        "100 keeps the asyncio event loop responsive while amortizing per-process overhead.",
    )
    timeout_seconds: FiniteFloat = Field(
        default=0,
        ge=0,
        description="Job timeout in seconds (0 = no timeout)",
    )
    ttl_seconds_after_finished: int | None = Field(
        default=300,
        ge=0,
        description="TTL after finished (seconds)",
    )
    results_ttl_days: int | None = Field(
        default=None,
        ge=1,
        le=365,
        description="Days to retain result files before cleanup",
    )
    keep_failed_pods: bool = Field(
        default=False,
        description="Preserve failed JobSet pod attempts for debugging.",
    )
    cancel: bool = Field(
        default=False,
        description="Set to true to cancel the job",
    )
    pod_template: PodTemplateConfig = Field(
        default_factory=PodTemplateConfig,
        description="Pod template configuration",
    )
    scheduling: SchedulingConfig = Field(
        default_factory=SchedulingConfig,
        description="Kueue gang-scheduling configuration. Set "
        "scheduling.queueName to a LocalQueue name to admit this job's "
        "controller + worker pods atomically via Kueue. When unset, the "
        "operator falls back to AIPERF_K8S_JOBSET_KUEUE_DEFAULT_QUEUE_NAME "
        "(operator-deploy env). Safe to leave unset on clusters without Kueue.",
    )
