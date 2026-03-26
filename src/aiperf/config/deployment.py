# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified Kubernetes deployment configuration models.

These models provide a single source of truth for all Kubernetes deployment
concerns (pod templates, scheduling, images) with camelCase aliases for
CRD round-tripping.
"""

from typing import Any, Literal

from pydantic import ConfigDict, Field

from aiperf.config._base import BaseConfig
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
        description="Kueue WorkloadPriorityClass name for scheduling priority",
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
    annotations: dict[str, str] = Field(
        default_factory=dict,
        description="Additional pod annotations",
    )
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Additional pod labels",
    )
    image_pull_secrets: list[str] = Field(
        default_factory=list,
        description="Image pull secret names",
    )
    service_account_name: str | None = Field(
        default=None,
        description="Service account name for pods",
    )
    container_security_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Container securityContext overrides (merged into each container spec)",
    )


class DeploymentConfig(BaseConfig):
    """Complete Kubernetes deployment configuration.

    Unifies image settings, pod template, and scheduling into a single model.
    """

    model_config = ConfigDict(extra="forbid")

    image: str = Field(
        default="nvcr.io/nvidia/aiperf:latest",
        description="Container image",
    )
    image_pull_policy: ImagePullPolicy | None = Field(
        default=None,
        description="Image pull policy (Always, Never, IfNotPresent)",
    )
    resource_mode: Literal["guaranteed", "none"] = Field(
        default="guaranteed",
        description="CPU/memory resource mode for controller and worker pods. "
        "'guaranteed' applies the standard requests==limits block. "
        "'none' omits CPU/memory requests and limits as an escape hatch.",
    )
    connections_per_worker: int = Field(
        default=100,
        ge=1,
        description="Maximum concurrent connections each worker handles. "
        "100 keeps the asyncio event loop responsive while amortizing per-process overhead.",
    )
    timeout_seconds: float = Field(
        default=0,
        ge=0,
        description="Job timeout in seconds (0 = no timeout)",
    )
    ttl_seconds_after_finished: int | None = Field(
        default=300,
        description="TTL after finished (seconds)",
    )
    results_ttl_days: int | None = Field(
        default=None,
        ge=1,
        le=365,
        description="Days to retain result files before cleanup",
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
        description="Kueue scheduling configuration",
    )
