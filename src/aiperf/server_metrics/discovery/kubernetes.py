# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes metrics endpoint discovery.

Discovery eligibility (per pod):
1) Label: nvidia.com/metrics-enabled=true   (Dynamo)
2) Annotation: prometheus.io/scrape=true   (standard)
3) User-provided label_selector (server-side filter; treated as fallback eligibility)

Prometheus annotations control scheme/port/path when present.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from kubernetes.client import CoreV1Api, V1Pod

_logger = logging.getLogger(__name__)

DYNAMO_METRICS_ENABLED: str = "nvidia.com/metrics-enabled"
PROM_SCRAPE: str = "prometheus.io/scrape"
PROM_PORT: str = "prometheus.io/port"
PROM_PATH: str = "prometheus.io/path"
PROM_SCHEME: str = "prometheus.io/scheme"


class KubernetesDiscoveryConfig(BaseModel):
    namespace: str | None = Field(
        default=None,
        description="Namespace to search. None searches all namespaces.",
    )
    label_selector: str | None = Field(
        default=None,
        description=(
            "Optional Kubernetes label selector applied server-side at list-time. "
            "Used as fallback eligibility for non-Dynamo/non-annotated pods."
        ),
    )
    default_scheme: str = Field(
        default="http",
        description="Default scheme if prometheus.io/scheme is not set.",
    )
    default_path: str = Field(
        default="/metrics",
        description="Default path if prometheus.io/path is not set.",
    )
    default_port: int | None = Field(
        default=None,
        description=(
            "Fallback port if prometheus.io/port is absent and no container port is found. "
            "If None, pods without a discoverable port are skipped."
        ),
    )
    prefer_port_name: str = Field(
        default="metrics",
        description="Preferred container port name to use when selecting a port.",
    )
    require_in_k8s: bool = Field(
        default=True,
        description="If True, return [] when not running inside Kubernetes.",
    )


def is_running_in_kubernetes() -> bool:
    """Return True if running inside Kubernetes."""
    return bool(os.environ.get("KUBERNETES_SERVICE_HOST"))


async def discover_kubernetes_endpoints(config: KubernetesDiscoveryConfig) -> list[str]:
    """Discover /metrics endpoints from Running pods."""
    if config.require_in_k8s and not is_running_in_kubernetes():
        _logger.debug("Not running in Kubernetes; skipping discovery")
        return []

    core_api = await _load_core_api()
    if core_api is None:
        return []

    pods = await _list_running_pods(core_api, config.namespace, config.label_selector)
    urls = {_pod_to_url(pod, config) for pod in pods}
    return sorted(url for url in urls if url)


async def _load_core_api() -> CoreV1Api | None:
    """Load Kubernetes CoreV1Api client (blocking work moved off event loop)."""
    try:
        return await asyncio.to_thread(_load_core_api_blocking)
    except Exception as e:
        _logger.warning(f"Failed to load Kubernetes client/config: {e}")
        return None


def _load_core_api_blocking() -> CoreV1Api:
    """Blocking loader for Kubernetes CoreV1Api client."""
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.CoreV1Api()


async def _list_running_pods(
    core_api: CoreV1Api,
    namespace: str | None,
    label_selector: str | None,
) -> list[V1Pod]:
    """List Running pods (async wrapper)."""
    try:
        pod_list = await asyncio.to_thread(
            _list_running_pods_blocking,
            core_api,
            namespace,
            label_selector,
        )
        return list(getattr(pod_list, "items", []) or [])
    except Exception as e:
        _logger.warning(f"Kubernetes pod list failed: {e}")
        return []


def _list_running_pods_blocking(
    core_api: CoreV1Api,
    namespace: str | None,
    label_selector: str | None,
) -> Any:
    """Blocking wrapper for listing Running pods."""
    kwargs: dict[str, Any] = {"field_selector": "status.phase=Running"}
    if label_selector:
        kwargs["label_selector"] = label_selector

    if namespace:
        return core_api.list_namespaced_pod(namespace, **kwargs)
    return core_api.list_pod_for_all_namespaces(**kwargs)


def _pod_to_url(pod: V1Pod, config: KubernetesDiscoveryConfig) -> str | None:
    """If eligible, build scrape URL from pod annotations/spec."""
    pod_ip = getattr(getattr(pod, "status", None), "pod_ip", None)
    if not pod_ip:
        return None

    metadata = getattr(pod, "metadata", None)
    if metadata is None:
        return None

    labels = metadata.labels or {}
    annotations = metadata.annotations or {}

    if not _is_eligible(labels, annotations, config):
        return None

    scheme = annotations.get(PROM_SCHEME, config.default_scheme)
    path = _normalize_path(annotations.get(PROM_PATH, config.default_path))

    port = _resolve_port(
        pod, annotations.get(PROM_PORT), config.prefer_port_name, config.default_port
    )
    if port is None:
        return None

    return f"{scheme}://{pod_ip}:{port}{path}"


def _is_eligible(
    labels: dict[str, str],
    annotations: dict[str, str],
    config: KubernetesDiscoveryConfig,
) -> bool:
    """Check discovery eligibility by label/annotation/selector."""
    if labels.get(DYNAMO_METRICS_ENABLED, "").lower() == "true":
        return True
    if annotations.get(PROM_SCRAPE, "").lower() == "true":
        return True
    # label_selector is already applied server-side; this is a fallback gate.
    return config.label_selector is not None


def _normalize_path(path: str) -> str:
    """Ensure metrics path starts with '/'."""
    return path if path.startswith("/") else f"/{path}"


def _resolve_port(
    pod: V1Pod,
    annotation_port: str | None,
    preferred_port_name: str,
    fallback: int | None,
) -> int | None:
    """Resolve port from annotation -> named port -> first port -> fallback."""
    if annotation_port:
        try:
            return int(annotation_port)
        except ValueError:
            pass

    spec = getattr(pod, "spec", None)
    containers = getattr(spec, "containers", None) if spec else None
    if not containers:
        return fallback

    first_port = None
    for container in containers:
        for port_spec in getattr(container, "ports", None) or []:
            container_port = getattr(port_spec, "container_port", None)
            if not container_port:
                continue
            if first_port is None:
                first_port = int(container_port)
            if getattr(port_spec, "name", None) == preferred_port_name:
                return int(container_port)

    return first_port if first_port is not None else fallback
