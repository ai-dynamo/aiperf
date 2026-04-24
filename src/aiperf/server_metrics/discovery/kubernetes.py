# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes metrics endpoint discovery using kubernetes_asyncio.

Discovers Prometheus /metrics endpoints from running pods. Eligibility per pod:
1) Label: nvidia.com/metrics-enabled=true   (Dynamo)
2) Annotation: prometheus.io/scrape=true   (standard)
3) User-provided label_selector (server-side filter; treated as fallback eligibility)

Prometheus annotations control scheme/port/path when present.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient

from aiperf.kubernetes.client import k8s_client

if TYPE_CHECKING:
    from kubernetes_asyncio.client import V1Pod

_logger = logging.getLogger(__name__)

DYNAMO_METRICS_ENABLED = "nvidia.com/metrics-enabled"
PROM_SCRAPE = "prometheus.io/scrape"
PROM_PORT = "prometheus.io/port"
PROM_PATH = "prometheus.io/path"
PROM_SCHEME = "prometheus.io/scheme"
AIPERF_METRICS_PATHS = "aiperf.nvidia.com/metrics-paths"

DEFAULT_SCHEME = "http"
DEFAULT_PATH = "/metrics"
PREFERRED_PORT_NAME = "metrics"


def is_running_in_kubernetes() -> bool:
    """Return True if running inside a Kubernetes cluster."""
    return bool(os.environ.get("KUBERNETES_SERVICE_HOST"))


async def discover_kubernetes_endpoints(
    *,
    namespace: str | None = None,
    label_selector: str | None = None,
) -> list[str]:
    """Discover /metrics endpoints from running pods.

    Args:
        namespace: Namespace to search. None searches all namespaces.
        label_selector: Optional K8s label selector applied server-side.

    Returns:
        Sorted, deduplicated list of discovered endpoint URLs.
    """
    try:
        async with k8s_client() as api:
            pods = await _list_running_pods(api, namespace, label_selector)
            urls: set[str] = set()
            for pod in pods:
                urls.update(_pod_to_urls(pod, label_selector))
            return sorted(urls)
    except Exception as e:  # noqa: BLE001 - defensive: unexpected errors must not crash metrics discovery
        _logger.warning("Failed to discover Kubernetes endpoints: %s", e)
        return []


async def _list_running_pods(
    api: ApiClient,
    namespace: str | None,
    label_selector: str | None,
) -> list[V1Pod]:
    """List Running pods, optionally filtered by namespace and labels."""
    try:
        core = client.CoreV1Api(api)
        kwargs: dict[str, Any] = {"field_selector": "status.phase=Running"}
        if label_selector:
            kwargs["label_selector"] = label_selector

        if namespace is None:
            pod_list = await core.list_pod_for_all_namespaces(**kwargs)
        else:
            pod_list = await core.list_namespaced_pod(namespace=namespace, **kwargs)
        return pod_list.items
    except Exception as e:  # noqa: BLE001 - defensive: discovery must not crash initialization
        _logger.warning("Kubernetes pod list failed: %s", e)
        return []


def _pod_to_urls(pod: V1Pod, label_selector: str | None) -> list[str]:
    """Build scrape URL(s) from pod if eligible, else empty list.

    When ``aiperf.nvidia.com/metrics-paths`` annotation is present, generates
    one URL per comma-separated path. Otherwise falls back to standard
    ``prometheus.io/path`` (single URL).
    """
    pod_ip = pod.status.pod_ip if pod.status else None
    if not pod_ip:
        return []

    labels: dict[str, str] = (pod.metadata.labels or {}) if pod.metadata else {}
    annotations: dict[str, str] = (
        (pod.metadata.annotations or {}) if pod.metadata else {}
    )

    if not _is_eligible(labels, annotations, label_selector):
        return []

    scheme = annotations.get(PROM_SCHEME, DEFAULT_SCHEME)
    port = _resolve_port(pod, annotations.get(PROM_PORT))
    if port is None:
        return []

    # Multi-path: aiperf.nvidia.com/metrics-paths annotation
    multi_paths = annotations.get(AIPERF_METRICS_PATHS)
    if multi_paths:
        paths = [
            _normalize_path(p.strip()) for p in multi_paths.split(",") if p.strip()
        ]
    else:
        paths = [_normalize_path(annotations.get(PROM_PATH, DEFAULT_PATH))]

    return [f"{scheme}://{pod_ip}:{port}{path}" for path in paths]


def _is_eligible(
    labels: dict[str, str],
    annotations: dict[str, str],
    label_selector: str | None,
) -> bool:
    """Check discovery eligibility by label/annotation/selector."""
    if labels.get(DYNAMO_METRICS_ENABLED, "").lower() == "true":
        return True
    if annotations.get(PROM_SCRAPE, "").lower() == "true":
        return True
    # label_selector already applied server-side; all returned pods are eligible
    return label_selector is not None


def _normalize_path(path: str) -> str:
    """Ensure metrics path starts with '/'."""
    return path if path.startswith("/") else f"/{path}"


def _resolve_port(pod: V1Pod, annotation_port: str | None) -> int | None:
    """Resolve port: annotation → named 'metrics' port → first container port."""
    if annotation_port:
        try:
            return int(annotation_port)
        except ValueError:
            pass

    containers = (pod.spec.containers or []) if pod.spec else []
    first_port: int | None = None

    for container in containers:
        for port_spec in container.ports or []:
            container_port = port_spec.container_port
            if not container_port:
                continue
            if first_port is None:
                first_port = int(container_port)
            if port_spec.name == PREFERRED_PORT_NAME:
                return int(container_port)

    return first_port
