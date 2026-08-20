# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover Prometheus endpoints exposed by inference-server pods."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import k8s_client

if TYPE_CHECKING:
    from kubernetes_asyncio.client import V1Pod

_logger = logging.getLogger(__name__)

DYNAMO_METRICS_ENABLED = "nvidia.com/metrics-enabled"
PROM_PORT = "prometheus.io/port"
PROM_PATH = "prometheus.io/path"
PROM_SCHEME = "prometheus.io/scheme"
AIPERF_METRICS_PATHS = "aiperf.nvidia.com/metrics-paths"

DEFAULT_SCHEME = "http"
DEFAULT_PATH = "/metrics"
PREFERRED_PORT_NAME = "metrics"
ALL_NAMESPACES = "*"
DEFAULT_DISCOVERY_TIMEOUT_S = 30.0
SERVICE_ACCOUNT_NAMESPACE_FILE = (
    "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
)

INFERENCE_SERVER_IMAGE_MARKERS: tuple[str, ...] = (
    "vllm",
    "sglang",
    "tritonserver",
    "triton-server",
    "triton-inference-server",
    "tensorrt-llm",
    "tensorrtllm",
    "trt-llm",
    "trtllm",
    "dynamo",
)


def is_running_in_kubernetes() -> bool:
    """Return whether Kubernetes injected its service environment."""
    return bool(os.environ.get("KUBERNETES_SERVICE_HOST"))


def resolve_own_namespace() -> str | None:
    """Resolve this pod's namespace from operator env or ServiceAccount mount."""
    namespace = os.environ.get("AIPERF_NAMESPACE")
    if namespace:
        return namespace
    try:
        # This fixed Kubernetes system path intentionally bypasses the user-template
        # path checker; projected ServiceAccount mounts use symlinks internally.
        namespace = (
            Path(SERVICE_ACCOUNT_NAMESPACE_FILE).read_text(encoding="utf-8").strip()
        )
    except OSError:
        return None
    return namespace or None


async def discover_kubernetes_endpoints(
    *,
    namespace: str | None = None,
    label_selector: str | None = None,
    request_timeout: float = DEFAULT_DISCOVERY_TIMEOUT_S,
) -> list[str]:
    """Return sorted scrape URLs for eligible Running pods.

    ``namespace=None`` is deliberately namespaced to the benchmark pod. Use
    ``"*"`` only when the ServiceAccount has an explicit cluster-wide pod-list
    grant.
    """
    if namespace is None:
        namespace = resolve_own_namespace()
        if namespace is None:
            _logger.warning(
                "Kubernetes metrics discovery skipped: cannot determine this pod's "
                "namespace (AIPERF_NAMESPACE unset and %s unreadable). Set "
                "server_metrics.discovery.namespace explicitly.",
                SERVICE_ACCOUNT_NAMESPACE_FILE,
            )
            return []

    try:
        async with k8s_client() as api:
            pods = await _list_running_pods(
                api, namespace, label_selector, request_timeout
            )
            urls: set[str] = set()
            for pod in pods:
                urls.update(_pod_to_urls(pod, label_selector))
            return sorted(urls)
    except Exception as exc:  # noqa: BLE001 - discovery is best-effort
        _logger.warning("Failed to discover Kubernetes endpoints: %s", exc)
        return []


async def _list_running_pods(
    api: ApiClient,
    namespace: str,
    label_selector: str | None,
    request_timeout: float = DEFAULT_DISCOVERY_TIMEOUT_S,
) -> list[V1Pod]:
    """List Running pods in one namespace or across all namespaces."""
    try:
        core = client.CoreV1Api(api)
        kwargs: dict[str, Any] = {
            "field_selector": "status.phase=Running",
            "_request_timeout": request_timeout,
        }
        if label_selector:
            kwargs["label_selector"] = label_selector

        if namespace == ALL_NAMESPACES:
            pod_list = await core.list_pod_for_all_namespaces(**kwargs)
        else:
            pod_list = await core.list_namespaced_pod(namespace=namespace, **kwargs)
        return pod_list.items
    except ApiException as exc:
        if exc.status == 403:
            _logger.warning(_forbidden_message(namespace))
        else:
            _logger.warning("Kubernetes pod list failed: %s", exc)
        return []
    except Exception as exc:  # noqa: BLE001 - discovery is best-effort
        _logger.warning("Kubernetes pod list failed: %s", exc)
        return []


def _forbidden_message(namespace: str) -> str:
    """Return actionable RBAC guidance for a denied pod list."""
    if namespace == ALL_NAMESPACES:
        return (
            "Server-metrics discovery was denied 'pods: list' across all "
            "namespaces (HTTP 403). discovery.namespace='*' requires a "
            "ClusterRole granting 'pods: list' bound to the benchmark "
            "ServiceAccount. Select one namespace or add the cluster-scoped grant."
        )
    return (
        "Server-metrics discovery was denied 'pods: list' in namespace "
        f"'{namespace}' (HTTP 403). The benchmark ServiceAccount needs a Role "
        "granting pods get/list there. Add the namespace and any custom "
        "ServiceAccounts to the chart's serverMetricsDiscoveryNamespaces value."
    )


def _pod_to_urls(pod: V1Pod, label_selector: str | None) -> list[str]:
    """Build scrape URLs for an eligible pod."""
    pod_ip = pod.status.pod_ip if pod.status else None
    if not pod_ip:
        return []

    labels: dict[str, str] = (pod.metadata.labels or {}) if pod.metadata else {}
    annotations: dict[str, str] = (
        (pod.metadata.annotations or {}) if pod.metadata else {}
    )
    if not _is_eligible(pod, labels, annotations, label_selector):
        return []

    port = _resolve_port(pod, annotations.get(PROM_PORT))
    if port is None:
        return []
    scheme = annotations.get(PROM_SCHEME, DEFAULT_SCHEME)
    multi_paths = annotations.get(AIPERF_METRICS_PATHS)
    if multi_paths:
        paths = [
            _normalize_path(path.strip())
            for path in multi_paths.split(",")
            if path.strip()
        ]
    else:
        paths = [_normalize_path(annotations.get(PROM_PATH, DEFAULT_PATH))]

    host = f"[{pod_ip}]" if ":" in pod_ip else pod_ip
    return [f"{scheme}://{host}:{port}{path}" for path in paths]


def _is_eligible(
    pod: V1Pod,
    labels: dict[str, str],
    annotations: dict[str, str],
    label_selector: str | None,
) -> bool:
    """Return whether the pod was explicitly selected or looks like inference."""
    if labels.get(DYNAMO_METRICS_ENABLED, "").lower() == "true":
        return True
    if annotations.get(AIPERF_METRICS_PATHS):
        return True
    if _has_inference_server_container(pod):
        return True
    return label_selector is not None


def _has_inference_server_container(pod: V1Pod) -> bool:
    """Return whether any container image identifies a supported server."""
    containers = (pod.spec.containers or []) if pod.spec else []
    return any(
        marker in (container.image or "").lower()
        for container in containers
        for marker in INFERENCE_SERVER_IMAGE_MARKERS
    )


def _normalize_path(path: str) -> str:
    """Ensure a metrics path begins with a slash."""
    return path if path.startswith("/") else f"/{path}"


def _resolve_port(pod: V1Pod, annotation_port: str | None) -> int | None:
    """Resolve annotation port, named metrics port, or first container port."""
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
