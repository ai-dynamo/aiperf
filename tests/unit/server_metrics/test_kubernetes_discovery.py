# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for inference-only Kubernetes metrics discovery."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.server_metrics.discovery.kubernetes import (
    ALL_NAMESPACES,
    _list_running_pods,
    _pod_to_urls,
    discover_kubernetes_endpoints,
    resolve_own_namespace,
)


def _pod(
    *,
    ip: str = "10.1.2.3",
    image: str = "vllm/vllm-openai:latest",
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
    ports: list[tuple[int, str | None]] | None = None,
) -> SimpleNamespace:
    container_ports = [
        SimpleNamespace(container_port=port, name=name)
        for port, name in (ports if ports is not None else [(8000, None)])
    ]
    return SimpleNamespace(
        metadata=SimpleNamespace(labels=labels or {}, annotations=annotations or {}),
        status=SimpleNamespace(pod_ip=ip, phase="Running"),
        spec=SimpleNamespace(
            containers=[SimpleNamespace(image=image, ports=container_ports)]
        ),
    )


@asynccontextmanager
async def _fake_client(api: MagicMock) -> AsyncIterator[MagicMock]:
    yield api


def test_resolve_own_namespace_prefers_operator_env(tmp_path) -> None:
    namespace_file = tmp_path / "namespace"
    namespace_file.write_text("mounted")
    with (
        patch.dict("os.environ", {"AIPERF_NAMESPACE": "benchmark"}),
        patch(
            "aiperf.server_metrics.discovery.kubernetes.SERVICE_ACCOUNT_NAMESPACE_FILE",
            str(namespace_file),
        ),
    ):
        assert resolve_own_namespace() == "benchmark"


def test_resolve_own_namespace_uses_serviceaccount_mount(tmp_path) -> None:
    namespace_file = tmp_path / "namespace"
    namespace_file.write_text("mounted\n")
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "aiperf.server_metrics.discovery.kubernetes.SERVICE_ACCOUNT_NAMESPACE_FILE",
            str(namespace_file),
        ),
    ):
        assert resolve_own_namespace() == "mounted"


def test_pod_to_urls_rejects_prometheus_annotation_without_inference_marker() -> None:
    pod = _pod(
        image="grafana/loki:latest",
        annotations={"prometheus.io/scrape": "true"},
        ports=[(3100, None)],
    )
    assert _pod_to_urls(pod, None) == []


def test_pod_to_urls_honors_inference_annotations_and_ipv6() -> None:
    pod = _pod(
        ip="fd00:10:244::7",
        annotations={
            "prometheus.io/port": "9090",
            "prometheus.io/scheme": "https",
            "aiperf.nvidia.com/metrics-paths": "/metrics,vllm/stats",
        },
    )
    assert _pod_to_urls(pod, None) == [
        "https://[fd00:10:244::7]:9090/metrics",
        "https://[fd00:10:244::7]:9090/vllm/stats",
    ]


def test_pod_to_urls_prefers_named_metrics_port() -> None:
    pod = _pod(ports=[(8000, "http"), (9090, "metrics")])
    assert _pod_to_urls(pod, None) == ["http://10.1.2.3:9090/metrics"]


@pytest.mark.asyncio
async def test_discovery_uses_k8s_client_context_and_deduplicates() -> None:
    api = MagicMock()
    pod = _pod()
    with (
        patch.dict("os.environ", {"AIPERF_NAMESPACE": "benchmark"}),
        patch(
            "aiperf.server_metrics.discovery.kubernetes.k8s_client",
            return_value=_fake_client(api),
        ) as client_factory,
        patch(
            "aiperf.server_metrics.discovery.kubernetes._list_running_pods",
            new=AsyncMock(return_value=[pod, pod]),
        ) as list_pods,
    ):
        urls = await discover_kubernetes_endpoints()

    assert urls == ["http://10.1.2.3:8000/metrics"]
    client_factory.assert_called_once_with()
    list_pods.assert_awaited_once_with(api, "benchmark", None, 30.0)


@pytest.mark.asyncio
async def test_discovery_never_falls_back_to_cluster_scope(tmp_path) -> None:
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "aiperf.server_metrics.discovery.kubernetes.SERVICE_ACCOUNT_NAMESPACE_FILE",
            str(tmp_path / "missing"),
        ),
        patch(
            "aiperf.server_metrics.discovery.kubernetes.k8s_client"
        ) as client_factory,
    ):
        assert await discover_kubernetes_endpoints() == []
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_namespaced_list_passes_selector_and_timeout() -> None:
    api = MagicMock()
    list_call = AsyncMock(return_value=SimpleNamespace(items=[]))
    with patch(
        "aiperf.server_metrics.discovery.kubernetes.client.CoreV1Api",
        return_value=SimpleNamespace(list_namespaced_pod=list_call),
    ):
        assert await _list_running_pods(api, "dynamo", "app=vllm", 7.5) == []
    list_call.assert_awaited_once_with(
        namespace="dynamo",
        field_selector="status.phase=Running",
        _request_timeout=7.5,
        label_selector="app=vllm",
    )


@pytest.mark.asyncio
async def test_star_namespace_requires_explicit_all_namespace_call() -> None:
    api = MagicMock()
    list_call = AsyncMock(return_value=SimpleNamespace(items=[]))
    with patch(
        "aiperf.server_metrics.discovery.kubernetes.client.CoreV1Api",
        return_value=SimpleNamespace(list_pod_for_all_namespaces=list_call),
    ):
        assert await _list_running_pods(api, ALL_NAMESPACES, None) == []
    list_call.assert_awaited_once_with(
        field_selector="status.phase=Running",
        _request_timeout=30.0,
    )


@pytest.mark.asyncio
async def test_forbidden_namespace_logs_actionable_rbac(caplog) -> None:
    api = MagicMock()
    list_call = AsyncMock(side_effect=ApiException(status=403))
    with (
        patch(
            "aiperf.server_metrics.discovery.kubernetes.client.CoreV1Api",
            return_value=SimpleNamespace(list_namespaced_pod=list_call),
        ),
        caplog.at_level("WARNING", logger="aiperf.server_metrics.discovery.kubernetes"),
    ):
        assert await _list_running_pods(api, "dynamo", None) == []
    assert "pods: list" in caplog.text
    assert "ServiceAccount" in caplog.text
    assert "serverMetricsDiscoveryNamespaces" in caplog.text
