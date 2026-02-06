# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.server_metrics.discovery.kubernetes import (
    KubernetesDiscoveryConfig,
    _list_running_pods,
    _list_running_pods_blocking,
    _load_core_api,
    _load_core_api_blocking,
    _pod_to_url,
    _resolve_port,
    discover_kubernetes_endpoints,
    is_running_in_kubernetes,
)


class TestIsRunningInKubernetes:
    """Test Kubernetes environment detection."""

    def test_returns_true_when_service_host_set(self):
        """Should return True when KUBERNETES_SERVICE_HOST is set."""
        with patch.dict("os.environ", {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}):
            assert is_running_in_kubernetes() is True

    def test_returns_false_when_service_host_missing(self):
        """Should return False when KUBERNETES_SERVICE_HOST is not set."""
        with patch.dict("os.environ", {}, clear=True):
            assert is_running_in_kubernetes() is False


class TestDiscoverKubernetesEndpoints:
    """Test main discovery function."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_not_in_kubernetes(self):
        """Should return empty list when not running in K8s."""
        with patch.dict("os.environ", {}, clear=True):
            urls = await discover_kubernetes_endpoints(KubernetesDiscoveryConfig())

        assert urls == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_core_api_unavailable(self):
        with (
            patch.dict("os.environ", {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}),
            patch(
                "aiperf.server_metrics.discovery.kubernetes._load_core_api",
                new=AsyncMock(return_value=None),
            ),
        ):
            urls = await discover_kubernetes_endpoints(KubernetesDiscoveryConfig())

        assert urls == []

    @pytest.mark.asyncio
    async def test_discovers_urls_from_pods(self):
        mock_pod = MagicMock()
        mock_pod.status.pod_ip = "10.1.2.3"
        mock_pod.metadata.annotations = {"prometheus.io/scrape": "true"}
        mock_pod.metadata.labels = {}
        mock_pod.spec.containers = [MagicMock(ports=[MagicMock(container_port=9100)])]

        with (
            patch.dict("os.environ", {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}),
            patch(
                "aiperf.server_metrics.discovery.kubernetes._load_core_api",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch(
                "aiperf.server_metrics.discovery.kubernetes._list_running_pods",
                new=AsyncMock(return_value=[mock_pod]),
            ),
        ):
            urls = await discover_kubernetes_endpoints(KubernetesDiscoveryConfig())

        assert urls == ["http://10.1.2.3:9100/metrics"]


class TestLoadCoreApi:
    @pytest.mark.asyncio
    async def test_load_core_api_returns_none_on_exception(self):
        with patch(
            "aiperf.server_metrics.discovery.kubernetes.asyncio.to_thread",
            new=AsyncMock(side_effect=Exception("boom")),
        ):
            assert await _load_core_api() is None

    def test_load_core_api_blocking_falls_back_to_kube_config(self):
        from kubernetes import config as k8s_config

        with (
            patch("kubernetes.config.load_incluster_config") as load_incluster,
            patch("kubernetes.config.load_kube_config") as load_kube,
            patch("kubernetes.client.CoreV1Api") as core_api,
        ):
            load_incluster.side_effect = k8s_config.ConfigException("no in-cluster")
            core_api.return_value = MagicMock()

            result = _load_core_api_blocking()

        load_kube.assert_called_once()
        assert result is core_api.return_value


class TestListRunningPods:
    @pytest.mark.asyncio
    async def test_list_running_pods_exception_returns_empty(self):
        core_api = MagicMock()
        with patch(
            "aiperf.server_metrics.discovery.kubernetes._list_running_pods_blocking",
            side_effect=Exception("boom"),
        ):
            pods = await _list_running_pods(core_api, None, None)
        assert pods == []

    @pytest.mark.asyncio
    async def test_list_running_pods_returns_items(self):
        core_api = MagicMock()
        pod_list = MagicMock()
        pod_list.items = ["pod1"]
        with patch(
            "aiperf.server_metrics.discovery.kubernetes._list_running_pods_blocking",
            return_value=pod_list,
        ):
            pods = await _list_running_pods(core_api, None, None)
        assert pods == ["pod1"]


class TestListRunningPodsBlocking:
    def test_list_running_pods_blocking_namespaced_with_selector(self):
        core_api = MagicMock()
        core_api.list_namespaced_pod.return_value = MagicMock(items=["pod1"])

        result = _list_running_pods_blocking(core_api, "ns1", "app=vllm")

        core_api.list_namespaced_pod.assert_called_once()
        assert result.items == ["pod1"]

    def test_list_running_pods_blocking_all_namespaces(self):
        core_api = MagicMock()
        core_api.list_pod_for_all_namespaces.return_value = MagicMock(items=["pod1"])

        result = _list_running_pods_blocking(core_api, None, None)

        core_api.list_pod_for_all_namespaces.assert_called_once()
        assert result.items == ["pod1"]


class TestPodToUrl:
    """Test URL extraction from pod metadata."""

    def test_extracts_url_with_prometheus_annotations(self):
        """Should use Prometheus annotations for port/path/scheme."""
        pod = MagicMock()
        pod.status.pod_ip = "10.1.2.3"
        pod.metadata.annotations = {
            "prometheus.io/scrape": "true",
            "prometheus.io/port": "9090",
            "prometheus.io/path": "/custom/metrics",
            "prometheus.io/scheme": "https",
        }
        pod.metadata.labels = {}
        pod.spec.containers = []

        url = _pod_to_url(pod, KubernetesDiscoveryConfig())

        assert url == "https://10.1.2.3:9090/custom/metrics"

    def test_uses_defaults_without_annotations(self):
        """Should use defaults when annotations are missing."""
        pod = MagicMock()
        pod.status.pod_ip = "10.1.2.3"
        pod.metadata.annotations = {}
        pod.metadata.labels = {"nvidia.com/metrics-enabled": "true"}
        pod.spec.containers = [MagicMock()]
        pod.spec.containers[0].ports = [MagicMock(container_port=8080)]

        url = _pod_to_url(pod, KubernetesDiscoveryConfig())

        assert url == "http://10.1.2.3:8080/metrics"

    def test_returns_none_when_ineligible(self):
        pod = MagicMock()
        pod.status.pod_ip = "10.1.2.3"
        pod.metadata.annotations = {}
        pod.metadata.labels = {}
        pod.spec.containers = [MagicMock(ports=[MagicMock(container_port=8080)])]

        assert _pod_to_url(pod, KubernetesDiscoveryConfig()) is None

    def test_returns_none_when_missing_pod_ip(self):
        pod = MagicMock()
        pod.status.pod_ip = None
        pod.metadata.annotations = {"prometheus.io/scrape": "true"}
        pod.metadata.labels = {}
        pod.spec.containers = []

        assert _pod_to_url(pod, KubernetesDiscoveryConfig()) is None

    def test_returns_none_when_missing_metadata(self):
        pod = MagicMock()
        pod.status.pod_ip = "10.1.2.3"
        pod.metadata = None

        assert _pod_to_url(pod, KubernetesDiscoveryConfig()) is None

    def test_returns_none_when_port_unresolved(self):
        pod = MagicMock()
        pod.status.pod_ip = "10.1.2.3"
        pod.metadata.annotations = {"prometheus.io/scrape": "true"}
        pod.metadata.labels = {}
        pod.spec.containers = []

        cfg = KubernetesDiscoveryConfig(default_port=None)
        assert _pod_to_url(pod, cfg) is None


class TestResolvePort:
    def test_invalid_annotation_uses_fallback(self):
        pod = MagicMock()
        pod.spec.containers = []
        assert _resolve_port(pod, "not-a-number", "metrics", 1234) == 1234

    def test_skips_missing_container_port_and_uses_named_port(self):
        port_missing = MagicMock(container_port=None, name="metrics")
        port_named = MagicMock(container_port=7777, name="metrics")
        pod = MagicMock()
        pod.spec.containers = [MagicMock(ports=[port_missing, port_named])]

        assert _resolve_port(pod, None, "metrics", None) == 7777
