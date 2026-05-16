# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.server_metrics.discovery.kubernetes import (
    _is_eligible,
    _normalize_path,
    _pod_to_urls,
    _resolve_port,
    discover_kubernetes_endpoints,
    is_running_in_kubernetes,
)


class _Port:
    def __init__(self, container_port: int | None = None, name: str | None = None):
        self.container_port = container_port
        self.name = name


class _Container:
    def __init__(self, ports=None, image: str | None = None):
        self.ports = ports
        self.image = image


class _PodSpec:
    def __init__(self, containers=None):
        self.containers = containers


class _PodStatus:
    def __init__(self, pod_ip=None, phase=None):
        self.pod_ip = pod_ip
        self.phase = phase


class _PodMeta:
    def __init__(self, labels=None, annotations=None):
        self.labels = labels
        self.annotations = annotations


class _Pod:
    def __init__(self, metadata=None, status=None, spec=None):
        self.metadata = metadata
        self.status = status
        self.spec = spec


class _PodList:
    def __init__(self, items):
        self.items = items


def _make_pod(
    *,
    pod_ip: str = "10.1.2.3",
    labels: dict | None = None,
    annotations: dict | None = None,
    ports: list[dict] | None = None,
    image: str | None = None,
) -> _Pod:
    """Build a typed-ish V1Pod stand-in with the given manifest pieces."""
    container_ports = None
    if ports is not None:
        container_ports = [
            _Port(
                container_port=p.get("containerPort"),
                name=p.get("name"),
            )
            for p in ports
        ]
    if ports is not None or image is not None:
        containers = [_Container(ports=container_ports, image=image)]
    else:
        containers = []
    return _Pod(
        metadata=_PodMeta(labels=labels or {}, annotations=annotations or {}),
        status=_PodStatus(pod_ip=pod_ip, phase="Running"),
        spec=_PodSpec(containers=containers),
    )


@asynccontextmanager
async def _fake_k8s_client(mock_api):
    yield mock_api


# ---------------------------------------------------------------------------
# is_running_in_kubernetes
# ---------------------------------------------------------------------------
class TestIsRunningInKubernetes:
    def test_returns_true_when_service_host_set(self):
        with patch.dict("os.environ", {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}):
            assert is_running_in_kubernetes() is True

    def test_returns_false_when_service_host_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            assert is_running_in_kubernetes() is False

    def test_returns_false_when_service_host_empty(self):
        with patch.dict("os.environ", {"KUBERNETES_SERVICE_HOST": ""}):
            assert is_running_in_kubernetes() is False


# ---------------------------------------------------------------------------
# _is_eligible
# ---------------------------------------------------------------------------
class TestIsEligible:
    def test_eligible_via_dynamo_label(self):
        pod = _make_pod(labels={"nvidia.com/metrics-enabled": "true"})
        assert _is_eligible(pod, pod.metadata.labels, {}, None) is True

    def test_eligible_via_dynamo_label_case_insensitive(self):
        pod = _make_pod(labels={"nvidia.com/metrics-enabled": "True"})
        assert _is_eligible(pod, pod.metadata.labels, {}, None) is True

    def test_eligible_via_aiperf_metrics_paths_annotation(self):
        annotations = {"aiperf.nvidia.com/metrics-paths": "/metrics,/vllm/metrics"}
        pod = _make_pod(annotations=annotations)
        assert _is_eligible(pod, {}, annotations, None) is True

    @pytest.mark.parametrize(
        "image",
        [
            param("vllm/vllm-openai:v0.6.3", id="vllm"),
            param("lmsysorg/sglang:latest", id="sglang"),
            param("nvcr.io/nvidia/tritonserver:24.05-py3", id="tritonserver"),
            param("nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3", id="trtllm"),
            param("nvcr.io/nvidia/dynamo:0.1.0", id="dynamo"),
            param("VLLM/Vllm-Openai:Latest", id="case-insensitive"),
        ],
    )  # fmt: skip
    def test_eligible_via_inference_server_image(self, image: str):
        pod = _make_pod(image=image, ports=[{"containerPort": 8000}])
        assert _is_eligible(pod, {}, {}, None) is True

    def test_eligible_via_label_selector_fallback(self):
        pod = _make_pod()
        assert _is_eligible(pod, {}, {}, "app=vllm") is True

    def test_not_eligible_without_markers_or_selector(self):
        pod = _make_pod()
        assert _is_eligible(pod, {}, {}, None) is False

    def test_not_eligible_when_dynamo_label_is_false(self):
        pod = _make_pod(labels={"nvidia.com/metrics-enabled": "false"})
        assert _is_eligible(pod, pod.metadata.labels, {}, None) is False

    def test_not_eligible_for_loki_with_prometheus_scrape(self):
        # Regression: Loki / Grafana / kube-state-metrics all set
        # prometheus.io/scrape=true but are not inference servers.
        annotations = {"prometheus.io/scrape": "true"}
        pod = _make_pod(
            annotations=annotations,
            image="grafana/loki:2.9.0",
            ports=[{"containerPort": 3100}],
        )
        assert _is_eligible(pod, {}, annotations, None) is False

    @pytest.mark.parametrize(
        "image",
        [
            param("grafana/loki:2.9.0", id="loki"),
            param("grafana/grafana:10.2.0", id="grafana"),
            param("quay.io/prometheus/node-exporter:v1.7.0", id="node-exporter"),
            param("registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.10.0", id="kube-state-metrics"),
            param("nginx:1.25", id="nginx"),
        ],
    )  # fmt: skip
    def test_not_eligible_for_non_inference_images(self, image: str):
        pod = _make_pod(image=image, ports=[{"containerPort": 9090}])
        assert _is_eligible(pod, {}, {}, None) is False


# ---------------------------------------------------------------------------
# _normalize_path
# ---------------------------------------------------------------------------
class TestNormalizePath:
    @pytest.mark.parametrize(
        "path, expected",
        [
            param("/metrics", "/metrics", id="already-slash"),
            param("metrics", "/metrics", id="missing-slash"),
            param("/custom/path", "/custom/path", id="custom-path"),
        ],
    )  # fmt: skip
    def test_normalize_path(self, path: str, expected: str):
        assert _normalize_path(path) == expected


# ---------------------------------------------------------------------------
# _resolve_port
# ---------------------------------------------------------------------------
class TestResolvePort:
    def test_annotation_port_takes_precedence(self):
        pod = _make_pod(ports=[{"containerPort": 8080}])
        assert _resolve_port(pod, "9090") == 9090

    def test_invalid_annotation_falls_through_to_container_port(self):
        pod = _make_pod(ports=[{"containerPort": 8080}])
        assert _resolve_port(pod, "not-a-number") == 8080

    def test_preferred_metrics_port_name(self):
        pod = _make_pod(
            ports=[
                {"containerPort": 8080, "name": "http"},
                {"containerPort": 9090, "name": "metrics"},
            ]
        )
        assert _resolve_port(pod, None) == 9090

    def test_first_container_port_when_no_named_match(self):
        pod = _make_pod(
            ports=[
                {"containerPort": 8080, "name": "http"},
                {"containerPort": 8443, "name": "https"},
            ]
        )
        assert _resolve_port(pod, None) == 8080

    def test_no_containers_returns_none(self):
        pod = _Pod(spec=_PodSpec(containers=[]), status=_PodStatus(pod_ip="x"))
        assert _resolve_port(pod, None) is None

    def test_no_ports_returns_none(self):
        pod = _make_pod(ports=[])
        assert _resolve_port(pod, None) is None

    def test_skips_port_without_container_port(self):
        pod = _make_pod(ports=[{"name": "metrics"}])
        assert _resolve_port(pod, None) is None

    def test_multiple_containers(self):
        cs = [
            _Container(ports=[_Port(container_port=8080, name="http")]),
            _Container(ports=[_Port(container_port=9090, name="metrics")]),
        ]
        pod = _Pod(
            metadata=_PodMeta(labels={}, annotations={}),
            status=_PodStatus(pod_ip="10.1.2.3"),
            spec=_PodSpec(containers=cs),
        )
        assert _resolve_port(pod, None) == 9090


# ---------------------------------------------------------------------------
# _pod_to_url
# ---------------------------------------------------------------------------
class TestPodToUrl:
    def test_url_with_prometheus_annotations(self):
        # prometheus.io annotations are honored when the pod is otherwise
        # eligible (here: a vLLM image triggers eligibility).
        pod = _make_pod(
            annotations={
                "prometheus.io/port": "9090",
                "prometheus.io/path": "/custom/metrics",
                "prometheus.io/scheme": "https",
            },
            image="vllm/vllm-openai:latest",
            ports=[{"containerPort": 8000}],
        )
        assert _pod_to_urls(pod, None) == ["https://10.1.2.3:9090/custom/metrics"]

    def test_url_with_dynamo_label_and_defaults(self):
        pod = _make_pod(
            labels={"nvidia.com/metrics-enabled": "true"},
            ports=[{"containerPort": 8080}],
        )
        assert _pod_to_urls(pod, None) == ["http://10.1.2.3:8080/metrics"]

    def test_url_with_inference_server_image(self):
        pod = _make_pod(
            image="nvcr.io/nvidia/tritonserver:24.05-py3",
            ports=[{"containerPort": 8002}],
        )
        assert _pod_to_urls(pod, None) == ["http://10.1.2.3:8002/metrics"]

    def test_url_with_label_selector_fallback(self):
        pod = _make_pod(ports=[{"containerPort": 8080}])
        assert _pod_to_urls(pod, "app=vllm") == ["http://10.1.2.3:8080/metrics"]

    def test_returns_none_when_ineligible(self):
        pod = _make_pod(ports=[{"containerPort": 8080}])
        assert _pod_to_urls(pod, None) == []

    def test_returns_none_for_loki_pod_with_prometheus_scrape(self):
        # Regression: prometheus.io/scrape=true alone must not trigger discovery.
        pod = _make_pod(
            annotations={"prometheus.io/scrape": "true"},
            image="grafana/loki:2.9.0",
            ports=[{"containerPort": 3100}],
        )
        assert _pod_to_urls(pod, None) == []

    def test_returns_none_when_missing_pod_ip(self):
        pod = _make_pod(
            pod_ip="",
            image="vllm/vllm-openai:latest",
            ports=[{"containerPort": 8000}],
        )
        assert _pod_to_urls(pod, None) == []

    def test_returns_none_when_no_port(self):
        pod = _make_pod(
            image="vllm/vllm-openai:latest",
            ports=[],
        )
        assert _pod_to_urls(pod, None) == []

    def test_returns_none_when_missing_metadata(self):
        pod = _Pod(
            metadata=None,
            status=_PodStatus(pod_ip="10.1.2.3"),
            spec=_PodSpec(containers=[]),
        )
        assert _pod_to_urls(pod, None) == []

    def test_multi_path_annotation(self):
        # The metrics-paths annotation is itself an opt-in signal.
        pod = _make_pod(
            annotations={
                "prometheus.io/port": "8000",
                "aiperf.nvidia.com/metrics-paths": "/metrics,/vllm/metrics,/sglang/metrics",
            },
        )
        urls = _pod_to_urls(pod, None)
        assert len(urls) == 3
        assert "http://10.1.2.3:8000/metrics" in urls
        assert "http://10.1.2.3:8000/vllm/metrics" in urls
        assert "http://10.1.2.3:8000/sglang/metrics" in urls

    def test_multi_path_overrides_single_path(self):
        pod = _make_pod(
            annotations={
                "prometheus.io/port": "8000",
                "prometheus.io/path": "/default",
                "aiperf.nvidia.com/metrics-paths": "/a,/b",
            },
        )
        urls = _pod_to_urls(pod, None)
        assert urls == ["http://10.1.2.3:8000/a", "http://10.1.2.3:8000/b"]


# ---------------------------------------------------------------------------
# discover_kubernetes_endpoints (integration-level)
# ---------------------------------------------------------------------------
class TestDiscoverKubernetesEndpoints:
    @pytest.mark.asyncio
    async def test_returns_empty_when_api_unavailable(self):
        def _raise(*_, **__):
            raise RuntimeError("no cluster")

        with patch(
            "aiperf.server_metrics.discovery.kubernetes.k8s_client",
            new=_raise,
        ):
            urls = await discover_kubernetes_endpoints()
        assert urls == []

    @pytest.mark.asyncio
    async def test_discovers_urls_from_pods(self):
        pod = _make_pod(
            image="vllm/vllm-openai:latest",
            ports=[{"containerPort": 9100}],
        )
        mock_api = MagicMock()

        with (
            patch(
                "aiperf.server_metrics.discovery.kubernetes.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            patch(
                "aiperf.server_metrics.discovery.kubernetes._list_running_pods",
                new=AsyncMock(return_value=[pod]),
            ),
        ):
            urls = await discover_kubernetes_endpoints()

        assert urls == ["http://10.1.2.3:9100/metrics"]

    @pytest.mark.asyncio
    async def test_deduplicates_urls(self):
        pod1 = _make_pod(
            pod_ip="10.1.2.3",
            image="vllm/vllm-openai:latest",
            ports=[{"containerPort": 9100}],
        )
        pod2 = _make_pod(
            pod_ip="10.1.2.3",
            image="vllm/vllm-openai:latest",
            ports=[{"containerPort": 9100}],
        )
        mock_api = MagicMock()

        with (
            patch(
                "aiperf.server_metrics.discovery.kubernetes.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            patch(
                "aiperf.server_metrics.discovery.kubernetes._list_running_pods",
                new=AsyncMock(return_value=[pod1, pod2]),
            ),
        ):
            urls = await discover_kubernetes_endpoints()

        assert urls == ["http://10.1.2.3:9100/metrics"]

    @pytest.mark.asyncio
    async def test_sorts_urls(self):
        pod_a = _make_pod(
            pod_ip="10.1.2.9",
            image="vllm/vllm-openai:latest",
            ports=[{"containerPort": 9100}],
        )
        pod_b = _make_pod(
            pod_ip="10.1.2.1",
            image="vllm/vllm-openai:latest",
            ports=[{"containerPort": 9100}],
        )
        mock_api = MagicMock()

        with (
            patch(
                "aiperf.server_metrics.discovery.kubernetes.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            patch(
                "aiperf.server_metrics.discovery.kubernetes._list_running_pods",
                new=AsyncMock(return_value=[pod_a, pod_b]),
            ),
        ):
            urls = await discover_kubernetes_endpoints()

        assert urls == [
            "http://10.1.2.1:9100/metrics",
            "http://10.1.2.9:9100/metrics",
        ]

    @pytest.mark.asyncio
    async def test_passes_namespace_and_selector(self):
        mock_api = MagicMock()

        with (
            patch(
                "aiperf.server_metrics.discovery.kubernetes.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            patch(
                "aiperf.server_metrics.discovery.kubernetes._list_running_pods",
                new=AsyncMock(return_value=[]),
            ) as mock_list,
        ):
            await discover_kubernetes_endpoints(
                namespace="inference", label_selector="app=vllm"
            )

        mock_list.assert_called_once_with(mock_api, "inference", "app=vllm")

    @pytest.mark.asyncio
    async def test_skips_ineligible_pods(self):
        eligible = _make_pod(
            pod_ip="10.1.2.3",
            image="vllm/vllm-openai:latest",
            ports=[{"containerPort": 9100}],
        )
        ineligible = _make_pod(
            pod_ip="10.1.2.4",
            ports=[{"containerPort": 8080}],
        )
        mock_api = MagicMock()

        with (
            patch(
                "aiperf.server_metrics.discovery.kubernetes.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            patch(
                "aiperf.server_metrics.discovery.kubernetes._list_running_pods",
                new=AsyncMock(return_value=[eligible, ineligible]),
            ),
        ):
            urls = await discover_kubernetes_endpoints()

        assert urls == ["http://10.1.2.3:9100/metrics"]


# ---------------------------------------------------------------------------
# _list_running_pods
# ---------------------------------------------------------------------------
class TestListRunningPods:
    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        mock_api = MagicMock()
        mock_core = MagicMock(
            list_pod_for_all_namespaces=AsyncMock(side_effect=RuntimeError("boom"))
        )

        with patch(
            "aiperf.server_metrics.discovery.kubernetes.client.CoreV1Api",
            return_value=mock_core,
        ):
            from aiperf.server_metrics.discovery.kubernetes import _list_running_pods

            pods = await _list_running_pods(mock_api, None, None)
            assert pods == []

    @pytest.mark.asyncio
    async def test_passes_field_selector_and_label_selector(self):
        mock_api = MagicMock()
        mock_core_list = AsyncMock(return_value=_PodList(items=[]))
        mock_core = MagicMock(list_namespaced_pod=mock_core_list)

        with patch(
            "aiperf.server_metrics.discovery.kubernetes.client.CoreV1Api",
            return_value=mock_core,
        ):
            from aiperf.server_metrics.discovery.kubernetes import _list_running_pods

            await _list_running_pods(mock_api, "test-ns", "app=vllm")

        kwargs = mock_core_list.call_args.kwargs
        assert kwargs["namespace"] == "test-ns"
        assert kwargs["field_selector"] == "status.phase=Running"
        assert kwargs["label_selector"] == "app=vllm"

    @pytest.mark.asyncio
    async def test_uses_all_namespaces_when_no_namespace(self):
        mock_api = MagicMock()
        mock_all_ns = AsyncMock(return_value=_PodList(items=[]))
        mock_core = MagicMock(list_pod_for_all_namespaces=mock_all_ns)

        with patch(
            "aiperf.server_metrics.discovery.kubernetes.client.CoreV1Api",
            return_value=mock_core,
        ):
            from aiperf.server_metrics.discovery.kubernetes import _list_running_pods

            await _list_running_pods(mock_api, None, None)

        mock_all_ns.assert_called_once()
        assert mock_all_ns.call_args.kwargs["field_selector"] == "status.phase=Running"
