# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def _make_pod(
    *,
    pod_ip: str = "10.1.2.3",
    labels: dict | None = None,
    annotations: dict | None = None,
    ports: list[dict] | None = None,
) -> MagicMock:
    """Build a mock kr8s pod with the given raw manifest."""
    containers = []
    if ports is not None:
        containers = [{"ports": ports}]

    raw = {
        "status": {"podIP": pod_ip},
        "metadata": {
            "labels": labels or {},
            "annotations": annotations or {},
        },
        "spec": {"containers": containers},
    }
    pod = MagicMock()
    pod.raw = raw
    return pod


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
        assert _is_eligible({"nvidia.com/metrics-enabled": "true"}, {}, None) is True

    def test_eligible_via_dynamo_label_case_insensitive(self):
        assert _is_eligible({"nvidia.com/metrics-enabled": "True"}, {}, None) is True

    def test_eligible_via_prometheus_annotation(self):
        assert _is_eligible({}, {"prometheus.io/scrape": "true"}, None) is True

    def test_eligible_via_label_selector_fallback(self):
        assert _is_eligible({}, {}, "app=vllm") is True

    def test_not_eligible_without_markers_or_selector(self):
        assert _is_eligible({}, {}, None) is False

    def test_not_eligible_when_dynamo_label_is_false(self):
        assert _is_eligible({"nvidia.com/metrics-enabled": "false"}, {}, None) is False


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
        raw = {"spec": {"containers": [{"ports": [{"containerPort": 8080}]}]}}
        assert (
            _resolve_port(
                raw,
                "9090",
            )
            == 9090
        )

    def test_invalid_annotation_falls_through_to_container_port(self):
        raw = {"spec": {"containers": [{"ports": [{"containerPort": 8080}]}]}}
        assert _resolve_port(raw, "not-a-number") == 8080

    def test_preferred_metrics_port_name(self):
        raw = {
            "spec": {
                "containers": [
                    {
                        "ports": [
                            {"containerPort": 8080, "name": "http"},
                            {"containerPort": 9090, "name": "metrics"},
                        ]
                    }
                ]
            }
        }
        assert _resolve_port(raw, None) == 9090

    def test_first_container_port_when_no_named_match(self):
        raw = {
            "spec": {
                "containers": [
                    {
                        "ports": [
                            {"containerPort": 8080, "name": "http"},
                            {"containerPort": 8443, "name": "https"},
                        ]
                    }
                ]
            }
        }
        assert _resolve_port(raw, None) == 8080

    def test_no_containers_returns_none(self):
        raw = {"spec": {"containers": []}}
        assert _resolve_port(raw, None) is None

    def test_no_ports_returns_none(self):
        raw = {"spec": {"containers": [{"ports": []}]}}
        assert _resolve_port(raw, None) is None

    def test_skips_port_without_container_port(self):
        raw = {"spec": {"containers": [{"ports": [{"name": "metrics"}]}]}}
        assert _resolve_port(raw, None) is None

    def test_multiple_containers(self):
        raw = {
            "spec": {
                "containers": [
                    {"ports": [{"containerPort": 8080, "name": "http"}]},
                    {"ports": [{"containerPort": 9090, "name": "metrics"}]},
                ]
            }
        }
        assert _resolve_port(raw, None) == 9090


# ---------------------------------------------------------------------------
# _pod_to_url
# ---------------------------------------------------------------------------
class TestPodToUrl:
    def test_url_with_prometheus_annotations(self):
        pod = _make_pod(
            annotations={
                "prometheus.io/scrape": "true",
                "prometheus.io/port": "9090",
                "prometheus.io/path": "/custom/metrics",
                "prometheus.io/scheme": "https",
            },
        )
        assert _pod_to_urls(pod, None) == ["https://10.1.2.3:9090/custom/metrics"]

    def test_url_with_dynamo_label_and_defaults(self):
        pod = _make_pod(
            labels={"nvidia.com/metrics-enabled": "true"},
            ports=[{"containerPort": 8080}],
        )
        assert _pod_to_urls(pod, None) == ["http://10.1.2.3:8080/metrics"]

    def test_url_with_label_selector_fallback(self):
        pod = _make_pod(ports=[{"containerPort": 8080}])
        assert _pod_to_urls(pod, "app=vllm") == ["http://10.1.2.3:8080/metrics"]

    def test_returns_none_when_ineligible(self):
        pod = _make_pod(ports=[{"containerPort": 8080}])
        assert _pod_to_urls(pod, None) == []

    def test_returns_none_when_missing_pod_ip(self):
        pod = _make_pod(pod_ip="", annotations={"prometheus.io/scrape": "true"})
        assert _pod_to_urls(pod, None) == []

    def test_returns_none_when_no_port(self):
        pod = _make_pod(
            annotations={"prometheus.io/scrape": "true"},
            ports=[],
        )
        assert _pod_to_urls(pod, None) == []

    def test_returns_none_when_missing_metadata(self):
        pod = MagicMock()
        pod.raw = {"status": {"podIP": "10.1.2.3"}}
        assert _pod_to_urls(pod, None) == []

    def test_multi_path_annotation(self):
        pod = _make_pod(
            annotations={
                "prometheus.io/scrape": "true",
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
                "prometheus.io/scrape": "true",
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
        with patch(
            "aiperf.server_metrics.discovery.kubernetes._get_api",
            new=AsyncMock(return_value=None),
        ):
            urls = await discover_kubernetes_endpoints()
        assert urls == []

    @pytest.mark.asyncio
    async def test_discovers_urls_from_pods(self):
        pod = _make_pod(
            annotations={"prometheus.io/scrape": "true"},
            ports=[{"containerPort": 9100}],
        )
        mock_api = MagicMock()

        with (
            patch(
                "aiperf.server_metrics.discovery.kubernetes._get_api",
                new=AsyncMock(return_value=mock_api),
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
            annotations={"prometheus.io/scrape": "true"},
            ports=[{"containerPort": 9100}],
        )
        pod2 = _make_pod(
            pod_ip="10.1.2.3",
            annotations={"prometheus.io/scrape": "true"},
            ports=[{"containerPort": 9100}],
        )
        mock_api = MagicMock()

        with (
            patch(
                "aiperf.server_metrics.discovery.kubernetes._get_api",
                new=AsyncMock(return_value=mock_api),
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
            annotations={"prometheus.io/scrape": "true"},
            ports=[{"containerPort": 9100}],
        )
        pod_b = _make_pod(
            pod_ip="10.1.2.1",
            annotations={"prometheus.io/scrape": "true"},
            ports=[{"containerPort": 9100}],
        )
        mock_api = MagicMock()

        with (
            patch(
                "aiperf.server_metrics.discovery.kubernetes._get_api",
                new=AsyncMock(return_value=mock_api),
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
                "aiperf.server_metrics.discovery.kubernetes._get_api",
                new=AsyncMock(return_value=mock_api),
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
            annotations={"prometheus.io/scrape": "true"},
            ports=[{"containerPort": 9100}],
        )
        ineligible = _make_pod(
            pod_ip="10.1.2.4",
            ports=[{"containerPort": 8080}],
        )
        mock_api = MagicMock()

        with (
            patch(
                "aiperf.server_metrics.discovery.kubernetes._get_api",
                new=AsyncMock(return_value=mock_api),
            ),
            patch(
                "aiperf.server_metrics.discovery.kubernetes._list_running_pods",
                new=AsyncMock(return_value=[eligible, ineligible]),
            ),
        ):
            urls = await discover_kubernetes_endpoints()

        assert urls == ["http://10.1.2.3:9100/metrics"]


# ---------------------------------------------------------------------------
# _get_api
# ---------------------------------------------------------------------------
class TestGetApi:
    @pytest.mark.asyncio
    async def test_suppresses_noisy_http_loggers_before_loading_client(self):
        mock_kr8s_asyncio = MagicMock()
        mock_kr8s_asyncio.api = AsyncMock(return_value=MagicMock())

        with patch.dict(
            "sys.modules",
            {"kr8s": MagicMock(), "kr8s.asyncio": mock_kr8s_asyncio},
        ):
            from aiperf.server_metrics.discovery import kubernetes as discovery

            with patch.object(
                discovery, "suppress_noisy_http_loggers"
            ) as suppress_mock:
                await discovery._get_api()

        suppress_mock.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        mock_kr8s_asyncio = MagicMock()
        mock_kr8s_asyncio.api = AsyncMock(side_effect=Exception("no cluster"))

        with patch.dict(
            "sys.modules",
            {"kr8s": MagicMock(), "kr8s.asyncio": mock_kr8s_asyncio},
        ):
            from aiperf.server_metrics.discovery.kubernetes import _get_api

            assert await _get_api() is None


# ---------------------------------------------------------------------------
# _list_running_pods
# ---------------------------------------------------------------------------
class TestListRunningPods:
    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        import kr8s

        mock_api = MagicMock()

        async def boom(*args, **kwargs):
            raise kr8s.ServerError("boom", response=MagicMock(status_code=500))

        mock_api.async_get = boom

        from aiperf.server_metrics.discovery.kubernetes import _list_running_pods

        pods = await _list_running_pods(mock_api, None, None)
        assert pods == []

    @pytest.mark.asyncio
    async def test_passes_field_selector_and_label_selector(self):
        mock_api = MagicMock()
        call_kwargs = {}

        async def capture(*args, **kwargs):
            call_kwargs.update(kwargs)
            return
            yield  # make it an async generator that yields nothing

        mock_api.async_get = capture

        from aiperf.server_metrics.discovery.kubernetes import _list_running_pods

        await _list_running_pods(mock_api, "test-ns", "app=vllm")

        assert call_kwargs["namespace"] == "test-ns"
        assert call_kwargs["field_selector"] == "status.phase=Running"
        assert call_kwargs["label_selector"] == "app=vllm"

    @pytest.mark.asyncio
    async def test_uses_kr8s_all_when_no_namespace(self):
        import kr8s

        mock_api = MagicMock()
        call_kwargs = {}

        async def capture(*args, **kwargs):
            call_kwargs.update(kwargs)
            return
            yield

        mock_api.async_get = capture

        from aiperf.server_metrics.discovery.kubernetes import _list_running_pods

        await _list_running_pods(mock_api, None, None)

        assert call_kwargs["namespace"] is kr8s.ALL
