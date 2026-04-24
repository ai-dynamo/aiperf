# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.client_pods — gaps not covered in test_client.py.

Focuses on:

- get_pods (not covered anywhere else)
- find_controller_pod selector string + ApiException propagation
- find_retrievable_pod when find_controller_pod returns None
- get_pod_summary pods with None status / None container_statuses
- get_pod_summary label_selector passed to the API
- cluster_version ApiException propagation
- find_operator_pod passes default label_selector + namespace
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException
from kubernetes_asyncio.client.models import (
    V1ContainerStatus,
    V1ObjectMeta,
    V1Pod,
    V1PodList,
    V1PodStatus,
)
from pytest import param

from aiperf.kubernetes.client_pods import (
    cluster_version,
    find_controller_pod,
    find_operator_pod,
    find_retrievable_pod,
    get_pod_summary,
    get_pods,
)
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.models import PodSummary


def _make_v1pod(
    name: str = "pod-0",
    namespace: str = "default",
    phase: str | None = "Running",
    container_statuses: list[dict[str, Any]] | None = None,
    status: V1PodStatus | None = None,
) -> V1Pod:
    """Build a V1Pod for list_namespaced_pod mocking."""
    if status is not None:
        return V1Pod(
            metadata=V1ObjectMeta(name=name, namespace=namespace), status=status
        )
    if container_statuses is None:
        container_statuses = [{"name": "c", "ready": True, "restart_count": 0}]
    css = [
        V1ContainerStatus(
            name=cs["name"],
            ready=cs["ready"],
            restart_count=cs.get("restart_count", 0),
            image="x",
            image_id="y",
            state={},
        )
        for cs in container_statuses
    ]
    return V1Pod(
        metadata=V1ObjectMeta(name=name, namespace=namespace),
        status=V1PodStatus(phase=phase, container_statuses=css or None),
    )


def _pod_list(pods: list[V1Pod]) -> V1PodList:
    """Wrap pods in a V1PodList."""
    return V1PodList(items=pods)


def _api_exception(status: int) -> ApiException:
    """Construct an ApiException with the given HTTP status code."""
    return ApiException(status=status, reason=f"err-{status}")


class TestGetPods:
    """Verify the typed get_pods wrapper."""

    @pytest.mark.asyncio
    async def test_returns_pod_items(self) -> None:
        """Returns the .items attribute of the pod list."""
        api = MagicMock(spec=ApiClient)
        pods = [_make_v1pod(name="a"), _make_v1pod(name="b")]
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list(pods))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await get_pods(api, "ns", "app=aiperf")
        assert [p.metadata.name for p in result] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_passes_namespace_and_label_selector(self) -> None:
        """Forwards namespace positionally and label_selector as kwarg."""
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([]))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            await get_pods(api, "my-ns", "foo=bar")
        args, kwargs = mock_core.list_namespaced_pod.call_args
        assert args == ("my-ns",)
        assert kwargs == {"label_selector": "foo=bar"}

    @pytest.mark.asyncio
    async def test_api_exception_propagates(self) -> None:
        """Errors from the k8s client bubble up (not suppressed)."""
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(side_effect=_api_exception(500))
        with (
            patch(
                "aiperf.kubernetes.client_pods.client.CoreV1Api",
                return_value=mock_core,
            ),
            pytest.raises(ApiException),
        ):
            await get_pods(api, "ns", "app=aiperf")

    @pytest.mark.asyncio
    async def test_empty_list_when_no_matches(self) -> None:
        """An empty pod list yields an empty result."""
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([]))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await get_pods(api, "ns", "app=aiperf")
        assert result == []


class TestGetPodSummaryEdges:
    """Edge cases for the pod readiness summary."""

    @pytest.mark.asyncio
    async def test_passes_jobset_name_label_selector(self) -> None:
        """Filters on the JobSetLabels.JOBSET_NAME label."""
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([]))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            await get_pod_summary(api, "my-js", "bench-ns")
        args, kwargs = mock_core.list_namespaced_pod.call_args
        assert args == ("bench-ns",)
        assert kwargs["label_selector"] == "jobset.sigs.k8s.io/jobset-name=my-js"

    @pytest.mark.asyncio
    async def test_pod_without_status_not_counted_ready(self) -> None:
        """A pod with status=None is counted in total but not ready."""
        api = MagicMock(spec=ApiClient)
        pod = V1Pod(metadata=V1ObjectMeta(name="p", namespace="default"), status=None)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([pod]))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await get_pod_summary(api, "js", "default")
        assert result == PodSummary(ready=0, total=1, restarts=0)

    @pytest.mark.asyncio
    async def test_pod_without_container_statuses_not_ready(self) -> None:
        """A Running pod with no container_statuses is not counted ready."""
        api = MagicMock(spec=ApiClient)
        pod = _make_v1pod(
            name="p", status=V1PodStatus(phase="Running", container_statuses=None)
        )
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([pod]))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await get_pod_summary(api, "js", "default")
        assert result == PodSummary(ready=0, total=1, restarts=0)

    @pytest.mark.asyncio
    async def test_running_pod_not_ready_when_any_container_not_ready(self) -> None:
        """All container statuses must be ready for the pod to be counted ready."""
        api = MagicMock(spec=ApiClient)
        pod = _make_v1pod(
            name="p",
            phase="Running",
            container_statuses=[
                {"name": "c1", "ready": True, "restart_count": 0},
                {"name": "c2", "ready": False, "restart_count": 5},
            ],
        )
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([pod]))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await get_pod_summary(api, "js", "default")
        assert result == PodSummary(ready=0, total=1, restarts=5)

    @pytest.mark.asyncio
    async def test_restart_count_none_treated_as_zero(self) -> None:
        """None restart_count values (rare) do not crash the sum."""
        api = MagicMock(spec=ApiClient)
        # V1ContainerStatus's model-side validation rejects None, but the code
        # guards with ``cs.restart_count or 0`` for defensive reasons. Use a
        # bare MagicMock to exercise that fallback path directly.
        fake_cs = MagicMock()
        fake_cs.ready = True
        fake_cs.restart_count = None
        fake_status = MagicMock()
        fake_status.phase = "Running"
        fake_status.container_statuses = [fake_cs]
        fake_pod = MagicMock()
        fake_pod.status = fake_status
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            return_value=MagicMock(items=[fake_pod])
        )
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await get_pod_summary(api, "js", "default")
        assert result == PodSummary(ready=1, total=1, restarts=0)


class TestFindOperatorPodSelectorArgs:
    """Verify find_operator_pod passes its default namespace + label_selector."""

    @pytest.mark.asyncio
    async def test_default_namespace_and_selector(self) -> None:
        """Default namespace 'aiperf-system' and operator label_selector."""
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([]))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_operator_pod(api)
        args, kwargs = mock_core.list_namespaced_pod.call_args
        assert args == ("aiperf-system",)
        assert kwargs["label_selector"] == "app.kubernetes.io/name=aiperf-operator"
        assert result is None

    @pytest.mark.asyncio
    async def test_custom_namespace_and_selector_override(self) -> None:
        """Caller overrides for both namespace and selector are honored."""
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([]))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            await find_operator_pod(api, namespace="ops", label_selector="a=b")
        args, kwargs = mock_core.list_namespaced_pod.call_args
        assert args == ("ops",)
        assert kwargs["label_selector"] == "a=b"


class TestFindControllerPodErrorPath:
    """Verify find_controller_pod surfaces ApiException + uses controller selector."""

    @pytest.mark.asyncio
    async def test_api_exception_propagates(self) -> None:
        """Any ApiException is not suppressed (callers decide)."""
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(side_effect=_api_exception(500))
        with (
            patch(
                "aiperf.kubernetes.client_pods.client.CoreV1Api",
                return_value=mock_core,
            ),
            pytest.raises(ApiException),
        ):
            await find_controller_pod(api, "ns", "j-1")

    @pytest.mark.asyncio
    async def test_uses_controller_selector(self) -> None:
        """Selector includes the replicatedjob-name=controller filter."""
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([]))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            await find_controller_pod(api, "ns", "j-1")
        selector = mock_core.list_namespaced_pod.call_args.kwargs["label_selector"]
        assert "jobset.sigs.k8s.io/replicatedjob-name=controller" in selector
        assert "aiperf.nvidia.com/job-id=j-1" in selector

    @pytest.mark.asyncio
    async def test_picks_first_pod_when_multiple(self) -> None:
        """Returns the first pod when the selector matches several."""
        api = MagicMock(spec=ApiClient)
        pods = [
            _make_v1pod(name="first", phase="Running"),
            _make_v1pod(name="second", phase="Running"),
        ]
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list(pods))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_controller_pod(api, "ns", "j-1")
        assert result == ("first", PodPhase.RUNNING)


class TestFindRetrievablePodNoPod:
    """Verify find_retrievable_pod short-circuits when no pod exists."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "require_running",
        [
            param(True, id="require_running_true"),
            param(False, id="require_running_false"),
        ],
    )  # fmt: skip
    async def test_no_pod_returns_none(self, require_running: bool) -> None:
        """Regardless of require_running, no pod -> None."""
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([]))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_retrievable_pod(
                api, "ns", "j-1", require_running=require_running
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_unknown_phase_not_retrievable(self) -> None:
        """PodPhase.UNKNOWN is not retrievable (no status.phase)."""
        api = MagicMock(spec=ApiClient)
        pod = V1Pod(
            metadata=V1ObjectMeta(name="ctrl", namespace="ns"),
            status=V1PodStatus(),  # no phase
        )
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([pod]))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_retrievable_pod(api, "ns", "j-1")
        assert result is None


class TestClusterVersionPropagatesError:
    """Verify cluster_version does not suppress ApiException."""

    @pytest.mark.asyncio
    async def test_api_exception_propagates(self) -> None:
        """Errors from VersionApi.get_code bubble up unchanged."""
        api = MagicMock(spec=ApiClient)
        mock_version_api = MagicMock()
        mock_version_api.get_code = AsyncMock(side_effect=_api_exception(500))
        with (
            patch(
                "aiperf.kubernetes.client_pods.client.VersionApi",
                return_value=mock_version_api,
            ),
            pytest.raises(ApiException),
        ):
            await cluster_version(api)
