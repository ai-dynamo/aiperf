# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes pod client helpers.

Focuses on:
- Pod summary tolerance for missing status and missing containerStatuses.
- Operator namespace discovery under duplicate installs and RBAC/API failures.
- Controller pod lookup selector shape, first-match semantics, and malformed phases.
- Retrievable pod phase gates for completed vs. still-running controller pods.

Out of scope:
- General client facade coverage; see tests/unit/kubernetes/test_client.py.
- Baseline client_pods happy paths; see tests/unit/kubernetes/test_client_pods.py.
"""

from __future__ import annotations

import logging
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
    find_controller_pod,
    find_operator_namespace,
    find_operator_pod,
    find_retrievable_pod,
    get_pod_summary,
    get_pods,
    resolve_operator_namespace,
)
from aiperf.kubernetes.client_selectors import controller_selector
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.models import PodSummary

# ============================================================
# Helpers
# ============================================================


def _container_status(
    *,
    name: str = "controller",
    ready: bool = True,
    restart_count: int | None = 0,
) -> V1ContainerStatus:
    """Build a container status shaped like the kubelet reports it."""
    return V1ContainerStatus(
        name=name,
        ready=ready,
        restart_count=restart_count,
        image="nvcr.io/nvidia/aiperf:adversarial",
        image_id="sha256:controller-adversarial",
        state={},
    )


def _pod(
    *,
    name: str = "aiperf-bench-7f2a-controller-0",
    namespace: str = "ml-lab",
    phase: str | None = "Running",
    statuses: list[V1ContainerStatus] | None = None,
    status: V1PodStatus | None = None,
) -> V1Pod:
    """Build a pod object with realistic AIPerf metadata."""
    if status is None:
        status = V1PodStatus(phase=phase, container_statuses=statuses)
    return V1Pod(
        metadata=V1ObjectMeta(name=name, namespace=namespace),
        status=status,
    )


def _pod_without_status(
    *,
    name: str = "aiperf-bench-7f2a-controller-no-status",
    namespace: str = "ml-lab",
) -> V1Pod:
    """Build a pod whose status object has not been populated by the apiserver yet."""
    return V1Pod(metadata=V1ObjectMeta(name=name, namespace=namespace), status=None)


def _pod_list(pods: list[V1Pod]) -> V1PodList:
    """Wrap pods in the Kubernetes list response object."""
    return V1PodList(items=pods)


def _api_exception(status: int, reason: str | None = None) -> ApiException:
    """Construct an ApiException with the supplied apiserver status."""
    return ApiException(status=status, reason=reason or f"apiserver-{status}")


# ============================================================
# Pod summary trust-boundary cases
# ============================================================


class TestGetPodSummaryAdversarial:
    """Summaries stay useful when kubelet status is partial or inconsistent."""

    @pytest.mark.asyncio
    async def test_get_pod_summary_partial_status_counts_total_without_false_ready(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        pods = [
            _pod_without_status(),
            _pod(name="aiperf-bench-7f2a-worker-empty", statuses=None),
            _pod(
                name="aiperf-bench-7f2a-controller-ready",
                statuses=[_container_status(ready=True, restart_count=2)],
            ),
            _pod(
                name="aiperf-bench-7f2a-worker-crashlooping",
                statuses=[_container_status(ready=False, restart_count=5)],
            ),
            _pod(
                name="aiperf-bench-7f2a-controller-complete",
                phase="Succeeded",
                statuses=[_container_status(ready=True, restart_count=3)],
            ),
        ]
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list(pods))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await get_pod_summary(api, "aiperf-bench-7f2a", "ml-lab")
        assert result == PodSummary(ready=1, total=5, restarts=10)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status",
        [
            param(403, id="forbidden-rbac-zero-summary"),
            param(404, id="namespace-or-jobset-gone-zero-summary"),
            param(500, id="apiserver-error-zero-summary"),
        ],
    )  # fmt: skip
    async def test_get_pod_summary_api_exception_returns_zero_summary(
        self,
        status: int,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(side_effect=_api_exception(status))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await get_pod_summary(api, "aiperf-bench-7f2a", "ml-lab")
        assert result == PodSummary(ready=0, total=0, restarts=0)


# ============================================================
# Operator pod and namespace discovery
# ============================================================


class TestFindOperatorNamespaceAdversarial:
    """Operator namespace auto-detection handles duplicates and RBAC limits."""

    @pytest.mark.asyncio
    async def test_find_operator_namespace_multiple_installs_warns_and_picks_first(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_pod_for_all_namespaces = AsyncMock(
            return_value=_pod_list(
                [
                    _pod(name="aiperf-operator-primary", namespace="aiperf-system"),
                    _pod(name="aiperf-operator-shadow", namespace="research-aiperf"),
                ]
            )
        )
        with (
            caplog.at_level(logging.WARNING, logger="aiperf.kubernetes.client_pods"),
            patch(
                "aiperf.kubernetes.client_pods.client.CoreV1Api",
                return_value=mock_core,
            ),
        ):
            result = await find_operator_namespace(api)
        assert result == "aiperf-system"
        assert "Multiple aiperf-operator installs detected" in caplog.text
        assert "research-aiperf" in caplog.text

    @pytest.mark.asyncio
    async def test_find_operator_namespace_404_propagates(self) -> None:
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_pod_for_all_namespaces = AsyncMock(
            side_effect=_api_exception(404, "pod resource not found")
        )
        with (
            patch(
                "aiperf.kubernetes.client_pods.client.CoreV1Api",
                return_value=mock_core,
            ),
            pytest.raises(ApiException) as exc_info,
        ):
            await find_operator_namespace(api)
        assert exc_info.value.status == 404

    @pytest.mark.asyncio
    async def test_resolve_operator_namespace_forbidden_uses_default(self) -> None:
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_pod_for_all_namespaces = AsyncMock(
            side_effect=_api_exception(403, "cluster-wide pods forbidden")
        )
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await resolve_operator_namespace(
                api,
                explicit=None,
                default="operator-fallback",
            )
        assert result == "operator-fallback"


class TestFindOperatorPodAdversarial:
    """Operator pod lookup keeps its first-match and phase contracts explicit."""

    @pytest.mark.asyncio
    async def test_find_operator_pod_multiple_matches_picks_first(self) -> None:
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            return_value=_pod_list(
                [
                    _pod(name="aiperf-operator-pending", phase="Pending"),
                    _pod(name="aiperf-operator-running", phase="Running"),
                ]
            )
        )
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_operator_pod(api, namespace="aiperf-system")
        assert result == ("aiperf-operator-pending", PodPhase.PENDING)

    @pytest.mark.asyncio
    async def test_find_operator_pod_lowercase_phase_normalizes(self) -> None:
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            return_value=_pod_list([_pod(name="aiperf-operator-0", phase="running")])
        )
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_operator_pod(api, namespace="aiperf-system")
        assert result == ("aiperf-operator-0", PodPhase.RUNNING)

    @pytest.mark.asyncio
    async def test_find_operator_pod_404_propagates(self) -> None:
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            side_effect=_api_exception(404, "operator namespace missing")
        )
        with (
            patch(
                "aiperf.kubernetes.client_pods.client.CoreV1Api",
                return_value=mock_core,
            ),
            pytest.raises(ApiException) as exc_info,
        ):
            await find_operator_pod(api, namespace="aiperf-system")
        assert exc_info.value.status == 404


# ============================================================
# Controller pod lookup and selector edges
# ============================================================


class TestFindControllerPodAdversarial:
    """Controller lookup is selector-driven and intentionally first-match."""

    @pytest.mark.asyncio
    async def test_find_controller_pod_uses_exact_controller_selector(self) -> None:
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([]))
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            await find_controller_pod(api, "ml-lab", "aiperf-bench-7f2a")
        args, kwargs = mock_core.list_namespaced_pod.call_args
        assert args == ("ml-lab",)
        assert kwargs == {
            "label_selector": controller_selector("aiperf-bench-7f2a"),
        }

    @pytest.mark.asyncio
    async def test_find_controller_pod_multiple_matches_picks_first_without_rescanning(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            return_value=_pod_list(
                [
                    _pod(name="aiperf-bench-7f2a-controller-old", phase="Pending"),
                    _pod(name="aiperf-bench-7f2a-controller-new", phase="Running"),
                ]
            )
        )
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_controller_pod(api, "ml-lab", "aiperf-bench-7f2a")
        assert result == ("aiperf-bench-7f2a-controller-old", PodPhase.PENDING)

    @pytest.mark.asyncio
    async def test_find_controller_pod_missing_status_returns_unknown(self) -> None:
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            return_value=_pod_list([_pod_without_status(name="controller-no-status")])
        )
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_controller_pod(api, "ml-lab", "aiperf-bench-7f2a")
        assert result == ("controller-no-status", PodPhase.UNKNOWN)

    @pytest.mark.asyncio
    async def test_find_controller_pod_kube_reason_phase_raises_value_error(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            return_value=_pod_list(
                [_pod(name="controller-evicted", phase="CrashLoopBackOff")]
            )
        )
        with (
            patch(
                "aiperf.kubernetes.client_pods.client.CoreV1Api",
                return_value=mock_core,
            ),
            pytest.raises(ValueError, match="CrashLoopBackOff"),
        ):
            await find_controller_pod(api, "ml-lab", "aiperf-bench-7f2a")


# ============================================================
# Retrievable pod phase semantics
# ============================================================


class TestFindRetrievablePodAdversarial:
    """Retrieval gates match kubectl-copy reality: Running or completed only."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phase,require_running,expected",
        [
            param("running", False, ("controller", PodPhase.RUNNING), id="lowercase-running-retrievable"),
            param("succeeded", False, ("controller", PodPhase.SUCCEEDED), id="lowercase-succeeded-retrievable"),
            param("succeeded", True, None, id="succeeded-rejected-when-running-required"),
            param("unknown", False, None, id="unknown-not-retrievable"),
        ],
    )  # fmt: skip
    async def test_find_retrievable_pod_phase_matrix_returns_expected(
        self,
        phase: str,
        require_running: bool,
        expected: tuple[str, PodPhase] | None,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            return_value=_pod_list([_pod(name="controller", phase=phase)])
        )
        with patch(
            "aiperf.kubernetes.client_pods.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_retrievable_pod(
                api,
                "ml-lab",
                "aiperf-bench-7f2a",
                require_running=require_running,
            )
        assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status",
        [
            param(403, id="forbidden-propagates"),
            param(404, id="pod-list-404-propagates"),
        ],
    )  # fmt: skip
    async def test_get_pods_api_exception_propagates(
        self,
        status: int,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(side_effect=_api_exception(status))
        with (
            patch(
                "aiperf.kubernetes.client_pods.client.CoreV1Api",
                return_value=mock_core,
            ),
            pytest.raises(ApiException) as exc_info,
        ):
            await get_pods(api, "ml-lab", controller_selector("aiperf-bench-7f2a"))
        assert exc_info.value.status == status
