# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.cli_commands.kube.debug module.

Focuses on:
- Pod problem detection: CrashLoopBackOff, ImagePullBackOff, OOMKilled, Unschedulable
- Namespace resolution: explicit, job-id, all-namespaces, fallback
- Event filtering and sorting
- Node resource extraction
- Verbose vs non-verbose log collection
- Report output structure
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.cli_commands.kube.debug import (
    _extract_pod_info,
    _get_event_severity_style,
    _get_namespace_events,
    _get_node_resources,
    _get_problem_pod_logs,
    _print_report,
)
from aiperf.kubernetes.models import JobSetInfo

# ============================================================
# Helpers
# ============================================================


def _make_pod(
    name: str = "test-pod",
    namespace: str = "default",
    phase: str = "Running",
    *,
    node: str = "node-1",
    container_statuses: list[dict[str, Any]] | None = None,
    init_container_statuses: list[dict[str, Any]] | None = None,
    conditions: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Create a mock Pod object exposing ``.name`` and ``.raw`` (legacy shape)."""
    pod = MagicMock()
    pod.name = name
    pod.raw = {
        "metadata": {"name": name, "namespace": namespace},
        "spec": {"nodeName": node},
        "status": {
            "phase": phase,
            "containerStatuses": container_statuses or [],
            "initContainerStatuses": init_container_statuses or [],
            "conditions": conditions or [],
        },
    }
    return pod


def _make_container_status(
    name: str = "main",
    restart_count: int = 0,
    waiting_reason: str = "",
    waiting_message: str = "",
    terminated_reason: str = "",
    terminated_message: str = "",
    last_terminated_reason: str = "",
) -> dict[str, Any]:
    """Build a container status dict."""
    cs: dict[str, Any] = {"name": name, "restartCount": restart_count, "state": {}}
    if waiting_reason:
        cs["state"]["waiting"] = {"reason": waiting_reason, "message": waiting_message}
    if terminated_reason:
        cs["state"]["terminated"] = {
            "reason": terminated_reason,
            "message": terminated_message,
        }
    if last_terminated_reason:
        cs["lastState"] = {
            "terminated": {"reason": last_terminated_reason, "message": ""}
        }
    return cs


def _make_event_raw(
    event_type: str = "Normal",
    reason: str = "Pulled",
    message: str = "Successfully pulled image",
    kind: str = "Pod",
    obj_name: str = "test-pod",
    count: int = 1,
    last_timestamp: str = "2026-03-11T10:00:00Z",
) -> dict[str, Any]:
    """Create a serialized event dict (as CoreV1Api would produce via sanitize)."""
    return {
        "type": event_type,
        "reason": reason,
        "message": message,
        "involvedObject": {"kind": kind, "name": obj_name},
        "count": count,
        "lastTimestamp": last_timestamp,
    }


def _make_node_raw(
    name: str = "node-1",
    *,
    ready: bool = True,
    cpu_capacity: str = "16",
    memory_capacity: str = "64Gi",
    gpu_capacity: str = "4",
    cpu_allocatable: str = "15",
    memory_allocatable: str = "62Gi",
    gpu_allocatable: str = "4",
    pressure_conditions: list[str] | None = None,
) -> dict[str, Any]:
    """Create a serialized node dict (as CoreV1Api would produce via sanitize)."""
    conditions = [
        {
            "type": "Ready",
            "status": "True" if ready else "False",
        }
    ]
    for ptype in pressure_conditions or []:
        conditions.append({"type": ptype, "status": "True"})

    return {
        "metadata": {"name": name},
        "status": {
            "capacity": {
                "cpu": cpu_capacity,
                "memory": memory_capacity,
                "nvidia.com/gpu": gpu_capacity,
            },
            "allocatable": {
                "cpu": cpu_allocatable,
                "memory": memory_allocatable,
                "nvidia.com/gpu": gpu_allocatable,
            },
            "conditions": conditions,
        },
    }


def _mock_listing_response(raws: list[dict[str, Any]]) -> MagicMock:
    """Build a CoreV1Api list response where each item.sanitize yields the raw dict.

    Since ``_get_namespace_events`` / ``_get_node_resources`` call
    ``ApiClient().sanitize_for_serialization(item)`` on each item, we patch
    ``sanitize_for_serialization`` at the test level to pass-through raws.
    """
    response = MagicMock()
    response.items = list(raws)
    return response


@asynccontextmanager
async def _fake_k8s_client(api: Any):
    """Drop-in replacement for ``k8s_client`` yielding the provided api object."""
    yield api


# ============================================================
# _extract_pod_info Tests
# ============================================================


class TestExtractPodInfo:
    """Verify pod info extraction and problem detection."""

    def test_healthy_running_pod_no_problems(self) -> None:
        pod = _make_pod(
            name="healthy-pod",
            phase="Running",
            container_statuses=[_make_container_status("main")],
        )
        info = _extract_pod_info(pod)

        assert info["name"] == "healthy-pod"
        assert info["phase"] == "Running"
        assert info["restarts"] == 0
        assert info["problems"] == []
        assert info["node"] == "node-1"

    def test_crashloopbackoff_detected_as_critical(self) -> None:
        pod = _make_pod(
            phase="Running",
            container_statuses=[
                _make_container_status(
                    "main",
                    restart_count=5,
                    waiting_reason="CrashLoopBackOff",
                    waiting_message="back-off 5m0s",
                )
            ],
        )
        info = _extract_pod_info(pod)

        assert len(info["problems"]) == 1
        problem = info["problems"][0]
        assert problem["severity"] == "CRITICAL"
        assert problem["state"] == "CrashLoopBackOff"
        assert problem["container"] == "main"
        assert "crash-looping" in problem["suggestion"]
        assert info["restarts"] == 5

    def test_imagepullbackoff_detected_as_critical(self) -> None:
        pod = _make_pod(
            phase="Pending",
            container_statuses=[
                _make_container_status("main", waiting_reason="ImagePullBackOff")
            ],
        )
        info = _extract_pod_info(pod)

        assert len(info["problems"]) == 1
        assert info["problems"][0]["severity"] == "CRITICAL"
        assert info["problems"][0]["state"] == "ImagePullBackOff"

    def test_errimagepull_detected_as_critical(self) -> None:
        pod = _make_pod(
            phase="Pending",
            container_statuses=[
                _make_container_status("main", waiting_reason="ErrImagePull")
            ],
        )
        info = _extract_pod_info(pod)

        assert len(info["problems"]) == 1
        assert info["problems"][0]["severity"] == "CRITICAL"
        assert info["problems"][0]["state"] == "ErrImagePull"

    def test_oomkilled_terminated_state(self) -> None:
        pod = _make_pod(
            phase="Running",
            container_statuses=[
                _make_container_status("main", terminated_reason="OOMKilled")
            ],
        )
        info = _extract_pod_info(pod)

        assert len(info["problems"]) == 1
        assert info["problems"][0]["severity"] == "CRITICAL"
        assert info["problems"][0]["state"] == "OOMKilled"
        assert "memory" in info["problems"][0]["suggestion"].lower()

    def test_oomkilled_last_terminated_state(self) -> None:
        pod = _make_pod(
            phase="Running",
            container_statuses=[
                _make_container_status("main", last_terminated_reason="OOMKilled")
            ],
        )
        info = _extract_pod_info(pod)

        assert len(info["problems"]) == 1
        assert info["problems"][0]["state"] == "OOMKilled (previous)"
        assert info["problems"][0]["severity"] == "CRITICAL"

    def test_unschedulable_condition_detected(self) -> None:
        pod = _make_pod(
            phase="Pending",
            conditions=[
                {
                    "type": "PodScheduled",
                    "status": "False",
                    "reason": "Unschedulable",
                    "message": "0/3 nodes available: insufficient nvidia.com/gpu",
                }
            ],
        )
        info = _extract_pod_info(pod)

        assert len(info["problems"]) == 1
        problem = info["problems"][0]
        assert problem["severity"] == "CRITICAL"
        assert problem["state"] == "Unschedulable"
        assert problem["container"] == "-"
        assert "insufficient nvidia.com/gpu" in problem["message"]

    def test_pending_unknown_waiting_reason_is_warning(self) -> None:
        """Unknown waiting reasons on Pending pods produce WARNING severity."""
        pod = _make_pod(
            phase="Pending",
            container_statuses=[
                _make_container_status("main", waiting_reason="ContainerCreating")
            ],
        )
        info = _extract_pod_info(pod)

        assert len(info["problems"]) == 1
        assert info["problems"][0]["severity"] == "WARNING"
        assert info["problems"][0]["state"] == "ContainerCreating"

    def test_unknown_waiting_reason_on_running_pod_no_problem(self) -> None:
        """Unknown waiting reasons on non-Pending pods are ignored."""
        pod = _make_pod(
            phase="Running",
            container_statuses=[
                _make_container_status("main", waiting_reason="SomeUnknownReason")
            ],
        )
        info = _extract_pod_info(pod)
        assert info["problems"] == []

    def test_init_container_problems_included(self) -> None:
        pod = _make_pod(
            phase="Pending",
            init_container_statuses=[
                _make_container_status(
                    "init-setup",
                    restart_count=3,
                    waiting_reason="CrashLoopBackOff",
                )
            ],
        )
        info = _extract_pod_info(pod)

        assert len(info["problems"]) == 1
        assert info["problems"][0]["container"] == "init-setup"
        assert info["restarts"] == 3

    def test_multiple_problems_in_single_pod(self) -> None:
        pod = _make_pod(
            phase="Pending",
            container_statuses=[
                _make_container_status("gpu-worker", waiting_reason="ImagePullBackOff"),
                _make_container_status("sidecar", terminated_reason="OOMKilled"),
            ],
            conditions=[
                {
                    "type": "PodScheduled",
                    "status": "False",
                    "reason": "Unschedulable",
                    "message": "No GPU nodes",
                }
            ],
        )
        info = _extract_pod_info(pod)

        assert len(info["problems"]) == 3
        states = {p["state"] for p in info["problems"]}
        assert states == {"ImagePullBackOff", "OOMKilled", "Unschedulable"}

    def test_restarts_summed_across_all_containers(self) -> None:
        pod = _make_pod(
            phase="Running",
            init_container_statuses=[_make_container_status("init", restart_count=2)],
            container_statuses=[
                _make_container_status("main", restart_count=3),
                _make_container_status("sidecar", restart_count=1),
            ],
        )
        info = _extract_pod_info(pod)
        assert info["restarts"] == 6

    def test_missing_status_fields_handled(self) -> None:
        """Pod with minimal/empty status doesn't crash."""
        pod = MagicMock()
        pod.name = "bare-pod"
        pod.raw = {"metadata": {}, "spec": {}, "status": {}}

        info = _extract_pod_info(pod)
        assert info["name"] == "bare-pod"
        assert info["phase"] == "Unknown"
        assert info["problems"] == []
        assert info["node"] == ""
        assert info["namespace"] == ""


# ============================================================
# _get_event_severity_style Tests
# ============================================================


class TestGetEventSeverityStyle:
    """Verify Rich style mapping for event types."""

    @pytest.mark.parametrize(
        "event_type,expected_style",
        [
            param("Warning", "yellow", id="warning"),
            param("Normal", "dim", id="normal"),
            param("", "dim", id="empty"),
        ],
    )  # fmt: skip
    def test_event_type_maps_to_correct_style(
        self, event_type: str, expected_style: str
    ) -> None:
        assert _get_event_severity_style(event_type) == expected_style


# ============================================================
# _get_namespace_events Tests
# ============================================================


class TestGetNamespaceEvents:
    """Verify event fetching and sorting."""

    @pytest.mark.asyncio
    async def test_events_sorted_newest_first(self) -> None:
        raws = [
            _make_event_raw(last_timestamp="2026-03-11T09:00:00Z", reason="Old"),
            _make_event_raw(last_timestamp="2026-03-11T11:00:00Z", reason="New"),
            _make_event_raw(last_timestamp="2026-03-11T10:00:00Z", reason="Mid"),
        ]
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_event = AsyncMock(
            return_value=_mock_listing_response(raws)
        )

        with (
            patch(
                "kubernetes_asyncio.client.CoreV1Api",
                return_value=mock_core,
            ),
            patch(
                "kubernetes_asyncio.client.ApiClient.sanitize_for_serialization",
                side_effect=lambda o: o,
            ),
        ):
            result = await _get_namespace_events(api, "default")

        assert len(result) == 3
        assert result[0]["reason"] == "New"
        assert result[1]["reason"] == "Mid"
        assert result[2]["reason"] == "Old"

    @pytest.mark.asyncio
    async def test_events_include_involved_object(self) -> None:
        raws = [_make_event_raw(kind="Pod", obj_name="my-pod")]
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_event = AsyncMock(
            return_value=_mock_listing_response(raws)
        )

        with (
            patch(
                "kubernetes_asyncio.client.CoreV1Api",
                return_value=mock_core,
            ),
            patch(
                "kubernetes_asyncio.client.ApiClient.sanitize_for_serialization",
                side_effect=lambda o: o,
            ),
        ):
            result = await _get_namespace_events(api, "default")

        assert result[0]["object"] == "Pod/my-pod"
        assert result[0]["count"] == 1

    @pytest.mark.asyncio
    async def test_api_error_returns_empty_list(self) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_event = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )

        with patch(
            "kubernetes_asyncio.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await _get_namespace_events(api, "default")
        assert result == []


# ============================================================
# _get_node_resources Tests
# ============================================================


class TestGetNodeResources:
    """Verify node resource extraction."""

    @pytest.mark.asyncio
    async def test_healthy_node_resources(self) -> None:
        raws = [_make_node_raw("gpu-node-1", gpu_capacity="8", gpu_allocatable="6")]
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_node = AsyncMock(return_value=_mock_listing_response(raws))

        with (
            patch(
                "kubernetes_asyncio.client.CoreV1Api",
                return_value=mock_core,
            ),
            patch(
                "kubernetes_asyncio.client.ApiClient.sanitize_for_serialization",
                side_effect=lambda o: o,
            ),
        ):
            result = await _get_node_resources(api)

        assert len(result) == 1
        node = result[0]
        assert node["name"] == "gpu-node-1"
        assert node["ready"] is True
        assert node["gpu_capacity"] == "8"
        assert node["gpu_allocatable"] == "6"
        assert node["pressure"] == []

    @pytest.mark.asyncio
    async def test_node_with_pressure_conditions(self) -> None:
        raws = [
            _make_node_raw(
                "stressed-node",
                pressure_conditions=["MemoryPressure", "DiskPressure"],
            )
        ]
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_node = AsyncMock(return_value=_mock_listing_response(raws))

        with (
            patch(
                "kubernetes_asyncio.client.CoreV1Api",
                return_value=mock_core,
            ),
            patch(
                "kubernetes_asyncio.client.ApiClient.sanitize_for_serialization",
                side_effect=lambda o: o,
            ),
        ):
            result = await _get_node_resources(api)

        assert result[0]["pressure"] == ["MemoryPressure", "DiskPressure"]

    @pytest.mark.asyncio
    async def test_not_ready_node(self) -> None:
        raws = [_make_node_raw("down-node", ready=False)]
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_node = AsyncMock(return_value=_mock_listing_response(raws))

        with (
            patch(
                "kubernetes_asyncio.client.CoreV1Api",
                return_value=mock_core,
            ),
            patch(
                "kubernetes_asyncio.client.ApiClient.sanitize_for_serialization",
                side_effect=lambda o: o,
            ),
        ):
            result = await _get_node_resources(api)
        assert result[0]["ready"] is False

    @pytest.mark.asyncio
    async def test_api_error_returns_empty_list(self) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_node = AsyncMock(side_effect=RuntimeError("timeout"))

        with patch(
            "kubernetes_asyncio.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await _get_node_resources(api)
        assert result == []


# ============================================================
# _get_problem_pod_logs Tests
# ============================================================


class TestGetProblemPodLogs:
    """Verify log fetching for pods with problems."""

    @pytest.mark.asyncio
    async def test_fetches_logs_from_problem_pods_only(self) -> None:
        healthy_pod = _make_pod(
            "healthy",
            namespace="ns",
            container_statuses=[_make_container_status("main")],
        )
        problem_pod = _make_pod(
            "broken",
            namespace="ns",
            container_statuses=[
                _make_container_status("main", waiting_reason="CrashLoopBackOff")
            ],
        )
        pod_infos = [_extract_pod_info(healthy_pod), _extract_pod_info(problem_pod)]

        api = MagicMock()
        mock_core = MagicMock()
        mock_core.read_namespaced_pod_log = AsyncMock(
            return_value="ERROR: segfault\nat line 42\n"
        )

        with patch(
            "kubernetes_asyncio.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await _get_problem_pod_logs(api, pod_infos, tail_lines=20)

        assert "broken" in result
        assert "healthy" not in result
        assert result["broken"]["main"] == "ERROR: segfault\nat line 42"
        # Verify the call was for the problem pod's container.
        mock_core.read_namespaced_pod_log.assert_called_once()
        call_kwargs = mock_core.read_namespaced_pod_log.call_args.kwargs
        assert call_kwargs["name"] == "broken"
        assert call_kwargs["container"] == "main"
        assert call_kwargs["tail_lines"] == 20

    @pytest.mark.asyncio
    async def test_api_exception_returns_unavailable(self) -> None:
        from kubernetes_asyncio.client.exceptions import ApiException

        pod = _make_pod(
            "err-pod",
            namespace="ns",
            container_statuses=[
                _make_container_status("main", waiting_reason="CrashLoopBackOff")
            ],
        )
        pod_infos = [_extract_pod_info(pod)]

        api = MagicMock()
        mock_core = MagicMock()
        mock_core.read_namespaced_pod_log = AsyncMock(
            side_effect=ApiException(status=500, reason="Internal")
        )

        with patch(
            "kubernetes_asyncio.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await _get_problem_pod_logs(api, pod_infos)

        assert result["err-pod"]["main"] == "<logs unavailable>"

    @pytest.mark.asyncio
    async def test_generic_error_returns_error_message(self) -> None:
        pod = _make_pod(
            "err-pod",
            namespace="ns",
            container_statuses=[
                _make_container_status("main", waiting_reason="CrashLoopBackOff")
            ],
        )
        pod_infos = [_extract_pod_info(pod)]

        api = MagicMock()
        mock_core = MagicMock()
        mock_core.read_namespaced_pod_log = AsyncMock(
            side_effect=RuntimeError("unexpected")
        )

        with patch(
            "kubernetes_asyncio.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await _get_problem_pod_logs(api, pod_infos)

        assert result["err-pod"]["main"] == "<error fetching logs>"


# ============================================================
# _print_report Tests
# ============================================================


class TestPrintReport:
    """Verify report output structure."""

    def test_no_pods_prints_warning(self) -> None:
        with (
            patch("aiperf.kubernetes.console.console"),
            patch("aiperf.kubernetes.console.print_warning") as mock_warn,
            patch("aiperf.kubernetes.console.print_header"),
            patch("aiperf.kubernetes.console.print_info"),
        ):
            _print_report(
                "default",
                pod_infos=[],
                events=[],
                node_resources=[],
                pod_logs={},
                verbose=False,
            )
            mock_warn.assert_any_call("No pods found")

    def test_no_problems_prints_success(self) -> None:
        pod = _make_pod("ok-pod", container_statuses=[_make_container_status("main")])
        pod_infos = [_extract_pod_info(pod)]

        with (
            patch("aiperf.kubernetes.console.console"),
            patch("aiperf.kubernetes.console.print_success") as mock_success,
            patch("aiperf.kubernetes.console.print_header"),
            patch("aiperf.kubernetes.console.print_info"),
        ):
            _print_report(
                "default",
                pod_infos=pod_infos,
                events=[],
                node_resources=[],
                pod_logs={},
                verbose=False,
            )
            mock_success.assert_called_once_with("No problems detected")

    def test_verbose_shows_pod_logs(self) -> None:
        pod = _make_pod(
            "crash-pod",
            container_statuses=[
                _make_container_status("main", waiting_reason="CrashLoopBackOff")
            ],
        )
        pod_infos = [_extract_pod_info(pod)]
        pod_logs = {"crash-pod": {"main": "Traceback: error at line 1"}}

        with (
            patch("aiperf.kubernetes.console.console"),
            patch("aiperf.kubernetes.console.print_header") as mock_header,
            patch("aiperf.kubernetes.console.print_info"),
            patch("aiperf.kubernetes.console.print_error"),
        ):
            _print_report(
                "default",
                pod_infos=pod_infos,
                events=[],
                node_resources=[],
                pod_logs=pod_logs,
                verbose=True,
            )
            header_calls = [str(c) for c in mock_header.call_args_list]
            assert any("Problem Pod Logs" in c for c in header_calls)

    def test_non_verbose_skips_logs_section(self) -> None:
        pod = _make_pod(
            "crash-pod",
            container_statuses=[
                _make_container_status("main", waiting_reason="CrashLoopBackOff")
            ],
        )
        pod_infos = [_extract_pod_info(pod)]
        pod_logs = {"crash-pod": {"main": "some logs"}}

        with (
            patch("aiperf.kubernetes.console.console"),
            patch("aiperf.kubernetes.console.print_header") as mock_header,
            patch("aiperf.kubernetes.console.print_info"),
            patch("aiperf.kubernetes.console.print_error"),
        ):
            _print_report(
                "default",
                pod_infos=pod_infos,
                events=[],
                node_resources=[],
                pod_logs=pod_logs,
                verbose=False,
            )
            header_calls = [str(c) for c in mock_header.call_args_list]
            assert not any("Problem Pod Logs" in c for c in header_calls)

    def test_warning_events_shown_in_non_verbose(self) -> None:
        events = [
            {
                "type": "Warning",
                "reason": "BackOff",
                "message": "Restarting",
                "object": "Pod/x",
                "count": 3,
                "last_seen": "2026-03-11T10:00:00Z",
            },
            {
                "type": "Normal",
                "reason": "Pulled",
                "message": "Pulled image",
                "object": "Pod/x",
                "count": 1,
                "last_seen": "2026-03-11T09:00:00Z",
            },
        ]

        with (
            patch("aiperf.kubernetes.console.console"),
            patch("aiperf.kubernetes.console.print_header") as mock_header,
            patch("aiperf.kubernetes.console.print_info"),
            patch("aiperf.kubernetes.console.print_success"),
            patch("aiperf.kubernetes.console.print_warning"),
        ):
            _print_report(
                "default",
                pod_infos=[],
                events=events,
                node_resources=[],
                pod_logs={},
                verbose=False,
            )
            header_calls = [str(c) for c in mock_header.call_args_list]
            assert any("Warning Events" in c for c in header_calls)


# ============================================================
# debug() Command Tests
# ============================================================


class TestDebugCommand:
    """Verify the debug() command orchestration."""

    @pytest.mark.asyncio
    async def test_explicit_namespace_collects_diagnostics(self) -> None:
        from aiperf.cli_commands.kube.debug import debug

        api = MagicMock()

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.get_pods",
                new=AsyncMock(return_value=[]),
            ) as mock_get_pods,
            patch(
                "aiperf.cli_commands.kube.debug._get_node_resources",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_namespace_events",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.cli_commands.kube.debug._print_report") as mock_report,
        ):
            await debug(namespace="my-ns")

            mock_get_pods.assert_called_once()
            call_args = mock_get_pods.call_args
            # get_pods(api, namespace, label_selector) positional
            assert call_args[0][1] == "my-ns"
            mock_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_job_id_resolves_namespace(self) -> None:
        from aiperf.cli_commands.kube.debug import debug

        api = MagicMock()
        jobset = JobSetInfo(
            name="aiperf-abc123",
            namespace="bench-ns",
            jobset={},
            status="Running",
        )

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=jobset),
            ) as mock_find,
            patch(
                "aiperf.kubernetes.client.get_pods",
                new=AsyncMock(return_value=[]),
            ) as mock_get_pods,
            patch(
                "aiperf.cli_commands.kube.debug._get_node_resources",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_namespace_events",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.cli_commands.kube.debug._print_report"),
        ):
            await debug(job_id="abc123")

            mock_find.assert_called_once_with(api, "abc123", None)
            call_ns = mock_get_pods.call_args[0][1]
            assert call_ns == "bench-ns"

    @pytest.mark.asyncio
    async def test_job_id_not_found_prints_error_and_returns(self) -> None:
        from aiperf.cli_commands.kube.debug import debug

        api = MagicMock()

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=None),
            ),
            patch("aiperf.kubernetes.console.print_error") as mock_print_error,
            patch("aiperf.cli_commands.kube.debug._print_report") as mock_report,
        ):
            await debug(job_id="nonexistent")

            mock_print_error.assert_called_once()
            assert "nonexistent" in mock_print_error.call_args[0][0]
            mock_report.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_namespaces_collects_from_multiple(self) -> None:
        from aiperf.cli_commands.kube.debug import debug

        api = MagicMock()
        jobsets = [
            JobSetInfo(name="js1", namespace="ns-a", jobset={}, status="Running"),
            JobSetInfo(name="js2", namespace="ns-b", jobset={}, status="Running"),
        ]

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.list_jobsets",
                new=AsyncMock(return_value=jobsets),
            ),
            patch(
                "aiperf.kubernetes.client.get_pods",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_node_resources",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_namespace_events",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.cli_commands.kube.debug._print_report") as mock_report,
        ):
            await debug(all_namespaces=True)

            assert mock_report.call_count == 2
            reported_ns = sorted(c[0][0] for c in mock_report.call_args_list)
            assert reported_ns == ["ns-a", "ns-b"]

    @pytest.mark.asyncio
    async def test_all_namespaces_no_deployments_warns(self) -> None:
        from aiperf.cli_commands.kube.debug import debug

        api = MagicMock()

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.list_jobsets",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.kubernetes.console.print_warning") as mock_warn,
            patch("aiperf.cli_commands.kube.debug._print_report") as mock_report,
        ):
            await debug(all_namespaces=True)

            mock_warn.assert_called_once()
            mock_report.assert_not_called()

    @pytest.mark.asyncio
    async def test_verbose_fetches_problem_pod_logs(self) -> None:
        from aiperf.cli_commands.kube.debug import debug

        crash_pod = _make_pod(
            "crash",
            container_statuses=[
                _make_container_status("main", waiting_reason="CrashLoopBackOff")
            ],
        )
        api = MagicMock()

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.get_pods",
                new=AsyncMock(return_value=[crash_pod]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_node_resources",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_namespace_events",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_problem_pod_logs",
                new=AsyncMock(return_value={"crash": {"main": "logs"}}),
            ) as mock_logs,
            patch("aiperf.cli_commands.kube.debug._print_report") as mock_report,
        ):
            await debug(namespace="test-ns", verbose=True)

            mock_logs.assert_called_once()
            report_kwargs = mock_report.call_args
            pod_logs = report_kwargs.kwargs["pod_logs"]
            assert pod_logs == {"crash": {"main": "logs"}}

    @pytest.mark.asyncio
    async def test_non_verbose_skips_log_fetching(self) -> None:
        from aiperf.cli_commands.kube.debug import debug

        api = MagicMock()

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.get_pods",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_node_resources",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_namespace_events",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.cli_commands.kube.debug._get_problem_pod_logs") as mock_logs,
            patch("aiperf.cli_commands.kube.debug._print_report"),
        ):
            await debug(namespace="test-ns", verbose=False)

            mock_logs.assert_not_called()

    @pytest.mark.asyncio
    async def test_job_id_uses_job_selector(self) -> None:
        from aiperf.cli_commands.kube.debug import debug

        api = MagicMock()
        jobset = JobSetInfo(
            name="aiperf-xyz", namespace="bench-ns", jobset={}, status="Running"
        )

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=jobset),
            ),
            patch(
                "aiperf.kubernetes.client.get_pods",
                new=AsyncMock(return_value=[]),
            ) as mock_get_pods,
            patch(
                "aiperf.cli_commands.kube.debug._get_node_resources",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_namespace_events",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.cli_commands.kube.debug._print_report"),
        ):
            await debug(job_id="xyz")

            selector = mock_get_pods.call_args[0][2]
            assert "job-id" in selector
