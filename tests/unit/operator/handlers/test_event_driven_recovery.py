# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for watch-driven AIPerfJob recovery."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import kopf
import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.constants import Annotations, Containers
from aiperf.kubernetes.cr_refs import AIPERF_JOB_API_VERSION
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.phase import Phase
from aiperf.operator import client_cache, main
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.handlers import monitor, pod_restarts
from aiperf.operator.handlers._job_identity import StaleAIPerfJobCallback
from aiperf.operator.status import StatusBuilder


def _timestamp(*, seconds_ago: float = 0) -> str:
    return (
        (datetime.now(UTC) - timedelta(seconds=seconds_ago))
        .isoformat()
        .replace("+00:00", "Z")
    )


def _aiperfjob_body(*, heartbeat: str | None = None) -> dict[str, Any]:
    annotations = {Annotations.CONTROLLER_HEARTBEAT: heartbeat} if heartbeat else {}
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {
            "name": "benchmark",
            "namespace": "bench-prod",
            "uid": "job-uid",
            "resourceVersion": "42",
            "creationTimestamp": _timestamp(seconds_ago=60),
            "annotations": annotations,
        },
        "status": {
            "phase": str(Phase.RUNNING),
            "jobSetName": "aiperf-benchmark",
            "jobId": "benchmark",
            "startTime": _timestamp(seconds_ago=60),
        },
    }


def _kopf_patch() -> MagicMock:
    patch = MagicMock()
    patch.status = {}
    patch.metadata = {}
    return patch


@pytest.mark.asyncio
async def test_controller_failure_push_terminalizes_parent_without_salvage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A controller-reported fatal error wins a concurrent completion race."""
    body = _aiperfjob_body()
    body["status"]["phase"] = str(Phase.COMPLETED)
    body["status"]["controllerFailure"] = "timing-manager: worker floor breached"
    status_patch = _install_live_status_api(monkeypatch, body)

    await monitor.handle_controller_failure_event(
        body=body,
        new="timing-manager: worker floor breached",
        namespace="bench-prod",
        name="benchmark",
    )

    status_patch.assert_awaited_once()
    assert body["status"]["phase"] == str(Phase.FAILED)
    assert (
        body["status"]["error"]
        == "Controller reported fatal failure: timing-manager: worker floor breached"
    )


@asynccontextmanager
async def _fake_k8s_client() -> AsyncIterator[MagicMock]:
    yield MagicMock(name="ApiClient")


def _install_status_api(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    custom = MagicMock(name="CustomObjectsApi")
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(monitor, "k8s_client", _fake_k8s_client)
    monkeypatch.setattr(monitor.client, "CustomObjectsApi", lambda _api: custom)
    monkeypatch.setattr(
        monitor,
        "current_aiperfjob_resource_version",
        AsyncMock(return_value="43"),
    )
    monkeypatch.setattr(
        monitor,
        "current_aiperfjob_body",
        AsyncMock(
            return_value={
                "metadata": {"resourceVersion": "43"},
                "spec": {},
                "status": {"phase": str(Phase.RUNNING)},
            }
        ),
    )
    return custom.patch_namespaced_custom_object_status


def _install_live_status_api(
    monkeypatch: pytest.MonkeyPatch,
    live_body: dict[str, Any],
    *,
    conflict_once: bool = False,
) -> AsyncMock:
    """Install a JSON-patch fake backed by one mutable live CR body."""

    async def _current_body(
        _namespace: str, _name: str, expected_uid: str
    ) -> dict[str, Any]:
        if live_body["metadata"]["uid"] != expected_uid:
            raise StaleAIPerfJobCallback("same-name parent was replaced")
        return deepcopy(live_body)

    async def _current_resource_version(*_: Any, **__: Any) -> str:
        return str(live_body["metadata"]["resourceVersion"])

    conflicted = False

    def _resolve_parent(pointer: str) -> tuple[dict[str, Any], str]:
        current: Any = live_body
        parts = [
            part.replace("~1", "/").replace("~0", "~")
            for part in pointer.removeprefix("/").split("/")
        ]
        for part in parts[:-1]:
            current = current[part]
        return current, parts[-1]

    async def _patch_parent(**kwargs: Any) -> dict[str, Any]:
        try:
            for operation in kwargs["body"]:
                parent, member = _resolve_parent(operation["path"])
                if operation["op"] == "test":
                    assert parent[member] == operation["value"]
                else:
                    assert operation["op"] == "add"
                    parent[member] = deepcopy(operation["value"])
        except (AssertionError, KeyError) as exc:
            raise ApiException(
                status=422, reason="json patch precondition lost"
            ) from exc
        live_body["metadata"]["resourceVersion"] = str(
            int(live_body["metadata"]["resourceVersion"]) + 1
        )
        return deepcopy(live_body)

    async def _patch_status(**kwargs: Any) -> None:
        nonlocal conflicted
        if conflict_once and not conflicted:
            conflicted = True
            live_body["metadata"]["resourceVersion"] = str(
                int(live_body["metadata"]["resourceVersion"]) + 1
            )
            raise ApiException(status=409, reason="controller heartbeat raced")
        for operation in kwargs["body"]:
            path = operation["path"]
            if operation["op"] == "test":
                if path == "/metadata/uid":
                    assert live_body["metadata"]["uid"] == operation["value"]
                elif path == "/metadata/resourceVersion":
                    assert (
                        live_body["metadata"]["resourceVersion"] == operation["value"]
                    )
                elif path == "/status/phase":
                    assert live_body["status"]["phase"] == operation["value"]
                elif path.startswith("/status/"):
                    key = (
                        path.removeprefix("/status/")
                        .replace("~1", "/")
                        .replace("~0", "~")
                    )
                    assert live_body["status"].get(key) == operation["value"]
                continue
            assert operation["op"] == "add"
            key = path.removeprefix("/status/").replace("~1", "/").replace("~0", "~")
            live_body["status"][key] = deepcopy(operation["value"])

    status_patch = AsyncMock(side_effect=_patch_status)
    custom = MagicMock(name="CustomObjectsApi")
    custom.patch_namespaced_custom_object = AsyncMock(side_effect=_patch_parent)
    custom.patch_namespaced_custom_object_status = status_patch
    monkeypatch.setattr(monitor, "k8s_client", _fake_k8s_client)
    monkeypatch.setattr(monitor.client, "CustomObjectsApi", lambda _api: custom)
    monkeypatch.setattr(
        monitor,
        "current_aiperfjob_resource_version",
        AsyncMock(side_effect=_current_resource_version),
    )
    monkeypatch.setattr(
        monitor,
        "current_aiperfjob_body",
        AsyncMock(side_effect=_current_body),
        raising=False,
    )
    return status_patch


@pytest.mark.asyncio
async def test_fresh_controller_heartbeat_skips_broad_monitor_and_cluster_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy heartbeat tick must remain a cached-body-only operation."""
    broad_monitor = AsyncMock(side_effect=AssertionError("broad monitor invoked"))
    cluster_client = MagicMock(side_effect=AssertionError("cluster I/O invoked"))
    monkeypatch.setattr(monitor, "monitor_progress", broad_monitor)
    monkeypatch.setattr(monitor, "k8s_client", cluster_client)
    body = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    patch = _kopf_patch()

    await monitor.heartbeat_watchdog(
        body=body,
        status=body["status"],
        spec={},
        name="benchmark",
        namespace="bench-prod",
        patch=patch,
    )

    assert patch.status == {}
    assert patch.metadata == {}
    broad_monitor.assert_not_awaited()
    cluster_client.assert_not_called()


@pytest.mark.asyncio
async def test_stale_controller_heartbeat_invokes_existing_recovery_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired heartbeat delegates to the proven broad recovery path."""
    broad_monitor = AsyncMock()
    monkeypatch.setattr(monitor, "monitor_progress", broad_monitor)
    body = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=3600))
    patch = _kopf_patch()

    await monitor.heartbeat_watchdog(
        body=body,
        status=body["status"],
        spec={},
        name="benchmark",
        namespace="bench-prod",
        patch=patch,
    )

    broad_monitor.assert_awaited_once_with(
        body=body,
        status=body["status"],
        spec={},
        name="benchmark",
        namespace="bench-prod",
        patch=patch,
    )


@pytest.mark.asyncio
async def test_explicit_timeout_invokes_recovery_even_with_fresh_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The event-driven design must preserve explicit timeoutSeconds behavior."""
    broad_monitor = AsyncMock()
    monkeypatch.setattr(monitor, "monitor_progress", broad_monitor)
    body = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    patch = _kopf_patch()

    await monitor.heartbeat_watchdog(
        body=body,
        status=body["status"],
        spec={"timeoutSeconds": 1},
        name="benchmark",
        namespace="bench-prod",
        patch=patch,
    )

    broad_monitor.assert_awaited_once()


def test_main_registers_watchdog_timer_instead_of_broad_monitor_timer() -> None:
    """Only the heartbeat gate may own the recurring AIPerfJob timer."""
    handlers = kopf.get_default_registry()._spawning.get_all_handlers()  # noqa: SLF001

    assert sum(handler.fn is main.heartbeat_watchdog for handler in handlers) == 1
    assert all(handler.fn is not main.monitor_progress for handler in handlers)
    watchdog = next(
        handler for handler in handlers if handler.fn is main.heartbeat_watchdog
    )
    assert watchdog.interval == OperatorEnvironment.MONITOR.INTERVAL
    assert watchdog.initial_delay == OperatorEnvironment.MONITOR.INITIAL_DELAY


@pytest.mark.asyncio
async def test_jobset_failure_event_reuses_failure_classifier_and_fenced_status_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A watched controller JobSet failure immediately terminalizes its parent."""
    status_patch = _install_status_api(monkeypatch)
    failed_event = MagicMock()
    close = AsyncMock()
    monkeypatch.setattr(monitor.events, "failed", failed_event)
    monkeypatch.setattr(monitor, "close_progress_client", close)
    body = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    jobset_body = {
        "metadata": {
            "name": "aiperf-benchmark",
            "uid": "jobset-uid",
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_JOB_API_VERSION,
                    "kind": "AIPerfJob",
                    "name": "benchmark",
                    "uid": "job-uid",
                    "controller": True,
                }
            ],
        },
        "status": {
            "conditions": [
                {
                    "type": "Failed",
                    "status": "True",
                    "message": "controller exited",
                }
            ],
            "replicatedJobsStatus": [{"name": "controller", "failed": 1}],
        },
    }

    await monitor.handle_jobset_failure_event(
        body=body,
        jobset_body=jobset_body,
        namespace="bench-prod",
        name="benchmark",
    )

    kwargs = status_patch.await_args.kwargs
    assert kwargs["name"] == "benchmark"
    operations = kwargs["body"]
    assert operations[:2] == [
        {"op": "test", "path": "/metadata/uid", "value": "job-uid"},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": "43",
        },
    ]
    assert operations[2] == {
        "op": "test",
        "path": "/status/phase",
        "value": str(Phase.RUNNING),
    }
    status_values = {
        operation["path"].removeprefix("/status/"): operation["value"]
        for operation in operations[2:]
        if operation["op"] == "add"
    }
    assert status_values["phase"] == str(Phase.FAILED)
    assert status_values["error"] == "controller exited"
    failed_event.assert_called_once()
    close.assert_awaited_once()


@pytest.mark.parametrize(
    ("winning_phase", "cancel_requested"),
    [
        pytest.param(Phase.COMPLETED, False, id="completion-wins"),
        pytest.param(Phase.CANCELLED, True, id="cancellation-wins"),
    ],
)
@pytest.mark.asyncio
async def test_jobset_failure_event_cannot_overwrite_concurrent_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
    winning_phase: Phase,
    cancel_requested: bool,
) -> None:
    """A terminal parent update winning during an await must remain authoritative."""
    body = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    live_body = deepcopy(body)
    live_body["spec"] = {"cancel": False}
    status_patch = _install_live_status_api(monkeypatch, live_body)
    entered_close = asyncio.Event()
    release_close = asyncio.Event()

    async def _blocking_close(_key: str) -> None:
        entered_close.set()
        await release_close.wait()

    monkeypatch.setattr(monitor.events, "failed", MagicMock())
    monkeypatch.setattr(monitor, "close_progress_client", _blocking_close)
    jobset_body = {
        "metadata": {
            "name": "aiperf-benchmark",
            "uid": "jobset-uid",
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_JOB_API_VERSION,
                    "kind": "AIPerfJob",
                    "name": "benchmark",
                    "uid": "job-uid",
                    "controller": True,
                }
            ],
        },
        "status": {
            "conditions": [
                {
                    "type": "Failed",
                    "status": "True",
                    "message": "controller exited",
                }
            ],
            "replicatedJobsStatus": [{"name": "controller", "failed": 1}],
        },
    }

    event_task = asyncio.create_task(
        monitor.handle_jobset_failure_event(
            body=body,
            jobset_body=jobset_body,
            namespace="bench-prod",
            name="benchmark",
        )
    )
    await entered_close.wait()
    live_body["metadata"]["resourceVersion"] = "44"
    live_body["status"]["phase"] = str(winning_phase)
    live_body["spec"]["cancel"] = cancel_requested
    release_close.set()
    await event_task

    assert live_body["status"]["phase"] == str(winning_phase)
    status_patch.assert_not_awaited()


@pytest.mark.parametrize(
    "winning_state",
    [
        pytest.param("completed", id="completion-wins"),
        pytest.param("cancelled", id="cancellation-wins"),
        pytest.param("replacement", id="same-name-replacement-wins"),
    ],
)
@pytest.mark.asyncio
async def test_controller_subphase_event_cannot_overwrite_live_parent_winner(
    monkeypatch: pytest.MonkeyPatch,
    winning_state: str,
) -> None:
    """A stale controller lifecycle event must not mutate a newer parent."""
    stale_body = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    stale_body["status"]["phase"] = str(Phase.INITIALIZING)
    stale_body["spec"] = {"cancel": False}
    live_body = deepcopy(stale_body)
    live_body["metadata"]["resourceVersion"] = "43"
    if winning_state == "completed":
        live_body["status"]["phase"] = str(Phase.COMPLETED)
    elif winning_state == "cancelled":
        live_body["status"]["phase"] = str(Phase.CANCELLED)
        live_body["spec"]["cancel"] = True
    else:
        live_body["metadata"]["uid"] = "replacement-uid"
    status_patch = _install_live_status_api(monkeypatch, live_body)
    kopf_patch = _kopf_patch()

    await main.on_controller_subphase(
        old="ready",
        new="profiling",
        body=stale_body,
        status=stale_body["status"],
        name="benchmark",
        namespace="bench-prod",
        patch=kopf_patch,
    )

    assert kopf_patch.status == {}
    status_patch.assert_not_awaited()


@pytest.mark.asyncio
async def test_jobset_completed_condition_never_claims_benchmark_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Job process exit is not evidence of the durable controller handshake."""
    claim = AsyncMock(return_value=True)
    completion = AsyncMock()
    close = AsyncMock()
    monkeypatch.setattr(monitor, "try_claim_completion", claim)
    monkeypatch.setattr(monitor, "handle_completion", completion)
    monkeypatch.setattr(monitor, "close_progress_client", close)
    patch = _kopf_patch()

    handled = await monitor._handle_jobset_terminal_condition(
        body=_aiperfjob_body(heartbeat=_timestamp(seconds_ago=3600)),
        status=_aiperfjob_body()["status"],
        jobset_status={
            "conditions": [{"type": "Completed", "status": "True"}],
            "replicatedJobsStatus": [],
        },
        namespace="bench-prod",
        name="benchmark",
        jobset_name="aiperf-benchmark",
        job_id="benchmark",
        key="bench-prod/benchmark@job-uid",
        sb=StatusBuilder(patch, _aiperfjob_body()["status"]),
    )

    assert handled is False
    claim.assert_not_awaited()
    completion.assert_not_awaited()
    close.assert_not_awaited()


def _startup_pod() -> dict[str, Any]:
    return {
        "metadata": {
            "name": "aiperf-benchmark-controller-0",
            "namespace": "bench-prod",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-benchmark"},
        },
        "status": {
            "containerStatuses": [
                {
                    "name": Containers.CONTROL_PLANE,
                    "restartCount": 0,
                    "state": {
                        "waiting": {
                            "reason": "ImagePullBackOff",
                            "message": "image unavailable",
                        }
                    },
                }
            ]
        },
    }


def _terminated_pod() -> dict[str, Any]:
    return {
        "metadata": {
            "name": "aiperf-benchmark-controller-0",
            "namespace": "bench-prod",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-benchmark"},
        },
        "status": {
            "containerStatuses": [
                {
                    "name": Containers.CONTROL_PLANE,
                    "restartCount": 0,
                    "state": {"terminated": {"exitCode": 137, "reason": "OOMKilled"}},
                },
                {
                    "name": Containers.RESULTS_SIDECAR,
                    "restartCount": 0,
                    "state": {"running": {"startedAt": _timestamp()}},
                },
            ]
        },
    }


def _healthy_pod() -> dict[str, Any]:
    pod = _terminated_pod()
    pod["status"]["containerStatuses"] = [
        {
            "name": Containers.CONTROL_PLANE,
            "restartCount": 0,
            "state": {"running": {"startedAt": _timestamp()}},
        },
        {
            "name": Containers.RESULTS_SIDECAR,
            "restartCount": 0,
            "state": {"running": {"startedAt": _timestamp()}},
        },
    ]
    return pod


@pytest.mark.asyncio
async def test_pod_startup_event_persists_focused_parent_diagnosis_without_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A watched startup blocker is classified from that Pod body directly."""
    status_patch = _install_status_api(monkeypatch)
    parent = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    pod = _startup_pod()
    list_pods = MagicMock(side_effect=AssertionError("pod list invoked"))
    monkeypatch.setattr(
        pod_restarts,
        "_lookup_aiperfjob_body",
        AsyncMock(return_value=parent),
    )
    monkeypatch.setattr(monitor.client, "CoreV1Api", list_pods)

    await monitor.handle_pod_recovery_event(
        body=pod,
        meta=pod["metadata"],
        namespace="bench-prod",
        name="aiperf-benchmark-controller-0",
    )

    operations = status_patch.await_args.kwargs["body"]
    status_values = {
        operation["path"].removeprefix("/status/"): operation["value"]
        for operation in operations[2:]
        if operation["op"] == "add"
    }
    assert status_values["startupIssue"]["reason"] == "ImagePullBackOff"
    assert status_values["startupIssue"]["podName"] == pod["metadata"]["name"]
    workers_ready = next(
        condition
        for condition in status_values["conditions"]
        if condition["type"] == "WorkersReady"
    )
    assert workers_ready["reason"] == "PodStartupBlocked"
    list_pods.assert_not_called()


@pytest.mark.asyncio
async def test_aged_pod_startup_event_defers_delete_to_claiming_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Pod watch must not enter destructive cleanup before the durable claim."""
    parent = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    fingerprint = f"ImagePull:aiperf-benchmark-controller-0:{Containers.CONTROL_PLANE}"
    parent["spec"] = {"cancel": False}
    parent["status"].update(
        {
            "phase": str(Phase.INITIALIZING),
            "startupIssue": {
                "fingerprint": fingerprint,
                "podName": "aiperf-benchmark-controller-0",
                "containerName": Containers.CONTROL_PLANE,
                "reason": "ImagePullBackOff",
                "message": "image unavailable",
                "category": "ImagePull",
                "terminalAfterThreshold": True,
                "firstObservedTime": _timestamp(
                    seconds_ago=(
                        K8sEnvironment.WATCHDOG.PENDING_CRITICAL_THRESHOLD_SECONDS + 1
                    )
                ),
                "warningEmitted": True,
            },
        }
    )
    pod = _startup_pod()
    _install_live_status_api(monkeypatch, parent)
    monkeypatch.setattr(
        pod_restarts,
        "_lookup_aiperfjob_body",
        AsyncMock(return_value=parent),
    )

    async def _delete_after_claim(*_: Any, **__: Any) -> bool:
        annotations = parent["metadata"].get("annotations") or {}
        assert annotations.get(Annotations.STARTUP_FAILURE_CLAIMED) == fingerprint
        return True

    delete_jobset = AsyncMock(side_effect=_delete_after_claim)
    monkeypatch.setattr(monitor, "_delete_jobset_or_retry", delete_jobset)
    monkeypatch.setattr(monitor, "close_progress_client", AsyncMock())
    monkeypatch.setattr(monitor.events, "failed", MagicMock())

    await monitor.handle_pod_recovery_event(
        body=pod,
        meta=pod["metadata"],
        namespace="bench-prod",
        name="aiperf-benchmark-controller-0",
    )

    delete_jobset.assert_not_awaited()
    assert parent["status"]["phase"] == str(Phase.INITIALIZING)

    await main.startup_issue_deadline(
        body=parent,
        status=parent["status"],
        spec=parent["spec"],
        name="benchmark",
        namespace="bench-prod",
        patch=_kopf_patch(),
    )

    delete_jobset.assert_awaited_once()
    assert parent["status"]["phase"] == str(Phase.FAILED)


@pytest.mark.asyncio
async def test_pod_recovery_event_clears_same_pod_startup_diagnosis_without_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy update for the diagnosed Pod clears the durable blocker."""
    status_patch = _install_status_api(monkeypatch)
    parent = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    parent["status"]["startupIssue"] = {
        "podName": "aiperf-benchmark-controller-0",
        "reason": "ImagePullBackOff",
    }
    parent["status"]["conditions"] = [
        {
            "type": "WorkersReady",
            "status": "False",
            "reason": "PodStartupBlocked",
            "message": "image unavailable",
            "lastTransitionTime": _timestamp(seconds_ago=30),
        }
    ]
    pod = _healthy_pod()
    list_pods = MagicMock(side_effect=AssertionError("pod list invoked"))
    monkeypatch.setattr(
        pod_restarts,
        "_lookup_aiperfjob_body",
        AsyncMock(return_value=parent),
    )
    monkeypatch.setattr(monitor.client, "CoreV1Api", list_pods)

    await monitor.handle_pod_recovery_event(
        body=pod,
        meta=pod["metadata"],
        namespace="bench-prod",
        name="aiperf-benchmark-controller-0",
    )

    operations = status_patch.await_args.kwargs["body"]
    status_values = {
        operation["path"].removeprefix("/status/"): operation["value"]
        for operation in operations[2:]
        if operation["op"] == "add"
    }
    assert status_values["startupIssue"] is None
    workers_ready = next(
        condition
        for condition in status_values["conditions"]
        if condition["type"] == "WorkersReady"
    )
    assert workers_ready["reason"] == "WorkersStarting"
    list_pods.assert_not_called()


@pytest.mark.asyncio
async def test_controller_termination_event_reaches_salvage_with_event_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A watched nonzero controller exit enters salvage without a Pod list."""
    parent = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    pod = _terminated_pod()
    recover = AsyncMock(return_value=True)
    list_pods = MagicMock(side_effect=AssertionError("pod list invoked"))
    monkeypatch.setattr(
        pod_restarts,
        "_lookup_aiperfjob_body",
        AsyncMock(return_value=parent),
    )
    monkeypatch.setattr(monitor, "_maybe_recover_terminated_controller", recover)
    monkeypatch.setattr(monitor.client, "CoreV1Api", list_pods)
    monkeypatch.setattr(monitor, "k8s_client", _fake_k8s_client)

    await monitor.handle_pod_recovery_event(
        body=pod,
        meta=pod["metadata"],
        namespace="bench-prod",
        name="aiperf-benchmark-controller-0",
    )

    assert recover.await_count == 1
    assert recover.await_args.kwargs["pod"] is pod
    assert recover.await_args.kwargs["jobset_verified"] is True
    list_pods.assert_not_called()


@pytest.mark.asyncio
async def test_main_pod_watch_dispatches_focused_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live kopf Pod watch routes non-deletion events to focused recovery."""
    pod = _startup_pod()
    focused = AsyncMock()
    monkeypatch.setattr(monitor, "handle_pod_recovery_event", focused, raising=False)
    monkeypatch.setattr(
        main.pod_restarts_handler,
        "handle_pod_restart",
        AsyncMock(),
    )

    await main.on_pod_container_status_change(
        event={"type": "MODIFIED"},
        body=pod,
        meta=pod["metadata"],
        namespace="bench-prod",
        name="aiperf-benchmark-controller-0",
    )

    focused.assert_awaited_once_with(
        body=pod,
        meta=pod["metadata"],
        namespace="bench-prod",
        name="aiperf-benchmark-controller-0",
    )


@pytest.mark.asyncio
async def test_fresh_heartbeat_composes_with_watches_for_coarse_healthy_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Healthy watches must advance worker and lifecycle status without polling."""
    parent = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    parent["spec"] = {"benchmark": {"runtime": {"workersPerPod": 1}}}
    parent["status"].update(
        {
            "phase": str(Phase.PENDING),
            "workers": {"ready": 0, "total": 2},
        }
    )
    broad_monitor = AsyncMock(side_effect=AssertionError("broad monitor invoked"))
    monkeypatch.setattr(monitor, "monitor_progress", broad_monitor)
    watchdog_patch = _kopf_patch()

    await monitor.heartbeat_watchdog(
        body=parent,
        status=parent["status"],
        spec=parent["spec"],
        name="benchmark",
        namespace="bench-prod",
        patch=watchdog_patch,
    )

    status_patch = _install_live_status_api(monkeypatch, parent)
    monkeypatch.setattr(
        main.jobset_terminal_handler,
        "_lookup_aiperfjob_body",
        AsyncMock(return_value=parent),
    )
    jobset = {
        "metadata": {
            "name": "aiperf-benchmark",
            "uid": "jobset-uid",
            "labels": {
                "app": "aiperf",
                "aiperf.nvidia.com/job-id": "benchmark",
            },
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_JOB_API_VERSION,
                    "kind": "AIPerfJob",
                    "name": "benchmark",
                    "uid": "job-uid",
                    "controller": True,
                }
            ],
        },
        "status": {
            "replicatedJobsStatus": [{"name": "workers", "ready": 1, "active": 1}]
        },
    }
    jobset_watch = getattr(main, "on_jobset_replicated_jobs_status", None)
    if callable(jobset_watch):
        await jobset_watch(
            old=[],
            new=jobset["status"]["replicatedJobsStatus"],
            namespace="bench-prod",
            name="aiperf-benchmark",
            body=jobset,
        )

    assert status_patch.await_count == 1
    assert parent["status"]["phase"] == str(Phase.INITIALIZING)
    assert parent["status"]["workers"] == {"ready": 1, "total": 2}
    workers_ready = next(
        condition
        for condition in parent["status"]["conditions"]
        if condition["type"] == "WorkersReady"
    )
    assert workers_ready["status"] == "True"

    lifecycle_patch = _kopf_patch()
    lifecycle_watch = getattr(main, "on_controller_subphase", None)
    if callable(lifecycle_watch):
        await lifecycle_watch(
            old="ready",
            new="profiling",
            body=parent,
            status=parent["status"],
            name="benchmark",
            namespace="bench-prod",
            patch=lifecycle_patch,
        )

    assert parent["status"]["phase"] == str(Phase.RUNNING)
    assert lifecycle_patch.status == {}
    assert status_patch.await_count == 2
    broad_monitor.assert_not_awaited()
    assert watchdog_patch.status == {}


@pytest.mark.asyncio
async def test_jobset_readiness_rebases_conditions_from_live_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A readiness event must preserve an unrelated concurrent condition."""
    stale_parent = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    stale_parent["spec"] = {"benchmark": {"runtime": {"workersPerPod": 1}}}
    stale_parent["status"].update(
        {
            "phase": str(Phase.PENDING),
            "workers": {"ready": 0, "total": 2},
            "conditions": [
                {
                    "type": "WorkersReady",
                    "status": "False",
                    "reason": "WorkersStarting",
                    "message": "waiting",
                    "lastTransitionTime": _timestamp(seconds_ago=30),
                }
            ],
        }
    )
    live_parent = deepcopy(stale_parent)
    live_parent["metadata"]["resourceVersion"] = "43"
    live_parent["status"]["conditions"].append(
        {
            "type": "ControllerObserved",
            "status": "True",
            "reason": "Heartbeat",
            "message": "controller heartbeat observed",
            "lastTransitionTime": _timestamp(seconds_ago=1),
        }
    )
    _install_live_status_api(monkeypatch, live_parent)
    jobset = {
        "metadata": {
            "name": "aiperf-benchmark",
            "uid": "jobset-uid",
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_JOB_API_VERSION,
                    "kind": "AIPerfJob",
                    "name": "benchmark",
                    "uid": "job-uid",
                    "controller": True,
                }
            ],
        },
        "status": {
            "replicatedJobsStatus": [{"name": "workers", "ready": 1, "active": 1}]
        },
    }

    await monitor.handle_jobset_progress_event(
        body=stale_parent,
        jobset_body=jobset,
        namespace="bench-prod",
        name="benchmark",
    )

    condition_types = {
        condition["type"] for condition in live_parent["status"]["conditions"]
    }
    assert condition_types == {"ControllerObserved", "WorkersReady"}
    workers_ready = next(
        condition
        for condition in live_parent["status"]["conditions"]
        if condition["type"] == "WorkersReady"
    )
    assert workers_ready["status"] == "True"


@pytest.mark.asyncio
async def test_jobset_readiness_preserves_arbitrary_base_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An event must not interpret a condition unknown to StatusBuilder as removed."""
    parent = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    parent["spec"] = {"benchmark": {"runtime": {"workersPerPod": 1}}}
    parent["status"].update(
        {
            "phase": str(Phase.PENDING),
            "workers": {"ready": 0, "total": 2},
            "conditions": [
                {
                    "type": "ControllerObserved",
                    "status": "True",
                    "reason": "Heartbeat",
                    "message": "controller heartbeat observed",
                    "lastTransitionTime": _timestamp(seconds_ago=1),
                },
                {
                    "type": "WorkersReady",
                    "status": "False",
                    "reason": "WorkersStarting",
                    "message": "waiting",
                    "lastTransitionTime": _timestamp(seconds_ago=30),
                },
            ],
        }
    )
    _install_live_status_api(monkeypatch, parent)
    jobset = {
        "metadata": {
            "name": "aiperf-benchmark",
            "uid": "jobset-uid",
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_JOB_API_VERSION,
                    "kind": "AIPerfJob",
                    "name": "benchmark",
                    "uid": "job-uid",
                    "controller": True,
                }
            ],
        },
        "status": {
            "replicatedJobsStatus": [{"name": "workers", "ready": 1, "active": 1}]
        },
    }

    await monitor.handle_jobset_progress_event(
        body=parent,
        jobset_body=jobset,
        namespace="bench-prod",
        name="benchmark",
    )

    assert {item["type"] for item in parent["status"]["conditions"]} == {
        "ControllerObserved",
        "WorkersReady",
    }


@pytest.mark.asyncio
async def test_jobset_readiness_retries_after_controller_heartbeat_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-shot readiness event must rebase after an unrelated heartbeat race."""
    parent = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    parent["spec"] = {"benchmark": {"runtime": {"workersPerPod": 1}}}
    parent["status"].update(
        {
            "phase": str(Phase.PENDING),
            "workers": {"ready": 0, "total": 2},
        }
    )
    status_patch = _install_live_status_api(monkeypatch, parent, conflict_once=True)
    monkeypatch.setattr(
        main.jobset_terminal_handler,
        "_lookup_aiperfjob_body",
        AsyncMock(side_effect=lambda *_: deepcopy(parent)),
    )
    jobset = {
        "metadata": {
            "name": "aiperf-benchmark",
            "uid": "jobset-uid",
            "labels": {
                "app": "aiperf",
                "aiperf.nvidia.com/job-id": "benchmark",
            },
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_JOB_API_VERSION,
                    "kind": "AIPerfJob",
                    "name": "benchmark",
                    "uid": "job-uid",
                    "controller": True,
                }
            ],
        },
        "status": {
            "replicatedJobsStatus": [{"name": "workers", "ready": 1, "active": 1}]
        },
    }

    with pytest.raises(kopf.TemporaryError):
        await main.on_jobset_replicated_jobs_status(
            old=[],
            new=jobset["status"]["replicatedJobsStatus"],
            namespace="bench-prod",
            name="aiperf-benchmark",
            body=jobset,
        )
    await main.on_jobset_replicated_jobs_status(
        old=[],
        new=jobset["status"]["replicatedJobsStatus"],
        namespace="bench-prod",
        name="aiperf-benchmark",
        body=jobset,
    )

    assert status_patch.await_count == 2
    assert parent["status"]["workers"] == {"ready": 1, "total": 2}
    assert parent["status"]["phase"] == str(Phase.INITIALIZING)


@pytest.mark.asyncio
async def test_healthy_pod_clear_retries_after_controller_heartbeat_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-shot healthy Pod event must rebase instead of losing its clear."""
    parent = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    parent["spec"] = {"cancel": False}
    parent["status"]["startupIssue"] = {
        "fingerprint": "ImagePull:aiperf-benchmark-controller-0:controller",
        "podName": "aiperf-benchmark-controller-0",
        "containerName": "controller",
        "reason": "ImagePullBackOff",
        "message": "image unavailable",
        "category": "ImagePull",
        "terminalAfterThreshold": True,
        "firstObservedTime": _timestamp(seconds_ago=30),
        "warningEmitted": False,
    }
    parent["status"]["conditions"] = [
        {
            "type": "WorkersReady",
            "status": "False",
            "reason": "PodStartupBlocked",
            "message": "image unavailable",
            "lastTransitionTime": _timestamp(seconds_ago=30),
        }
    ]
    status_patch = _install_live_status_api(monkeypatch, parent, conflict_once=True)
    monkeypatch.setattr(
        pod_restarts,
        "_lookup_aiperfjob_body",
        AsyncMock(side_effect=lambda *_: deepcopy(parent)),
    )
    pod = _healthy_pod()

    with pytest.raises(kopf.TemporaryError):
        await monitor.handle_pod_recovery_event(
            body=pod,
            meta=pod["metadata"],
            namespace="bench-prod",
            name="aiperf-benchmark-controller-0",
        )
    await monitor.handle_pod_recovery_event(
        body=pod,
        meta=pod["metadata"],
        namespace="bench-prod",
        name="aiperf-benchmark-controller-0",
    )

    assert status_patch.await_count == 2
    assert parent["status"]["startupIssue"] is None


@pytest.mark.asyncio
async def test_cached_startup_blocker_deadline_fails_with_fresh_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stable cached blocker must age to failure without another Pod event."""
    body = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    body["spec"] = {}
    body["status"].update(
        {
            "phase": str(Phase.INITIALIZING),
            "startupIssue": {
                "fingerprint": ("ImagePull:aiperf-benchmark-controller-0:controller"),
                "podName": "aiperf-benchmark-controller-0",
                "containerName": "controller",
                "reason": "ImagePullBackOff",
                "message": "image unavailable",
                "category": "ImagePull",
                "terminalAfterThreshold": True,
                "firstObservedTime": _timestamp(
                    seconds_ago=(
                        K8sEnvironment.WATCHDOG.PENDING_CRITICAL_THRESHOLD_SECONDS + 1
                    )
                ),
                "warningEmitted": True,
            },
        }
    )
    broad_monitor = AsyncMock(side_effect=AssertionError("broad monitor invoked"))
    cluster_client = MagicMock(side_effect=AssertionError("cluster I/O invoked"))
    monkeypatch.setattr(monitor, "monitor_progress", broad_monitor)
    monkeypatch.setattr(monitor, "k8s_client", cluster_client)
    watchdog_patch = _kopf_patch()

    await monitor.heartbeat_watchdog(
        body=body,
        status=body["status"],
        spec=body["spec"],
        name="benchmark",
        namespace="bench-prod",
        patch=watchdog_patch,
    )

    _install_live_status_api(monkeypatch, body)

    async def _delete_after_claim(*_: Any, **__: Any) -> bool:
        annotations = body["metadata"].get("annotations") or {}
        assert (
            annotations.get("aiperf.nvidia.com/startup-failure-claimed")
            == (body["status"]["startupIssue"]["fingerprint"])
        )
        return True

    delete_jobset = AsyncMock(side_effect=_delete_after_claim)
    close = AsyncMock()
    monkeypatch.setattr(monitor, "_delete_jobset_or_retry", delete_jobset)
    monkeypatch.setattr(monitor, "close_progress_client", close)
    monkeypatch.setattr(monitor.events, "failed", MagicMock())
    deadline_patch = _kopf_patch()
    deadline = getattr(main, "startup_issue_deadline", None)
    if callable(deadline):
        await deadline(
            body=body,
            status=body["status"],
            spec=body["spec"],
            name="benchmark",
            namespace="bench-prod",
            patch=deadline_patch,
        )

    assert body["status"]["phase"] == str(Phase.FAILED)
    assert deadline_patch.status == {}
    delete_jobset.assert_awaited_once()
    close.assert_awaited_once()
    broad_monitor.assert_not_awaited()
    assert watchdog_patch.status == {}


@pytest.mark.parametrize(
    "winning_state",
    [
        pytest.param("recovered", id="blocker-recovered"),
        pytest.param("cancelled", id="cancellation-wins"),
        pytest.param("completed", id="completion-wins"),
        pytest.param("completion-claim", id="completion-claim-wins"),
        pytest.param("replacement", id="same-name-replacement-wins"),
    ],
)
@pytest.mark.asyncio
async def test_startup_deadline_claim_serializes_before_delete(
    monkeypatch: pytest.MonkeyPatch,
    winning_state: str,
) -> None:
    """A parent winner during the claim race must prevent cleanup and failure."""
    stale_body = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    stale_body["spec"] = {"cancel": False}
    stale_body["status"].update(
        {
            "phase": str(Phase.INITIALIZING),
            "startupIssue": {
                "fingerprint": "ImagePull:aiperf-benchmark-controller-0:controller",
                "podName": "aiperf-benchmark-controller-0",
                "containerName": "controller",
                "reason": "ImagePullBackOff",
                "message": "image unavailable",
                "category": "ImagePull",
                "terminalAfterThreshold": True,
                "firstObservedTime": _timestamp(
                    seconds_ago=(
                        K8sEnvironment.WATCHDOG.PENDING_CRITICAL_THRESHOLD_SECONDS + 1
                    )
                ),
                "warningEmitted": True,
            },
        }
    )
    live_body = deepcopy(stale_body)
    entered_claim = asyncio.Event()
    release_claim = asyncio.Event()
    entered_delete = asyncio.Event()
    release_delete = asyncio.Event()

    def _pointer_parent(
        document: dict[str, Any], pointer: str
    ) -> tuple[dict[str, Any], str]:
        current: Any = document
        parts = [
            part.replace("~1", "/").replace("~0", "~")
            for part in pointer.removeprefix("/").split("/")
        ]
        for part in parts[:-1]:
            current = current[part]
        return current, parts[-1]

    async def _patch_parent(*, body: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        entered_claim.set()
        await release_claim.wait()
        try:
            for operation in body:
                parent, member = _pointer_parent(live_body, operation["path"])
                if operation["op"] == "test":
                    assert parent[member] == operation["value"]
                else:
                    assert operation["op"] == "add"
                    parent[member] = deepcopy(operation["value"])
        except (AssertionError, KeyError) as exc:
            raise ApiException(status=422, reason="claim precondition lost") from exc
        live_body["metadata"]["resourceVersion"] = str(
            int(live_body["metadata"]["resourceVersion"]) + 1
        )
        return deepcopy(live_body)

    status_patch = AsyncMock()
    custom = MagicMock(name="CustomObjectsApi")
    custom.patch_namespaced_custom_object = AsyncMock(side_effect=_patch_parent)
    custom.patch_namespaced_custom_object_status = status_patch
    monkeypatch.setattr(monitor, "k8s_client", _fake_k8s_client)
    monkeypatch.setattr(client_cache, "k8s_client", _fake_k8s_client)
    monkeypatch.setattr(monitor.client, "CustomObjectsApi", lambda _api: custom)

    async def _current_body(
        _namespace: str, _name: str, expected_uid: str
    ) -> dict[str, Any]:
        if live_body["metadata"]["uid"] != expected_uid:
            raise StaleAIPerfJobCallback("same-name parent was replaced")
        return deepcopy(live_body)

    monkeypatch.setattr(
        monitor,
        "current_aiperfjob_body",
        AsyncMock(side_effect=_current_body),
    )

    async def _delete(*_: Any, **__: Any) -> bool:
        entered_delete.set()
        await release_delete.wait()
        return True

    delete_jobset = AsyncMock(side_effect=_delete)
    monkeypatch.setattr(monitor, "_delete_jobset_or_retry", delete_jobset)
    monkeypatch.setattr(monitor, "close_progress_client", AsyncMock())
    monkeypatch.setattr(monitor.events, "failed", MagicMock())
    deadline_task = asyncio.create_task(
        main.startup_issue_deadline(
            body=stale_body,
            status=stale_body["status"],
            spec=stale_body["spec"],
            name="benchmark",
            namespace="bench-prod",
            patch=_kopf_patch(),
        )
    )
    claim_wait = asyncio.create_task(entered_claim.wait())
    delete_wait = asyncio.create_task(entered_delete.wait())
    done, pending = await asyncio.wait(
        {claim_wait, delete_wait}, timeout=1, return_when=asyncio.FIRST_COMPLETED
    )
    assert done
    first_boundary = "claim" if entered_claim.is_set() else "delete"

    if first_boundary == "claim":
        live_body["metadata"]["resourceVersion"] = "43"
        if winning_state == "recovered":
            live_body["status"]["startupIssue"] = None
        elif winning_state == "cancelled":
            live_body["status"]["phase"] = str(Phase.CANCELLED)
            live_body["spec"]["cancel"] = True
        elif winning_state == "completed":
            live_body["status"]["phase"] = str(Phase.COMPLETED)
        elif winning_state == "completion-claim":
            live_body["metadata"]["annotations"][Annotations.COMPLETION_CLAIMED] = (
                _timestamp()
            )
        else:
            live_body["metadata"]["uid"] = "replacement-uid"
        release_claim.set()
    else:
        release_delete.set()

    await deadline_task
    for waiter in pending:
        waiter.cancel()

    assert first_boundary == "claim"
    delete_jobset.assert_not_awaited()
    status_patch.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_deadline_existing_failure_claim_resumes_fenced_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after claiming must resume cleanup under the durable claim."""
    body = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    fingerprint = "ImagePull:aiperf-benchmark-controller-0:controller"
    body["metadata"]["annotations"]["aiperf.nvidia.com/startup-failure-claimed"] = (
        fingerprint
    )
    body["spec"] = {"cancel": False}
    body["status"].update(
        {
            "phase": str(Phase.INITIALIZING),
            "startupIssue": {
                "fingerprint": fingerprint,
                "podName": "aiperf-benchmark-controller-0",
                "containerName": "controller",
                "reason": "ImagePullBackOff",
                "message": "image unavailable",
                "category": "ImagePull",
                "terminalAfterThreshold": True,
                "firstObservedTime": _timestamp(
                    seconds_ago=(
                        K8sEnvironment.WATCHDOG.PENDING_CRITICAL_THRESHOLD_SECONDS + 1
                    )
                ),
                "warningEmitted": True,
            },
        }
    )
    status_patch = _install_live_status_api(monkeypatch, body)
    delete_jobset = AsyncMock(return_value=True)
    monkeypatch.setattr(monitor, "_delete_jobset_or_retry", delete_jobset)
    monkeypatch.setattr(monitor, "close_progress_client", AsyncMock())
    monkeypatch.setattr(monitor.events, "failed", MagicMock())

    await main.startup_issue_deadline(
        body=body,
        status=body["status"],
        spec=body["spec"],
        name="benchmark",
        namespace="bench-prod",
        patch=_kopf_patch(),
    )

    delete_jobset.assert_awaited_once()
    claim_path = "/metadata/annotations/aiperf.nvidia.com~1startup-failure-claimed"
    operations = status_patch.await_args.kwargs["body"]
    assert {"op": "test", "path": claim_path, "value": fingerprint} in operations
    assert body["status"]["phase"] == str(Phase.FAILED)


@pytest.mark.parametrize(
    "winning_state",
    [
        pytest.param("recovered", id="blocker-recovered"),
        pytest.param("cancelled", id="cancellation-wins"),
        pytest.param("completed", id="completion-wins"),
        pytest.param("replacement", id="same-name-replacement-wins"),
    ],
)
@pytest.mark.asyncio
async def test_startup_deadline_revalidates_before_delete_and_status_commit(
    monkeypatch: pytest.MonkeyPatch,
    winning_state: str,
) -> None:
    """A stale cached deadline must not delete or fail a newer live parent."""
    stale_body = _aiperfjob_body(heartbeat=_timestamp(seconds_ago=1))
    stale_body["spec"] = {"cancel": False}
    stale_body["status"].update(
        {
            "phase": str(Phase.INITIALIZING),
            "startupIssue": {
                "fingerprint": ("ImagePull:aiperf-benchmark-controller-0:controller"),
                "podName": "aiperf-benchmark-controller-0",
                "containerName": "controller",
                "reason": "ImagePullBackOff",
                "message": "image unavailable",
                "category": "ImagePull",
                "terminalAfterThreshold": True,
                "firstObservedTime": _timestamp(
                    seconds_ago=(
                        K8sEnvironment.WATCHDOG.PENDING_CRITICAL_THRESHOLD_SECONDS + 1
                    )
                ),
                "warningEmitted": True,
            },
        }
    )
    live_body = deepcopy(stale_body)
    status_patch = _install_live_status_api(monkeypatch, live_body)
    entered_delete_boundary = asyncio.Event()
    release_delete_boundary = asyncio.Event()

    @asynccontextmanager
    async def _blocking_k8s_client() -> AsyncIterator[MagicMock]:
        entered_delete_boundary.set()
        await release_delete_boundary.wait()
        yield MagicMock(name="ApiClient")

    monkeypatch.setattr(monitor, "k8s_client", _blocking_k8s_client)
    delete_jobset = AsyncMock(return_value=True)
    monkeypatch.setattr(monitor, "_delete_jobset_or_retry", delete_jobset)
    monkeypatch.setattr(monitor, "close_progress_client", AsyncMock())
    monkeypatch.setattr(monitor.events, "failed", MagicMock())
    kopf_patch = _kopf_patch()
    deadline_task = asyncio.create_task(
        main.startup_issue_deadline(
            body=stale_body,
            status=stale_body["status"],
            spec=stale_body["spec"],
            name="benchmark",
            namespace="bench-prod",
            patch=kopf_patch,
        )
    )
    await entered_delete_boundary.wait()
    live_body["metadata"]["resourceVersion"] = "43"
    if winning_state == "recovered":
        live_body["status"]["startupIssue"] = None
    elif winning_state == "cancelled":
        live_body["status"]["phase"] = str(Phase.CANCELLED)
        live_body["spec"]["cancel"] = True
    elif winning_state == "completed":
        live_body["status"]["phase"] = str(Phase.COMPLETED)
    else:
        live_body["metadata"]["uid"] = "replacement-uid"
    release_delete_boundary.set()
    await deadline_task

    delete_jobset.assert_not_awaited()
    status_patch.assert_not_awaited()
    assert kopf_patch.status == {}
