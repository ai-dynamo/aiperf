# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProgressRouter push-based AIPerfJob status updates."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import SystemState
from aiperf.common.hooks import AIPerfHook, BackgroundTaskParams, HookAttrs
from aiperf.common.messages import (
    BaseServiceErrorMessage,
    ResultsExportedMessage,
    SystemStateChangedMessage,
)
from aiperf.common.models.error_models import ErrorDetails
from aiperf.kubernetes.constants import CONTROLLER_HEARTBEAT_INTERVAL_SECONDS


def _make_router():
    """Build a minimal ProgressRouter for testing without full service init."""
    from aiperf.api.routers.progress import ProgressRouter

    r = ProgressRouter.__new__(ProgressRouter)
    r._system_state = SystemState.INITIALIZING
    r._results_exported = False
    r._controller_failure = None
    r._metrics = []
    r._progress_tracker = MagicMock()
    r._progress_tracker._phases = {}
    r._k8s_patching_enabled = True
    r._k8s_job_id = "test-job"
    r._k8s_job_uid = "uid-123"
    r._k8s_namespace = "default"
    r._last_patched_jobset_annotations = {}
    r._last_patched_aiperfjob_annotations = {}
    return r


@pytest.mark.asyncio
async def test_state_change_fires_status_push():
    """SYSTEM_STATE_CHANGED updates _system_state and fires a status push."""
    r = _make_router()
    pushed = []
    r._patch_aiperfjob_status = AsyncMock(side_effect=lambda: pushed.append(True))

    msg = SystemStateChangedMessage(service_id="ctrl", state=SystemState.PROFILING)
    await r._on_system_state_changed(msg)
    await asyncio.sleep(0)

    assert r._system_state == SystemState.PROFILING
    assert len(pushed) == 1


@pytest.mark.asyncio
async def test_results_exported_fires_status_push():
    """RESULTS_EXPORTED sets _results_exported and fires a status push."""
    r = _make_router()
    pushed = []
    r._patch_aiperfjob_status = AsyncMock(side_effect=lambda: pushed.append(True))

    msg = ResultsExportedMessage(service_id="ctrl")
    await r._on_results_exported(msg)
    await asyncio.sleep(0)

    assert r._results_exported is True
    assert len(pushed) == 1


@pytest.mark.asyncio
async def test_service_error_pushes_controller_failure_to_aiperfjob():
    """A controller service failure is pushed before the controller exits."""
    r = _make_router()
    pushed = []
    r._patch_aiperfjob_status = AsyncMock(side_effect=lambda: pushed.append(True))

    await r._on_service_error(
        BaseServiceErrorMessage(
            service_id="timing-manager",
            error=ErrorDetails(message="Fatal worker availability threshold breached"),
        )
    )
    await asyncio.sleep(0)

    assert (
        r._controller_failure
        == "timing-manager: Fatal worker availability threshold breached"
    )
    assert pushed == [True]


@pytest.mark.asyncio
async def test_patch_disabled_when_k8s_patching_off():
    """_patch_aiperfjob_status is a no-op when _k8s_patching_enabled is False."""

    r = _make_router()
    r._k8s_patching_enabled = False

    push_called = False

    async def _fake_push(**_kwargs):
        nonlocal push_called
        push_called = True

    # Monkeypatch the module-level function so it would fail if called
    import aiperf.api.routers.progress as progress_mod

    original = progress_mod._push_aiperfjob_status
    progress_mod._push_aiperfjob_status = _fake_push
    try:
        await r._patch_aiperfjob_status()
    finally:
        progress_mod._push_aiperfjob_status = original

    assert not push_called, (
        "_push_aiperfjob_status must not be called when patching is disabled"
    )


@pytest.mark.asyncio
async def test_push_aiperfjob_status_uid_fence(monkeypatch):
    """UID mismatch raises ValueError — prevents stale controller from clobbering new CR."""
    import kubernetes_asyncio

    from aiperf.api.routers.progress import _push_aiperfjob_status

    fake_resource = {"metadata": {"uid": "different-uid"}}

    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(return_value=fake_resource)

    monkeypatch.setattr(
        kubernetes_asyncio,
        "client",
        SimpleNamespace(CustomObjectsApi=lambda _api: custom),
        raising=False,
    )

    @asynccontextmanager
    async def fake_k8s_client():
        yield MagicMock(name="ApiClient")

    import aiperf.kubernetes.client as kclient

    monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)

    with pytest.raises(ValueError, match="UID mismatch"):
        await _push_aiperfjob_status(
            job_id="j",
            job_uid="expected-uid",
            namespace="ns",
            phases={},
            system_state=SystemState.PROFILING,
            results_exported=False,
        )


@pytest.mark.asyncio
async def test_push_aiperfjob_status_patches_uid_fenced_controller_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every direct status push also refreshes the exact parent's heartbeat."""
    import kubernetes_asyncio

    from aiperf.api.routers.progress import _push_aiperfjob_status
    from aiperf.kubernetes.constants import Annotations

    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "metadata": {
                "name": "j",
                "namespace": "ns",
                "uid": "uid-123",
                "resourceVersion": "42",
                "annotations": {"example.com/existing": "preserved"},
            },
        }
    )
    custom.patch_namespaced_custom_object = AsyncMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        kubernetes_asyncio,
        "client",
        SimpleNamespace(CustomObjectsApi=lambda _api: custom),
        raising=False,
    )

    @asynccontextmanager
    async def fake_k8s_client():
        yield MagicMock(name="ApiClient")

    import aiperf.kubernetes.client as kclient

    monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)

    before = datetime.now(UTC)
    await _push_aiperfjob_status(
        job_id="j",
        job_uid="uid-123",
        namespace="ns",
        phases={},
        system_state=SystemState.PROFILING,
        results_exported=False,
        controller_failure="Fatal worker availability threshold breached",
    )
    after = datetime.now(UTC)

    custom.patch_namespaced_custom_object.assert_awaited_once()
    annotation_call = custom.patch_namespaced_custom_object.await_args.kwargs
    assert annotation_call["name"] == "j"
    assert annotation_call["namespace"] == "ns"
    assert annotation_call["_content_type"] == "application/json-patch+json"
    patch_body = annotation_call["body"]
    assert patch_body[0] == {
        "op": "test",
        "path": "/metadata/uid",
        "value": "uid-123",
    }
    assert len(patch_body) == 2
    heartbeat_op = patch_body[1]
    assert heartbeat_op["op"] == "add"
    assert heartbeat_op["path"] == (
        "/metadata/annotations/"
        + Annotations.CONTROLLER_HEARTBEAT.replace("~", "~0").replace("/", "~1")
    )
    heartbeat = datetime.fromisoformat(heartbeat_op["value"].replace("Z", "+00:00"))
    assert heartbeat.tzinfo == UTC
    assert before <= heartbeat <= after

    status_call = custom.patch_namespaced_custom_object_status.await_args.kwargs
    assert status_call["body"] == {
        "status": {
            "subPhase": "profiling",
            "phases": {},
            "controllerFailure": "Fatal worker availability threshold breached",
        }
    }
    assert status_call["_content_type"] == "application/merge-patch+json"


@pytest.mark.asyncio
async def test_periodic_push_refreshes_heartbeat_when_progress_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quiet live controller refreshes heartbeat on every timer invocation."""
    import kubernetes_asyncio

    from aiperf.api.routers.progress import ProgressRouter
    from aiperf.kubernetes.constants import Annotations

    custom = MagicMock()

    async def get_resource(*, name: str, **_: object) -> dict[str, object]:
        assert name == "test-job"
        return {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "metadata": {
                "name": "test-job",
                "namespace": "default",
                "uid": "uid-123",
                "resourceVersion": "42",
                "annotations": {},
            },
        }

    custom.get_namespaced_custom_object = AsyncMock(side_effect=get_resource)
    custom.patch_namespaced_custom_object = AsyncMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        kubernetes_asyncio,
        "client",
        SimpleNamespace(CustomObjectsApi=lambda _api: custom),
        raising=False,
    )

    @asynccontextmanager
    async def fake_k8s_client() -> AsyncIterator[MagicMock]:
        yield MagicMock(name="ApiClient")

    import aiperf.kubernetes.client as kclient
    import aiperf.kubernetes.phase as phase

    monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)
    timestamps = iter(("2026-08-17T12:00:00Z", "2026-08-17T12:00:10Z"))
    monkeypatch.setattr(phase, "format_timestamp", lambda: next(timestamps))

    router = _make_router()
    hook_type = getattr(
        ProgressRouter._patch_aiperfjob_status, HookAttrs.HOOK_TYPE, None
    )
    params = getattr(
        ProgressRouter._patch_aiperfjob_status, HookAttrs.HOOK_PARAMS, None
    )
    assert hook_type is AIPerfHook.BACKGROUND_TASK
    assert isinstance(params, BackgroundTaskParams)
    assert isinstance(params.interval, int | float)
    assert 0 < params.interval <= CONTROLLER_HEARTBEAT_INTERVAL_SECONDS
    assert params.immediate is False

    await router._patch_aiperfjob_status()
    await router._patch_aiperfjob_status()

    heartbeat_path = (
        "/metadata/annotations/"
        + Annotations.CONTROLLER_HEARTBEAT.replace("~", "~0").replace("/", "~1")
    )
    heartbeat_values = [
        operation["value"]
        for call in custom.patch_namespaced_custom_object.await_args_list
        if call.kwargs["name"] == "test-job"
        for operation in call.kwargs["body"]
        if operation.get("path") == heartbeat_path
    ]
    assert heartbeat_values == [
        "2026-08-17T12:00:00Z",
        "2026-08-17T12:00:10Z",
    ]
    assert custom.patch_namespaced_custom_object_status.await_count == 2
