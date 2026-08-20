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
from pytest import param

from aiperf.common.enums import SystemState
from aiperf.common.hooks import AIPerfHook, HookAttrs
from aiperf.common.messages import (
    BaseServiceErrorMessage,
    ResultsExportedMessage,
    SystemStateChangedMessage,
)
from aiperf.common.models import MetricResult
from aiperf.common.models.error_models import ErrorDetails
from aiperf.kubernetes.constants import CONTROLLER_HEARTBEAT_INTERVAL_SECONDS


def _make_router():
    """Build a minimal ProgressRouter for testing without full service init."""
    from aiperf.api.routers.progress import ProgressRouter

    r = ProgressRouter.__new__(ProgressRouter)
    r._system_state = SystemState.INITIALIZING
    r._results_exported = False
    r._controller_failure = None
    r._server_metrics = None
    r._metrics = []
    r._progress_tracker = MagicMock()
    r._progress_tracker._phases = {}
    r._k8s_patching_enabled = True
    r._k8s_job_id = "test-job"
    r._k8s_job_uid = "uid-123"
    r._k8s_namespace = "default"
    r._last_patched_jobset_annotations = {}
    r._last_patched_aiperfjob_annotations = {}
    r._stop_requested_event = asyncio.Event()
    return r


@pytest.mark.asyncio
async def test_non_kubernetes_run_schedules_no_status_push_task():
    """Outside Kubernetes, bus handlers must not spawn a push task at all.

    The handlers fire on messages published in every run mode -- two of them at
    per-tick rates -- so a task created only to return on its first line is
    pure local-run overhead.
    """
    from aiperf.api.routers.progress import ProgressRouter

    r = _make_router()
    r._k8s_patching_enabled = False
    r._patch_aiperfjob_status = AsyncMock()

    await r._on_system_state_changed(
        SystemStateChangedMessage(service_id="ctrl", state=SystemState.PROFILING)
    )
    await r._on_credit_phase_progress_status_push(MagicMock())
    await r._on_realtime_metrics_status_push(MagicMock())
    await r._on_credit_phase_complete_status_push(MagicMock())
    await r._on_results_exported(MagicMock())
    await asyncio.sleep(0)

    # State is still tracked; only the k8s round-trip is skipped.
    assert r._system_state == SystemState.PROFILING
    assert r._results_exported is True
    r._patch_aiperfjob_status.assert_not_called()

    # ...and neither CR-patching loop is started.
    r.start_background_task = MagicMock()
    await ProgressRouter._start_k8s_patch_loops(r)
    r.start_background_task.assert_not_called()


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
    # The heartbeat loop is no longer a @background_task: it is registered from
    # _start_k8s_patch_loops so a non-Kubernetes run never starts it at all.
    assert (
        getattr(ProgressRouter._patch_aiperfjob_status, HookAttrs.HOOK_TYPE, None)
        is not AIPerfHook.BACKGROUND_TASK
    )
    router.start_background_task = MagicMock()
    await ProgressRouter._start_k8s_patch_loops(router)
    intervals = [
        call.kwargs["interval"] for call in router.start_background_task.call_args_list
    ]
    assert len(intervals) == 2
    assert any(0 < i <= CONTROLLER_HEARTBEAT_INTERVAL_SECONDS for i in intervals)
    assert all(
        call.kwargs["immediate"] is False
        for call in router.start_background_task.call_args_list
    )
    del router.start_background_task

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


def _install_fake_k8s(
    monkeypatch: pytest.MonkeyPatch, resource: dict[str, object]
) -> MagicMock:
    """Point _push_aiperfjob_status at an in-memory CustomObjectsApi double."""
    import kubernetes_asyncio

    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(return_value=resource)
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

    monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)
    return custom


def _cr(status: dict[str, object] | None = None) -> dict[str, object]:
    """A live AIPerfJob body the UID fence accepts."""
    body: dict[str, object] = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {
            "name": "j",
            "namespace": "ns",
            "uid": "uid-123",
            "resourceVersion": "42",
            "annotations": {},
        },
    }
    if status is not None:
        body["status"] = status
    return body


def _phases(**overrides: object) -> dict[str, object]:
    """Two-phase progress: warmup finished, a named profiling phase running."""
    from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats

    phases = {
        "warmup": CombinedPhaseStats(
            phase="warmup",
            phase_kind="warmup",
            total_expected_requests=10,
            requests_sent=10,
            requests_completed=10,
            start_ns=1_000,
            last_update_ns=2_000,
        ),
        "steady_state": CombinedPhaseStats(
            phase="profiling",
            phase_name="steady_state",
            phase_kind="profiling",
            total_expected_requests=100,
            requests_sent=30,
            requests_completed=25,
            start_ns=5_000,
            last_update_ns=6_000,
        ),
    }
    phases.update(overrides)
    return phases


async def _push(**overrides: object) -> None:
    from aiperf.api.routers.progress import _push_aiperfjob_status

    kwargs: dict[str, object] = {
        "job_id": "j",
        "job_uid": "uid-123",
        "namespace": "ns",
        "phases": _phases(),
        "system_state": SystemState.PROFILING,
        "results_exported": False,
    }
    kwargs.update(overrides)
    await _push_aiperfjob_status(**kwargs)  # type: ignore[arg-type]


def _status_body(custom: MagicMock) -> dict[str, object]:
    """Extract the status payload regardless of which patch shape was used."""
    body = custom.patch_namespaced_custom_object_status.await_args.kwargs["body"]
    if isinstance(body, dict):
        return body["status"]
    return {
        op["path"].removeprefix("/status/"): op["value"]
        for op in body
        if op["op"] == "add"
    }


@pytest.mark.asyncio
async def test_push_aiperfjob_status_emits_current_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The push names the most recently started phase, not the alphabetized one.

    Without this key ``_requests_progress_percent`` falls back to alphabetized
    iteration over ``status.phases`` and reports warmup's 100% for a job that
    is 25% into profiling.
    """
    custom = _install_fake_k8s(monkeypatch, _cr({"phase": "Running"}))

    await _push()

    status = _status_body(custom)
    assert status["currentPhase"] == "steady_state"


@pytest.mark.asyncio
async def test_push_aiperfjob_status_current_phase_is_a_key_of_emitted_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """currentPhase must always resolve inside the phases map it ships with.

    A phase that has started but not yet sent a request wins the
    most-recently-started comparison while ``_build_phase_progress`` drops it,
    so the push has to emit a zeroed entry rather than a dangling pointer.
    """
    from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats

    custom = _install_fake_k8s(monkeypatch, _cr({"phase": "Running"}))

    await _push(
        phases=_phases(
            ramp=CombinedPhaseStats(
                phase="profiling",
                phase_name="ramp",
                phase_kind="profiling",
                total_expected_requests=None,
                requests_sent=0,
                requests_completed=0,
                start_ns=9_000,
            )
        )
    )

    status = _status_body(custom)
    assert status["currentPhase"] == "ramp"
    assert status["currentPhase"] in status["phases"]
    assert status["phases"]["ramp"]["requestsTotal"] == 0
    assert status["phases"]["ramp"]["requestsCompleted"] == 0


@pytest.mark.asyncio
async def test_push_aiperfjob_status_omits_current_phase_without_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no phases at all the key is omitted, never written as null.

    A merge-patch null would clobber a previously good value.
    """
    custom = _install_fake_k8s(monkeypatch, _cr({"phase": "Running"}))

    await _push(phases={})

    assert "currentPhase" not in _status_body(custom)


@pytest.mark.asyncio
async def test_push_aiperfjob_status_on_terminal_cr_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A push observing a terminal CR must not resurrect the cleared keys.

    kopf clears both currentPhase and subPhase when it stamps a terminal
    phase; an untracked push task fired on SHUTDOWN can land afterwards.
    """
    custom = _install_fake_k8s(
        monkeypatch, _cr({"phase": "Completed", "currentPhase": None, "subPhase": None})
    )

    await _push(system_state=SystemState.SHUTDOWN)

    custom.patch_namespaced_custom_object_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_push_aiperfjob_status_fences_json_patch_on_observed_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live CR is patched with a test op so the apiserver settles the race."""
    custom = _install_fake_k8s(monkeypatch, _cr({"phase": "Running"}))

    await _push()

    call = custom.patch_namespaced_custom_object_status.await_args.kwargs
    assert call["_content_type"] == "application/json-patch+json"
    assert call["body"][0] == {
        "op": "test",
        "path": "/status/phase",
        "value": "Running",
    }
    assert all(op["op"] == "add" for op in call["body"][1:])
    assert {op["path"] for op in call["body"][1:]} >= {
        "/status/subPhase",
        "/status/phases",
        "/status/currentPhase",
    }


@pytest.mark.asyncio
async def test_push_aiperfjob_status_merges_existing_status_into_json_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON-patch add replaces a member outright, so values are pre-merged.

    Without this the fenced write would silently drop sibling keys the merge
    patch it replaces would have preserved.
    """
    custom = _install_fake_k8s(
        monkeypatch,
        _cr({"phase": "Running", "phases": {"stale_phase": {"requestsTotal": 7}}}),
    )

    await _push()

    phases = _status_body(custom)["phases"]
    assert phases["stale_phase"] == {"requestsTotal": 7}
    assert "steady_state" in phases


def _api_exception(status: int, reason: str, body: str = ""):
    """Build an ApiException carrying a response body, as the apiserver would."""
    from kubernetes_asyncio.client.exceptions import ApiException

    exc = ApiException(status=status, reason=reason)
    exc.body = body
    return exc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, reason, body",
    [
        param(
            409,
            "Conflict",
            '{"message": "jsonpatch test operation does not apply"}',
            id="409-test-op-does-not-apply",
        ),
        param(
            422,
            "Unprocessable Entity",
            '{"message": "testing value /status/phase failed"}',
            id="422-testing-value-failed",
        ),
        param(
            422,
            "Unprocessable Entity",
            b'{"message": "jsonpatch test operation does not apply"}',
            id="422-bytes-body",
        ),
    ],
)  # fmt: skip
async def test_push_aiperfjob_status_swallows_lost_fence_race(
    monkeypatch: pytest.MonkeyPatch, status: int, reason: str, body: str | bytes
) -> None:
    """A rejected test op is the success path: the CR terminalized mid-push.

    The apiserver's wording has changed across evanphx/json-patch versions, so
    every form it has used must be recognized.
    """
    custom = _install_fake_k8s(monkeypatch, _cr({"phase": "Running"}))
    custom.patch_namespaced_custom_object_status = AsyncMock(
        side_effect=_api_exception(status, reason, body)  # type: ignore[arg-type]
    )

    await _push()

    custom.patch_namespaced_custom_object_status.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, reason, body",
    [
        param(
            422,
            "Unprocessable Entity",
            '{"message": "status.workers: Invalid value: \"string\": '
            'status.workers in body must be of type integer"}',
            id="422-crd-schema-violation",
        ),
        param(409, "Conflict", '{"message": "the object has been modified"}', id="409-plain-conflict"),
        param(500, "Internal Server Error", "", id="500-server-error"),
    ],
)  # fmt: skip
async def test_push_aiperfjob_status_reraises_non_fence_rejections(
    monkeypatch: pytest.MonkeyPatch, status: int, reason: str, body: str
) -> None:
    """A rejected payload must never masquerade as a lost race.

    422 also covers CRD structural-schema violations. Swallowing those at debug
    would stop status updates silently on every CR that has a phase -- the exact
    failure class the fence exists to prevent -- the moment anyone adds a
    status key of the wrong type.
    """
    from kubernetes_asyncio.client.exceptions import ApiException

    custom = _install_fake_k8s(monkeypatch, _cr({"phase": "Running"}))
    custom.patch_namespaced_custom_object_status = AsyncMock(
        side_effect=_api_exception(status, reason, body)
    )

    with pytest.raises(ApiException):
        await _push()


@pytest.mark.asyncio
async def test_push_aiperfjob_status_prefers_concrete_phase_over_legacy_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later-started legacy aggregate entry must not win over a named phase.

    ``JobProgress._concrete_phases`` filters to entries with an explicit
    identity before comparing start times; a bare max over all phases would
    name the aggregate here.
    """
    from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats

    custom = _install_fake_k8s(monkeypatch, _cr({"phase": "Running"}))

    await _push(
        phases={
            "steady_state": CombinedPhaseStats(
                phase="profiling",
                phase_name="steady_state",
                phase_kind="profiling",
                total_expected_requests=100,
                requests_sent=30,
                requests_completed=25,
                start_ns=5_000,
                last_update_ns=6_000,
            ),
            # Legacy aggregate: key equals str(phase) and no phase_name.
            "profiling": CombinedPhaseStats(
                phase="profiling",
                total_expected_requests=100,
                requests_sent=30,
                requests_completed=25,
                start_ns=99_000,
                last_update_ns=99_500,
            ),
        }
    )

    status = _status_body(custom)
    assert status["currentPhase"] == "steady_state"
    assert status["currentPhase"] in status["phases"]


def test_merge_patch_value_strips_nulls_when_target_is_not_an_object() -> None:
    """RFC 7386: a dict patch against a non-object target starts from {}.

    Returning the patch verbatim would leak null members into the CR status.
    """
    from aiperf.api.routers.progress import _merge_patch_value

    assert _merge_patch_value(None, {"a": 1, "b": None}) == {"a": 1}
    assert _merge_patch_value("scalar", {"a": {"b": None, "c": 2}}) == {"a": {"c": 2}}


@pytest.mark.asyncio
async def test_push_aiperfjob_status_uses_merge_patch_before_a_phase_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A phaseless CR keeps the merge patch: a test op on an absent path is 422."""
    custom = _install_fake_k8s(monkeypatch, _cr({}))

    await _push()

    call = custom.patch_namespaced_custom_object_status.await_args.kwargs
    assert call["_content_type"] == "application/merge-patch+json"
    assert call["body"]["status"]["currentPhase"] == "steady_state"


def _server_metrics_payload() -> dict[str, object]:
    """One scrape in RealtimeServerMetricsMessage dump shape."""
    return {
        "endpoint_summaries": {
            "http://svc:8000/metrics": {
                "endpoint_url": "http://svc:8000/metrics",
                "info": {},
                "metrics": {
                    "dynamo_frontend_requests": {
                        "type": "counter",
                        "description": "HELP text",
                        "series": [{"labels": {"model": "m"}, "stats": {"rate": 9.0}}],
                    },
                    "node_cpu_seconds_total": {
                        "type": "counter",
                        "series": [{"stats": {"rate": 1.0}}],
                    },
                },
            }
        }
    }


@pytest.mark.asyncio
async def test_push_aiperfjob_status_writes_curated_server_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """status.serverMetrics is the dashboard's non-WebSocket fallback; the push owns it.

    Zero writers existed at HEAD, so the REST fallback panel stayed blank for
    any viewer whose per-job WebSocket was blocked.
    """
    custom = _install_fake_k8s(monkeypatch, _cr({"phase": "Running"}))

    await _push(server_metrics=_server_metrics_payload())

    status = _status_body(custom)
    server_metrics = status["serverMetrics"]
    assert set(server_metrics["metrics"]) == {"dynamo_frontend_requests"}
    assert server_metrics["summary"]["endpoints_successful"] == [
        "http://svc:8000/metrics"
    ]


@pytest.mark.asyncio
async def test_push_aiperfjob_status_omits_server_metrics_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No scrape yet must leave the key absent rather than blanking a prior value."""
    custom = _install_fake_k8s(monkeypatch, _cr({"phase": "Running"}))

    await _push()

    assert "serverMetrics" not in _status_body(custom)


@pytest.mark.asyncio
async def test_realtime_server_metrics_caches_payload_without_pushing() -> None:
    """The scrape is cached for the next progress push, not pushed on its own.

    Server metrics scrape at ~3Hz on their own timer; firing a CR round-trip per
    scrape would multiply apiserver load for data that already rides along.
    """
    from aiperf.common.messages import RealtimeServerMetricsMessage

    r = _make_router()
    r._patch_aiperfjob_status = AsyncMock()

    await r._on_realtime_server_metrics(
        RealtimeServerMetricsMessage(service_id="sm", snapshot={"x": 1.0})
    )
    await asyncio.sleep(0)

    assert r._server_metrics is not None
    assert r._server_metrics["snapshot"] == {"x": 1.0}
    r._patch_aiperfjob_status.assert_not_called()


@pytest.mark.asyncio
async def test_push_aiperfjob_status_replaces_server_metrics_instead_of_merging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """serverMetrics is a snapshot of one scrape, not an accumulation.

    Every other status key is pre-resolved through _merge_patch_value so the
    fenced JSON patch stays equivalent to the merge patch it replaces. That
    recursively unions dicts, and this key maps metric name -> dict-valued
    stats, so a metric the current tick does not project -- exactly what the
    projection's caps produce -- would linger in the CR forever with stale
    values indistinguishable from live ones.
    """
    stale = {
        "summary": {"endpoints_configured": ["http://old/metrics"]},
        "metrics": {
            "sglang:token_usage": {"series": [{"stats": {"max": 0.99}}]},
            "dynamo_frontend_requests": {"series": [{"stats": {"rate": 1.0}}]},
        },
    }
    custom = _install_fake_k8s(
        monkeypatch, _cr({"phase": "Running", "serverMetrics": stale})
    )

    await _push(server_metrics=_server_metrics_payload())

    written = _status_body(custom)["serverMetrics"]
    assert set(written["metrics"]) == {"dynamo_frontend_requests"}
    assert "sglang:token_usage" not in written["metrics"]
    assert written["summary"]["endpoints_configured"] == ["http://svc:8000/metrics"]


@pytest.mark.asyncio
async def test_push_aiperfjob_status_still_merges_non_snapshot_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot exemption is one key wide; merge stays the default."""
    custom = _install_fake_k8s(
        monkeypatch,
        _cr({"phase": "Running", "summary": {"kept_by_merge": {"avg": 1.0}}}),
    )

    await _push(
        metrics=[
            MetricResult(tag="request_latency", header="Latency", unit="ms", avg=2.0)
        ],
        server_metrics=_server_metrics_payload(),
    )

    assert "kept_by_merge" in _status_body(custom)["summary"]


@pytest.mark.asyncio
async def test_realtime_server_metrics_is_not_dumped_outside_kubernetes() -> None:
    """The ~3Hz dump is pure waste in a local run where no CR carries it."""
    from aiperf.common.messages import RealtimeServerMetricsMessage

    r = _make_router()
    r._k8s_patching_enabled = False

    await r._on_realtime_server_metrics(
        RealtimeServerMetricsMessage(service_id="sm", snapshot={"x": 1.0})
    )

    assert r._server_metrics is None
