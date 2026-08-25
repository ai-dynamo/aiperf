# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProgressRouter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from kubernetes_asyncio.client.exceptions import ApiException
from starlette.testclient import TestClient

from aiperf.api.routers.progress import ProgressRouter
from aiperf.common.enums import SystemState
from aiperf.common.messages import (
    CommandSuccessResponse,
    GetPodStatesCommand,
    SystemStateChangedMessage,
    WorkerPodStateMessage,
)
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.config import AIPerfConfig
from aiperf.controller.system_controller_models import PodStateSnapshot


@pytest.fixture
def progress_router(mock_zmq, router_config: AIPerfConfig) -> ProgressRouter:
    return ProgressRouter(
        run=router_config,
    )


@pytest.fixture
def progress_client(progress_router: ProgressRouter) -> TestClient:
    app = FastAPI()
    app.state.progress = progress_router
    app.include_router(progress_router.get_router())
    return TestClient(app)


def _pod(pod_index: str, *, declared: int, ready: int) -> WorkerPodStateMessage:
    return WorkerPodStateMessage(
        service_id=f"wgm-{pod_index}",
        pod_index=pod_index,
        benchmark_generation="generation-1",
        dataset_generation="dataset-1",
        declared_workers=declared,
        declared_record_processors=1,
        router_connected_workers=ready,
        dispatchable_workers=ready,
        ready_workers=ready,
        ready_record_processors=1,
        degraded_workers=max(0, declared - ready),
        degraded_record_processors=0,
        pod_state="ready" if ready else "starting",
        admission_state="dispatchable" if ready else "admitting",
    )


def _controller_service(pods: dict[str, WorkerPodStateMessage]) -> object:
    command = GetPodStatesCommand(service_id="api-service")
    response = CommandSuccessResponse.from_command_message(
        command,
        "system-controller",
        data={
            "pod_states": {
                pod_index: pod.model_dump(mode="json")
                for pod_index, pod in pods.items()
            },
            "worker_startup_states": {},
        },
    )

    class _Service:
        service_id = "api-service"

        async def send_command_and_wait_for_response(
            self, query: GetPodStatesCommand, timeout: float
        ) -> object:
            assert isinstance(query, GetPodStatesCommand)
            assert timeout > 0
            return response

    return _Service()


class TestProgressEndpoint:
    """Test the /api/progress endpoint."""

    def test_progress_empty(self, progress_client: TestClient) -> None:
        response = progress_client.get("/api/progress")
        assert response.status_code == 200
        data = response.json()
        assert data["phases"] == {}

    def test_progress_with_phases(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        progress_router._progress_tracker._phases = {
            "warmup": CombinedPhaseStats(
                phase="warmup",
                total_expected_requests=100,
                requests_completed=50,
                start_ns=1000,
                last_update_ns=2000,
            )
        }
        response = progress_client.get("/api/progress")
        data = response.json()
        assert "warmup" in data["phases"]
        warmup = data["phases"]["warmup"]
        assert warmup["total_expected_requests"] == 100
        assert warmup["requests_completed"] == 50


class TestProgressWorkerStateQuery:
    """Controller snapshots survive late subscription and pub/sub loss."""

    def test_authoritative_query_populates_workers_with_empty_local_cache(
        self, progress_client: TestClient
    ) -> None:
        progress_client.app.state.service = _controller_service(
            {"0": _pod("0", declared=4, ready=4)}
        )

        workers = progress_client.get("/api/progress").json()["workers"]

        assert workers["ready"] == 4
        assert workers["total"] == 4
        assert workers["total_pods"] == 1

    def test_local_controller_handle_remains_supported(
        self, progress_client: TestClient
    ) -> None:
        class _LocalController:
            def get_pod_state_snapshot(self) -> PodStateSnapshot:
                return PodStateSnapshot(
                    pod_states={"0": _pod("0", declared=2, ready=1)}
                )

        progress_client.app.state.controller = _LocalController()

        workers = progress_client.get("/api/progress").json()["workers"]

        assert workers["ready"] == 1
        assert workers["total"] == 2

    @pytest.mark.asyncio
    async def test_empty_authoritative_state_overrides_stale_cache(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        await progress_router._on_worker_pod_state(_pod("stale", declared=8, ready=8))
        progress_client.app.state.service = _controller_service({})

        workers = progress_client.get("/api/progress").json()["workers"]

        assert workers["ready"] == 0
        assert workers["total"] == 0
        assert workers["total_pods"] == 0

    @pytest.mark.asyncio
    async def test_query_failure_falls_back_to_bus_cache(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        await progress_router._on_worker_pod_state(_pod("0", declared=3, ready=2))

        class _UnavailableService:
            service_id = "api-service"

            async def send_command_and_wait_for_response(
                self, _query: GetPodStatesCommand, timeout: float
            ) -> object:
                raise TimeoutError(f"controller missed {timeout}s deadline")

        progress_client.app.state.service = _UnavailableService()

        workers = progress_client.get("/api/progress").json()["workers"]

        assert workers["ready"] == 2
        assert workers["total"] == 3
        assert workers["total_pods"] == 1


class TestProgressRouterSystemState:
    """Tests for SYSTEM_STATE_CHANGED handling and system_state on /api/progress."""

    def test_default_system_state_is_initializing(
        self, progress_router: ProgressRouter
    ) -> None:
        assert progress_router._system_state == SystemState.INITIALIZING

    @pytest.mark.asyncio
    async def test_on_system_state_changed_updates_attribute(
        self, progress_router: ProgressRouter
    ) -> None:
        await progress_router._on_system_state_changed(
            SystemStateChangedMessage(
                service_id="system_controller",
                state=SystemState.PROFILING,
            )
        )
        assert progress_router._system_state == SystemState.PROFILING

    def test_progress_response_initializes_system_state_initializing(
        self, progress_client: TestClient
    ) -> None:
        data = progress_client.get("/api/progress").json()
        assert data["system_state"] == SystemState.INITIALIZING.value

    def test_progress_response_reflects_latest_system_state(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        import asyncio

        async def feed() -> None:
            for state in (
                SystemState.CONFIGURING,
                SystemState.READY,
                SystemState.PROFILING,
            ):
                await progress_router._on_system_state_changed(
                    SystemStateChangedMessage(
                        service_id="system_controller", state=state
                    )
                )

        asyncio.run(feed())
        data = progress_client.get("/api/progress").json()
        assert data["system_state"] == SystemState.PROFILING.value


@pytest.mark.asyncio
async def test_patch_jobset_annotations_uses_uid_fenced_json_patch(
    monkeypatch,
) -> None:
    from contextlib import asynccontextmanager

    import kubernetes_asyncio

    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={
            "metadata": {
                "uid": "jobset-uid",
                "annotations": {},
                "ownerReferences": [
                    {
                        "apiVersion": "aiperf.nvidia.com/v1alpha1",
                        "kind": "AIPerfJob",
                        "name": "job-1",
                        "uid": "job-uid",
                        "controller": True,
                    }
                ],
            }
        }
    )
    custom.patch_namespaced_custom_object = AsyncMock()
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

    from aiperf.api.routers.progress import _patch_jobset_annotations

    await _patch_jobset_annotations(
        job_id="job-1",
        job_uid="job-uid",
        namespace="ns",
        annotations={"k": "v"},
    )

    kwargs = custom.patch_namespaced_custom_object.call_args.kwargs
    assert kwargs["body"] == [
        {"op": "test", "path": "/metadata/uid", "value": "jobset-uid"},
        {
            "op": "add",
            "path": "/metadata/annotations/k",
            "value": "v",
        },
    ]
    assert kwargs["_content_type"] == "application/json-patch+json"


class TestBuildProgressAnnotationsSystemState:
    """`_build_progress_annotations` always emits the SYSTEM_STATE key."""

    def test_includes_system_state_when_phases_empty(self) -> None:
        from aiperf.api.routers.progress import _build_progress_annotations
        from aiperf.kubernetes.constants import ProgressAnnotations

        ann = _build_progress_annotations({}, SystemState.INITIALIZING)
        assert ann[ProgressAnnotations.STATUS] == "initializing"
        assert ann[ProgressAnnotations.SYSTEM_STATE] == "initializing"

    def test_includes_system_state_when_phases_present(self) -> None:
        from aiperf.api.routers.progress import _build_progress_annotations
        from aiperf.kubernetes.constants import ProgressAnnotations

        phases = {
            "profiling": CombinedPhaseStats(
                phase="profiling",
                total_expected_requests=100,
                requests_completed=25,
                start_ns=1000,
                last_update_ns=2000,
            )
        }
        ann = _build_progress_annotations(phases, SystemState.PROFILING)
        assert ann[ProgressAnnotations.PHASE] == "profiling"
        assert ann[ProgressAnnotations.STATUS] == "running"
        assert ann[ProgressAnnotations.SYSTEM_STATE] == "profiling"

    @pytest.mark.parametrize(
        "state, expected",
        [
            pytest.param(SystemState.INITIALIZING, "initializing", id="initializing"),
            pytest.param(SystemState.CONFIGURING, "configuring", id="configuring"),
            pytest.param(SystemState.READY, "ready", id="ready"),
            pytest.param(SystemState.PROFILING, "profiling", id="profiling"),
            pytest.param(SystemState.PROCESSING, "processing", id="processing"),
            pytest.param(SystemState.STOPPING, "stopping", id="stopping"),
            pytest.param(SystemState.SHUTDOWN, "shutdown", id="shutdown"),
        ],
    )  # fmt: skip
    def test_system_state_value_propagates_to_annotation(
        self, state: SystemState, expected: str
    ) -> None:
        from aiperf.api.routers.progress import _build_progress_annotations
        from aiperf.kubernetes.constants import ProgressAnnotations

        ann_empty = _build_progress_annotations({}, state)
        assert ann_empty[ProgressAnnotations.SYSTEM_STATE] == expected

        phases = {
            "warmup": CombinedPhaseStats(
                phase="warmup",
                total_expected_requests=10,
                requests_completed=5,
                start_ns=1000,
                last_update_ns=2000,
            )
        }
        ann_phases = _build_progress_annotations(phases, state)
        assert ann_phases[ProgressAnnotations.SYSTEM_STATE] == expected

    def test_system_state_only_change_breaks_dedup_equality(self) -> None:
        """Dedup gate must treat a system_state-only transition as a change.

        If phases are unchanged but the controller transitions
        configuring -> ready, the annotation dict must differ so
        `_patch_jobset_progress` does not skip the patch.
        """
        from aiperf.api.routers.progress import _build_progress_annotations

        phases = {
            "profiling": CombinedPhaseStats(
                phase="profiling",
                total_expected_requests=100,
                requests_completed=10,
                start_ns=1000,
                last_update_ns=2000,
            )
        }
        before = _build_progress_annotations(phases, SystemState.CONFIGURING)
        after = _build_progress_annotations(phases, SystemState.READY)
        assert before != after


@pytest.mark.asyncio
async def test_patch_aiperfjob_annotations_uses_aiperfjob_crd_refs(
    monkeypatch,
) -> None:
    """Verifies the AIPerfJob mirror patches the AIPerf CRD (not JobSet).

    `name=job_id` must be passed verbatim — the AIPerfJob CR name equals
    job_id, unlike the JobSet which is `aiperf-{job_id}`.
    """
    from contextlib import asynccontextmanager

    import kubernetes_asyncio

    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={
            "metadata": {
                "uid": "job-uid",
                "annotations": {},
            }
        }
    )
    custom.patch_namespaced_custom_object = AsyncMock()
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

    from aiperf.api.routers.progress import _patch_aiperfjob_annotations
    from aiperf.kubernetes.cr_refs import (
        AIPERF_GROUP,
        AIPERF_PLURAL,
        AIPERF_VERSION,
    )

    await _patch_aiperfjob_annotations(
        job_id="job-1",
        job_uid="job-uid",
        namespace="ns",
        annotations={"k": "v"},
    )

    kwargs = custom.patch_namespaced_custom_object.call_args.kwargs
    assert kwargs["group"] == AIPERF_GROUP
    assert kwargs["version"] == AIPERF_VERSION
    assert kwargs["plural"] == AIPERF_PLURAL
    assert kwargs["namespace"] == "ns"
    assert (
        kwargs["name"] == "job-1"
    )  # NOT aiperf-job-1 — AIPerfJob name is the bare job_id
    assert kwargs["body"] == [
        {"op": "test", "path": "/metadata/uid", "value": "job-uid"},
        {
            "op": "add",
            "path": "/metadata/annotations/k",
            "value": "v",
        },
    ]
    assert kwargs["_content_type"] == "application/json-patch+json"


def test_uid_fenced_annotation_patch_handles_absent_annotations() -> None:
    from aiperf.api.routers.progress import _uid_fenced_annotation_patch

    assert _uid_fenced_annotation_patch(
        metadata={"uid": "resource-uid", "resourceVersion": "31"},
        expected_uid="resource-uid",
        annotations={"aiperf.nvidia.com/progress": "10.0"},
    ) == [
        {"op": "test", "path": "/metadata/uid", "value": "resource-uid"},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": "31",
        },
        {
            "op": "add",
            "path": "/metadata/annotations",
            "value": {"aiperf.nvidia.com/progress": "10.0"},
        },
    ]


@pytest.mark.asyncio
async def test_aiperfjob_replacement_between_read_and_patch_is_rejected(
    monkeypatch,
) -> None:
    """The API server applies the UID test against the live replacement."""
    from contextlib import asynccontextmanager

    import kubernetes_asyncio

    class ReplacementAwareCustomApi:
        def __init__(self) -> None:
            self.live_uid = "old-uid"
            self.patch_attempts = 0

        async def get_namespaced_custom_object(self, **_kwargs: Any) -> dict[str, Any]:
            resource = {"metadata": {"uid": self.live_uid, "annotations": {}}}
            self.live_uid = "replacement-uid"
            return resource

        async def patch_namespaced_custom_object(self, **kwargs: Any) -> None:
            self.patch_attempts += 1
            uid_test = kwargs["body"][0]
            if uid_test["value"] != self.live_uid:
                raise ApiException(status=422, reason="JSON Patch test failed")

    custom = ReplacementAwareCustomApi()
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

    from aiperf.api.routers.progress import _patch_aiperfjob_annotations

    with pytest.raises(ApiException, match="JSON Patch test failed"):
        await _patch_aiperfjob_annotations(
            job_id="job-1",
            job_uid="old-uid",
            namespace="ns",
            annotations={"k": "v"},
        )

    assert custom.patch_attempts == 1


@pytest.mark.asyncio
async def test_jobset_owned_by_replacement_aiperfjob_is_not_patched(
    monkeypatch,
) -> None:
    from contextlib import asynccontextmanager

    import kubernetes_asyncio

    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={
            "metadata": {
                "uid": "replacement-jobset-uid",
                "annotations": {},
                "ownerReferences": [
                    {
                        "apiVersion": "aiperf.nvidia.com/v1alpha1",
                        "kind": "AIPerfJob",
                        "name": "job-1",
                        "uid": "replacement-job-uid",
                        "controller": True,
                    }
                ],
            }
        }
    )
    custom.patch_namespaced_custom_object = AsyncMock()
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

    from aiperf.api.routers.progress import _patch_jobset_annotations

    with pytest.raises(ValueError, match="not owned by the expected"):
        await _patch_jobset_annotations(
            job_id="job-1",
            job_uid="old-job-uid",
            namespace="ns",
            annotations={"k": "v"},
        )

    custom.patch_namespaced_custom_object.assert_not_awaited()


class TestPatchJobsetProgressMirrorsBoth:
    """`_patch_jobset_progress` patches BOTH the JobSet AND AIPerfJob CR."""

    def test_patching_is_disabled_without_exact_job_uid(
        self, monkeypatch, mock_zmq, router_config: AIPerfConfig
    ) -> None:
        monkeypatch.setenv("AIPERF_JOB_ID", "job-xyz")
        monkeypatch.setenv("AIPERF_NAMESPACE", "ns-xyz")
        monkeypatch.delenv("AIPERF_JOB_UID", raising=False)

        router = ProgressRouter(run=router_config)

        assert router._k8s_patching_enabled is False

    @pytest.mark.asyncio
    async def test_patches_both_jobset_and_aiperfjob_with_same_annotations(
        self, monkeypatch, progress_router: ProgressRouter
    ) -> None:
        progress_router._k8s_job_id = "job-xyz"
        progress_router._k8s_job_uid = "uid-job-xyz"
        progress_router._k8s_namespace = "ns-xyz"
        progress_router._k8s_patching_enabled = True
        progress_router._system_state = SystemState.READY

        jobset_calls: list[dict] = []
        aiperfjob_calls: list[dict] = []

        async def fake_jobset(job_id, job_uid, namespace, annotations):
            jobset_calls.append(
                {
                    "job_id": job_id,
                    "job_uid": job_uid,
                    "namespace": namespace,
                    "annotations": annotations,
                }
            )

        async def fake_aiperfjob(job_id, job_uid, namespace, annotations):
            aiperfjob_calls.append(
                {
                    "job_id": job_id,
                    "job_uid": job_uid,
                    "namespace": namespace,
                    "annotations": annotations,
                }
            )

        import aiperf.api.routers.progress as progress_mod

        monkeypatch.setattr(progress_mod, "_patch_jobset_annotations", fake_jobset)
        monkeypatch.setattr(
            progress_mod, "_patch_aiperfjob_annotations", fake_aiperfjob
        )

        await progress_router._patch_jobset_progress()

        assert len(jobset_calls) == 1
        assert len(aiperfjob_calls) == 1
        assert jobset_calls[0]["annotations"] == aiperfjob_calls[0]["annotations"]
        assert aiperfjob_calls[0]["job_id"] == "job-xyz"
        assert aiperfjob_calls[0]["job_uid"] == "uid-job-xyz"
        assert aiperfjob_calls[0]["namespace"] == "ns-xyz"
        # Annotations always carry SYSTEM_STATE.
        from aiperf.kubernetes.constants import ProgressAnnotations

        assert (
            aiperfjob_calls[0]["annotations"][ProgressAnnotations.SYSTEM_STATE]
            == "ready"
        )

    @pytest.mark.asyncio
    async def test_aiperfjob_patch_failure_does_not_crash_loop(
        self, monkeypatch, progress_router: ProgressRouter
    ) -> None:
        """AIPerfJob mirror is best-effort; failures must be swallowed."""
        progress_router._k8s_job_id = "job-xyz"
        progress_router._k8s_job_uid = "uid-job-xyz"
        progress_router._k8s_namespace = "ns-xyz"
        progress_router._k8s_patching_enabled = True

        async def ok_jobset(*_a: object, **_kw: object) -> None:
            return None

        async def boom_aiperfjob(*_a, **_kw):
            raise RuntimeError("apiserver rejected patch")

        import aiperf.api.routers.progress as progress_mod

        monkeypatch.setattr(progress_mod, "_patch_jobset_annotations", ok_jobset)
        monkeypatch.setattr(
            progress_mod, "_patch_aiperfjob_annotations", boom_aiperfjob
        )

        # Must not raise.
        await progress_router._patch_jobset_progress()

    @pytest.mark.asyncio
    async def test_aiperfjob_patch_failure_retries_unchanged_annotations(
        self, monkeypatch, progress_router: ProgressRouter
    ) -> None:
        """A successful JobSet patch must not suppress the failed CR retry."""
        progress_router._k8s_job_id = "job-xyz"
        progress_router._k8s_job_uid = "uid-job-xyz"
        progress_router._k8s_namespace = "ns-xyz"
        progress_router._k8s_patching_enabled = True

        jobset_calls = 0
        aiperfjob_calls = 0

        async def ok_jobset(*_a: object, **_kw: object) -> None:
            nonlocal jobset_calls
            jobset_calls += 1

        async def flaky_aiperfjob(*_a: object, **_kw: object) -> None:
            nonlocal aiperfjob_calls
            aiperfjob_calls += 1
            if aiperfjob_calls == 1:
                raise RuntimeError("transient CR patch failure")

        import aiperf.api.routers.progress as progress_mod

        monkeypatch.setattr(progress_mod, "_patch_jobset_annotations", ok_jobset)
        monkeypatch.setattr(
            progress_mod, "_patch_aiperfjob_annotations", flaky_aiperfjob
        )

        await progress_router._patch_jobset_progress()
        await progress_router._patch_jobset_progress()

        assert jobset_calls == 1
        assert aiperfjob_calls == 2

    @pytest.mark.asyncio
    async def test_jobset_patch_failure_retries_unchanged_annotations(
        self, monkeypatch, progress_router: ProgressRouter
    ) -> None:
        """A successful CR patch must not suppress the failed JobSet retry."""
        progress_router._k8s_job_id = "job-xyz"
        progress_router._k8s_job_uid = "uid-job-xyz"
        progress_router._k8s_namespace = "ns-xyz"
        progress_router._k8s_patching_enabled = True

        jobset_calls = 0
        aiperfjob_calls = 0

        async def flaky_jobset(*_a: object, **_kw: object) -> None:
            nonlocal jobset_calls
            jobset_calls += 1
            if jobset_calls == 1:
                raise RuntimeError("transient JobSet patch failure")

        async def ok_aiperfjob(*_a: object, **_kw: object) -> None:
            nonlocal aiperfjob_calls
            aiperfjob_calls += 1

        import aiperf.api.routers.progress as progress_mod

        monkeypatch.setattr(progress_mod, "_patch_jobset_annotations", flaky_jobset)
        monkeypatch.setattr(progress_mod, "_patch_aiperfjob_annotations", ok_aiperfjob)

        await progress_router._patch_jobset_progress()
        await progress_router._patch_jobset_progress()

        assert jobset_calls == 2
        assert aiperfjob_calls == 1

    @pytest.mark.asyncio
    async def test_dedup_skips_when_system_state_and_phases_unchanged(
        self, monkeypatch, progress_router: ProgressRouter
    ) -> None:
        """Two consecutive ticks with identical (phases, system_state) → 1 patch each."""
        progress_router._k8s_job_id = "job-xyz"
        progress_router._k8s_job_uid = "uid-job-xyz"
        progress_router._k8s_namespace = "ns-xyz"
        progress_router._k8s_patching_enabled = True
        progress_router._system_state = SystemState.PROFILING

        jobset_calls = 0
        aiperfjob_calls = 0

        async def fake_jobset(*_a, **_kw):
            nonlocal jobset_calls
            jobset_calls += 1

        async def fake_aiperfjob(*_a, **_kw):
            nonlocal aiperfjob_calls
            aiperfjob_calls += 1

        import aiperf.api.routers.progress as progress_mod

        monkeypatch.setattr(progress_mod, "_patch_jobset_annotations", fake_jobset)
        monkeypatch.setattr(
            progress_mod, "_patch_aiperfjob_annotations", fake_aiperfjob
        )

        await progress_router._patch_jobset_progress()
        await progress_router._patch_jobset_progress()
        # Second tick deduped (identical state) → only one JobSet patch total.
        assert jobset_calls == 1
        # AIPerfJob mirror is gated by the same dedup check.
        assert aiperfjob_calls == 1

    @pytest.mark.asyncio
    async def test_system_state_only_change_triggers_patch(
        self, monkeypatch, progress_router: ProgressRouter
    ) -> None:
        """system_state changing while phases stay constant must trigger a patch."""
        progress_router._k8s_job_id = "job-xyz"
        progress_router._k8s_job_uid = "uid-job-xyz"
        progress_router._k8s_namespace = "ns-xyz"
        progress_router._k8s_patching_enabled = True
        progress_router._system_state = SystemState.CONFIGURING

        jobset_calls: list[dict] = []
        aiperfjob_calls: list[dict] = []

        async def fake_jobset(job_id, job_uid, namespace, annotations):
            jobset_calls.append(dict(annotations))

        async def fake_aiperfjob(job_id, job_uid, namespace, annotations):
            aiperfjob_calls.append(dict(annotations))

        import aiperf.api.routers.progress as progress_mod

        monkeypatch.setattr(progress_mod, "_patch_jobset_annotations", fake_jobset)
        monkeypatch.setattr(
            progress_mod, "_patch_aiperfjob_annotations", fake_aiperfjob
        )

        await progress_router._patch_jobset_progress()
        # Same phases (none) but system_state transitions configuring -> ready.
        progress_router._system_state = SystemState.READY
        await progress_router._patch_jobset_progress()

        from aiperf.kubernetes.constants import ProgressAnnotations

        assert len(jobset_calls) == 2
        assert len(aiperfjob_calls) == 2
        assert jobset_calls[0][ProgressAnnotations.SYSTEM_STATE] == "configuring"
        assert jobset_calls[1][ProgressAnnotations.SYSTEM_STATE] == "ready"
