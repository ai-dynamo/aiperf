# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Override-based coverage for operator policy environment consumers."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import kopf
import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.kubernetes.phase import Phase
from aiperf.operator.environment import (
    OperatorEnvironment,
    _MonitorSettings,
    _ProgressSettings,
    _ReconcileSettings,
    _ResultsSettings,
    _SweepControllerSettings,
)


@pytest.mark.parametrize(
    "settings_type,env_name,field_name,raw_value,expected",
    [
        param(_ProgressSettings, "AIPERF_OPERATOR_PROGRESS_REQUEST_TIMEOUT_SECONDS", "REQUEST_TIMEOUT_SECONDS", "17", 17.0, id="progress-timeout"),
        param(_ResultsSettings, "AIPERF_RESULTS_DOWNLOAD_TIMEOUT_SECONDS", "DOWNLOAD_TIMEOUT_SECONDS", "41", 41.0, id="download-timeout"),
        param(_ResultsSettings, "AIPERF_RESULTS_DOWNLOAD_MAX_CONCURRENCY", "DOWNLOAD_MAX_CONCURRENCY", "7", 7, id="download-concurrency"),
        param(_ResultsSettings, "AIPERF_RESULTS_RETRY_MAX_DELAY_SECONDS", "RETRY_MAX_DELAY_SECONDS", "44", 44.0, id="retry-max-delay"),
        param(_ResultsSettings, "AIPERF_RESULTS_RETRY_BACKOFF_MULTIPLIER", "RETRY_BACKOFF_MULTIPLIER", "3", 3.0, id="retry-multiplier"),
        param(_ResultsSettings, "AIPERF_RESULTS_CLEANUP_INTERVAL_SECONDS", "CLEANUP_INTERVAL_SECONDS", "7200", 7200.0, id="cleanup-interval"),
        param(_ResultsSettings, "AIPERF_RESULTS_CLEANUP_INITIAL_DELAY_SECONDS", "CLEANUP_INITIAL_DELAY_SECONDS", "120", 120.0, id="cleanup-initial-delay"),
        param(_ResultsSettings, "AIPERF_RESULTS_CLEANUP_IDLE_SECONDS", "CLEANUP_IDLE_SECONDS", "240", 240.0, id="cleanup-idle"),
        param(_ResultsSettings, "AIPERF_RESULTS_GZIP_MINIMUM_SIZE_BYTES", "GZIP_MINIMUM_SIZE_BYTES", "777", 777, id="gzip-minimum"),
        param(_MonitorSettings, "AIPERF_OPERATOR_MONITOR_MISSING_JOBSET_SETTLE_DELAY_SECONDS", "MISSING_JOBSET_SETTLE_DELAY_SECONDS", "0.25", 0.25, id="missing-jobset-settle"),
        param(_SweepControllerSettings, "AIPERF_SWEEP_CONTROLLER_CHILD_POLL_INTERVAL_SECONDS", "CHILD_POLL_INTERVAL_SECONDS", "4", 4.0, id="child-poll"),
        param(_SweepControllerSettings, "AIPERF_SWEEP_CONTROLLER_CANCEL_POLL_INTERVAL_SECONDS", "CANCEL_POLL_INTERVAL_SECONDS", "6", 6.0, id="cancel-poll"),
        param(_SweepControllerSettings, "AIPERF_SWEEP_CONTROLLER_RECOVERY_SUMMARY_CONCURRENCY", "RECOVERY_SUMMARY_CONCURRENCY", "11", 11, id="recovery-concurrency"),
        param(_SweepControllerSettings, "AIPERF_SWEEP_CONTROLLER_OPERATOR_API_MAX_ATTEMPTS", "OPERATOR_API_MAX_ATTEMPTS", "4", 4, id="operator-api-attempts"),
        param(_SweepControllerSettings, "AIPERF_SWEEP_CONTROLLER_OPERATOR_API_REQUEST_TIMEOUT_SECONDS", "OPERATOR_API_REQUEST_TIMEOUT_SECONDS", "12", 12.0, id="operator-api-timeout"),
        param(_SweepControllerSettings, "AIPERF_SWEEP_CONTROLLER_OPERATOR_API_INITIAL_BACKOFF_SECONDS", "OPERATOR_API_INITIAL_BACKOFF_SECONDS", "0.75", 0.75, id="operator-api-backoff"),
        param(_SweepControllerSettings, "AIPERF_SWEEP_CONTROLLER_OPERATOR_API_BACKOFF_MULTIPLIER", "OPERATOR_API_BACKOFF_MULTIPLIER", "2.5", 2.5, id="operator-api-multiplier"),
        param(_ReconcileSettings, "AIPERF_OPERATOR_RECONCILE_CONFLICT_RETRY_DELAY_SECONDS", "CONFLICT_RETRY_DELAY_SECONDS", "0.2", 0.2, id="reconcile-conflict"),
        param(_ReconcileSettings, "AIPERF_OPERATOR_RECONCILE_RUNS_CAS_MAX_ATTEMPTS", "RUNS_CAS_MAX_ATTEMPTS", "9", 9, id="runs-cas"),
        param(_ReconcileSettings, "AIPERF_OPERATOR_RECONCILE_EVENT_RETRY_DELAY_SECONDS", "EVENT_RETRY_DELAY_SECONDS", "1.2", 1.2, id="reconcile-event"),
        param(_ReconcileSettings, "AIPERF_OPERATOR_RECONCILE_PERSISTENCE_RETRY_DELAY_SECONDS", "PERSISTENCE_RETRY_DELAY_SECONDS", "2.2", 2.2, id="reconcile-persistence"),
        param(_ReconcileSettings, "AIPERF_OPERATOR_RECONCILE_STATE_RETRY_DELAY_SECONDS", "STATE_RETRY_DELAY_SECONDS", "3.2", 3.2, id="reconcile-state"),
        param(_ReconcileSettings, "AIPERF_OPERATOR_RECONCILE_CREATE_HARVEST_RETRY_DELAY_SECONDS", "CREATE_HARVEST_RETRY_DELAY_SECONDS", "4.2", 4.2, id="reconcile-create-harvest"),
        param(_ReconcileSettings, "AIPERF_OPERATOR_RECONCILE_TTL_DELETE_RETRY_DELAY_SECONDS", "TTL_DELETE_RETRY_DELAY_SECONDS", "5.2", 5.2, id="reconcile-ttl-delete"),
    ],
)  # fmt: skip
def test_policy_environment_override_reaches_field(
    settings_type: type[Any],
    env_name: str,
    field_name: str,
    raw_value: str,
    expected: float | int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env_name, raw_value)

    assert getattr(settings_type(), field_name) == expected


def test_results_retry_max_delay_rejects_below_initial_delay() -> None:
    with pytest.raises(ValidationError, match="RETRY_MAX_DELAY_SECONDS"):
        _ResultsSettings(RETRY_DELAY=4.0, RETRY_MAX_DELAY_SECONDS=3.0)


@pytest.mark.parametrize(
    "deadline_field",
    [
        param("CANCEL_GRACE_SECONDS", id="cancel-grace"),
        param("CHILD_MISSING_TIMEOUT_SECONDS", id="missing-child"),
    ],
)
def test_child_poll_interval_rejects_larger_deadline(deadline_field: str) -> None:
    with pytest.raises(ValidationError, match="CHILD_POLL_INTERVAL_SECONDS"):
        _SweepControllerSettings(
            CHILD_POLL_INTERVAL_SECONDS=6.0,
            **{deadline_field: 5.0},
        )


@pytest.mark.asyncio
async def test_progress_request_timeout_override_reaches_aiohttp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiperf.operator import progress_client as progress_mod

    captured: dict[str, Any] = {}

    class FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def close(self) -> None:
            return None

    monkeypatch.setattr(OperatorEnvironment.PROGRESS, "REQUEST_TIMEOUT_SECONDS", 13.5)
    monkeypatch.setattr(progress_mod, "create_tcp_connector", lambda: object())
    monkeypatch.setattr(progress_mod.aiohttp, "ClientSession", FakeSession)

    async with progress_mod.ProgressClient():
        pass

    assert captured["timeout"].total == 13.5


@pytest.mark.asyncio
async def test_result_download_timeout_override_reaches_aiohttp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aiperf.operator import progress_client as progress_mod

    captured: dict[str, Any] = {}

    class FakeResponse:
        status = 404

        async def __aenter__(self) -> FakeResponse:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def get(self, _url: str, **_kwargs: Any) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DOWNLOAD_TIMEOUT_SECONDS", 23.0)
    monkeypatch.setattr(progress_mod.aiohttp, "ClientSession", FakeSession)
    client = progress_mod.ProgressClient()
    client._session = SimpleNamespace(connector=object())  # type: ignore[assignment]

    downloaded = await client._stream_result_file(
        "http://controller/result", "result.json", tmp_path / "result.json"
    )

    assert downloaded is False
    assert captured["timeout"].total == 23.0


@pytest.mark.asyncio
async def test_result_download_concurrency_override_reaches_semaphore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aiperf.operator import progress_client as progress_mod

    limits: list[int] = []
    real_semaphore = asyncio.Semaphore

    def capture_semaphore(limit: int) -> asyncio.Semaphore:
        limits.append(limit)
        return real_semaphore(limit)

    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DOWNLOAD_MAX_CONCURRENCY", 3)
    monkeypatch.setattr(progress_mod.asyncio, "Semaphore", capture_semaphore)
    monkeypatch.setattr(
        progress_mod.ProgressClient,
        "get_results_list",
        AsyncMock(return_value=[{"name": "result.json"}]),
    )
    monkeypatch.setattr(
        progress_mod.ProgressClient,
        "download_result_file",
        AsyncMock(return_value=True),
    )

    downloaded = await progress_mod.ProgressClient().download_all_results(
        "controller", tmp_path
    )

    assert downloaded == ["result.json"]
    assert limits == [3]


@pytest.mark.asyncio
async def test_result_retry_policy_override_controls_backoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aiperf.operator.handlers import _completion_fetch as fetch_mod

    sleeps: list[float] = []

    async def fail() -> Any:
        raise OSError("not ready")

    async def capture_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(OperatorEnvironment.RESULTS, "RETRY_MAX_DELAY_SECONDS", 4.0)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "RETRY_BACKOFF_MULTIPLIER", 3.0)
    monkeypatch.setattr(fetch_mod.asyncio, "sleep", capture_sleep)
    monkeypatch.setattr(fetch_mod.random, "uniform", lambda _low, _high: 1.0)

    with pytest.raises(OSError, match="not ready"):
        await fetch_mod._fetch_with_progress_aware_retry(
            fail,
            dest_dir=tmp_path,
            job_id="job",
            initial_delay=2.0,
            description="test fetch",
            stagnation_limit=3,
        )

    assert sleeps == [2.0, 4.0]


def test_cleanup_timer_delay_overrides_reach_kopf_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kopf._core.intents import registries

    from aiperf.operator import main as operator_main

    isolated_registry = registries.OperatorRegistry()
    with monkeypatch.context() as context:
        context.setattr(
            OperatorEnvironment.RESULTS, "CLEANUP_INITIAL_DELAY_SECONDS", 17.0
        )
        context.setattr(OperatorEnvironment.RESULTS, "CLEANUP_IDLE_SECONDS", 23.0)
        context.setattr(registries, "get_default_registry", lambda: isolated_registry)
        importlib.reload(operator_main)

    timer = next(
        handler
        for handler in isolated_registry._spawning.get_all_handlers()  # noqa: SLF001
        if handler.fn is operator_main.cleanup_old_results
    )
    assert timer.initial_delay == 17.0
    assert timer.idle == 23.0

    restored_registry = registries.OperatorRegistry()
    with monkeypatch.context() as context:
        context.setattr(registries, "get_default_registry", lambda: restored_registry)
        importlib.reload(operator_main)


@pytest.mark.asyncio
async def test_sweep_cleanup_interval_override_reaches_retention_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aiperf.operator import main as operator_main

    timeouts: list[float | None] = []

    async def capture_wait_for(_awaitable: Any, *, timeout: float | None) -> None:
        timeouts.append(timeout)
        _awaitable.close()
        raise asyncio.CancelledError

    bootstrap_task = asyncio.create_task(asyncio.sleep(0))
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "CLEANUP_INTERVAL_SECONDS", 321.0)
    monkeypatch.setattr(operator_main.cleanup, "reconcile_sweep_results", AsyncMock())
    monkeypatch.setattr(operator_main.asyncio, "wait_for", capture_wait_for)

    with pytest.raises(asyncio.CancelledError):
        await operator_main._run_sweep_results_retention(tmp_path, bootstrap_task)

    assert timeouts == [321.0]


@pytest.mark.asyncio
async def test_missing_jobset_settle_override_reaches_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiperf.operator.handlers import monitor

    sleeps: list[float] = []

    async def capture_sleep(delay: float) -> None:
        sleeps.append(delay)

    custom = SimpleNamespace(
        get_namespaced_custom_object=AsyncMock(
            return_value={
                "metadata": {"uid": "uid"},
                "status": {"phase": Phase.COMPLETED},
            }
        )
    )
    monkeypatch.setattr(
        OperatorEnvironment.MONITOR, "MISSING_JOBSET_SETTLE_DELAY_SECONDS", 0.4
    )
    monkeypatch.setattr(monitor.asyncio, "sleep", capture_sleep)

    short_circuit = await monitor._reconcile_missing_jobset(
        custom,
        body={"metadata": {"uid": "uid"}},
        namespace="ns",
        name="job",
        jobset_name="aiperf-job",
        current_phase=Phase.RUNNING,
        sb=MagicMock(),
    )

    assert short_circuit is True
    assert sleeps == [0.4]


@pytest.mark.asyncio
async def test_sweep_cancel_poll_override_reaches_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiperf.sweep_controller import main as sweep_main

    sleeps: list[float] = []
    responses = [RuntimeError("temporary"), {"spec": {"cancel": True}}]

    async def read_parent(**_kwargs: Any) -> dict[str, Any]:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def capture_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        OperatorEnvironment.SWEEP_CONTROLLER, "CANCEL_POLL_INTERVAL_SECONDS", 0.3
    )
    monkeypatch.setattr(sweep_main.asyncio, "sleep", capture_sleep)
    flag = {"requested": False}

    await sweep_main._poll_cancel_flag(
        SimpleNamespace(get_namespaced_custom_object=read_parent),
        namespace="ns",
        name="sweep",
        flag=flag,
    )

    assert flag["requested"] is True
    assert sleeps == [0.3]


@pytest.mark.asyncio
async def test_sweep_child_poll_override_reaches_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiperf.sweep_controller import k8s_executor as executor_mod

    executor = object.__new__(executor_mod.K8sChildJobExecutor)
    executor.sweep_uid = "sweep-uid"
    executor.sweep_name = "sweep"
    executor.sweep_run_epoch = "1770000000"
    monkeypatch.setattr(
        executor_mod.K8sChildJobExecutor,
        "_try_read_child",
        AsyncMock(
            side_effect=[
                {"status": {"phase": "Running"}},
                {"status": {"phase": "Completed"}},
            ]
        ),
    )
    monkeypatch.setattr(executor_mod, "is_my_child", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        OperatorEnvironment.SWEEP_CONTROLLER, "CHILD_POLL_INTERVAL_SECONDS", 0.6
    )
    sleep = AsyncMock()
    monkeypatch.setattr(executor_mod.asyncio, "sleep", sleep)

    result = await executor._wait_until_terminal(
        "child", MagicMock(), expected_child_uid="child-uid"
    )

    assert result is None
    sleep.assert_awaited_once_with(0.6)


@pytest.mark.asyncio
async def test_operator_api_retry_overrides_reach_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiperf.sweep_controller import k8s_executor as executor_mod

    timeouts: list[float | None] = []
    sleeps: list[float] = []

    class FakeResponse:
        status = 503

        async def __aenter__(self) -> FakeResponse:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class FakeSession:
        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def get(self, _url: str, *, timeout: aiohttp.ClientTimeout) -> FakeResponse:
            timeouts.append(timeout.total)
            return FakeResponse()

    async def capture_sleep(delay: float) -> None:
        sleeps.append(delay)

    settings = OperatorEnvironment.SWEEP_CONTROLLER
    monkeypatch.setattr(settings, "OPERATOR_API_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(settings, "OPERATOR_API_REQUEST_TIMEOUT_SECONDS", 7.0)
    monkeypatch.setattr(settings, "OPERATOR_API_INITIAL_BACKOFF_SECONDS", 0.25)
    monkeypatch.setattr(settings, "OPERATOR_API_BACKOFF_MULTIPLIER", 3.0)
    monkeypatch.setattr(executor_mod.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(executor_mod.asyncio, "sleep", capture_sleep)

    executor = object.__new__(executor_mod.K8sChildJobExecutor)
    summary = await executor._fetch_summary_from_operator(
        {
            "metadata": {"namespace": "ns", "name": "child"},
            "status": {"runEpoch": "1770000000"},
        }
    )

    assert summary == {}
    assert timeouts == [7.0, 7.0]
    assert sleeps == [0.25]


@pytest.mark.asyncio
async def test_runs_cas_and_event_delay_overrides_reach_retry_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiperf.operator.handlers.sweep import _child_runs

    read_state = AsyncMock(return_value=([], 2, True, True, False, "rv"))
    append_patch = AsyncMock(return_value="retry")
    monkeypatch.setattr(_child_runs, "_read_runs_state", read_state)
    monkeypatch.setattr(_child_runs, "_append_run_entry_patch", append_patch)
    monkeypatch.setattr(OperatorEnvironment.RECONCILE, "RUNS_CAS_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(
        OperatorEnvironment.RECONCILE, "EVENT_RETRY_DELAY_SECONDS", 1.75
    )

    with pytest.raises(kopf.TemporaryError) as exc_info:
        await _child_runs.append_run_entry(
            "ns",
            "sweep",
            {"childName": "child"},
            api=MagicMock(),
        )

    assert read_state.await_count == 2
    assert append_patch.await_count == 2
    assert exc_info.value.delay == 1.75


@pytest.mark.asyncio
async def test_conflict_delay_override_reaches_event_status_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kubernetes_asyncio.client import ApiException

    from aiperf.operator.handlers import monitor

    patch_status = AsyncMock(
        side_effect=ApiException(status=409, reason="resourceVersion conflict")
    )
    custom = SimpleNamespace(patch_namespaced_custom_object_status=patch_status)

    @asynccontextmanager
    async def fake_k8s_client() -> AsyncIterator[MagicMock]:
        yield MagicMock(name="ApiClient")

    monkeypatch.setattr(
        OperatorEnvironment.RECONCILE, "CONFLICT_RETRY_DELAY_SECONDS", 0.375
    )
    monkeypatch.setattr(
        monitor,
        "_live_event_status_fence",
        AsyncMock(return_value=({"status": {}}, "resource-version", Phase.RUNNING)),
    )
    monkeypatch.setattr(monitor, "k8s_client", fake_k8s_client)
    monkeypatch.setattr(monitor.client, "CustomObjectsApi", lambda _api: custom)

    with pytest.raises(kopf.TemporaryError) as exc_info:
        await monitor._patch_event_status(
            body={"metadata": {"name": "job", "uid": "job-uid"}},
            namespace="ns",
            name="job",
            status_patch_builder=lambda _body: {"activeWorkers": 1},
        )

    assert exc_info.value.delay == 0.375
    patch_status.assert_awaited_once()


@pytest.mark.asyncio
async def test_persistence_delay_override_reaches_create_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiperf.operator.handlers import create

    monkeypatch.setattr(
        OperatorEnvironment.RECONCILE, "PERSISTENCE_RETRY_DELAY_SECONDS", 2.25
    )
    monkeypatch.setattr(
        create,
        "save_job_spec_file",
        AsyncMock(side_effect=OSError("PVC unavailable")),
    )

    with pytest.raises(kopf.TemporaryError) as exc_info:
        await create._persist_spec_and_index(
            {},
            "ns",
            "job",
            "job-id",
            body={"metadata": {"creationTimestamp": "2026-08-25T12:00:00Z"}},
        )

    assert exc_info.value.delay == 2.25


@pytest.mark.asyncio
async def test_create_harvest_delay_override_reaches_create_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kubernetes_asyncio.client import ApiException

    from aiperf.operator.handlers import create

    monkeypatch.setattr(
        OperatorEnvironment.RECONCILE,
        "CREATE_HARVEST_RETRY_DELAY_SECONDS",
        4.25,
    )
    monkeypatch.setattr(
        create,
        "_create_resources",
        AsyncMock(side_effect=ApiException(status=503, reason="Unavailable")),
    )
    patch = MagicMock()
    patch.status = {}

    with pytest.raises(kopf.TemporaryError) as exc_info:
        await create.on_create(
            body={"metadata": {"name": "job", "uid": "job-uid"}},
            spec={},
            name="job",
            namespace="ns",
            uid="job-uid",
            patch=patch,
        )

    assert exc_info.value.delay == 4.25


@pytest.mark.asyncio
async def test_ttl_delete_delay_override_reaches_sweep_reaper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kubernetes_asyncio
    from kubernetes_asyncio.client import ApiException

    import aiperf.kubernetes.client as kclient
    from aiperf.operator.handlers.sweep import lifecycle

    delete = AsyncMock(side_effect=ApiException(status=503, reason="Unavailable"))
    custom = SimpleNamespace(delete_namespaced_custom_object=delete)
    fake_k8s_module = SimpleNamespace(
        CustomObjectsApi=lambda _api: custom,
        V1Preconditions=lambda *, uid: SimpleNamespace(uid=uid),
        V1DeleteOptions=lambda *, preconditions: SimpleNamespace(
            preconditions=preconditions
        ),
    )

    @asynccontextmanager
    async def fake_k8s_client() -> AsyncIterator[MagicMock]:
        yield MagicMock(name="ApiClient")

    monkeypatch.setattr(
        OperatorEnvironment.RECONCILE, "TTL_DELETE_RETRY_DELAY_SECONDS", 8.5
    )
    monkeypatch.setattr(kubernetes_asyncio, "client", fake_k8s_module)
    monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)

    with pytest.raises(kopf.TemporaryError) as exc_info:
        await lifecycle.maybe_reap_finished(
            body={
                "metadata": {"name": "sweep", "namespace": "ns", "uid": "sweep-uid"},
                "spec": {"ttlSecondsAfterFinished": 0},
            },
            status={
                "phase": "Succeeded",
                "completionTime": "2020-01-01T00:00:00Z",
            },
            name="sweep",
            namespace="ns",
        )

    assert exc_info.value.delay == 8.5
    delete.assert_awaited_once()


def test_state_retry_override_reaches_identity_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiperf.operator.handlers import _job_identity

    monkeypatch.setattr(
        OperatorEnvironment.RECONCILE, "STATE_RETRY_DELAY_SECONDS", 2.75
    )

    with pytest.raises(kopf.TemporaryError) as exc_info:
        _job_identity.aiperfjob_jobset_uid(
            {
                "metadata": {
                    "name": "aiperf-job",
                    "ownerReferences": [
                        {
                            "apiVersion": "aiperf.nvidia.com/v1alpha1",
                            "kind": "AIPerfJob",
                            "name": "job",
                            "uid": "parent-uid",
                            "controller": True,
                        }
                    ],
                }
            },
            jobset_name="aiperf-job",
            parent_name="job",
            parent_uid="parent-uid",
        )

    assert exc_info.value.delay == 2.75
