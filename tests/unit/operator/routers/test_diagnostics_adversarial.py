# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator diagnostics API endpoints.

Focuses on:
- events endpoint contracts for missing CRs, partial pod failures, API 404 vs 500,
  filtering, sorting, caps, and schema stability
- logs endpoint contracts for pod ownership, tail-line limits, default containers,
  and apiserver error propagation
- conditions surfaced through the job-detail diagnostics payload

Out of scope: browser rendering of diagnostics tabs, covered by
``tests/unit/ui/test_operator_logs_events_edges.py`` and conditions UI tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from aiperf.operator.routers.jobs import create_jobs_router
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.models import AIPerfJobInfo

# ============================================================
# Helpers
# ============================================================


_EVENT_BASE_TS = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)


def _app(api: object | None, results_dir: Path) -> FastAPI:
    """Build the jobs router with the production Kubernetes exception shape."""
    app = FastAPI()

    @app.exception_handler(ApiException)
    async def _api_exception_handler(
        request: Request, exc: ApiException
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=exc.status or 500,
            content={"detail": str(exc.body or exc.reason or "Kubernetes API error")},
        )

    app.include_router(create_jobs_router([api], results_dir))
    return app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client for diagnostics routes with a live ApiClient token."""
    transport = httpx.ASGITransport(
        app=_app(object(), tmp_path), raise_app_exceptions=False
    )
    async with httpx.AsyncClient(
        transport=transport, base_url="http://aiperf.operator.local"
    ) as c:
        yield c


def _live_job_info(
    *,
    namespace: str = "aiperf-benchmarks",
    name: str = "llama-3-8b-diagnostics",
) -> AIPerfJobInfo:
    """Return a real display model matching the diagnostics detail schema."""
    return AIPerfJobInfo(
        name=name,
        namespace=namespace,
        phase="Running",
        job_id=name,
        jobset_name=f"aiperf-{name}",
        workers_ready=2,
        workers_total=3,
        current_phase="profiling",
        created="2026-05-18T12:00:00Z",
        source="live",
        model="meta-llama/Llama-3-8B",
        endpoint="http://vllm-router.aiperf-system:8000/v1",
    )


def _raw_aiperf_job(
    *,
    namespace: str = "aiperf-benchmarks",
    name: str = "llama-3-8b-diagnostics",
) -> dict[str, object]:
    """Return the minimal raw CR body the events endpoint needs."""
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {},
        "status": {"phase": "Running", "jobId": name},
    }


def _pod(
    name: str,
    *,
    containers: list[str] | None = None,
    default_container: str | None = None,
) -> MagicMock:
    """Build a V1Pod-shaped mock for owned diagnostics pods."""
    pod = MagicMock()
    pod.metadata = MagicMock()
    pod.metadata.name = name
    pod.metadata.annotations = {}
    if default_container is not None:
        pod.metadata.annotations["kubectl.kubernetes.io/default-container"] = (
            default_container
        )
    pod.spec = MagicMock()
    pod.spec.containers = [
        SimpleNamespace(name=container_name)
        for container_name in (containers or ["controller", "event-bus"])
    ]
    pod.status = MagicMock()
    pod.status.phase = "Running"
    pod.status.container_statuses = []
    return pod


def _event(
    *,
    reason: str,
    message: str,
    type_: str = "Warning",
    involved_kind: str = "Pod",
    involved_name: str = "llama-3-8b-diagnostics-controller-0",
    minutes_after_base: int = 0,
    first_timestamp: datetime | None = _EVENT_BASE_TS,
    last_timestamp: datetime | None = _EVENT_BASE_TS,
    event_time: datetime | None = None,
    count: int = 1,
) -> MagicMock:
    """Build a V1Event-shaped mock with the attributes the router serializes."""
    ev = MagicMock()
    ev.type = type_
    ev.reason = reason
    ev.message = message
    ev.first_timestamp = first_timestamp
    ev.last_timestamp = (
        None
        if last_timestamp is None
        else last_timestamp + timedelta(minutes=minutes_after_base)
    )
    ev.event_time = event_time
    ev.count = count
    ev.involved_object = SimpleNamespace(
        kind=involved_kind,
        name=involved_name,
        namespace="aiperf-benchmarks",
    )
    ev.source = SimpleNamespace(component="kubelet", host="dgx-node-01")
    return ev


# ============================================================
# Events endpoint errors and partial failures
# ============================================================


class TestDiagnosticsEventsErrors:
    """Events distinguish legitimate disappearance from apiserver failure."""

    @pytest.mark.asyncio
    async def test_list_job_events_missing_cr_returns_empty_stable_schema(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs as jobs_module

        monkeypatch.setattr(
            jobs_module, "get_raw_aiperfjob", AsyncMock(return_value=None)
        )
        event_lookup = AsyncMock()
        monkeypatch.setattr(jobs_module, "list_events_for_object", event_lookup)

        response = await client.get(
            "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-gone/events"
        )

        assert response.status_code == 200, response.text
        assert response.json() == {"events": []}
        event_lookup.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,expected_status,expected_detail",
        [
            param(404, 200, None, id="cr-404-is-archived-empty-events"),
            param(500, 500, "etcd leader unavailable", id="cr-500-surfaces-api-failure"),
        ],
    )  # fmt: skip
    async def test_list_job_events_cr_lookup_404_vs_500_keeps_distinct_contracts(
        self,
        tmp_path: Path,
        status: int,
        expected_status: int,
        expected_detail: str | None,
    ) -> None:
        api_error = ApiException(status=status, reason="Kubernetes API error")
        api_error.body = "etcd leader unavailable"
        mock_custom = MagicMock(
            get_namespaced_custom_object=AsyncMock(side_effect=api_error)
        )
        transport = httpx.ASGITransport(
            app=_app(object(), tmp_path), raise_app_exceptions=False
        )

        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://aiperf.operator.local"
            ) as c:
                response = await c.get(
                    "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-diagnostics/events"
                )

        assert response.status_code == expected_status, response.text
        if expected_detail is None:
            assert response.json() == {"events": []}
        else:
            assert expected_detail in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_list_job_events_one_evicted_pod_404_keeps_remaining_events(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs as jobs_module

        async def fake_events(
            api: object, namespace: str, object_name: str
        ) -> list[MagicMock]:
            del api, namespace
            if object_name == "llama-3-8b-diagnostics-controller-1":
                raise ApiException(status=404, reason="pod no longer exists")
            return [
                _event(
                    reason="Started",
                    message=f"Started container for {object_name}",
                    type_="Normal",
                    involved_name=object_name,
                )
            ]

        monkeypatch.setattr(
            jobs_module,
            "get_raw_aiperfjob",
            AsyncMock(return_value=_raw_aiperf_job()),
        )
        monkeypatch.setattr(
            jobs_module,
            "get_pods",
            AsyncMock(
                return_value=[
                    _pod("llama-3-8b-diagnostics-controller-0"),
                    _pod("llama-3-8b-diagnostics-controller-1"),
                ]
            ),
        )
        monkeypatch.setattr(jobs_module, "list_events_for_object", fake_events)

        response = await client.get(
            "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-diagnostics/events"
        )

        assert response.status_code == 200, response.text
        names = [
            event["involved_object"]["name"] for event in response.json()["events"]
        ]
        assert names == [
            "llama-3-8b-diagnostics",
            "llama-3-8b-diagnostics-controller-0",
        ]


# ============================================================
# Events filtering, caps, counts, and schema
# ============================================================


class TestDiagnosticsEventsResponseShape:
    """Returned events stay bounded, sorted, filtered, and schema-stable."""

    @pytest.mark.asyncio
    async def test_list_job_events_filters_noise_sorts_newest_first_and_caps_at_200(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs as jobs_module

        noise = _event(
            reason="PolicyViolation",
            message="policy validating-node-p4sa-audience failed on request.userInfo",
            involved_name="llama-3-8b-diagnostics-controller-0",
        )
        raw_events = [
            _event(
                reason=f"BackOff{i:03d}",
                message=f"container restart backoff {i}",
                involved_name="llama-3-8b-diagnostics-controller-0",
                minutes_after_base=i,
            )
            for i in range(205)
        ]
        raw_events.insert(17, noise)
        monkeypatch.setattr(
            jobs_module,
            "get_raw_aiperfjob",
            AsyncMock(return_value=_raw_aiperf_job()),
        )
        monkeypatch.setattr(jobs_module, "get_pods", AsyncMock(return_value=[]))
        monkeypatch.setattr(
            jobs_module, "list_events_for_object", AsyncMock(return_value=raw_events)
        )

        response = await client.get(
            "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-diagnostics/events"
        )

        assert response.status_code == 200, response.text
        events = response.json()["events"]
        assert len(events) == 200
        assert events[0]["reason"] == "BackOff204"
        assert events[-1]["reason"] == "BackOff005"
        assert "PolicyViolation" not in {event["reason"] for event in events}

    @pytest.mark.asyncio
    async def test_list_job_events_warning_and_normal_type_counts_are_preserved(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs as jobs_module

        events = [
            _event(
                reason="Scheduled", message="assigned to dgx-node-01", type_="Normal"
            ),
            _event(
                reason="Pulling", message="pulling controller image", type_="Normal"
            ),
            _event(reason="FailedMount", message="timed out waiting for PVC"),
            _event(reason="BackOff", message="back-off restarting failed container"),
        ]
        monkeypatch.setattr(
            jobs_module,
            "get_raw_aiperfjob",
            AsyncMock(return_value=_raw_aiperf_job()),
        )
        monkeypatch.setattr(jobs_module, "get_pods", AsyncMock(return_value=[]))
        monkeypatch.setattr(
            jobs_module, "list_events_for_object", AsyncMock(return_value=events)
        )

        response = await client.get(
            "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-diagnostics/events"
        )

        assert response.status_code == 200, response.text
        types = [event["type"] for event in response.json()["events"]]
        assert types.count("Normal") == 2
        assert types.count("Warning") == 2

    @pytest.mark.asyncio
    async def test_list_job_events_event_time_fallback_keeps_nullable_schema_keys(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs as jobs_module

        event_time = datetime(2026, 5, 18, 12, 7, 30, tzinfo=UTC)
        monkeypatch.setattr(
            jobs_module,
            "get_raw_aiperfjob",
            AsyncMock(return_value=_raw_aiperf_job()),
        )
        monkeypatch.setattr(jobs_module, "get_pods", AsyncMock(return_value=[]))
        monkeypatch.setattr(
            jobs_module,
            "list_events_for_object",
            AsyncMock(
                return_value=[
                    _event(
                        reason="Created",
                        message="Created pod sandbox",
                        type_="Normal",
                        first_timestamp=None,
                        last_timestamp=None,
                        event_time=event_time,
                        count=3,
                    )
                ]
            ),
        )

        response = await client.get(
            "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-diagnostics/events"
        )

        assert response.status_code == 200, response.text
        event = response.json()["events"][0]
        assert set(event) == {
            "type",
            "reason",
            "message",
            "source",
            "involved_object",
            "first_timestamp",
            "last_timestamp",
            "count",
        }
        assert event["first_timestamp"] == "2026-05-18T12:07:30+00:00"
        assert event["last_timestamp"] == "2026-05-18T12:07:30+00:00"
        assert event["count"] == 3
        assert event["source"] == {"component": "kubelet", "host": "dgx-node-01"}


# ============================================================
# Logs endpoint ownership, limits, and apiserver errors
# ============================================================


class TestDiagnosticsLogsContracts:
    """Pod logs enforce ownership and surface actionable Kubernetes failures."""

    @pytest.mark.asyncio
    async def test_get_pod_logs_missing_owned_pod_returns_404_with_job_identity(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        monkeypatch.setattr(jobs_logs, "get_pods", AsyncMock(return_value=[]))

        response = await client.get(
            "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-diagnostics/logs",
            params={"pod": "stray-controller-0"},
        )

        assert response.status_code == 404
        assert "stray-controller-0" in response.json()["detail"]
        assert "aiperf-benchmarks/llama-3-8b-diagnostics" in response.json()["detail"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tail_lines",
        [
            param(0, id="zero-rejected"),
            param(10_001, id="above-max-rejected"),
        ],
    )  # fmt: skip
    async def test_get_pod_logs_out_of_range_tail_lines_returns_400_before_pod_lookup(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        tail_lines: int,
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        pod_lookup = AsyncMock()
        monkeypatch.setattr(jobs_logs, "get_pods", pod_lookup)

        response = await client.get(
            "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-diagnostics/logs",
            params={
                "pod": "llama-3-8b-diagnostics-controller-0",
                "tail_lines": tail_lines,
            },
        )

        assert response.status_code == 400
        assert "tail_lines must be in [1, 10000]" in response.json()["detail"]
        pod_lookup.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_pod_logs_valid_max_tail_uses_default_container_and_plain_text(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        pod = _pod(
            "llama-3-8b-diagnostics-controller-0",
            containers=["event-bus", "controller"],
            default_container="controller",
        )
        monkeypatch.setattr(jobs_logs, "get_pods", AsyncMock(return_value=[pod]))
        read_log = AsyncMock(return_value="profile complete\nrecords flushed\n")
        mock_core = MagicMock(read_namespaced_pod_log=read_log)
        monkeypatch.setattr(
            jobs_logs.client, "CoreV1Api", MagicMock(return_value=mock_core)
        )

        response = await client.get(
            "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-diagnostics/logs",
            params={
                "pod": "llama-3-8b-diagnostics-controller-0",
                "tail_lines": 10_000,
            },
        )

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == "profile complete\nrecords flushed\n"
        read_log.assert_awaited_once()
        assert read_log.call_args.kwargs["tail_lines"] == 10_000
        assert read_log.call_args.kwargs["container"] == "controller"
        assert read_log.call_args.kwargs["_preload_content"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,detail",
        [
            param(404, "pod was evicted", id="pod-log-404-preserved"),
            param(500, "apiserver log stream timeout", id="pod-log-500-preserved"),
        ],
    )  # fmt: skip
    async def test_get_pod_logs_read_api_error_surfaces_status_and_detail(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        status: int,
        detail: str,
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        api_error = ApiException(status=status, reason="Kubernetes API error")
        api_error.body = detail
        pod = _pod("llama-3-8b-diagnostics-controller-0")
        monkeypatch.setattr(jobs_logs, "get_pods", AsyncMock(return_value=[pod]))
        mock_core = MagicMock(read_namespaced_pod_log=AsyncMock(side_effect=api_error))
        monkeypatch.setattr(
            jobs_logs.client, "CoreV1Api", MagicMock(return_value=mock_core)
        )

        response = await client.get(
            "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-diagnostics/logs",
            params={"pod": "llama-3-8b-diagnostics-controller-0"},
        )

        assert response.status_code == status
        assert detail in response.json()["detail"]


# ============================================================
# Conditions surfaced through diagnostics detail
# ============================================================


class TestDiagnosticsConditionsSchema:
    """The detail response keeps condition rows intact for the UI conditions tab."""

    @pytest.mark.asyncio
    async def test_get_job_detail_conditions_preserve_terminal_counterpart_rows(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs as jobs_module

        conditions = [
            {
                "type": "Complete",
                "status": "False",
                "reason": "BenchmarkStillRunning",
                "message": "Benchmark llama-3-8b-diagnostics is still profiling.",
                "lastTransitionTime": "2026-05-18T12:00:00Z",
            },
            {
                "type": "Failed",
                "status": "False",
                "reason": "BenchmarkHealthy",
                "message": "No failure has been observed for this benchmark.",
                "lastTransitionTime": "2026-05-18T12:00:00Z",
            },
            {
                "type": "ResultsAvailable",
                "status": "True",
                "reason": "SummaryWritten",
                "message": "profile_export_aiperf.json is available on the results PVC.",
                "lastTransitionTime": "2026-05-18T12:04:00Z",
            },
        ]
        monkeypatch.setattr(
            jobs_module,
            "find_any_job",
            AsyncMock(return_value=_live_job_info()),
        )
        monkeypatch.setattr(
            jobs_module,
            "get_raw_aiperfjob_status",
            AsyncMock(return_value={"phase": "Running", "conditions": conditions}),
        )
        monkeypatch.setattr(jobs_module, "get_pods", AsyncMock(return_value=[]))

        response = await client.get(
            "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-diagnostics"
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"]["conditions"] == conditions
        assert body["pods"] == []
        assert body["job"]["name"] == "llama-3-8b-diagnostics"
