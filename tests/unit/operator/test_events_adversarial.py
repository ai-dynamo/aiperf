# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes Event diagnostics surfaces.

Focuses on:
- Event emission helper contracts for kopf-backed Normal and Warning events.
- UI-facing EventEntry conversion at the Kubernetes trust boundary.
- Namespace/name routing for AIPerfJob and Pod event lookups.
- Noisy admission-policy event filtering without dropping workload policy signals.
- Event ``count`` preservation for diagnostic badges and duplicate-series display.
- Best-effort behavior when one Kubernetes Event API read fails.

Out of scope: Pod restart threshold and dedup state machines; see sibling
``tests/unit/operator/handlers/test_pod_restarts_adversarial.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiperf.operator.routers import jobs
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes import client_pods
from aiperf.operator import events

# =============================================================================
# Helpers
# =============================================================================


def _aiperfjob_body(
    *,
    name: str = "llama3-8b-throughput",
    namespace: str = "bench-prod",
    job_id: str = "aiperf-bench-7f2a",
) -> dict[str, object]:
    """Build a realistic AIPerfJob resource body for kopf event emission."""
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "uid": f"job-{job_id}",
        },
        "status": {"jobId": job_id},
    }


def _raw_event(
    *,
    reason: str = "FailedScheduling",
    message: str = "0/4 nodes are available: insufficient nvidia.com/gpu",
    type_: str = "Warning",
    involved_kind: str = "Pod",
    involved_name: str = "llama3-controller-0",
    involved_namespace: str = "bench-prod",
    component: str | None = "default-scheduler",
    host: str | None = "dgx-node-01",
    first_timestamp: datetime | None = None,
    last_timestamp: datetime | None = None,
    event_time: datetime | None = None,
    count: int | None = 1,
) -> SimpleNamespace:
    """Build a V1Event-shaped object with only fields the router reads."""
    involved = SimpleNamespace(
        kind=involved_kind,
        name=involved_name,
        namespace=involved_namespace,
    )
    source = None
    if component is not None or host is not None:
        source = SimpleNamespace(component=component, host=host)
    return SimpleNamespace(
        type=type_,
        reason=reason,
        message=message,
        source=source,
        involved_object=involved,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        event_time=event_time,
        count=count,
    )


def _pod(name: str) -> SimpleNamespace:
    """Build the minimal Pod shape ``_list_events_impl`` needs."""
    return SimpleNamespace(metadata=SimpleNamespace(name=name))


# =============================================================================
# Event emission helper contracts
# =============================================================================


class TestKopfEventEmissionAdversarial:
    """Kopf event helpers should preserve resource identity and stay best-effort."""

    @pytest.mark.parametrize(
        "event_type,reason,emitter_name",
        [
            (events.EventType.NORMAL, events.EventReason.CREATED, "info"),
            (events.EventType.WARNING, events.EventReason.FAILED, "warn"),
        ],
    )  # fmt: skip
    def test_post_event_kopf_emitter_preserves_resource_body_and_camelcase_reason(
        self,
        monkeypatch: pytest.MonkeyPatch,
        event_type: events.EventType,
        reason: events.EventReason,
        emitter_name: str,
    ) -> None:
        body = _aiperfjob_body(name="llama3-8b-throughput")
        info = MagicMock()
        warn = MagicMock()
        monkeypatch.setattr(events.kopf, "info", info)
        monkeypatch.setattr(events.kopf, "warn", warn)

        events.post_event(
            body,
            reason,
            "operator surfaced a diagnostic event",
            event_type,
        )

        called = info if emitter_name == "info" else warn
        not_called = warn if emitter_name == "info" else info
        called.assert_called_once_with(
            body,
            reason=str(reason),
            message="operator surfaced a diagnostic event",
        )
        not_called.assert_not_called()

    def test_post_event_kopf_lookup_error_logs_and_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        body = _aiperfjob_body(name="llama3-event-context-missing")
        monkeypatch.setattr(events.kopf, "info", MagicMock(side_effect=LookupError))

        events.post_event(body, events.EventReason.CREATED, "created benchmark job")

        assert "Could not post event: kopf context unavailable" in caplog.text

    def test_post_event_kopf_api_failure_logs_reason_and_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        body = _aiperfjob_body(name="llama3-event-api-failure")
        monkeypatch.setattr(
            events.kopf,
            "warn",
            MagicMock(side_effect=RuntimeError("apiserver rejected event create")),
        )

        events.post_event(
            body,
            events.EventReason.POD_RESTARTS,
            "Pod controller-0 has restarted 4 times: CrashLoopBackOff",
            events.EventType.WARNING,
        )

        assert (
            "Failed to post event PodRestarts: apiserver rejected event create"
            in caplog.text
        )


# =============================================================================
# EventEntry conversion at the Kubernetes trust boundary
# =============================================================================


class TestEventEntryConversionAdversarial:
    """Raw Kubernetes Event shapes should map to stable UI diagnostics entries."""

    def test_event_to_entry_new_api_event_time_preserves_involved_object_and_count(
        self,
    ) -> None:
        event_time = datetime(2026, 5, 18, 12, 30, 15, tzinfo=UTC)
        raw = _raw_event(
            reason="BackOff",
            message="Back-off restarting failed container controller",
            involved_kind="AIPerfJob",
            involved_name="llama3-8b-throughput",
            involved_namespace="bench-prod",
            component="kopf",
            host=None,
            first_timestamp=None,
            last_timestamp=None,
            event_time=event_time,
            count=17,
        )

        entry = jobs._event_to_entry(raw)

        assert entry.reason == "BackOff"
        assert entry.message == "Back-off restarting failed container controller"
        assert entry.source.component == "kopf"
        assert entry.source.host is None
        assert entry.involved_object.kind == "AIPerfJob"
        assert entry.involved_object.name == "llama3-8b-throughput"
        assert entry.involved_object.namespace == "bench-prod"
        assert entry.first_timestamp == event_time.isoformat()
        assert entry.last_timestamp == event_time.isoformat()
        assert entry.count == 17

    def test_event_to_entry_missing_source_and_involved_object_returns_null_fields(
        self,
    ) -> None:
        raw = _raw_event(component=None, host=None)
        raw.source = None
        raw.involved_object = None

        entry = jobs._event_to_entry(raw)

        assert entry.source.component is None
        assert entry.source.host is None
        assert entry.involved_object.kind is None
        assert entry.involved_object.name is None
        assert entry.involved_object.namespace is None


# =============================================================================
# Namespace/name routing and diagnostic event lists
# =============================================================================


class TestEventListRoutingAdversarial:
    """Event list helpers must query the intended namespace and involvedObject names."""

    @pytest.mark.asyncio
    async def test_list_events_for_object_uses_namespace_and_involved_object_field_selector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        core = MagicMock()
        core.list_namespaced_event = AsyncMock(return_value=SimpleNamespace(items=[]))
        monkeypatch.setattr(
            client_pods.client,
            "CoreV1Api",
            MagicMock(return_value=core),
        )

        result = await client_pods.list_events_for_object(
            object(),
            "bench-prod",
            "llama3-8b-throughput-v02-t1",
        )

        assert result == []
        core.list_namespaced_event.assert_awaited_once_with(
            namespace="bench-prod",
            field_selector="involvedObject.name=llama3-8b-throughput-v02-t1",
        )

    @pytest.mark.asyncio
    async def test_list_events_impl_queries_cr_and_each_owned_pod_in_same_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str]] = []

        async def list_events(
            _api: object, namespace: str, name: str
        ) -> list[SimpleNamespace]:
            calls.append((namespace, name))
            return [_raw_event(involved_name=name, involved_namespace=namespace)]

        monkeypatch.setattr(
            jobs, "get_raw_aiperfjob", AsyncMock(return_value=_aiperfjob_body())
        )
        monkeypatch.setattr(
            jobs,
            "get_pods",
            AsyncMock(
                return_value=[
                    _pod("llama3-controller-0"),
                    _pod("llama3-worker-0-0"),
                ]
            ),
        )
        monkeypatch.setattr(jobs, "list_events_for_object", list_events)

        response = await jobs._list_events_impl(
            object(), "bench-prod", "llama3-8b-throughput"
        )

        assert calls == [
            ("bench-prod", "llama3-8b-throughput"),
            ("bench-prod", "llama3-controller-0"),
            ("bench-prod", "llama3-worker-0-0"),
        ]
        assert [event.involved_object.name for event in response.events] == [
            "llama3-8b-throughput",
            "llama3-controller-0",
            "llama3-worker-0-0",
        ]

    @pytest.mark.asyncio
    async def test_list_events_impl_missing_live_cr_returns_empty_diagnostics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        list_events = AsyncMock(return_value=[_raw_event()])
        get_pods = AsyncMock(return_value=[_pod("llama3-controller-0")])
        monkeypatch.setattr(jobs, "get_raw_aiperfjob", AsyncMock(return_value=None))
        monkeypatch.setattr(jobs, "list_events_for_object", list_events)
        monkeypatch.setattr(jobs, "get_pods", get_pods)

        response = await jobs._list_events_impl(
            object(), "bench-prod", "archived-llama3"
        )

        assert response.events == []
        list_events.assert_not_awaited()
        get_pods.assert_not_awaited()


# =============================================================================
# Duplicate/noisy events and best-effort diagnostics
# =============================================================================


class TestEventDiagnosticsNoiseAndFailuresAdversarial:
    """Diagnostics event panes should filter known noise and preserve useful signal."""

    @pytest.mark.asyncio
    async def test_list_events_impl_filters_known_gke_noise_but_keeps_workload_policy_violation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        noisy = _raw_event(
            reason="PolicyViolation",
            message=(
                "validating-node-p4sa-audience denied unrelated kubelet identity: "
                "no such key: username"
            ),
        )
        real_policy = _raw_event(
            reason="PolicyViolation",
            message="policy gpu-workload-guardrails rejected privileged container",
        )
        scheduling = _raw_event(reason="FailedScheduling")
        monkeypatch.setattr(
            jobs, "get_raw_aiperfjob", AsyncMock(return_value=_aiperfjob_body())
        )
        monkeypatch.setattr(jobs, "get_pods", AsyncMock(return_value=[]))
        monkeypatch.setattr(
            jobs,
            "list_events_for_object",
            AsyncMock(return_value=[noisy, real_policy, scheduling]),
        )

        response = await jobs._list_events_impl(
            object(), "bench-prod", "llama3-8b-throughput"
        )

        assert [event.reason for event in response.events] == [
            "PolicyViolation",
            "FailedScheduling",
        ]
        assert "gpu-workload-guardrails" in (response.events[0].message or "")

    @pytest.mark.asyncio
    async def test_list_events_impl_preserves_kubernetes_event_count_for_diagnostic_badges(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repeated = _raw_event(
            reason="BackOff",
            message="Back-off restarting failed container controller",
            last_timestamp=datetime(2026, 5, 18, 12, 31, tzinfo=UTC),
            count=23,
        )
        monkeypatch.setattr(
            jobs, "get_raw_aiperfjob", AsyncMock(return_value=_aiperfjob_body())
        )
        monkeypatch.setattr(jobs, "get_pods", AsyncMock(return_value=[]))
        monkeypatch.setattr(
            jobs, "list_events_for_object", AsyncMock(return_value=[repeated])
        )

        response = await jobs._list_events_impl(
            object(), "bench-prod", "llama3-8b-throughput"
        )

        assert len(response.events) == 1
        assert response.events[0].reason == "BackOff"
        assert response.events[0].count == 23

    @pytest.mark.asyncio
    async def test_list_events_impl_same_reason_on_cr_and_pod_keeps_both_involved_objects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def list_events(
            _api: object, _namespace: str, name: str
        ) -> list[SimpleNamespace]:
            return [
                _raw_event(
                    reason="FailedScheduling",
                    message="0/4 nodes are available: insufficient nvidia.com/gpu",
                    involved_kind="AIPerfJob"
                    if name == "llama3-8b-throughput"
                    else "Pod",
                    involved_name=name,
                    last_timestamp=datetime(2026, 5, 18, 12, 31, tzinfo=UTC),
                )
            ]

        monkeypatch.setattr(
            jobs, "get_raw_aiperfjob", AsyncMock(return_value=_aiperfjob_body())
        )
        monkeypatch.setattr(
            jobs, "get_pods", AsyncMock(return_value=[_pod("llama3-controller-0")])
        )
        monkeypatch.setattr(jobs, "list_events_for_object", list_events)

        response = await jobs._list_events_impl(
            object(), "bench-prod", "llama3-8b-throughput"
        )

        assert [
            (e.involved_object.kind, e.involved_object.name) for e in response.events
        ] == [
            ("AIPerfJob", "llama3-8b-throughput"),
            ("Pod", "llama3-controller-0"),
        ]

    @pytest.mark.asyncio
    async def test_list_events_impl_pod_event_api_failure_returns_available_cr_events(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One pod-level Event API failure should not blank the whole diagnostics pane."""
        cr_event = _raw_event(
            reason="Created",
            message="Created benchmark job aiperf-bench-7f2a with 8 workers",
            involved_kind="AIPerfJob",
            involved_name="llama3-8b-throughput",
        )

        async def list_events(
            _api: object, _namespace: str, name: str
        ) -> list[SimpleNamespace]:
            if name == "llama3-controller-0":
                raise ApiException(status=403, reason="events forbidden for pod")
            return [cr_event]

        monkeypatch.setattr(
            jobs, "get_raw_aiperfjob", AsyncMock(return_value=_aiperfjob_body())
        )
        monkeypatch.setattr(
            jobs, "get_pods", AsyncMock(return_value=[_pod("llama3-controller-0")])
        )
        monkeypatch.setattr(jobs, "list_events_for_object", list_events)

        response = await jobs._list_events_impl(
            object(), "bench-prod", "llama3-8b-throughput"
        )

        assert [event.reason for event in response.events] == ["Created"]
