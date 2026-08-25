# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for watch-driven Pod restart event handling.

Focuses on:
- JobSet-label routing assumptions at the Pod event trust boundary.
- In-process dedup across noisy repeated watch events and per-parent keys.
- Restart-count threshold and delta-like behavior without kopf field patches.
- Missing containerStatuses and malformed Pod bodies from broad watch events.
- Parent AIPerfJob event target shape and no pods:patch dependency.

Out of scope: monitor-timer restart polling removed from this path; see sibling
``tests/unit/operator/test_monitor_state_machine_edges.py`` for timer behavior.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import kopf
import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.operator import main
from aiperf.operator.client_cache import _warned_pod_restarts
from aiperf.operator.handlers import pod_restarts

# =============================================================================
# Helpers
# =============================================================================


@asynccontextmanager
async def _fake_k8s_client() -> AsyncIterator[MagicMock]:
    """Yield a fake ApiClient for lazy ``k8s_client`` call sites."""
    yield MagicMock(name="ApiClient")


def _pod_meta(
    *,
    name: str = "llama3-controller-0",
    namespace: str = "bench-prod",
    jobset_name: str | None = "aiperf-llama3-8b-throughput",
) -> dict[str, Any]:
    """Build realistic Pod metadata from a JobSet-owned controller pod."""
    labels: dict[str, str] = {"app.kubernetes.io/name": "aiperf-controller"}
    if jobset_name is not None:
        labels["jobset.sigs.k8s.io/jobset-name"] = jobset_name
    return {
        "name": name,
        "namespace": namespace,
        "uid": f"pod-{name}-7f2a",
        "labels": labels,
    }


def _pod_body(
    *,
    meta: dict[str, Any] | None = None,
    statuses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a Pod body with optional containerStatuses."""
    body: dict[str, Any] = {"metadata": meta or _pod_meta()}
    if statuses is not None:
        body["status"] = {"containerStatuses": statuses}
    return body


def _container_status(
    *,
    name: str = "controller",
    restart_count: int = 3,
    waiting_reason: str | None = None,
    terminated_reason: str | None = "OOMKilled",
) -> dict[str, Any]:
    """Build one Kubernetes containerStatus entry with restart context."""
    status: dict[str, Any] = {"name": name, "restartCount": restart_count}
    if terminated_reason is not None:
        status["lastState"] = {"terminated": {"reason": terminated_reason}}
    if waiting_reason is not None:
        status["state"] = {"waiting": {"reason": waiting_reason}}
    return status


def _aiperfjob_body(
    *,
    name: str = "llama3-8b-throughput",
    namespace: str = "bench-prod",
    job_id: str = "aiperf-bench-7f2a",
) -> dict[str, Any]:
    """Build the parent AIPerfJob body used as the Kubernetes Event target."""
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


def _owned_pod_body(*, jobset_uid: str = "jobset-7f2a") -> dict[str, Any]:
    """Build a Pod with an exact batch Job controller owner."""
    meta = _pod_meta()
    meta["ownerReferences"] = [
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": "aiperf-llama3-8b-throughput-controller-0",
            "uid": "batch-job-7f2a",
            "controller": True,
        }
    ]
    return _pod_body(meta=meta)


def _owned_batch_job(*, jobset_uid: str = "jobset-7f2a") -> dict[str, Any]:
    """Build the intermediate batch Job owned by the candidate JobSet."""
    return {
        "metadata": {
            "name": "aiperf-llama3-8b-throughput-controller-0",
            "uid": "batch-job-7f2a",
            "ownerReferences": [
                {
                    "apiVersion": "jobset.x-k8s.io/v1alpha2",
                    "kind": "JobSet",
                    "name": "aiperf-llama3-8b-throughput",
                    "uid": jobset_uid,
                    "controller": True,
                }
            ],
        }
    }


def _owned_jobset(*, uid: str = "jobset-7f2a") -> dict[str, Any]:
    """Build the exact JobSet owned by the AIPerfJob fixture."""
    return {
        "metadata": {
            "name": "aiperf-llama3-8b-throughput",
            "uid": uid,
            "ownerReferences": [
                {
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfJob",
                    "name": "llama3-8b-throughput",
                    "uid": "job-aiperf-bench-7f2a",
                    "controller": True,
                }
            ],
        }
    }


def _install_custom_objects_api(
    monkeypatch: pytest.MonkeyPatch,
    *,
    get_results: list[dict[str, Any]] | None = None,
    batch_job: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Install fake k8s client factories and return captured API methods."""
    get = AsyncMock(
        name="get_namespaced_custom_object",
        side_effect=get_results,
    )
    patch = AsyncMock(name="patch_namespaced_custom_object")
    custom = MagicMock(name="CustomObjectsApi")
    custom.get_namespaced_custom_object = get
    custom.patch_namespaced_custom_object = patch

    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client", lambda: _fake_k8s_client()
    )
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi",
        MagicMock(return_value=custom),
    )
    batch = MagicMock(name="BatchV1Api")
    batch.read_namespaced_job = AsyncMock(return_value=batch_job)
    monkeypatch.setattr(
        "kubernetes_asyncio.client.BatchV1Api",
        MagicMock(return_value=batch),
    )
    return SimpleNamespace(custom=custom, get=get, patch=patch, batch=batch)


@pytest.fixture(autouse=True)
def _clear_warned_restarts() -> None:
    """Reset pod-restart dedup state around every adversarial case."""
    _warned_pod_restarts.clear()
    yield
    _warned_pod_restarts.clear()


# =============================================================================
# Label filtering and no pods:patch dependency
# =============================================================================


class TestPodRestartEventRouting:
    """Pod watch events should route only through the JobSet label shortcut."""

    @pytest.mark.parametrize(
        "meta",
        [
            param(_pod_meta(jobset_name=None), id="missing-jobset-label"),
            param({"name": "llama3-controller-0", "namespace": "bench-prod"}, id="missing-labels-map"),
            param(_pod_meta(jobset_name=""), id="empty-jobset-label"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_handle_pod_restart_missing_jobset_label_skips_lookup_and_event(
        self, meta: dict[str, Any]
    ) -> None:
        lookup = AsyncMock(return_value=_aiperfjob_body())
        event = MagicMock()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(pod_restarts, "_lookup_aiperfjob_body", lookup)
            monkeypatch.setattr(pod_restarts.events, "pod_restarts", event)
            await pod_restarts.handle_pod_restart(
                old=[],
                new=[_container_status(restart_count=5)],
                body=_pod_body(meta=meta),
                meta=meta,
                namespace="bench-prod",
                name="llama3-controller-0",
                threshold=3,
            )

        lookup.assert_not_awaited()
        event.assert_not_called()
        assert _warned_pod_restarts == {}

    @pytest.mark.asyncio
    async def test_lookup_aiperfjob_body_unprefixed_jobset_skips_api_get(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install_custom_objects_api(monkeypatch)

        result = await pod_restarts._lookup_aiperfjob_body(
            "bench-prod",
            "llama3-8b-throughput",
            _owned_pod_body(),
        )

        assert result is None
        fake.get.assert_not_awaited()
        fake.patch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_lookup_aiperfjob_body_prefixed_jobset_uses_get_not_patch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install_custom_objects_api(
            monkeypatch,
            get_results=[_owned_jobset(), _aiperfjob_body()],
            batch_job=_owned_batch_job(),
        )

        result = await pod_restarts._lookup_aiperfjob_body(
            "bench-prod",
            "aiperf-llama3-8b-throughput",
            _owned_pod_body(),
        )

        assert result == _aiperfjob_body()
        assert fake.get.await_count == 2
        kwargs = fake.get.await_args_list[-1].kwargs
        assert kwargs["namespace"] == "bench-prod"
        assert kwargs["plural"] == "aiperfjobs"
        assert kwargs["name"] == "llama3-8b-throughput"
        fake.batch.read_namespaced_job.assert_awaited_once()
        fake.patch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_lookup_aiperfjob_body_rejects_replacement_jobset_uid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A label-matching JobSet replacement cannot inherit an old Pod event."""
        fake = _install_custom_objects_api(
            monkeypatch,
            get_results=[_owned_jobset(uid="replacement-jobset")],
            batch_job=_owned_batch_job(jobset_uid="old-jobset"),
        )

        result = await pod_restarts._lookup_aiperfjob_body(
            "bench-prod",
            "aiperf-llama3-8b-throughput",
            _owned_pod_body(),
        )

        assert result is None
        assert fake.get.await_count == 1

    @pytest.mark.asyncio
    async def test_lookup_aiperfjob_body_rejects_replacement_parent_uid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A same-name AIPerfJob replacement cannot receive the old Pod event."""
        replacement = _aiperfjob_body()
        replacement["metadata"]["uid"] = "replacement-parent"
        fake = _install_custom_objects_api(
            monkeypatch,
            get_results=[_owned_jobset(), replacement],
            batch_job=_owned_batch_job(),
        )

        result = await pod_restarts._lookup_aiperfjob_body(
            "bench-prod",
            "aiperf-llama3-8b-throughput",
            _owned_pod_body(),
        )

        assert result is None
        assert fake.get.await_count == 2

    @pytest.mark.asyncio
    async def test_lookup_aiperfjob_body_transient_api_failure_requests_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A one-shot Pod event must not be dropped on a transient owner read."""
        fake = _install_custom_objects_api(
            monkeypatch,
            batch_job=_owned_batch_job(),
        )
        fake.batch.read_namespaced_job.side_effect = ApiException(
            status=503, reason="apiserver unavailable"
        )

        with pytest.raises(kopf.TemporaryError) as excinfo:
            await pod_restarts._lookup_aiperfjob_body(
                "bench-prod",
                "aiperf-llama3-8b-throughput",
                _owned_pod_body(),
            )

        assert "retry" in str(excinfo.value).lower()

    def test_on_pod_container_status_change_signature_has_no_patch_dependency(
        self,
    ) -> None:
        """The kopf Pod shortcut must not require pods:patch RBAC injection."""
        parameters = inspect.signature(main.on_pod_container_status_change).parameters

        assert "patch" not in parameters
        assert "event" in parameters
        assert "body" in parameters
        assert parameters["_"].kind is inspect.Parameter.VAR_KEYWORD


# =============================================================================
# Main kopf wrapper body handling
# =============================================================================


class TestPodRestartKopfWrapper:
    """The decorated wrapper narrows noisy Pod events before handler dispatch."""

    @pytest.mark.asyncio
    async def test_on_pod_container_status_change_deleted_event_skips_handler(
        self,
    ) -> None:
        handler = AsyncMock()
        meta = _pod_meta()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                main.pod_restarts_handler, "handle_pod_restart", handler
            )
            await main.on_pod_container_status_change(
                event={"type": "DELETED"},
                body=_pod_body(
                    meta=meta, statuses=[_container_status(restart_count=7)]
                ),
                meta=meta,
                namespace="bench-prod",
                name="llama3-controller-0",
                patch=object(),
            )

        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_pod_container_status_change_missing_container_statuses_passes_empty_list(
        self,
    ) -> None:
        handler = AsyncMock()
        meta = _pod_meta()
        body = _pod_body(meta=meta)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                main.pod_restarts_handler, "handle_pod_restart", handler
            )
            await main.on_pod_container_status_change(
                event={"type": "MODIFIED"},
                body=body,
                meta=meta,
                namespace="bench-prod",
                name="llama3-controller-0",
                logger=MagicMock(),
            )

        handler.assert_awaited_once()
        assert handler.await_args.kwargs["old"] == []
        assert handler.await_args.kwargs["new"] == []
        assert handler.await_args.kwargs["body"] is body
        assert handler.await_args.kwargs["meta"] is meta


# =============================================================================
# Dedup, restart counts, and parent event shape
# =============================================================================


class TestPodRestartDedupAndEventShape:
    """Restart events are keyed by parent job and emitted against the parent CR."""

    @pytest.mark.asyncio
    async def test_handle_pod_restart_threshold_boundary_emits_only_at_or_above_threshold(
        self,
    ) -> None:
        meta = _pod_meta()
        event = MagicMock()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                pod_restarts,
                "_lookup_aiperfjob_body",
                AsyncMock(return_value=_aiperfjob_body()),
            )
            monkeypatch.setattr(pod_restarts.events, "pod_restarts", event)
            await pod_restarts.handle_pod_restart(
                old=[],
                new=[
                    _container_status(name="controller", restart_count=2),
                    _container_status(name="worker", restart_count=3),
                ],
                body=_pod_body(meta=meta),
                meta=meta,
                namespace="bench-prod",
                name="llama3-controller-0",
                threshold=3,
            )

        event.assert_called_once()
        assert event.call_args.args[1:] == ("llama3-controller-0", 3, "OOMKilled")

    @pytest.mark.asyncio
    async def test_handle_pod_restart_noisy_repeated_events_same_count_emit_once(
        self,
    ) -> None:
        meta = _pod_meta()
        event = MagicMock()
        status = [_container_status(restart_count=5, waiting_reason="CrashLoopBackOff")]

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                pod_restarts,
                "_lookup_aiperfjob_body",
                AsyncMock(return_value=_aiperfjob_body()),
            )
            monkeypatch.setattr(pod_restarts.events, "pod_restarts", event)
            for _ in range(25):
                await pod_restarts.handle_pod_restart(
                    old=status,
                    new=status,
                    body=_pod_body(meta=meta),
                    meta=meta,
                    namespace="bench-prod",
                    name="llama3-controller-0",
                    threshold=3,
                )

        event.assert_called_once()
        assert event.call_args.args[3] == "CrashLoopBackOff"
        assert _warned_pod_restarts == {
            "bench-prod/aiperf-bench-7f2a@job-aiperf-bench-7f2a": {
                ("llama3-controller-0", 5)
            }
        }

    @pytest.mark.asyncio
    async def test_handle_pod_restart_increasing_restart_counts_emit_each_new_count(
        self,
    ) -> None:
        meta = _pod_meta()
        event = MagicMock()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                pod_restarts,
                "_lookup_aiperfjob_body",
                AsyncMock(return_value=_aiperfjob_body()),
            )
            monkeypatch.setattr(pod_restarts.events, "pod_restarts", event)
            for restart_count in [2, 3, 3, 4, 4, 5]:
                await pod_restarts.handle_pod_restart(
                    old=[],
                    new=[_container_status(restart_count=restart_count)],
                    body=_pod_body(meta=meta),
                    meta=meta,
                    namespace="bench-prod",
                    name="llama3-controller-0",
                    threshold=3,
                )

        assert [call.args[2] for call in event.call_args_list] == [3, 4, 5]

    @pytest.mark.asyncio
    async def test_handle_pod_restart_same_pod_count_under_different_parent_jobs_emit_separately(
        self,
    ) -> None:
        meta = _pod_meta()
        event = MagicMock()
        parents = [
            _aiperfjob_body(job_id="aiperf-bench-7f2a"),
            _aiperfjob_body(name="llama3-70b-throughput", job_id="aiperf-bench-91ab"),
        ]

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                pod_restarts,
                "_lookup_aiperfjob_body",
                AsyncMock(side_effect=parents),
            )
            monkeypatch.setattr(pod_restarts.events, "pod_restarts", event)
            for _parent in parents:
                await pod_restarts.handle_pod_restart(
                    old=[],
                    new=[_container_status(restart_count=5)],
                    body=_pod_body(meta=meta),
                    meta=meta,
                    namespace="bench-prod",
                    name="llama3-controller-0",
                    threshold=3,
                )

        assert event.call_count == 2
        assert {call.args[0]["status"]["jobId"] for call in event.call_args_list} == {
            "aiperf-bench-7f2a",
            "aiperf-bench-91ab",
        }
        assert set(_warned_pod_restarts) == {
            "bench-prod/aiperf-bench-7f2a@job-aiperf-bench-7f2a",
            "bench-prod/aiperf-bench-91ab@job-aiperf-bench-91ab",
        }

    @pytest.mark.asyncio
    async def test_handle_pod_restart_event_targets_parent_aiperfjob_not_pod_body(
        self,
    ) -> None:
        meta = _pod_meta()
        pod_body = _pod_body(meta=meta)
        parent_body = _aiperfjob_body()
        event = MagicMock()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                pod_restarts,
                "_lookup_aiperfjob_body",
                AsyncMock(return_value=parent_body),
            )
            monkeypatch.setattr(pod_restarts.events, "pod_restarts", event)
            await pod_restarts.handle_pod_restart(
                old=[],
                new=[
                    _container_status(
                        restart_count=6,
                        terminated_reason="OOMKilled",
                        waiting_reason="CrashLoopBackOff",
                    )
                ],
                body=pod_body,
                meta=meta,
                namespace="bench-prod",
                name="llama3-controller-0",
                threshold=3,
            )

        event.assert_called_once()
        assert event.call_args.args == (
            parent_body,
            "llama3-controller-0",
            6,
            "CrashLoopBackOff",
        )
        assert event.call_args.args[0] is not pod_body

    @pytest.mark.asyncio
    async def test_handle_pod_restart_missing_container_statuses_do_not_leak_dedup_state(
        self,
    ) -> None:
        meta = _pod_meta()
        lookup = AsyncMock(return_value=_aiperfjob_body())
        event = MagicMock()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(pod_restarts, "_lookup_aiperfjob_body", lookup)
            monkeypatch.setattr(pod_restarts.events, "pod_restarts", event)
            await pod_restarts.handle_pod_restart(
                old=[],
                new=[],
                body=_pod_body(meta=meta),
                meta=meta,
                namespace="bench-prod",
                name="llama3-controller-0",
                threshold=3,
            )

        lookup.assert_not_awaited()
        event.assert_not_called()
        assert _warned_pod_restarts == {}
