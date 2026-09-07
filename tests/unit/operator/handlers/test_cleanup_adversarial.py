# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial cleanup/delete tests for Kubernetes operator handlers.

Locks delete-path contracts that are easy to regress during cleanup refactors:

1. AIPerfJob deletion requests cooperative cancellation before closing the
   cached ProgressClient and before dropping runs-index rows.
2. AIPerfJob deletion stays finalizer-free with respect to owned resources:
   JobSet, ConfigMap, Role, and RoleBinding cleanup is left to Kubernetes
   ownerReference GC, not direct delete calls from the handler.
3. AIPerfSweep deletion best-effort patches child AIPerfJobs with
   ``spec.cancel=true``; 404/409 races and namespace-deletion list failures do
   not block parent deletion.
4. Re-fired delete handlers are idempotent and keep the cancellation flag set.

Out of scope: TTL result-directory pruning lives in
``tests/unit/operator/test_cleanup_handler.py``; sweep phase rollup and cancel
condition state machines live in ``tests/unit/operator/test_sweep_handler_adversarial.py``.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import kopf
import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.phase import Phase
from aiperf.operator.client_cache import (
    _reset_for_testing,
    is_cancellation_requested,
)
from aiperf.operator.handlers import lifecycle
from aiperf.operator.handlers.sweep import lifecycle as sweep_lifecycle

# =============================================================================
# Helpers
# =============================================================================


class _FakeSweepDeleteK8s(SimpleNamespace):
    """Captured fake Kubernetes API for AIPerfSweep on_delete tests."""

    list_objs: AsyncMock
    patch_obj: AsyncMock
    custom: MagicMock


@pytest.fixture(autouse=True)
def _reset_client_cache_state() -> Generator[None, None, None]:
    """Reset sticky cancellation flags and ProgressClient caches between tests."""
    _reset_for_testing()
    yield
    _reset_for_testing()


_SWEEP_NAME = "latency-sweep"
_SWEEP_UID = "sweep-uid-7f2a"
_RUN_EPOCH = "1778027124"


def _sweep_body(*, run_epoch: Any = _RUN_EPOCH) -> dict[str, Any]:
    """Build the AIPerfSweep body kopf hands to ``on_delete``.

    ``status.runEpoch`` is part of the child-discovery identity: without it the
    handler cannot distinguish children of this run from children left behind
    by a prior same-name sweep, so it refuses to patch anything.
    """
    body: dict[str, Any] = {"metadata": {"uid": _SWEEP_UID}}
    if run_epoch is not None:
        body["status"] = {"runEpoch": run_epoch}
    return body


def _child_job(
    name: str | None,
    *,
    uid: str | None = "child-uid-0001",
    owner_kind: str = "AIPerfSweep",
    owner_name: str = _SWEEP_NAME,
    owner_uid: str = _SWEEP_UID,
    owner_api_version: str = "aiperf.nvidia.com/v1alpha1",
    owner_controller: bool = True,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a child AIPerfJob list item carrying full owner identity.

    Discovery labels alone are user-writable, so the handler additionally
    requires a controller ownerReference matching the deleting sweep's
    apiVersion, kind, name, and immutable UID, plus a child UID for the
    json-patch ``test`` precondition.
    """
    metadata: dict[str, Any] = {
        "ownerReferences": [
            {
                "apiVersion": owner_api_version,
                "kind": owner_kind,
                "name": owner_name,
                "uid": owner_uid,
                "controller": owner_controller,
            }
        ],
        "labels": labels
        if labels is not None
        else {
            "aiperf.nvidia.com/sweep": _SWEEP_NAME,
            "aiperf.nvidia.com/sweep-uid": _SWEEP_UID,
            "aiperf.nvidia.com/sweep-run-epoch": _RUN_EPOCH,
        },
    }
    if name is not None:
        metadata["name"] = name
    if uid is not None:
        metadata["uid"] = uid
    return {"metadata": metadata}


def _cancel_patch(child_uid: str) -> list[dict[str, Any]]:
    """Expected json-patch body for a cooperative child cancel."""
    return [
        {"op": "test", "path": "/metadata/uid", "value": child_uid},
        {"op": "add", "path": "/spec/cancel", "value": True},
    ]


def _install_fake_sweep_delete_k8s(
    monkeypatch: pytest.MonkeyPatch,
    *,
    children: list[dict[str, Any]] | None = None,
    list_side_effect: BaseException | None = None,
    patch_side_effect: BaseException | list[BaseException | None] | None = None,
) -> _FakeSweepDeleteK8s:
    """Install fake k8s_client and CustomObjectsApi used by sweep on_delete."""
    list_objs = AsyncMock()
    if list_side_effect is not None:
        list_objs.side_effect = list_side_effect
    else:
        list_objs.return_value = {"items": children or []}

    patch_obj = AsyncMock()
    if isinstance(patch_side_effect, list) or patch_side_effect is not None:
        patch_obj.side_effect = patch_side_effect

    custom = MagicMock()
    custom.list_namespaced_custom_object = list_objs
    custom.patch_namespaced_custom_object = patch_obj

    fake_k8s_module = SimpleNamespace(CustomObjectsApi=lambda _api: custom)

    @asynccontextmanager
    async def fake_k8s_client() -> Any:
        yield MagicMock(name="ApiClient")

    import kubernetes_asyncio

    import aiperf.kubernetes.client as kclient

    monkeypatch.setattr(kubernetes_asyncio, "client", fake_k8s_module, raising=False)
    monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)

    return _FakeSweepDeleteK8s(
        list_objs=list_objs,
        patch_obj=patch_obj,
        custom=custom,
    )


# =============================================================================
# AIPerfJob delete: cooperative cancellation and finalizer-free cleanup
# =============================================================================


class TestAIPerfJobOnDeleteCleanup:
    """Delete-path contracts for a single AIPerfJob CR."""

    @pytest.mark.asyncio
    async def test_on_delete_requests_cancellation_before_client_close_and_index_drop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancellation must be visible before cache close and index cleanup await."""
        call_order: list[str] = []

        def fake_request_cancellation(key: str) -> None:
            call_order.append(f"cancel:{key}")

        async def fake_close_progress_client(key: str) -> None:
            call_order.append(f"close:{key}")

        async def fake_index_cleanup(
            namespace: str, name: str, status: dict[str, Any]
        ) -> None:
            call_order.append(
                f"index:{namespace}/{name}/{status.get('jobId', '<missing>')}"
            )

        monkeypatch.setattr(
            lifecycle, "request_cancellation", fake_request_cancellation
        )
        monkeypatch.setattr(
            lifecycle, "close_progress_client", fake_close_progress_client
        )
        monkeypatch.setattr(
            lifecycle,
            "on_aiperfjob_delete_index_cleanup",
            fake_index_cleanup,
        )

        await lifecycle.on_delete(
            name="bench-latency-7f2a",
            namespace="benchmark-prod",
            status={"jobId": "aiperf-bench-7f2a", "phase": Phase.RUNNING},
        )

        assert call_order == [
            "cancel:benchmark-prod/aiperf-bench-7f2a",
            "close:benchmark-prod/aiperf-bench-7f2a",
            "index:benchmark-prod/bench-latency-7f2a/aiperf-bench-7f2a",
        ]

    @pytest.mark.asyncio
    async def test_on_delete_without_job_id_falls_back_to_cr_name_for_cancellation(
        self,
    ) -> None:
        """Pre-start deletes still cancel by CR name when status.jobId is absent."""
        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                new_callable=AsyncMock,
            ) as close_progress_client,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.on_aiperfjob_delete_index_cleanup",
                new_callable=AsyncMock,
            ) as index_cleanup,
        ):
            await lifecycle.on_delete(
                name="queued-benchmark-42",
                namespace="benchmark-dev",
                status={},
            )

        assert is_cancellation_requested("benchmark-dev/queued-benchmark-42") is True
        close_progress_client.assert_awaited_once_with(
            "benchmark-dev/queued-benchmark-42"
        )
        index_cleanup.assert_awaited_once_with(
            "benchmark-dev", "queued-benchmark-42", {}
        )

    @pytest.mark.asyncio
    async def test_on_delete_does_not_directly_delete_jobset_or_aux_resources(
        self,
    ) -> None:
        """Owned JobSet/RBAC/ConfigMap cleanup is delegated to ownerReference GC."""
        custom_api = MagicMock()
        custom_api.delete_namespaced_custom_object = AsyncMock()
        core_api = MagicMock()
        core_api.delete_namespaced_config_map = AsyncMock()
        rbac_api = MagicMock()
        rbac_api.delete_namespaced_role = AsyncMock()
        rbac_api.delete_namespaced_role_binding = AsyncMock()

        with (
            mock_patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=custom_api,
            ) as custom_factory,
            mock_patch(
                "kubernetes_asyncio.client.CoreV1Api",
                return_value=core_api,
            ) as core_factory,
            mock_patch(
                "kubernetes_asyncio.client.RbacAuthorizationV1Api",
                return_value=rbac_api,
            ) as rbac_factory,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.on_aiperfjob_delete_index_cleanup",
                new_callable=AsyncMock,
            ),
        ):
            await lifecycle.on_delete(
                name="bench-cleanup-9c3a",
                namespace="benchmark-prod",
                status={
                    "jobId": "bench-cleanup-9c3a",
                    "jobSetName": "bench-cleanup-9c3a-js",
                },
            )

        custom_factory.assert_not_called()
        core_factory.assert_not_called()
        rbac_factory.assert_not_called()
        custom_api.delete_namespaced_custom_object.assert_not_awaited()
        core_api.delete_namespaced_config_map.assert_not_awaited()
        rbac_api.delete_namespaced_role.assert_not_awaited()
        rbac_api.delete_namespaced_role_binding.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_delete_repeated_refire_is_idempotent_and_keeps_cancel_set(
        self,
    ) -> None:
        """Repeated kopf delete delivery must not clear the sticky cancel flag."""
        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                new_callable=AsyncMock,
            ) as close_progress_client,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.on_aiperfjob_delete_index_cleanup",
                new_callable=AsyncMock,
            ) as index_cleanup,
        ):
            await lifecycle.on_delete(
                name="bench-repeat-13b7",
                namespace="benchmark-prod",
                status={"jobId": "bench-repeat-13b7"},
            )
            await lifecycle.on_delete(
                name="bench-repeat-13b7",
                namespace="benchmark-prod",
                status={"jobId": "bench-repeat-13b7"},
            )

        assert is_cancellation_requested("benchmark-prod/bench-repeat-13b7") is True
        assert close_progress_client.await_count == 2
        assert index_cleanup.await_count == 2


# =============================================================================
# AIPerfSweep delete: child cooperative cancellation races
# =============================================================================


class TestAIPerfSweepOnDeleteCleanup:
    """Best-effort child cancellation during AIPerfSweep CR deletion."""

    @pytest.mark.asyncio
    async def test_on_delete_rejects_label_spoofs_without_exact_owner_identity(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A user-writable sweep label cannot authorize cancellation."""
        fake = _install_fake_sweep_delete_k8s(
            monkeypatch,
            children=[
                _child_job("owned-child"),
                _child_job("wrong-uid", owner_uid="attacker-controlled-uid"),
                _child_job("wrong-name", owner_name="another-sweep"),
                _child_job("wrong-kind", owner_kind="AIPerfJob"),
                _child_job("wrong-api-version", owner_api_version="example.com/v1"),
                _child_job("not-controller", owner_controller=False),
                _child_job("no-child-uid", uid=None),
                _child_job(
                    "stale-epoch-label",
                    labels={
                        "aiperf.nvidia.com/sweep": _SWEEP_NAME,
                        "aiperf.nvidia.com/sweep-uid": _SWEEP_UID,
                        "aiperf.nvidia.com/sweep-run-epoch": "1778027000",
                    },
                ),
                {"metadata": {"name": "ownerless-label-spoof", "uid": "spoof-uid"}},
            ],
        )

        await sweep_lifecycle.on_delete(
            body=_sweep_body(),
            uid=_SWEEP_UID,
            name=_SWEEP_NAME,
            namespace="benchmark-prod",
        )

        fake.patch_obj.assert_awaited_once()
        assert fake.patch_obj.await_args.kwargs["name"] == "owned-child"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "run_epoch",
        [
            param(None, id="absent"),
            param("", id="empty"),
            param("not-an-epoch", id="non-decimal"),
            param(True, id="bool-masquerading-as-int"),
        ],
    )  # fmt: skip
    async def test_on_delete_without_usable_run_epoch_patches_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        run_epoch: Any,
    ) -> None:
        """No run epoch means no way to fence children of a prior same-name run."""
        fake = _install_fake_sweep_delete_k8s(
            monkeypatch,
            children=[_child_job("latency-sweep-v00-t0")],
        )

        await sweep_lifecycle.on_delete(
            body=_sweep_body(run_epoch=run_epoch),
            uid=_SWEEP_UID,
            name=_SWEEP_NAME,
            namespace="benchmark-prod",
        )

        fake.list_objs.assert_not_awaited()
        fake.patch_obj.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_delete_patches_named_children_with_cancel_true_in_namespace(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only named child AIPerfJobs are patched, with namespace-scoped selector."""
        fake = _install_fake_sweep_delete_k8s(
            monkeypatch,
            children=[
                _child_job("latency-sweep-v00-t0", uid="child-uid-0000"),
                _child_job(None),
                _child_job("latency-sweep-v01-t0", uid="child-uid-0001"),
            ],
        )

        await sweep_lifecycle.on_delete(
            body=_sweep_body(run_epoch=int(_RUN_EPOCH)),
            uid=_SWEEP_UID,
            name=_SWEEP_NAME,
            namespace="benchmark-prod",
        )

        fake.list_objs.assert_awaited_once_with(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace="benchmark-prod",
            plural="aiperfjobs",
            label_selector=(
                "aiperf.nvidia.com/sweep=latency-sweep,"
                "aiperf.nvidia.com/sweep-uid=sweep-uid-7f2a,"
                "aiperf.nvidia.com/sweep-run-epoch=1778027124"
            ),
        )
        patched_names = [call.kwargs["name"] for call in fake.patch_obj.await_args_list]
        assert patched_names == ["latency-sweep-v00-t0", "latency-sweep-v01-t0"]
        for call, child_uid in zip(
            fake.patch_obj.await_args_list,
            ["child-uid-0000", "child-uid-0001"],
            strict=True,
        ):
            assert call.kwargs["namespace"] == "benchmark-prod"
            assert call.kwargs["plural"] == "aiperfjobs"
            assert call.kwargs["body"] == _cancel_patch(child_uid)
            assert call.kwargs["_content_type"] == "application/json-patch+json"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code",
        [
            param(404, id="child-already-gc-d-404"),
            param(409, id="child-update-conflict-409"),
        ],
    )  # fmt: skip
    async def test_on_delete_child_patch_404_or_409_race_continues_to_next_child(
        self,
        monkeypatch: pytest.MonkeyPatch,
        status_code: int,
    ) -> None:
        """Deletion races on one child must not prevent cancelling later children."""
        fake = _install_fake_sweep_delete_k8s(
            monkeypatch,
            children=[
                _child_job("latency-sweep-v00-t0"),
                _child_job("latency-sweep-v01-t0"),
            ],
            patch_side_effect=[
                ApiException(status=status_code, reason="delete race"),
                None,
            ],
        )

        await sweep_lifecycle.on_delete(
            body=_sweep_body(),
            uid=_SWEEP_UID,
            name=_SWEEP_NAME,
            namespace="benchmark-prod",
        )

        assert fake.patch_obj.await_count == 2
        assert (
            fake.patch_obj.await_args_list[1].kwargs["name"] == "latency-sweep-v01-t0"
        )

    @pytest.mark.asyncio
    async def test_on_delete_child_patch_non_404_409_raises_temporary_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-race child patch failures must surface so kopf retries deletion."""
        fake = _install_fake_sweep_delete_k8s(
            monkeypatch,
            children=[
                _child_job("latency-sweep-v00-t0"),
                _child_job("latency-sweep-v01-t0"),
            ],
            patch_side_effect=[
                ApiException(status=500, reason="apiserver unavailable"),
                None,
            ],
        )

        with pytest.raises(kopf.TemporaryError, match="latency-sweep-v00-t0"):
            await sweep_lifecycle.on_delete(
                body=_sweep_body(),
                uid=_SWEEP_UID,
                name=_SWEEP_NAME,
                namespace="benchmark-prod",
            )

        fake.patch_obj.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "list_error",
        [
            param(ApiException(status=404, reason="namespace terminating"), id="list-404"),
            param(ApiException(status=409, reason="resource-version-expired"), id="list-409"),
            param(ConnectionError("apiserver connection reset"), id="connection-reset"),
            param(TimeoutError("apiserver watch cache timed out"), id="timeout"),
        ],
    )  # fmt: skip
    async def test_on_delete_namespace_deletion_list_failures_are_swallowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        list_error: BaseException,
    ) -> None:
        """Namespace deletion can make the child list fail; parent delete must continue."""
        fake = _install_fake_sweep_delete_k8s(
            monkeypatch,
            list_side_effect=list_error,
        )

        with caplog.at_level("WARNING", logger=sweep_lifecycle.__name__):
            await sweep_lifecycle.on_delete(
                body=_sweep_body(),
                uid=_SWEEP_UID,
                name=_SWEEP_NAME,
                namespace="benchmark-prod",
            )

        fake.patch_obj.assert_not_awaited()
        assert (
            "cooperative-cancel best-effort failed for benchmark-prod/latency-sweep"
            in caplog.text
        )

    @pytest.mark.asyncio
    async def test_on_delete_repeated_refire_reapplies_child_cancel_idempotently(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeated sweep deletes issue the same json-patch shape without state drift."""
        fake = _install_fake_sweep_delete_k8s(
            monkeypatch,
            children=[_child_job("latency-sweep-v00-t0")],
        )

        await sweep_lifecycle.on_delete(
            body=_sweep_body(),
            uid=_SWEEP_UID,
            name=_SWEEP_NAME,
            namespace="benchmark-prod",
        )
        await sweep_lifecycle.on_delete(
            body=_sweep_body(),
            uid=_SWEEP_UID,
            name=_SWEEP_NAME,
            namespace="benchmark-prod",
        )

        assert fake.list_objs.await_count == 2
        assert fake.patch_obj.await_count == 2
        assert [call.kwargs["body"] for call in fake.patch_obj.await_args_list] == [
            _cancel_patch("child-uid-0001"),
            _cancel_patch("child-uid-0001"),
        ]
