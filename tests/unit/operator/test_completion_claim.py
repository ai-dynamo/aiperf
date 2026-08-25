# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the durable completion-claim dedup mechanism.

The operator replaces the in-memory ``_shutdown_sent`` set with a CR-level
annotation (``Annotations.COMPLETION_CLAIMED``). These tests cover the
``try_claim_completion`` / ``is_completion_claimed`` helpers and verify
that the call-site handlers skip ``handle_completion`` when the claim
already exists.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import kopf
import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.constants import Annotations
from aiperf.kubernetes.phase import Phase
from aiperf.operator.client_cache import (
    _reset_for_testing,
    _shutdown_sent,
    get_claim_timestamp,
    is_completion_claimed,
    try_claim_completion,
)
from aiperf.operator.handlers._completion_retry import _claim_age_seconds


@pytest.fixture(autouse=True)
def _clean_state():
    _reset_for_testing()
    yield
    _reset_for_testing()


def _body_with_annotation() -> dict:
    return {
        "metadata": {
            "annotations": {Annotations.COMPLETION_CLAIMED: "2026-01-01T00:00:00Z"},
        },
    }


def _body_without_annotation() -> dict:
    return {"metadata": {"annotations": {}}}


@asynccontextmanager
async def _fake_k8s_client(mock_api: MagicMock):
    """Helper that yields the supplied mock ApiClient."""
    yield mock_api


class TestIsCompletionClaimed:
    def test_returns_true_when_annotation_set(self) -> None:
        assert is_completion_claimed(_body_with_annotation()) is True

    def test_returns_false_when_annotation_missing(self) -> None:
        assert is_completion_claimed(_body_without_annotation()) is False

    def test_handles_missing_annotations_dict(self) -> None:
        assert is_completion_claimed({"metadata": {}}) is False

    def test_handles_null_annotations(self) -> None:
        assert is_completion_claimed({"metadata": {"annotations": None}}) is False


class TestTryClaimCompletion:
    @pytest.mark.asyncio
    async def test_annotation_present_with_live_claim_loses_race(self) -> None:
        """A body-snapshot claim confirmed against LIVE state is a lost race.

        We must NOT submit an overwriting claim patch in this case: the
        ``test``+``add`` patch would otherwise re-claim an already-claimed CR
        (the ``add`` overwrites the existing annotation) and re-run the
        non-idempotent ``handle_completion``."""
        body = _body_with_annotation()
        submit = AsyncMock(return_value=True)  # would (wrongly) win if reached
        live = AsyncMock(return_value=True)

        with (
            mock_patch(
                "aiperf.operator.client_cache._submit_claim_patch",
                new=submit,
            ),
            mock_patch(
                "aiperf.operator.client_cache._read_live_completion_claimed",
                new=live,
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is False
        live.assert_awaited_once()
        submit.assert_not_awaited()
        # A lost race still populates the in-process cache.
        assert "ns/j" in _shutdown_sent

    @pytest.mark.asyncio
    async def test_forged_annotation_does_not_suppress_claim(self) -> None:
        """A snapshot annotation absent from LIVE state (forged/stale) must NOT
        suppress completion: we still submit the atomic claim. The annotation
        is user-writable, so it cannot be trusted as a skip on its own."""
        body = _body_with_annotation()
        submit = AsyncMock(return_value=True)
        live = AsyncMock(return_value=False)

        with (
            mock_patch(
                "aiperf.operator.client_cache._submit_claim_patch",
                new=submit,
            ),
            mock_patch(
                "aiperf.operator.client_cache._read_live_completion_claimed",
                new=live,
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is True
        live.assert_awaited_once()
        submit.assert_awaited_once()
        assert "ns/j" in _shutdown_sent

    @pytest.mark.asyncio
    async def test_fast_path_in_process_cache(self) -> None:
        """If key already in _shutdown_sent this process, return False without touching body/API."""
        _shutdown_sent.add("ns/j")

        def _raise(*_, **__):
            raise AssertionError("should not open a k8s client")

        with mock_patch(
            "aiperf.operator.client_cache.k8s_client",
            new=_raise,
        ):
            result = await try_claim_completion("ns", "j", _body_without_annotation())

        assert result is False

    @pytest.mark.asyncio
    async def test_successful_claim_patches_annotation(self) -> None:
        """Slow path: mock k8s_client + CustomObjectsApi, verify patch called and True returned."""
        body = _body_without_annotation()
        mock_api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.patch_namespaced_custom_object = AsyncMock(return_value={})

        with (
            mock_patch(
                "aiperf.operator.client_cache.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is True
        assert "ns/j" in _shutdown_sent
        mock_custom.patch_namespaced_custom_object.assert_awaited_once()
        kwargs = mock_custom.patch_namespaced_custom_object.call_args.kwargs
        patch_ops = kwargs["body"]
        assert kwargs.get("_content_type") == "application/json-patch+json"
        assert any(op.get("op") == "test" for op in patch_ops)
        assert any(
            op.get("op") == "add" and "completion-claimed" in op.get("path", "")
            for op in patch_ops
        )

    @pytest.mark.asyncio
    async def test_successful_claim_with_null_annotations(self) -> None:
        """If metadata.annotations is missing, patch adds the dict first."""
        body = {"metadata": {}}
        mock_api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.patch_namespaced_custom_object = AsyncMock(return_value={})

        with (
            mock_patch(
                "aiperf.operator.client_cache.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is True
        patch_ops = mock_custom.patch_namespaced_custom_object.call_args.kwargs["body"]
        # Three-op sequence: test current metadata, add dict, add annotation.
        assert len(patch_ops) == 3
        assert patch_ops[0]["op"] == "test"
        assert patch_ops[0]["path"] == "/metadata"
        assert patch_ops[0]["value"] == {}
        assert patch_ops[1]["op"] == "add"
        assert patch_ops[1]["path"] == "/metadata/annotations"

    @pytest.mark.asyncio
    async def test_conflict_409_without_live_claim_is_retryable(self) -> None:
        """409 from an unrelated resourceVersion change must not poison the cache."""
        body = {"metadata": {"resourceVersion": "42"}}
        mock_api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=409, reason="conflict")
        )
        mock_custom.get_namespaced_custom_object = AsyncMock(
            return_value={"metadata": {"resourceVersion": "43", "annotations": {}}}
        )

        with (
            mock_patch(
                "aiperf.operator.client_cache.k8s_client",
                side_effect=lambda: _fake_k8s_client(mock_api),
            ),
            mock_patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is False
        assert "ns/j" not in _shutdown_sent
        mock_custom.get_namespaced_custom_object.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_conflict_409_with_live_claim_marks_lost_race(self) -> None:
        """409 with the live claim annotation is still a completion-race loss."""
        body = {"metadata": {"resourceVersion": "42"}}
        mock_api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=409, reason="conflict")
        )
        mock_custom.get_namespaced_custom_object = AsyncMock(
            return_value={
                "metadata": {
                    "resourceVersion": "43",
                    "annotations": {
                        Annotations.COMPLETION_CLAIMED: "2026-01-01T00:00:00Z"
                    },
                }
            }
        )

        with (
            mock_patch(
                "aiperf.operator.client_cache.k8s_client",
                side_effect=lambda: _fake_k8s_client(mock_api),
            ),
            mock_patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is False
        assert "ns/j" in _shutdown_sent
        mock_custom.get_namespaced_custom_object.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unprocessable_422_returns_false(self) -> None:
        """On 422, return False without poisoning the retry cache."""
        body = _body_without_annotation()
        mock_api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=422, reason="test op failed")
        )

        with (
            mock_patch(
                "aiperf.operator.client_cache.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is False
        assert "ns/j" not in _shutdown_sent

    @pytest.mark.asyncio
    async def test_other_server_error_returns_false_without_marking(self) -> None:
        """Non-409/422 errors return False but do not poison the in-process cache.

        We must be able to retry later; a transient 500 should not latch
        the claim off for the lifetime of the operator process.
        """
        body = _body_without_annotation()
        mock_api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=500, reason="server error")
        )

        with (
            mock_patch(
                "aiperf.operator.client_cache.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is False
        assert "ns/j" not in _shutdown_sent

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_false(self) -> None:
        """Any unexpected error returns False fail-safe (don't double-complete)."""
        body = _body_without_annotation()

        def _raise(*_, **__):
            raise RuntimeError("boom")

        with mock_patch(
            "aiperf.operator.client_cache.k8s_client",
            new=_raise,
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is False
        assert "ns/j" not in _shutdown_sent


class TestLifecycleIntegration:
    """Verify on_benchmark_complete honours the durable claim annotation."""

    @pytest.mark.asyncio
    async def test_annotation_preset_on_body_skips_handle_completion(self) -> None:
        """If the body carries the claim AND it is confirmed in LIVE state,
        the handler loses the race and does not re-run handle_completion."""
        from aiperf.operator.handlers.lifecycle import on_benchmark_complete

        patch = MagicMock()
        patch.status = {}

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new_callable=AsyncMock,
            ) as mock_handle,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.get_or_create_progress_client",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.client_cache._read_live_completion_claimed",
                new=AsyncMock(return_value=True),
            ),
        ):
            await on_benchmark_complete(
                body=_body_with_annotation(),
                status={"phase": Phase.RUNNING, "jobId": "j", "jobSetName": "js"},
                name="j",
                namespace="ns",
                patch=patch,
            )

        mock_handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_claim_failure_skips_handle_completion(self) -> None:
        """If try_claim_completion returns False, handle_completion is not called."""
        from aiperf.operator.handlers.lifecycle import on_benchmark_complete

        patch = MagicMock()
        patch.status = {}

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.try_claim_completion",
                new_callable=AsyncMock,
                return_value=False,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            await on_benchmark_complete(
                body=_body_without_annotation(),
                status={"phase": Phase.RUNNING, "jobId": "j", "jobSetName": "js"},
                name="j",
                namespace="ns",
                patch=patch,
            )

        mock_handle.assert_not_called()


class TestMonitorIntegration:
    """Verify monitor_progress' JobSet-completed path honours the durable claim.

    Two regimes for an existing ``completion-claimed`` annotation:

    - ``phase`` is already terminal (Completed / Failed / Cancelled) → the
      annotation is authoritative evidence that a prior handler finished;
      ``handle_completion`` must NOT run again.
    - ``phase`` is non-terminal (Running / Initializing) → the previous
      handler crashed between claim-patch and status-patch, orphaning the
      CR. ``monitor_progress`` re-runs ``handle_completion`` so the CR
      converges to Completed instead of staying stuck. See
      tests/kubernetes/chaos/test_chaos_operator_resilience.py::
      test_c5_orphaned_claim_recovers.
    """

    @pytest.mark.asyncio
    async def test_terminal_phase_with_annotation_skips_handle_completion(
        self,
    ) -> None:
        """Annotation + terminal phase = prior handler finished; do NOT re-enter."""
        from aiperf.operator.handlers.monitor import monitor_progress

        patch = MagicMock()
        patch.status = {}

        jobset_raw = {
            "status": {
                "conditions": [{"type": "Completed", "status": "True"}],
                "replicatedJobsStatus": [],
            },
            "metadata": {"labels": {}},
            "spec": {},
        }
        mock_api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(return_value=jobset_raw)

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new_callable=AsyncMock,
            ) as mock_handle,
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new_callable=AsyncMock,
            ),
        ):
            # monitor_progress has an early-return for terminal phases, so this
            # exits before the JobSet condition check even reads the body.
            await monitor_progress(
                body=_body_with_annotation(),
                status={
                    "phase": Phase.COMPLETED,
                    "jobId": "j",
                    "jobSetName": "js",
                },
                spec={},
                name="j",
                namespace="ns",
                patch=patch,
            )

        mock_handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_orphaned_claim_triggers_handle_completion_recovery(self) -> None:
        """Annotation set + non-terminal phase + benchmark complete = recover.

        The orphan-recovery branch is gated on
        ``_benchmark_appears_complete`` returning True; otherwise firing
        the recovery during an in-flight benchmark drives
        ``handle_completion`` into a retry-stagnation loop and marks the
        CR Failed even when the benchmark would have succeeded.
        """
        from aiperf.operator.handlers.monitor import monitor_progress

        patch = MagicMock()
        patch.status = {}

        mock_api = MagicMock()
        mock_progress_client = AsyncMock()
        mock_progress_client.send_shutdown = AsyncMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._benchmark_appears_complete",
                new_callable=AsyncMock,
                return_value=True,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new_callable=AsyncMock,
            ) as mock_handle,
            mock_patch(
                "aiperf.operator.handlers.monitor.get_or_create_progress_client",
                new_callable=AsyncMock,
                return_value=mock_progress_client,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new_callable=AsyncMock,
            ),
        ):
            await monitor_progress(
                body=_body_with_annotation(),
                status={
                    "phase": Phase.RUNNING,
                    "jobId": "j",
                    "jobSetName": "js",
                },
                spec={},
                name="j",
                namespace="ns",
                patch=patch,
            )

        mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_orphaned_claim_deferred_while_benchmark_in_flight(self) -> None:
        """Annotation set + non-terminal phase + benchmark NOT complete = defer recovery.

        Pre-stamping the claim while the benchmark is still running (see
        C5 chaos scenario) must not trigger ``handle_completion``: the
        fetch path would stagnate on missing export files and mark the
        CR Failed. Recovery should wait for the next tick, when the
        benchmark has actually finished.
        """
        from aiperf.operator.handlers.monitor import monitor_progress

        patch = MagicMock()
        patch.status = {}

        jobset_raw = {
            "status": {
                "conditions": [],
                "replicatedJobsStatus": [],
            },
            "metadata": {"labels": {}},
            "spec": {},
        }
        mock_api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(return_value=jobset_raw)

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._benchmark_appears_complete",
                new_callable=AsyncMock,
                return_value=False,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new_callable=AsyncMock,
            ) as mock_handle,
            mock_patch(
                "aiperf.operator.handlers.monitor._run_worker_and_progress_phase",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new_callable=AsyncMock,
            ),
        ):
            await monitor_progress(
                body=_body_with_annotation(),
                status={
                    "phase": Phase.RUNNING,
                    "jobId": "j",
                    "jobSetName": "js",
                },
                spec={},
                name="j",
                namespace="ns",
                patch=patch,
            )

        mock_handle.assert_not_called()


class TestClaimLatchWithReadOnlyBody:
    """kopf hands handlers a read-only ``kopf.Body``, not a ``dict``.

    Every ``try_claim_completion`` call site in ``handlers/`` is reached from a
    kopf handler, so the success branch must not assume the body snapshot is
    mutable. It previously called ``body.setdefault(...)``, which raises
    ``AttributeError`` on ``kopf.Body`` (a ``Mapping``), killing the monitor
    timer AFTER the apiserver claim patch had already landed -- leaving an
    orphaned claim that only the recovery path could converge.
    """

    @pytest.mark.asyncio
    async def test_claim_succeeds_with_kopf_body(self) -> None:
        body = kopf.Body({"metadata": {"annotations": {}}})
        assert not isinstance(body, MutableMapping)

        with (
            mock_patch(
                "aiperf.operator.client_cache._submit_claim_patch",
                new=AsyncMock(return_value=True),
            ),
            mock_patch(
                "aiperf.operator.client_cache._post_dashboard_refresh",
                new=AsyncMock(),
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is True
        assert "ns/j" in _shutdown_sent

    @pytest.mark.asyncio
    async def test_claim_age_readable_after_immutable_body_claim(self) -> None:
        """The same-tick retry gate must still resolve a claim age.

        The latch cannot be written into an immutable body, so the claim
        timestamp has to come from the process-local registry instead.
        """
        body = kopf.Body({"metadata": {"annotations": {}}})

        with (
            mock_patch(
                "aiperf.operator.client_cache._submit_claim_patch",
                new=AsyncMock(return_value=True),
            ),
            mock_patch(
                "aiperf.operator.client_cache._post_dashboard_refresh",
                new=AsyncMock(),
            ),
        ):
            assert await try_claim_completion("ns", "j", body) is True

        assert get_claim_timestamp("ns/j") is not None
        age = _claim_age_seconds(dict(body), "ns", "j")
        assert age is not None and age >= 0

    @pytest.mark.asyncio
    async def test_mutable_body_still_latched_in_place(self) -> None:
        """Plain dicts keep the annotation written back into the snapshot."""
        body: dict = {"metadata": {"annotations": {}}}

        with (
            mock_patch(
                "aiperf.operator.client_cache._submit_claim_patch",
                new=AsyncMock(return_value=True),
            ),
            mock_patch(
                "aiperf.operator.client_cache._post_dashboard_refresh",
                new=AsyncMock(),
            ),
        ):
            assert await try_claim_completion("ns", "j", body) is True

        assert body["metadata"]["annotations"][Annotations.COMPLETION_CLAIMED]
        assert _claim_age_seconds(body, "ns", "j") is not None
