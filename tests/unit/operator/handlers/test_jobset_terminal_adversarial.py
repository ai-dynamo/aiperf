# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for watch-driven JobSet terminal fast-path handling.

Focuses on:
- Completed/Failed condition parsing at the kopf trust boundary.
- Parent AIPerfJob lookup and benchmark-complete annotation patch shape.
- Duplicate event idempotence and controller-race skip semantics.
- Name/namespace derivation for JobSet-to-AIPerfJob routing.
- API-error handling on lookup and patch operations.

Out of scope: monitor-timer recovery for Failed JobSets; see sibling
``tests/unit/operator/test_monitor_state_machine_edges.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.constants import AIPerfLabels, Annotations
from aiperf.kubernetes.cr_refs import AIPERF_JOB_API_VERSION
from aiperf.operator.handlers import jobset_terminal

# =============================================================================
# Helpers
# =============================================================================


def _completed_condition(**overrides: Any) -> dict[str, Any]:
    """Build a JobSet Completed condition with realistic controller fields."""
    condition: dict[str, Any] = {
        "type": "Completed",
        "status": "True",
        "reason": "AllJobsCompleted",
        "message": "JobSet aiperf-llama3-8b-throughput completed all jobs",
    }
    condition.update(overrides)
    return condition


def _failed_condition(**overrides: Any) -> dict[str, Any]:
    """Build a JobSet Failed condition owned by monitor-timer recovery."""
    condition: dict[str, Any] = {
        "type": "Failed",
        "status": "True",
        "reason": "ReplicatedJobFailed",
        "message": "controller exited before artifacts were harvested",
    }
    condition.update(overrides)
    return condition


def _aiperfjob_body(
    *,
    name: str = "llama3-8b-throughput",
    namespace: str = "bench-prod",
    annotations: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a minimal parent AIPerfJob body returned by the apiserver."""
    metadata: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "uid": "aiperfjob-7f2a",
    }
    if annotations is not None:
        metadata["annotations"] = annotations
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": metadata,
    }


def _trusted_jobset_body(
    *,
    name: str = "aiperf-llama3-8b-throughput",
    parent_name: str = "llama3-8b-throughput",
    parent_uid: str = "aiperfjob-7f2a",
) -> dict[str, Any]:
    """Build a JobSet body with the ownership proof the handler requires."""
    return {
        "metadata": {
            "name": name,
            "labels": {
                AIPerfLabels.APP_KEY: AIPerfLabels.APP_VALUE,
                AIPerfLabels.JOB_ID: parent_name,
            },
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_JOB_API_VERSION,
                    "kind": "AIPerfJob",
                    "name": parent_name,
                    "uid": parent_uid,
                }
            ],
        }
    }


@asynccontextmanager
async def _fake_k8s_client() -> AsyncIterator[MagicMock]:
    """Yield a fake ApiClient for lazy ``k8s_client`` call sites."""
    yield MagicMock(name="ApiClient")


def _install_custom_objects_api(
    monkeypatch: pytest.MonkeyPatch,
    *,
    get_result: dict[str, Any] | BaseException | None = None,
    patch_result: object | BaseException | None = None,
) -> SimpleNamespace:
    """Install fake k8s client factories and return captured API methods."""
    get = AsyncMock(name="get_namespaced_custom_object")
    if isinstance(get_result, BaseException):
        get.side_effect = get_result
    else:
        get.return_value = get_result

    patch = AsyncMock(name="patch_namespaced_custom_object")
    if isinstance(patch_result, BaseException):
        patch.side_effect = patch_result
    else:
        patch.return_value = patch_result

    custom = MagicMock(name="CustomObjectsApi")
    custom.get_namespaced_custom_object = get
    custom.patch_namespaced_custom_object = patch

    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client",
        lambda: _fake_k8s_client(),
    )
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi",
        MagicMock(return_value=custom),
    )
    return SimpleNamespace(custom=custom, get=get, patch=patch)


# =============================================================================
# Completed/Failed condition parsing
# =============================================================================


class TestJobSetTerminalConditionParsing:
    """Malformed JobSet conditions must not trigger false terminal-success."""

    @pytest.mark.parametrize(
        "conditions,expected",
        [
            (None, False),
            ([], False),
            param([None, "not-a-condition", 17], False, id="non-dict-entries"),
            ([{"type": "Completed"}], False),
            ([{"type": "Completed", "status": "False"}], False),
            param([{"type": "Completed", "status": "true"}], False, id="lowercase-true"),
            ([_failed_condition()], False),
            param([_failed_condition(), _completed_condition()], True, id="completed-wins-over-failed"),
        ],
    )  # fmt: skip
    def test_has_completed_condition_malformed_or_failed_inputs_return_expected(
        self, conditions: list[object] | None, expected: bool
    ) -> None:
        assert jobset_terminal._has_completed_condition(conditions) is expected

    @pytest.mark.asyncio
    async def test_handle_jobset_conditions_failed_true_does_not_patch_or_lookup(
        self,
    ) -> None:
        """Failed/True is intentionally left to monitor-timer salvage logic."""
        lookup = AsyncMock(return_value=_aiperfjob_body())
        setter = AsyncMock()

        with (
            pytest.MonkeyPatch.context() as monkeypatch,
        ):
            monkeypatch.setattr(jobset_terminal, "_lookup_aiperfjob_body", lookup)
            monkeypatch.setattr(
                jobset_terminal,
                "_set_benchmark_complete_annotation",
                setter,
            )
            await jobset_terminal.handle_jobset_conditions(
                old=[],
                new=[_failed_condition()],
                namespace="bench-prod",
                jobset_name="aiperf-llama3-8b-throughput",
            )

        lookup.assert_not_awaited()
        setter.assert_not_awaited()


# =============================================================================
# Parent lookup and annotation patching
# =============================================================================


class TestJobSetTerminalParentLookupAndPatch:
    """Completed JobSets route to exactly one parent annotation patch."""

    @pytest.mark.asyncio
    async def test_lookup_aiperfjob_body_prefixed_jobset_uses_derived_parent_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install_custom_objects_api(
            monkeypatch,
            get_result=_aiperfjob_body(name="llama3-8b-throughput"),
        )

        result = await jobset_terminal._lookup_aiperfjob_body(
            "bench-prod",
            "aiperf-llama3-8b-throughput",
        )

        assert result == _aiperfjob_body(name="llama3-8b-throughput")
        fake.get.assert_awaited_once()
        kwargs = fake.get.await_args.kwargs
        assert kwargs["namespace"] == "bench-prod"
        assert kwargs["plural"] == "aiperfjobs"
        assert kwargs["name"] == "llama3-8b-throughput"

    @pytest.mark.asyncio
    async def test_lookup_aiperfjob_body_unprefixed_jobset_skips_api_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install_custom_objects_api(monkeypatch, get_result=_aiperfjob_body())

        result = await jobset_terminal._lookup_aiperfjob_body(
            "bench-prod",
            "llama3-8b-throughput",
        )

        assert result is None
        fake.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_set_benchmark_complete_annotation_uses_metadata_patch_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install_custom_objects_api(monkeypatch)

        await jobset_terminal._set_benchmark_complete_annotation(
            "bench-prod",
            "llama3-8b-throughput",
        )

        fake.patch.assert_awaited_once()
        kwargs = fake.patch.await_args.kwargs
        assert kwargs["namespace"] == "bench-prod"
        assert kwargs["plural"] == "aiperfjobs"
        assert kwargs["name"] == "llama3-8b-throughput"
        assert kwargs["body"] == {
            "metadata": {
                "annotations": {Annotations.BENCHMARK_COMPLETE: "true"},
            },
        }

    @pytest.mark.asyncio
    async def test_handle_jobset_conditions_completed_patches_derived_parent_once(
        self,
    ) -> None:
        lookup = AsyncMock(return_value=_aiperfjob_body(name="llama3-8b-throughput"))
        setter = AsyncMock()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(jobset_terminal, "_lookup_aiperfjob_body", lookup)
            monkeypatch.setattr(
                jobset_terminal,
                "_set_benchmark_complete_annotation",
                setter,
            )
            await jobset_terminal.handle_jobset_conditions(
                old=[],
                new=[_completed_condition()],
                namespace="bench-prod",
                jobset_name="aiperf-llama3-8b-throughput",
                jobset_body=_trusted_jobset_body(),
            )

        lookup.assert_awaited_once_with("bench-prod", "aiperf-llama3-8b-throughput")
        setter.assert_awaited_once_with("bench-prod", "llama3-8b-throughput")


# =============================================================================
# Missing owner/labels, duplicate events, and wrong name/namespace
# =============================================================================


class TestJobSetTerminalRoutingAdversaries:
    """Fast-path routing must avoid annotating the wrong AIPerfJob."""

    @pytest.mark.asyncio
    async def test_handle_jobset_conditions_existing_completion_annotation_skips_patch(
        self,
    ) -> None:
        """Controller-pod annotation wins the duplicate event race."""
        lookup = AsyncMock(
            return_value=_aiperfjob_body(
                annotations={Annotations.BENCHMARK_COMPLETE: "true"},
            )
        )
        setter = AsyncMock()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(jobset_terminal, "_lookup_aiperfjob_body", lookup)
            monkeypatch.setattr(
                jobset_terminal,
                "_set_benchmark_complete_annotation",
                setter,
            )
            await jobset_terminal.handle_jobset_conditions(
                old=[],
                new=[_completed_condition()],
                namespace="bench-prod",
                jobset_name="aiperf-llama3-8b-throughput",
            )

        lookup.assert_awaited_once()
        setter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handle_jobset_conditions_duplicate_completed_old_skips_lookup(
        self,
    ) -> None:
        """A re-fired Completed event must not spend another CR get."""
        completed = [_completed_condition()]
        lookup = AsyncMock(return_value=_aiperfjob_body())
        setter = AsyncMock()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(jobset_terminal, "_lookup_aiperfjob_body", lookup)
            monkeypatch.setattr(
                jobset_terminal,
                "_set_benchmark_complete_annotation",
                setter,
            )
            await jobset_terminal.handle_jobset_conditions(
                old=completed,
                new=completed,
                namespace="bench-prod",
                jobset_name="aiperf-llama3-8b-throughput",
            )

        lookup.assert_not_awaited()
        setter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handle_jobset_conditions_missing_parent_body_skips_patch(
        self,
    ) -> None:
        """Sweep-owned or already-deleted JobSets must not annotate anything."""
        lookup = AsyncMock(return_value=None)
        setter = AsyncMock()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(jobset_terminal, "_lookup_aiperfjob_body", lookup)
            monkeypatch.setattr(
                jobset_terminal,
                "_set_benchmark_complete_annotation",
                setter,
            )
            await jobset_terminal.handle_jobset_conditions(
                old=[],
                new=[_completed_condition()],
                namespace="bench-prod",
                jobset_name="aiperf-missing-parent",
            )

        lookup.assert_awaited_once_with("bench-prod", "aiperf-missing-parent")
        setter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handle_jobset_conditions_wrong_namespace_stays_in_event_namespace(
        self,
    ) -> None:
        """The patch target namespace must be the JobSet event namespace, not parent metadata."""
        lookup = AsyncMock(
            return_value=_aiperfjob_body(
                name="llama3-8b-throughput",
                namespace="bench-staging",
            )
        )
        setter = AsyncMock()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(jobset_terminal, "_lookup_aiperfjob_body", lookup)
            monkeypatch.setattr(
                jobset_terminal,
                "_set_benchmark_complete_annotation",
                setter,
            )
            await jobset_terminal.handle_jobset_conditions(
                old=[],
                new=[_completed_condition()],
                namespace="bench-prod",
                jobset_name="aiperf-llama3-8b-throughput",
                jobset_body=_trusted_jobset_body(),
            )

        setter.assert_awaited_once_with("bench-prod", "llama3-8b-throughput")

    @pytest.mark.asyncio
    async def test_handle_jobset_conditions_missing_owner_or_labels_does_not_patch_parent(
        self,
    ) -> None:
        """A name-colliding JobSet without AIPerf ownership must not annotate a CR.

        Kopf field events include only old/new conditions plus resource name in the
        current wrapper. This test encodes the trust-boundary contract: name alone
        is insufficient evidence that the JobSet belongs to the AIPerfJob.
        """
        lookup = AsyncMock(return_value=_aiperfjob_body(name="llama3-8b-throughput"))
        setter = AsyncMock()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(jobset_terminal, "_lookup_aiperfjob_body", lookup)
            monkeypatch.setattr(
                jobset_terminal,
                "_set_benchmark_complete_annotation",
                setter,
            )
            await jobset_terminal.handle_jobset_conditions(
                old=[],
                new=[_completed_condition()],
                namespace="bench-prod",
                jobset_name="aiperf-llama3-8b-throughput",
            )

        setter.assert_not_awaited()


# =============================================================================
# API-error handling
# =============================================================================


class TestJobSetTerminalApiErrorHandling:
    """Lookup and patch API failures must have explicit retry/no-retry semantics."""

    @pytest.mark.asyncio
    async def test_lookup_aiperfjob_body_404_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_custom_objects_api(
            monkeypatch,
            get_result=ApiException(status=404, reason="Not Found"),
        )

        result = await jobset_terminal._lookup_aiperfjob_body(
            "bench-prod",
            "aiperf-deleted-parent",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_aiperfjob_body_503_propagates_for_kopf_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Transient apiserver lookup failures are not evidence of a missing parent."""
        _install_custom_objects_api(
            monkeypatch,
            get_result=ApiException(status=503, reason="apiserver unavailable"),
        )

        with pytest.raises(ApiException) as excinfo:
            await jobset_terminal._lookup_aiperfjob_body(
                "bench-prod",
                "aiperf-llama3-8b-throughput",
            )

        assert excinfo.value.status == 503

    @pytest.mark.asyncio
    async def test_set_benchmark_complete_annotation_api_error_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _install_custom_objects_api(
            monkeypatch,
            patch_result=ApiException(status=409, reason="conflict"),
        )

        await jobset_terminal._set_benchmark_complete_annotation(
            "bench-prod",
            "llama3-8b-throughput",
        )

        assert "Failed to set benchmark-complete annotation" in caplog.text
        assert "bench-prod/llama3-8b-throughput" in caplog.text
