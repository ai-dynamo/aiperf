# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for AIPerfJob create/finalize behavior.

Focuses on:
- cooperative cancellation after awaited create-handler steps
- malformed CR specs and config-validation status contracts
- partial Kubernetes resource failures before finalization
- observedGeneration stamping only on successful finalize
- idempotent retry shape for RBAC, ConfigMap, and JobSet creation

Out of scope: monitor/completion state-machine transitions, covered by
``tests/unit/operator/test_monitor_state_machine_edges.py`` and
``tests/unit/operator/test_completion_handler.py``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Callable, Generator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call
from unittest.mock import patch as mock_patch

import kopf
import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.operator.client_cache import (
    _reset_for_testing,
    is_cancellation_requested,
    job_key,
    request_cancellation,
)
from aiperf.operator.handlers.create import on_create
from tests.harness.operator import build_minimal_aiperfjob_spec, build_sample_body

# =============================================================================
# Helpers
# =============================================================================


def _patch_obj() -> MagicMock:
    """Build a kopf-like patch object with a mutable status mapping."""
    patch = MagicMock()
    patch.status = {}
    return patch


def _body_with_generation(generation: int = 17) -> dict[str, Any]:
    """Build a realistic AIPerfJob body with metadata.generation set."""
    body = build_sample_body()
    body["metadata"].update(
        {
            "name": "aiperf-bench-7f2a",
            "namespace": "production",
            "uid": "uid-aiperf-bench-7f2a",
            "generation": generation,
        }
    )
    return body


def _spec_with(**overrides: Any) -> dict[str, Any]:
    """Mutate the canonical validated AIPerfJob baseline via dict spread."""
    return {**build_minimal_aiperfjob_spec(), **overrides}


@asynccontextmanager
async def _fake_k8s_client() -> AsyncIterator[MagicMock]:
    """Yield a mock ApiClient without opening real Kubernetes connections."""
    yield MagicMock(name="ApiClient")


def _preflight_result(*, passed: bool = True) -> MagicMock:
    """Build a preflight result object matching OperatorPreflightChecker output."""
    from aiperf.kubernetes.preflight import CheckStatus

    check = MagicMock()
    check.status = CheckStatus.PASS if passed else CheckStatus.FAIL
    check.name = "gpu-quota"
    check.message = "quota admits 1 worker pod"

    result = MagicMock()
    result.passed = passed
    result.checks = [check]
    return result


def _conditions_by_type(patch: MagicMock) -> dict[str, dict[str, Any]]:
    """Return finalized status conditions keyed by condition type."""
    return {
        condition["type"]: condition for condition in patch.status.get("conditions", [])
    }


class _CreateHarness(SimpleNamespace):
    """Captured mocks installed around ``on_create`` for behavior assertions."""

    health: AsyncMock
    configmap: AsyncMock
    role: AsyncMock
    role_binding: AsyncMock
    jobset: AsyncMock
    save_spec: AsyncMock
    upsert_index: AsyncMock
    preflight_run_all: AsyncMock


@asynccontextmanager
async def _patched_create_dependencies(
    *,
    health_side_effect: Callable[[], object] | None = None,
    configmap_side_effect: Callable[[], object] | BaseException | None = None,
    jobset_side_effect: Callable[[], object] | BaseException | None = None,
) -> AsyncIterator[_CreateHarness]:
    """Patch all external create-handler dependencies and yield captured mocks."""
    harness = _CreateHarness(
        health=AsyncMock(return_value=MagicMock(reachable=True, error=None)),
        configmap=AsyncMock(),
        role=AsyncMock(),
        role_binding=AsyncMock(),
        jobset=AsyncMock(),
        save_spec=AsyncMock(),
        upsert_index=AsyncMock(),
        preflight_run_all=AsyncMock(return_value=_preflight_result()),
    )
    if health_side_effect is not None:
        harness.health.side_effect = health_side_effect
    if configmap_side_effect is not None:
        harness.configmap.side_effect = configmap_side_effect
    if jobset_side_effect is not None:
        harness.jobset.side_effect = jobset_side_effect

    runs_index = MagicMock(upsert_run_created=harness.upsert_index)

    with (
        mock_patch(
            "aiperf.operator.handlers.create.check_endpoint_health", new=harness.health
        ),
        mock_patch(
            "aiperf.operator.handlers.create.k8s_client",
            return_value=_fake_k8s_client(),
        ),
        mock_patch(
            "aiperf.operator.preflight.OperatorPreflightChecker"
        ) as preflight_cls,
        mock_patch(
            "aiperf.operator.handlers.create.create_idempotent_config_map",
            new=harness.configmap,
        ),
        mock_patch(
            "aiperf.operator.handlers.create.create_idempotent_role",
            new=harness.role,
        ),
        mock_patch(
            "aiperf.operator.handlers.create.create_idempotent_role_binding",
            new=harness.role_binding,
        ),
        mock_patch(
            "aiperf.operator.handlers.create.create_idempotent_custom_object",
            new=harness.jobset,
        ),
        mock_patch(
            "aiperf.operator.handlers.create.save_job_spec_file", new=harness.save_spec
        ),
        mock_patch("aiperf.operator.handlers.create.runs_index", new=runs_index),
        mock_patch("aiperf.operator.handlers.create.events.spec_valid"),
        mock_patch("aiperf.operator.handlers.create.events.spec_invalid"),
        mock_patch("aiperf.operator.handlers.create.events.endpoint_reachable"),
        mock_patch("aiperf.operator.handlers.create.events.preflight_passed"),
        mock_patch("aiperf.operator.handlers.create.events.resources_created"),
        mock_patch("aiperf.operator.handlers.create.events.created"),
        mock_patch("aiperf.operator.handlers.create.events.failed"),
    ):
        preflight_cls.return_value.run_all = harness.preflight_run_all
        yield harness


async def _call_on_create(
    *,
    spec: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    patch: kopf.Patch | MagicMock | None = None,
) -> tuple[dict[str, Any], kopf.Patch | MagicMock]:
    """Run ``on_create`` with realistic identifiers and return result plus patch."""
    body = body or _body_with_generation()
    patch = _patch_obj() if patch is None else patch
    result = await on_create(
        body=body,
        spec=spec or build_minimal_aiperfjob_spec(),
        name="aiperf-bench-7f2a",
        namespace="production",
        uid="uid-aiperf-bench-7f2a",
        patch=patch,
    )
    return result, patch


@pytest.fixture(autouse=True)
def _reset_operator_caches() -> Generator[None, None, None]:
    """Clear cancellation flags and client caches before/after each adversarial test."""
    _reset_for_testing()
    yield
    _reset_for_testing()


# =============================================================================
# Validation and malformed specs
# =============================================================================


class TestCreateValidationAdversarial:
    """Malformed specs fail before side-effectful Kubernetes calls."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "spec,match",
        [
            param(
                _spec_with(benchmark=["not-a-mapping"]),
                r"(?s)Invalid spec: .*benchmark",
                id="benchmark-list-rejected",
            ),
            param(
                _spec_with(
                    sweep={
                        "type": "grid",
                        "parameters": {"phases.profiling.concurrency": [1, 2]},
                    }
                ),
                r"(?s)Invalid spec: .*AIPerfJob\.spec\.sweep",
                id="aiperfjob-sweep-block-rejected",
            ),
        ],
    )  # fmt: skip
    async def test_on_create_malformed_spec_raises_permanent_error_before_k8s(
        self, spec: dict[str, Any], match: str
    ) -> None:
        patch = _patch_obj()

        with (
            mock_patch("aiperf.operator.handlers.create.k8s_client") as k8s_client,
            pytest.raises(kopf.PermanentError, match=match),
        ):
            await _call_on_create(spec=spec, patch=patch)

        k8s_client.assert_not_called()
        assert patch.status["phase"] == "Failed"
        assert "completionTime" in patch.status
        assert "observedGeneration" not in patch.status
        config_valid = _conditions_by_type(patch)["ConfigValid"]
        assert config_valid["status"] == "False"
        assert "Invalid spec" in patch.status["error"]

    @pytest.mark.asyncio
    async def test_on_create_endpoint_bad_scheme_reports_field_and_bad_value(
        self,
    ) -> None:
        spec = build_minimal_aiperfjob_spec()
        spec["benchmark"] = {
            **spec["benchmark"],
            "endpoint": {"urls": ["grpc://aiperf.invalid/v1/chat/completions"]},
        }
        patch = _patch_obj()

        with pytest.raises(
            kopf.PermanentError,
            match=r"(?s)Invalid spec: .*endpoint.*grpc://aiperf\.invalid",
        ):
            await _call_on_create(spec=spec, patch=patch)

        assert patch.status["phase"] == "Failed"
        assert "completionTime" in patch.status
        assert "observedGeneration" not in patch.status
        assert "grpc://aiperf.invalid" in patch.status["error"]

    @pytest.mark.asyncio
    async def test_on_create_unknown_jinja_variable_is_permanent_spec_error(
        self,
    ) -> None:
        spec = build_minimal_aiperfjob_spec()
        spec["benchmark"]["phases"]["concurrency"] = "{{ typo }}"
        patch = _patch_obj()

        with pytest.raises(kopf.PermanentError, match=r"Invalid spec: .*typo"):
            await _call_on_create(spec=spec, patch=patch)

        config_valid = _conditions_by_type(patch)["ConfigValid"]
        assert config_valid["status"] == "False"
        assert patch.status["phase"] == "Failed"
        assert "completionTime" in patch.status


# =============================================================================
# Success finalization and observedGeneration
# =============================================================================


class TestCreateFinalizeSuccess:
    """Successful create patches the CR with user-visible resource metadata."""

    @pytest.mark.asyncio
    async def test_on_create_success_stamps_observed_generation_and_patch_shape(
        self,
    ) -> None:
        async with _patched_create_dependencies() as harness:
            result, patch = await _call_on_create(body=_body_with_generation(23))

        assert result == {"jobSetName": "aiperf-aiperf-bench-7f2a", "workers": 1}
        assert patch.status["observedGeneration"] == 23
        assert patch.status["phase"] == "Pending"
        assert patch.status["jobId"] == "aiperf-bench-7f2a"
        assert patch.status["jobSetName"] == "aiperf-aiperf-bench-7f2a"
        assert patch.status["workers"] == {"ready": 0, "total": 1}
        assert "startTime" in patch.status

        resources = _conditions_by_type(patch)["ResourcesCreated"]
        assert resources["status"] == "True"
        assert (
            "Created ConfigMap/aiperf-aiperf-bench-7f2a-config" in resources["message"]
        )
        assert "JobSet/aiperf-aiperf-bench-7f2a" in resources["message"]
        harness.jobset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_create_renders_envelope_jinja_before_spec_validation(
        self,
    ) -> None:
        spec = build_minimal_aiperfjob_spec()
        spec["variables"] = {
            "concurrency_per_worker": 2,
            "workers": 3,
            "total_concurrency": "{{ concurrency_per_worker * workers }}",
        }
        spec["randomSeed"] = 42
        spec["benchmark"]["phases"]["concurrency"] = "{{ total_concurrency }}"

        async with _patched_create_dependencies():
            result, patch = await _call_on_create(spec=spec)

        assert result["workers"] == 1
        assert patch.status["phase"] == "Pending"
        assert _conditions_by_type(patch)["ConfigValid"]["status"] == "True"

    @pytest.mark.asyncio
    async def test_on_create_success_without_generation_skips_observed_generation(
        self,
    ) -> None:
        body = _body_with_generation()
        body["metadata"].pop("generation")

        async with _patched_create_dependencies():
            _, patch = await _call_on_create(body=body)

        assert patch.status["phase"] == "Pending"
        assert "observedGeneration" not in patch.status


# =============================================================================
# Partial resource failures and retry safety
# =============================================================================


class TestCreatePartialFailure:
    """Failures after partial resource creation remain retryable and unfinalized."""

    @pytest.mark.asyncio
    async def test_on_create_jobset_failure_raises_temporary_error_without_success_status(
        self,
    ) -> None:
        api_exc = ApiException(status=500, reason="jobset webhook unavailable")
        patch = _patch_obj()

        async with _patched_create_dependencies(jobset_side_effect=api_exc) as harness:
            with pytest.raises(
                kopf.TemporaryError,
                match=r"Transient error creating AIPerfJob production/aiperf-bench-7f2a",
            ):
                await _call_on_create(patch=patch)

        assert "observedGeneration" not in patch.status
        assert "ResourcesCreated" not in _conditions_by_type(patch)
        harness.save_spec.assert_awaited_once()
        harness.upsert_index.assert_awaited_once()
        harness.jobset.assert_awaited_once()
        assert harness.role.await_count == 1
        assert harness.role_binding.await_count == 1
        assert harness.configmap.await_count == 1

    @pytest.mark.asyncio
    async def test_on_create_unexpected_failure_stamps_completion_time(self) -> None:
        """A non-retryable create failure is terminal for status and TTL."""
        patch = _patch_obj()

        with (
            mock_patch(
                "aiperf.operator.handlers.create._create_resources",
                new=AsyncMock(side_effect=RuntimeError("manifest invariant failed")),
            ),
            mock_patch("aiperf.operator.handlers.create.events.failed"),
            pytest.raises(kopf.PermanentError, match="manifest invariant failed"),
        ):
            await _call_on_create(patch=patch)

        assert patch.status["phase"] == "Failed"
        assert patch.status["error"] == "manifest invariant failed"
        assert "completionTime" in patch.status
        assert "observedGeneration" not in patch.status

    @pytest.mark.asyncio
    async def test_on_create_sqlite_error_from_index_upsert_raises_temporary_error(
        self,
    ) -> None:
        """A transient sqlite failure (e.g. locked index DB while bootstrap
        runs) must surface as ``kopf.TemporaryError`` so the valid job is
        retried — not fall through to ``on_create``'s catch-all, which
        converts it to ``kopf.PermanentError`` and permanently fails the job.
        """
        patch = _patch_obj()

        async with _patched_create_dependencies() as harness:
            harness.upsert_index.side_effect = sqlite3.OperationalError(
                "database is locked"
            )
            with pytest.raises(
                kopf.TemporaryError, match=r"Persisting job spec/index failed"
            ):
                await _call_on_create(patch=patch)

        assert "observedGeneration" not in patch.status
        harness.save_spec.assert_awaited_once()
        harness.jobset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_create_idempotent_retry_reinvokes_all_resource_helpers(
        self,
    ) -> None:
        async with _patched_create_dependencies() as first:
            first_result, first_patch = await _call_on_create()
        async with _patched_create_dependencies() as second:
            second_result, second_patch = await _call_on_create()

        assert (
            first_result
            == second_result
            == {
                "jobSetName": "aiperf-aiperf-bench-7f2a",
                "workers": 1,
            }
        )
        assert first_patch.status["phase"] == second_patch.status["phase"] == "Pending"
        assert first.role.await_count == second.role.await_count == 1
        assert first.role_binding.await_count == second.role_binding.await_count == 1
        assert first.configmap.await_count == second.configmap.await_count == 1
        assert first.jobset.await_count == second.jobset.await_count == 1


# =============================================================================
# Cancellation at awaited boundaries
# =============================================================================


class TestCreateCancellationBoundaries:
    """Cooperative cancellation short-circuits create at relevant boundaries."""

    @pytest.mark.asyncio
    async def test_on_create_initial_cancel_terminalizes_without_external_work(
        self,
    ) -> None:
        """An initially-cancelled CR skips creation and publishes a terminal status.

        The status patch is deliberately NOT fenced with
        ``metadata.resourceVersion`` -- see ``lifecycle.on_cancel``: the fence
        409s kopf's merge PATCH and silently drops the cancel status. The
        UID-fenced identity re-reads still run, which is what the await list
        below asserts.
        """
        key = job_key("production", "aiperf-bench-7f2a", "uid-aiperf-bench-7f2a")
        parent_fence = AsyncMock(return_value="19")
        patch = kopf.Patch()

        async with _patched_create_dependencies() as harness:
            with (
                mock_patch(
                    "aiperf.operator.handlers.lifecycle.current_aiperfjob_resource_version",
                    new=parent_fence,
                ),
                mock_patch(
                    "aiperf.operator.handlers.lifecycle.close_progress_client",
                    new_callable=AsyncMock,
                ) as close_progress_client,
                mock_patch("aiperf.operator.handlers.lifecycle.events.cancelled"),
            ):
                result, patch = await _call_on_create(
                    spec=_spec_with(cancel=True), patch=patch
                )

        assert result == {}
        assert is_cancellation_requested(key) is True
        assert patch.status["phase"] == "Cancelled"
        assert patch.status["currentPhase"] is None
        assert patch.status["subPhase"] is None
        assert "completionTime" in patch.status
        assert "resourceVersion" not in (patch.get("metadata") or {})
        assert parent_fence.await_args_list == [
            call("production", "aiperf-bench-7f2a", "uid-aiperf-bench-7f2a"),
            call("production", "aiperf-bench-7f2a", "uid-aiperf-bench-7f2a"),
        ]
        close_progress_client.assert_awaited_once_with(key)
        harness.health.assert_not_awaited()
        harness.preflight_run_all.assert_not_awaited()
        harness.role.assert_not_awaited()
        harness.role_binding.assert_not_awaited()
        harness.configmap.assert_not_awaited()
        harness.save_spec.assert_not_awaited()
        harness.upsert_index.assert_not_awaited()
        harness.jobset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_create_cancellation_after_endpoint_probe_skips_k8s_resources(
        self,
    ) -> None:
        # Cancellation is keyed by immutable CR identity, so the UID is required:
        # a same-name replacement must not inherit a dead job's cancel flag.
        key = job_key("production", "aiperf-bench-7f2a", "uid-aiperf-bench-7f2a")

        def cancel_after_probe(*_args: object, **_kwargs: object) -> MagicMock:
            request_cancellation(key)
            return MagicMock(reachable=True, error=None)

        async with _patched_create_dependencies(
            health_side_effect=cancel_after_probe,
        ) as harness:
            result, patch = await _call_on_create()

        assert result == {}
        assert is_cancellation_requested(key) is True
        assert patch.status == {}
        harness.preflight_run_all.assert_not_awaited()
        harness.role.assert_not_awaited()
        harness.configmap.assert_not_awaited()
        harness.save_spec.assert_not_awaited()
        harness.jobset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_create_cancellation_after_configmap_skips_persistence_and_jobset(
        self,
    ) -> None:
        # Cancellation is keyed by immutable CR identity, so the UID is required:
        # a same-name replacement must not inherit a dead job's cancel flag.
        key = job_key("production", "aiperf-bench-7f2a", "uid-aiperf-bench-7f2a")

        def cancel_after_configmap(*_args: object, **_kwargs: object) -> None:
            request_cancellation(key)

        async with _patched_create_dependencies(
            configmap_side_effect=cancel_after_configmap,
        ) as harness:
            result, patch = await _call_on_create()

        assert result == {}
        assert is_cancellation_requested(key) is True
        assert patch.status == {}
        harness.role.assert_awaited_once()
        harness.role_binding.assert_awaited_once()
        harness.configmap.assert_awaited_once()
        harness.save_spec.assert_not_awaited()
        harness.upsert_index.assert_not_awaited()
        harness.jobset.assert_not_awaited()
