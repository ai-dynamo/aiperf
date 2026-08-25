# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for kube cancel/delete trust boundaries.

Focuses on:
- explicit target resolution versus the persisted last benchmark fallback.
- namespace, kubeconfig, and kube-context propagation before mutating Kubernetes.
- text/JSON cleanliness knobs that suppress fallback chatter for machine output.
- idempotent 404 delete behavior versus non-404 conflict propagation.
- confirmation-prompt abort semantics and last-benchmark clearing guards.
- AIPerfJob versus AIPerfSweep target distinction before a destructive action.

Out of scope: live apiserver behavior and operator reconciliation after a cancel;
Kubernetes client integration tests cover those paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.cli_helpers import (
    ResolvedJob,
    ResolvedSweep,
    confirm_action,
    resolve_job_id_and_namespace,
    resolve_target,
)
from aiperf.kubernetes.client_jobs import cancel_aiperf_job
from aiperf.kubernetes.client_jobsets import delete_jobset
from aiperf.kubernetes.console import LastBenchmarkInfo
from aiperf.kubernetes.models import AIPerfJobInfo, AIPerfSweepInfo, JobSetInfo

# =============================================================================
# Helpers
# =============================================================================


@dataclass(slots=True)
class _DeleteApis:
    """Fake Kubernetes API surfaces used by delete-jobset workflow tests."""

    custom: MagicMock
    core: MagicMock
    rbac: MagicMock


def _api_exception(status: int) -> ApiException:
    """Build a Kubernetes ApiException with stable status and reason."""
    return ApiException(status=status, reason=f"aiperf-api-{status}")


def _job_info(
    *,
    name: str = "llama3-throughput-v07",
    namespace: str = "bench-prod",
    job_id: str = "job-2026-05-18-9c3a",
) -> AIPerfJobInfo:
    """Build a realistic AIPerfJobInfo resolved from an AIPerfJob CR."""
    return AIPerfJobInfo(
        name=name,
        namespace=namespace,
        phase="Running",
        job_id=job_id,
        jobset_name=f"{name}-jobset",
        created="2026-05-18T10:30:00Z",
        model="meta-llama/Llama-3-8B",
        endpoint="http://localhost:8000",
    )


def _sweep_info(
    *,
    name: str = "latency-grid-search",
    namespace: str = "bench-prod",
) -> AIPerfSweepInfo:
    """Build a realistic AIPerfSweepInfo resolved from an AIPerfSweep CR."""
    return AIPerfSweepInfo(
        name=name,
        namespace=namespace,
        phase="Running",
        run_epoch=1770001234,
        total_variations=6,
        max_total_runs=18,
        completed_runs=2,
        failed_runs=0,
        created="2026-05-18T10:30:00Z",
    )


def _jobset_info(
    *,
    name: str = "legacy-direct-jobset",
    namespace: str = "bench-prod",
) -> JobSetInfo:
    """Build a direct-mode JobSet fallback for target resolution tests."""
    return JobSetInfo(
        name=name,
        namespace=namespace,
        jobset={
            "metadata": {
                "name": name,
                "namespace": namespace,
                "creationTimestamp": "2026-05-18T10:30:00Z",
                "labels": {"aiperf.nvidia.com/job-id": name},
                "annotations": {},
            },
            "status": {},
        },
        status="Running",
        model="meta-llama/Llama-3-8B",
        endpoint="http://localhost:8000",
    )


def _delete_apis(
    *,
    jobset_side_effect: Exception | None = None,
    aux_side_effect: Exception | None = None,
) -> _DeleteApis:
    """Create fake Kubernetes delete API clients with async methods."""
    custom = MagicMock()
    core = MagicMock()
    rbac = MagicMock()
    custom.delete_namespaced_custom_object = AsyncMock(side_effect=jobset_side_effect)
    core.delete_namespaced_config_map = AsyncMock(side_effect=aux_side_effect)
    rbac.delete_namespaced_role = AsyncMock(side_effect=aux_side_effect)
    rbac.delete_namespaced_role_binding = AsyncMock(side_effect=aux_side_effect)
    return _DeleteApis(custom=custom, core=core, rbac=rbac)


# =============================================================================
# Explicit versus last-benchmark resolution
# =============================================================================


class TestCancelDeleteTargetResolution:
    """Resolution decides which Kubernetes object a mutating workflow touches."""

    def test_resolve_job_id_and_namespace_explicit_target_ignores_stale_last_benchmark(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark",
            return_value=LastBenchmarkInfo(
                job_id="stale-benchmark-4f2a", namespace="stale-ns"
            ),
        ):
            resolved = resolve_job_id_and_namespace(
                "llama3-throughput-v07", "bench-prod"
            )

        assert resolved == ("llama3-throughput-v07", "bench-prod")
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(
        ("namespace", "expected_namespace"),
        [
            (None, "recorded-prod"),
            param("override-review", "override-review", id="explicit-namespace-wins"),
        ],
    )  # fmt: skip
    def test_resolve_job_id_and_namespace_last_benchmark_preserves_namespace_policy(
        self,
        namespace: str | None,
        expected_namespace: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark",
            return_value=LastBenchmarkInfo(
                job_id="last-latency-benchmark", namespace="recorded-prod"
            ),
        ):
            resolved = resolve_job_id_and_namespace(None, namespace)

        assert resolved == ("last-latency-benchmark", expected_namespace)
        assert "Using last benchmark: last-latency-benchmark in recorded-prod" in (
            capsys.readouterr().out
        )

    def test_resolve_job_id_and_namespace_quiet_last_benchmark_keeps_json_stdout_empty(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark",
            return_value=LastBenchmarkInfo(
                job_id="json-safe-last-benchmark", namespace="bench-prod"
            ),
        ):
            resolved = resolve_job_id_and_namespace(None, None, quiet=True)

        assert resolved == ("json-safe-last-benchmark", "bench-prod")
        assert capsys.readouterr().out == ""

    @pytest.mark.asyncio
    async def test_resolve_target_explicit_job_propagates_namespace_and_kube_context(
        self,
    ) -> None:
        api = MagicMock()
        job = _job_info(name="llama3-throughput-v07", namespace="bench-prod")
        open_api = AsyncMock(return_value=api)
        find_job = AsyncMock(return_value=job)
        find_sweep = AsyncMock()
        find_jobset = AsyncMock()

        with (
            patch("aiperf.kubernetes.cli_helpers._open_api_client", new=open_api),
            patch("aiperf.kubernetes.client.find_aiperf_job", new=find_job),
            patch("aiperf.kubernetes.client.find_aiperf_sweep", new=find_sweep),
            patch("aiperf.kubernetes.client.find_jobset", new=find_jobset),
        ):
            resolved = await resolve_target(
                "llama3-throughput-v07",
                namespace="bench-prod",
                kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
                kube_context="dgx-prod-admin",
            )

        assert isinstance(resolved, ResolvedJob)
        assert resolved.job_info is job
        open_api.assert_awaited_once_with(
            kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
            kube_context="dgx-prod-admin",
        )
        find_job.assert_awaited_once_with(api, "llama3-throughput-v07", "bench-prod")
        find_sweep.assert_not_awaited()
        find_jobset.assert_not_awaited()


# =============================================================================
# Job versus sweep destructive-target distinction
# =============================================================================


class TestCancelDeleteJobSweepDistinction:
    """A sweep name must not silently fall through to direct-mode JobSet cleanup."""

    @pytest.mark.asyncio
    async def test_resolve_target_sweep_found_before_jobset_fallback_returns_sweep(
        self,
    ) -> None:
        api = MagicMock()
        sweep = _sweep_info(name="latency-grid-search", namespace="bench-prod")
        with (
            patch(
                "aiperf.kubernetes.cli_helpers._open_api_client",
                new=AsyncMock(return_value=api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ) as mock_find_job,
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(return_value=sweep),
            ) as mock_find_sweep,
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=_jobset_info(name="latency-grid-search")),
            ) as mock_find_jobset,
        ):
            resolved = await resolve_target(
                "latency-grid-search", namespace="bench-prod"
            )

        assert isinstance(resolved, ResolvedSweep)
        assert resolved.sweep_info is sweep
        mock_find_job.assert_awaited_once_with(api, "latency-grid-search", "bench-prod")
        mock_find_sweep.assert_awaited_once_with(
            api, "latency-grid-search", "bench-prod"
        )
        mock_find_jobset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resolve_target_missing_name_closes_api_and_names_both_candidate_kinds(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api = MagicMock()
        api.close = AsyncMock()
        with (
            patch(
                "aiperf.kubernetes.cli_helpers._open_api_client",
                new=AsyncMock(return_value=api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=None),
            ),
        ):
            resolved = await resolve_target("missing-target", namespace="bench-prod")

        assert resolved is None
        api.close.assert_awaited_once()
        out = capsys.readouterr().out
        assert "AIPerfJob" in out
        assert "AIPerfSweep" in out
        assert "missing-target" in out
        assert "bench-prod" in out


# =============================================================================
# Kubernetes mutation error boundaries
# =============================================================================


class TestCancelDeleteMutationBoundaries:
    """Kubernetes API statuses retain idempotent and retryable meaning."""

    @pytest.mark.asyncio
    async def test_cancel_aiperf_job_uses_merge_patch_and_propagates_conflict(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        custom = MagicMock()
        custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(409)
        )

        with (
            patch(
                "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
                return_value=custom,
            ),
            pytest.raises(ApiException) as exc_info,
        ):
            await cancel_aiperf_job(api, "llama3-throughput-v07", "bench-prod")

        assert exc_info.value.status == 409
        custom.patch_namespaced_custom_object.assert_awaited_once_with(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfjobs",
            namespace="bench-prod",
            name="llama3-throughput-v07",
            body={"spec": {"cancel": True}},
            _content_type="application/merge-patch+json",
        )

    @pytest.mark.asyncio
    async def test_delete_jobset_404_is_idempotent_and_still_checks_aux_resources(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        apis = _delete_apis(jobset_side_effect=_api_exception(404))

        with (
            patch(
                "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
                return_value=apis.custom,
            ),
            patch(
                "aiperf.kubernetes.client_jobsets.client.CoreV1Api",
                return_value=apis.core,
            ),
            patch(
                "aiperf.kubernetes.client_jobsets.client.RbacAuthorizationV1Api",
                return_value=apis.rbac,
            ),
            patch("aiperf.kubernetes.client_jobsets.print_warning"),
            patch("aiperf.kubernetes.client_jobsets.print_success"),
        ):
            await delete_jobset(api, "llama3-throughput-v07-jobset", "bench-prod")

        apis.custom.delete_namespaced_custom_object.assert_awaited_once()
        apis.core.delete_namespaced_config_map.assert_awaited_once_with(
            name="llama3-throughput-v07-jobset-config", namespace="bench-prod"
        )
        apis.rbac.delete_namespaced_role.assert_awaited_once_with(
            name="llama3-throughput-v07-jobset-role", namespace="bench-prod"
        )
        apis.rbac.delete_namespaced_role_binding.assert_awaited_once_with(
            name="llama3-throughput-v07-jobset-binding", namespace="bench-prod"
        )

    @pytest.mark.asyncio
    async def test_delete_jobset_non_404_primary_error_propagates_before_aux_deletes(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        apis = _delete_apis(jobset_side_effect=_api_exception(403))

        with (
            patch(
                "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
                return_value=apis.custom,
            ),
            patch(
                "aiperf.kubernetes.client_jobsets.client.CoreV1Api",
                return_value=apis.core,
            ),
            patch(
                "aiperf.kubernetes.client_jobsets.client.RbacAuthorizationV1Api",
                return_value=apis.rbac,
            ),
            pytest.raises(ApiException) as exc_info,
        ):
            await delete_jobset(api, "llama3-throughput-v07-jobset", "bench-prod")

        assert exc_info.value.status == 403
        apis.core.delete_namespaced_config_map.assert_not_awaited()
        apis.rbac.delete_namespaced_role.assert_not_awaited()
        apis.rbac.delete_namespaced_role_binding.assert_not_awaited()


# =============================================================================
# Confirmation and last-benchmark clearing
# =============================================================================


class TestConfirmationAndLastBenchmarkGuards:
    """User intent and persisted defaults are guarded before cleanup side effects."""

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("y", True),
            param("Y", True, id="uppercase-y-accepted"),
            param("yes", False, id="yes-word-rejected-to-avoid-accidental-delete"),
            param("", False, id="empty-defaults-to-abort"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_confirm_action_accepts_only_single_letter_y(
        self,
        response: str,
        expected: bool,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with patch("builtins.input", return_value=response) as mock_input:
            result = await confirm_action("Delete bench-prod/llama3-throughput-v07?")

        assert result is expected
        mock_input.assert_called_once_with(
            "Delete bench-prod/llama3-throughput-v07? [y/N] "
        )
        out = capsys.readouterr().out
        if expected:
            assert "Aborted" not in out
        else:
            assert "Aborted" in out
