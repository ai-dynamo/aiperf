# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for Kueue/gang-scheduling support in AIPerf.

These tests verify end-to-end flows from CLI options through manifest generation
to operator handling. They use mocks for actual k8s API calls but test real code
paths through KubernetesDeployment, JobSetSpec, and operator handlers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import pytest
from pytest import param

from aiperf.config import AIPerfConfig
from aiperf.config.deployment import DeploymentConfig, SchedulingConfig
from aiperf.config.kube import KubeOptions
from aiperf.kubernetes.constants import KueueLabels, Labels
from aiperf.kubernetes.resources import KubernetesDeployment
from aiperf.operator.spec_converter import AIPerfJobSpecConverter
from aiperf.operator.status import Phase


@asynccontextmanager
async def _fake_k8s_client(api):
    """Yield the given mock ApiClient inside an async context."""
    yield api


def _mock_custom_api(return_value=None, side_effect=None):
    """Build a CustomObjectsApi mock that returns ``return_value`` from
    ``get_namespaced_custom_object`` or raises ``side_effect``."""
    mock_custom = MagicMock()
    mock_custom.get_namespaced_custom_object = AsyncMock(
        return_value=return_value, side_effect=side_effect
    )
    mock_custom.delete_namespaced_custom_object = AsyncMock(return_value={})
    return mock_custom


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def kube_config() -> AIPerfConfig:
    """Create a minimal AIPerfConfig for manifest generation."""
    return AIPerfConfig(
        models=["test-model"],
        endpoint={"urls": ["http://llm-server:8000"]},
        datasets={
            "default": {
                "type": "synthetic",
                "entries": 10,
                "prompts": {"isl": 32, "osl": 16},
            }
        },
        phases={"default": {"type": "concurrency", "requests": 100, "concurrency": 10}},
    )


@pytest.fixture
def kube_options_with_queue() -> KubeOptions:
    """Create KubeOptions with Kueue queue_name set."""
    return KubeOptions(
        image="aiperf:test",
        queue_name="test-queue",
        priority_class="high-priority",
    )


@pytest.fixture
def kube_options_without_queue() -> KubeOptions:
    """Create KubeOptions without Kueue scheduling."""
    return KubeOptions(image="aiperf:test")


@pytest.fixture
def mock_all_operator_events():
    """Mock all operator event functions to avoid kopf context issues."""
    with (
        mock_patch("aiperf.operator.events.spec_valid"),
        mock_patch("aiperf.operator.events.spec_invalid"),
        mock_patch("aiperf.operator.events.endpoint_reachable"),
        mock_patch("aiperf.operator.events.endpoint_unreachable"),
        mock_patch("aiperf.operator.events.resources_created"),
        mock_patch("aiperf.operator.events.created"),
        mock_patch("aiperf.operator.events.failed"),
        mock_patch("aiperf.operator.events.started"),
        mock_patch("aiperf.operator.events.workers_ready"),
    ):
        yield


@pytest.fixture
def full_spec_with_scheduling() -> dict[str, Any]:
    """Create a complete AIPerfJob spec with Kueue scheduling config."""
    return {
        "image": "aiperf:test",
        "scheduling": {
            "queueName": "team-gpu-queue",
            "priorityClass": "high-priority",
        },
        "benchmark": {
            "models": ["gpt-4"],
            "endpoint": {
                "urls": ["http://api.example.com"],
            },
            "datasets": {
                "default": {
                    "type": "synthetic",
                    "entries": 10,
                    "prompts": {"isl": 32, "osl": 16},
                }
            },
            "phases": {
                "default": {"type": "concurrency", "requests": 100, "concurrency": 10}
            },
        },
    }


# =============================================================================
# TestKueueManifestGeneration
# =============================================================================


class TestKueueManifestGeneration:
    """Test that Kueue options flow correctly from CLI through to generated manifests."""

    def test_runner_generates_kueue_labels_when_queue_name_set(
        self, kube_config: AIPerfConfig
    ) -> None:
        """Verify JobSet manifest includes kueue label and suspend=true when queue_name set."""
        deployment = KubernetesDeployment(
            job_id="abc12345",
            config=kube_config,
            deployment=DeploymentConfig(
                image="aiperf:test",
                scheduling=SchedulingConfig(queue_name="test-queue"),
            ),
        )

        jobset_manifest = deployment.get_jobset_spec().to_k8s_manifest()

        labels = jobset_manifest["metadata"]["labels"]
        assert KueueLabels.QUEUE_NAME in labels
        assert labels[KueueLabels.QUEUE_NAME] == "test-queue"
        assert jobset_manifest["spec"]["suspend"] is True

    def test_runner_omits_kueue_labels_when_queue_name_not_set(
        self, kube_config: AIPerfConfig
    ) -> None:
        """Verify no kueue labels and no suspend when queue_name is None."""
        deployment = KubernetesDeployment(
            job_id="abc12345",
            config=kube_config,
            deployment=DeploymentConfig(image="aiperf:test"),
        )

        jobset_manifest = deployment.get_jobset_spec().to_k8s_manifest()

        labels = jobset_manifest["metadata"]["labels"]
        assert KueueLabels.QUEUE_NAME not in labels
        assert KueueLabels.PRIORITY_CLASS not in labels
        assert "suspend" not in jobset_manifest["spec"]

    def test_full_deployment_manifests_include_kueue_on_jobset_only(
        self, kube_config: AIPerfConfig
    ) -> None:
        """Verify kueue labels appear only on JobSet, not on ConfigMap/Namespace/RBAC."""
        deployment = KubernetesDeployment(
            job_id="abc12345",
            config=kube_config,
            deployment=DeploymentConfig(
                image="aiperf:test",
                scheduling=SchedulingConfig(
                    queue_name="test-queue", priority_class="high-priority"
                ),
            ),
        )

        manifests = deployment.get_all_manifests()

        for manifest in manifests:
            kind = manifest["kind"]
            labels = manifest["metadata"].get("labels", {})

            if kind == "JobSet":
                assert KueueLabels.QUEUE_NAME in labels
                assert labels[KueueLabels.QUEUE_NAME] == "test-queue"
                assert KueueLabels.PRIORITY_CLASS in labels
                assert labels[KueueLabels.PRIORITY_CLASS] == "high-priority"
                assert manifest["spec"]["suspend"] is True
            else:
                assert KueueLabels.QUEUE_NAME not in labels
                assert KueueLabels.PRIORITY_CLASS not in labels

    def test_priority_class_flows_through_to_manifest(
        self, kube_config: AIPerfConfig
    ) -> None:
        """Verify priority_class label appears on JobSet manifest."""
        deployment = KubernetesDeployment(
            job_id="abc12345",
            config=kube_config,
            deployment=DeploymentConfig(
                image="aiperf:test",
                scheduling=SchedulingConfig(
                    queue_name="test-queue", priority_class="batch-low"
                ),
            ),
        )

        jobset_manifest = deployment.get_jobset_spec().to_k8s_manifest()

        labels = jobset_manifest["metadata"]["labels"]
        assert KueueLabels.PRIORITY_CLASS in labels
        assert labels[KueueLabels.PRIORITY_CLASS] == "batch-low"

    @pytest.mark.parametrize(
        "queue_name,priority_class,expect_suspend",
        [
            param("my-queue", None, True, id="queue_only"),
            param("my-queue", "high", True, id="queue_and_priority"),
            param(None, "high", False, id="priority_only_no_suspend"),
            param(None, None, False, id="neither"),
        ],
    )  # fmt: skip
    def test_suspend_only_when_queue_name_set(
        self,
        kube_config: AIPerfConfig,
        queue_name: str | None,
        priority_class: str | None,
        expect_suspend: bool,
    ) -> None:
        """Verify suspend=true only when queue_name is set (Kueue requires it)."""
        deployment = KubernetesDeployment(
            job_id="abc12345",
            config=kube_config,
            deployment=DeploymentConfig(
                image="aiperf:test",
                scheduling=SchedulingConfig(
                    queue_name=queue_name, priority_class=priority_class
                ),
            ),
        )

        jobset_manifest = deployment.get_jobset_spec().to_k8s_manifest()

        if expect_suspend:
            assert jobset_manifest["spec"]["suspend"] is True
        else:
            assert "suspend" not in jobset_manifest["spec"]


# =============================================================================
# TestKueueOperatorFlow
# =============================================================================


class TestKueueOperatorFlow:
    """Test operator behavior with Kueue-managed jobs."""

    @pytest.mark.asyncio
    async def test_monitor_sets_queued_phase_when_jobset_suspended_with_kueue_label(
        self,
    ) -> None:
        """Verify QUEUED phase when JobSet has kueue label and suspend=true."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        jobset_raw = {
            "metadata": {
                "labels": {
                    KueueLabels.QUEUE_NAME: "test-queue",
                    Labels.APP_KEY: Labels.APP_VALUE,
                },
            },
            "spec": {"suspend": True},
            "status": {
                "conditions": [],
                "replicatedJobsStatus": [],
            },
        }

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=_mock_custom_api(return_value=jobset_raw),
            ),
        ):
            await monitor_progress(
                body={},
                status={
                    "phase": Phase.PENDING,
                    "jobSetName": "aiperf-test123",
                    "jobId": "test123",
                },
                spec={},
                name="test123",
                namespace="default",
                patch=kopf_patch,
            )

        assert kopf_patch.status["phase"] == Phase.QUEUED

    @pytest.mark.asyncio
    async def test_monitor_transitions_from_queued_to_initializing_when_unsuspended(
        self,
    ) -> None:
        """Verify transition from QUEUED to INITIALIZING when Kueue unsuspends the job."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        jobset_raw = {
            "metadata": {
                "labels": {
                    KueueLabels.QUEUE_NAME: "test-queue",
                    Labels.APP_KEY: Labels.APP_VALUE,
                },
            },
            "spec": {"suspend": False},
            "status": {
                "conditions": [],
                "replicatedJobsStatus": [
                    {
                        "name": "workers",
                        "ready": 1,
                        "active": 0,
                        "succeeded": 0,
                        "failed": 0,
                        "suspended": 0,
                    },
                ],
            },
        }

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=_mock_custom_api(return_value=jobset_raw),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._check_pod_restarts",
                new_callable=AsyncMock,
            ),
        ):
            await monitor_progress(
                body={},
                status={
                    "phase": Phase.QUEUED,
                    "jobSetName": "aiperf-test123",
                    "jobId": "test123",
                    "workers": {"total": 2, "ready": 0},
                },
                spec={},
                name="test123",
                namespace="default",
                patch=kopf_patch,
            )

        assert kopf_patch.status["phase"] == Phase.INITIALIZING

    @pytest.mark.asyncio
    async def test_monitor_ignores_suspension_without_kueue_label(self) -> None:
        """Verify suspended JobSet WITHOUT kueue label does NOT set QUEUED phase."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        jobset_raw = {
            "metadata": {
                "labels": {
                    Labels.APP_KEY: Labels.APP_VALUE,
                },
            },
            "spec": {"suspend": True},
            "status": {
                "conditions": [],
                "replicatedJobsStatus": [],
            },
        }

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=_mock_custom_api(return_value=jobset_raw),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._check_pod_restarts",
                new_callable=AsyncMock,
            ),
        ):
            await monitor_progress(
                body={},
                status={
                    "phase": Phase.PENDING,
                    "jobSetName": "aiperf-test123",
                    "jobId": "test123",
                    "workers": {"total": 2, "ready": 0},
                },
                spec={},
                name="test123",
                namespace="default",
                patch=kopf_patch,
            )

        assert kopf_patch.status.get("phase") != Phase.QUEUED

    @pytest.mark.asyncio
    async def test_on_create_passes_scheduling_from_crd_spec(
        self,
        mock_all_operator_events: None,
        full_spec_with_scheduling: dict[str, Any],
    ) -> None:
        """Verify on_create extracts scheduling config and passes it to KubernetesDeployment."""
        from aiperf.operator.handlers.create import on_create

        body = {"metadata": {"name": "test-job", "namespace": "default"}}
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        captured_jobset_manifest: dict[str, Any] = {}

        async def capture_custom(*, api, group, version, plural, body, namespace):
            """Pass-through that captures the JobSet manifest submitted."""
            if plural == "jobsets":
                captured_jobset_manifest.update(body)

        # Mock preflight to always pass (avoids complex API mocking)
        mock_preflight_result = MagicMock()
        mock_preflight_result.passed = True
        mock_preflight_result.checks = []
        mock_preflight = MagicMock()
        mock_preflight.run_all = AsyncMock(return_value=mock_preflight_result)

        with (
            mock_patch(
                "aiperf.operator.handlers.create.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.check_endpoint_health",
                new_callable=AsyncMock,
                return_value=MagicMock(reachable=True, error=""),
            ),
            mock_patch(
                "aiperf.operator.preflight.OperatorPreflightChecker",
                return_value=mock_preflight,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role_binding",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_config_map",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.save_job_spec_file",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.index_job_created",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_custom_object",
                side_effect=capture_custom,
            ),
        ):
            result = await on_create(
                body=body,
                spec=full_spec_with_scheduling,
                name="test-job",
                namespace="default",
                uid="test-uid",
                patch=kopf_patch,
            )

        assert result is not None

        # Verify JobSet manifest was generated with scheduling labels
        assert captured_jobset_manifest, "JobSet manifest was not captured"
        labels = captured_jobset_manifest["metadata"]["labels"]
        assert labels[KueueLabels.QUEUE_NAME] == "team-gpu-queue"
        assert labels[KueueLabels.PRIORITY_CLASS] == "high-priority"
        assert captured_jobset_manifest["spec"]["suspend"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "current_phase",
        [
            param(Phase.PENDING, id="from_pending"),
            param(Phase.QUEUED, id="from_queued"),
        ],
    )  # fmt: skip
    async def test_monitor_queued_only_from_pending_or_queued(
        self,
        current_phase: str,
    ) -> None:
        """Verify QUEUED phase is only set from PENDING or QUEUED, not other phases."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        jobset_raw = {
            "metadata": {
                "labels": {KueueLabels.QUEUE_NAME: "test-queue"},
            },
            "spec": {"suspend": True},
            "status": {
                "conditions": [],
                "replicatedJobsStatus": [],
            },
        }

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=_mock_custom_api(return_value=jobset_raw),
            ),
        ):
            await monitor_progress(
                body={},
                status={
                    "phase": current_phase,
                    "jobSetName": "aiperf-test123",
                    "jobId": "test123",
                },
                spec={},
                name="test123",
                namespace="default",
                patch=kopf_patch,
            )

        assert kopf_patch.status["phase"] == Phase.QUEUED

    @pytest.mark.asyncio
    async def test_monitor_does_not_set_queued_from_running_phase(self) -> None:
        """Verify a RUNNING job is NOT set back to QUEUED even if suspend=true."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        jobset_raw = {
            "metadata": {
                "labels": {KueueLabels.QUEUE_NAME: "test-queue"},
            },
            "spec": {"suspend": True},
            "status": {
                "conditions": [],
                "replicatedJobsStatus": [],
            },
        }

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=_mock_custom_api(return_value=jobset_raw),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._check_pod_restarts",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._fetch_progress",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await monitor_progress(
                body={},
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "aiperf-test123",
                    "jobId": "test123",
                    "workers": {"total": 2, "ready": 2},
                },
                spec={},
                name="test123",
                namespace="default",
                patch=kopf_patch,
            )

        assert kopf_patch.status.get("phase") != Phase.QUEUED

    def test_spec_converter_extracts_scheduling_config(
        self,
        full_spec_with_scheduling: dict[str, Any],
    ) -> None:
        """Verify AIPerfJobSpecConverter correctly extracts scheduling fields."""
        converter = AIPerfJobSpecConverter(
            full_spec_with_scheduling, "test-job", "default"
        )
        scheduling = converter.to_deployment_config().scheduling

        assert scheduling.queue_name == "team-gpu-queue"
        assert scheduling.priority_class == "high-priority"

    def test_spec_converter_returns_none_without_scheduling(self) -> None:
        """Verify scheduling fields are None when not in spec."""
        spec: dict[str, Any] = {
            "benchmark": {
                "models": ["test"],
                "endpoint": {"urls": ["http://localhost:8000"]},
                "datasets": {
                    "default": {
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 32, "osl": 16},
                    }
                },
                "phases": {
                    "default": {"type": "concurrency", "requests": 10, "concurrency": 1}
                },
            },
        }
        converter = AIPerfJobSpecConverter(spec, "test-job", "default")
        scheduling = converter.to_deployment_config().scheduling

        assert scheduling.queue_name is None
        assert scheduling.priority_class is None
