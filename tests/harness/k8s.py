# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes test helpers — mock API, sample data builders, error factories.

These are pure functions with no pytest coupling. Conftest files wrap them
as fixtures; test files can also import them directly.
"""

from __future__ import annotations

import copy
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from aiperf.config.benchmark import BenchmarkRun

from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.config import AIPerfConfig
from aiperf.config.deployment import PodTemplateConfig

# =============================================================================
# Mock ApiClient + accessor patchers
# =============================================================================


def build_mock_api() -> MagicMock:
    """Return a MagicMock(spec=ApiClient) with AsyncMock close()."""
    api = MagicMock(spec=ApiClient)
    api.close = AsyncMock()
    return api


@contextmanager
def patch_api_accessors(
    *,
    core: MagicMock | None = None,
    custom: MagicMock | None = None,
    apps: MagicMock | None = None,
    rbac: MagicMock | None = None,
    version: MagicMock | None = None,
    module_targets: list[str] | None = None,
) -> Iterator[dict[str, MagicMock]]:
    """Patch kubernetes_asyncio typed-API constructors in listed modules.

    ``module_targets`` is the list of fully-qualified module paths to patch.
    Each target is patched so that ``client.CoreV1Api(api)`` / etc. return the
    supplied mock. If a mock is not provided, a ``MagicMock`` is created.
    """
    targets = module_targets or ["aiperf.kubernetes.client"]
    accessors: dict[str, MagicMock] = {
        "core": core or MagicMock(),
        "custom": custom or MagicMock(),
        "apps": apps or MagicMock(),
        "rbac": rbac or MagicMock(),
        "version": version or MagicMock(),
    }

    stack = ExitStack()
    try:
        for module in targets:
            stack.enter_context(
                patch(f"{module}.client.CoreV1Api", return_value=accessors["core"])
            )
            stack.enter_context(
                patch(
                    f"{module}.client.CustomObjectsApi",
                    return_value=accessors["custom"],
                )
            )
            stack.enter_context(
                patch(f"{module}.client.AppsV1Api", return_value=accessors["apps"])
            )
            stack.enter_context(
                patch(
                    f"{module}.client.RbacAuthorizationV1Api",
                    return_value=accessors["rbac"],
                )
            )
            stack.enter_context(
                patch(f"{module}.client.VersionApi", return_value=accessors["version"])
            )
        yield accessors
    finally:
        stack.close()


# =============================================================================
# Error Factories
# =============================================================================


def create_api_exception(
    status_code: int, reason: str = "Error", body: str | None = None
) -> ApiException:
    """Create an ApiException for testing kubernetes_asyncio error paths."""
    return ApiException(status=status_code, reason=reason, http_resp=None)


# =============================================================================
# Response Builders
# =============================================================================


def create_jobset_list_response(jobsets: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a JobSet list API response (CustomObjectsApi shape)."""
    return {"items": jobsets}


# =============================================================================
# Sample Data Builders
# =============================================================================


def build_sample_jobset() -> dict[str, Any]:
    """Create a sample JobSet dict for testing."""
    return {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            "name": "aiperf-test-job",
            "namespace": "default",
            "creationTimestamp": "2026-01-15T10:30:00Z",
            "labels": {
                "app": "aiperf",
                "aiperf.nvidia.com/job-id": "test-job-123",
            },
        },
        "status": {
            "conditions": [],
            "ready": 0,
        },
    }


def build_running_jobset(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a sample running JobSet."""
    jobset = copy.deepcopy(base or build_sample_jobset())
    jobset["status"]["ready"] = 1
    return jobset


def build_completed_jobset(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a sample completed JobSet."""
    jobset = copy.deepcopy(base or build_sample_jobset())
    jobset["status"]["conditions"] = [
        {"type": "Completed", "status": "True", "reason": "JobsCompleted"}
    ]
    return jobset


def build_failed_jobset(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a sample failed JobSet."""
    jobset = copy.deepcopy(base or build_sample_jobset())
    jobset["status"]["conditions"] = [
        {"type": "Failed", "status": "True", "reason": "BackoffLimitExceeded"}
    ]
    jobset["status"]["replicatedJobsStatus"] = [
        {"name": "controller", "failed": 1, "ready": 0, "succeeded": 0}
    ]
    return jobset


def build_sample_pod() -> dict[str, Any]:
    """Create a sample Pod dict (raw K8s API format) for testing."""
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "aiperf-test-job-controller-0-0",
            "namespace": "default",
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {
                    "name": "system-controller",
                    "ready": True,
                    "restartCount": 0,
                }
            ],
        },
        "spec": {
            "containers": [{"name": "system-controller"}],
        },
    }


def build_pending_pod(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a sample pending Pod."""
    pod = copy.deepcopy(base or build_sample_pod())
    pod["status"]["phase"] = "Pending"
    pod["status"]["containerStatuses"][0]["ready"] = False
    return pod


def build_succeeded_pod(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a sample succeeded Pod."""
    pod = copy.deepcopy(base or build_sample_pod())
    pod["status"]["phase"] = "Succeeded"
    return pod


# =============================================================================
# Config Builders
# =============================================================================


def build_sample_config() -> AIPerfConfig:
    """Create a minimal AIPerfConfig for testing."""
    return AIPerfConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets={
            "main": {
                "type": "synthetic",
                "entries": 10,
                "prompts": {"isl": 32, "osl": 16},
            }
        },
        phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
    )


def build_sample_run(config: AIPerfConfig | None = None) -> BenchmarkRun:
    """Create a minimal BenchmarkRun for testing."""
    from pathlib import Path

    from aiperf.config.benchmark import BenchmarkRun

    if config is None:
        config = build_sample_config()
    return BenchmarkRun(
        benchmark_id="test-run-001",
        cfg=config,
        artifact_dir=Path("/tmp/test-artifacts"),
    )


def build_sample_pod_template() -> PodTemplateConfig:
    """Create a sample PodTemplateConfig for testing."""
    return PodTemplateConfig(
        node_selector={"gpu": "true"},
        tolerations=[
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
        ],
        annotations={"custom/annotation": "value"},
        labels={"custom-label": "value"},
        image_pull_secrets=["my-registry-secret"],
        env=[
            {"name": "CUSTOM_VAR", "value": "custom_value"},
            {
                "name": "API_KEY",
                "valueFrom": {
                    "secretKeyRef": {"name": "my-secret", "key": "api-key"},
                },
            },
        ],
        volumes=[
            {"name": "secret-my-secret", "secret": {"secretName": "my-secret"}},
        ],
        volume_mounts=[
            {
                "name": "secret-my-secret",
                "mountPath": "/etc/secrets",
                "readOnly": True,
            },
        ],
        service_account_name="my-service-account",
    )
