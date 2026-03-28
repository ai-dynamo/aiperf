# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes test helpers — mock API, sample data builders, error factories.

These are pure functions with no pytest coupling. Conftest files wrap them
as fixtures; test files can also import them directly.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

if TYPE_CHECKING:
    from aiperf.config.benchmark import BenchmarkRun

import kr8s

from aiperf.config import AIPerfConfig
from aiperf.config.deployment import PodTemplateConfig
from aiperf.kubernetes.client import AIPerfKubeClient

# =============================================================================
# Mock kr8s API
# =============================================================================


def build_mock_kr8s_api() -> MagicMock:
    """Create a mock kr8s async API client."""
    api = MagicMock(spec=kr8s.Api)
    api.async_get = AsyncMock(return_value=[])
    return api


def build_mock_kube_client(api: MagicMock | None = None) -> AIPerfKubeClient:
    """Create a mock AIPerfKubeClient wrapping a mock kr8s API."""
    if api is None:
        api = build_mock_kr8s_api()
    return AIPerfKubeClient(api)


# =============================================================================
# Error Factories
# =============================================================================


def create_server_error(status_code: int, reason: str = "Error") -> kr8s.ServerError:
    """Create a kr8s ServerError for testing error handling."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.reason_phrase = reason
    mock_response.text = f'{{"message": "{reason}"}}'
    return kr8s.ServerError(reason, response=mock_response)


def create_not_found_error(name: str = "resource") -> kr8s.NotFoundError:
    """Create a kr8s NotFoundError for testing."""
    return kr8s.NotFoundError(f"{name} not found")


# =============================================================================
# Mock kr8s Object
# =============================================================================


def make_kr8s_object(raw: dict[str, Any]) -> MagicMock:
    """Create a mock kr8s object with a .raw attribute from a dict."""
    obj = MagicMock()
    obj.raw = raw
    obj.name = raw.get("metadata", {}).get("name", "")
    obj.namespace = raw.get("metadata", {}).get("namespace", "")
    return obj


async def async_list(items: list) -> Any:
    """Create an async generator from a list (for mocking api.async_get)."""
    for item in items:
        yield item


# =============================================================================
# Response Builders
# =============================================================================


def create_jobset_list_response(jobsets: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a JobSet list API response."""
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
    """Create a sample Pod dict (kr8s .raw format) for testing."""
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
