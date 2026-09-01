# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live acceptance coverage for named-phase operator status tracking."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio
import yaml

from tests.kubernetes.conftest import _gpu_node_tolerations
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

# Session-unique namespace keeps concurrent sessions from purging each other's jobs.
_SESSION_ID = uuid.uuid4().hex[:8]


@pytest_asyncio.fixture
async def phases_namespace(kubectl: KubectlClient) -> AsyncGenerator[str, None]:
    ns = f"aiperf-phases-{_SESSION_ID}"
    await kubectl.run("create", "namespace", ns, check=False)
    yield ns
    await kubectl.run(
        "delete", "namespace", ns, "--ignore-not-found", "--wait=false", check=False
    )


def _named_phase_manifest(
    *,
    name: str,
    namespace: str,
    image: str,
    image_pull_policy: str = "IfNotPresent",
    tolerations: list[dict] | None = None,
) -> dict[str, Any]:
    config = AIPerfJobConfig(
        request_count=5,
        warmup_request_count=0,
        concurrency=1,
        image=image,
        image_pull_policy=image_pull_policy,
        tolerations=tolerations or [],
    )
    manifest = yaml.safe_load(config.to_cr_manifest(name, namespace))
    manifest["spec"]["benchmark"]["phases"] = [
        {
            "name": "cache_prime",
            "kind": "warmup",
            "type": "concurrency",
            "concurrency": 1,
            "requests": 2,
        },
        {
            "name": "baseline",
            "kind": "profiling",
            "type": "concurrency",
            "concurrency": 1,
            "requests": 5,
        },
        {
            "name": "cooldown",
            "kind": "warmup",
            "type": "concurrency",
            "concurrency": 1,
            "requests": 2,
        },
    ]
    return manifest


@pytest.mark.timeout(1200)
@pytest.mark.asyncio
async def test_named_phases_are_complete_and_profiling_results_are_filtered(
    operator_ready: OperatorDeployer,
    k8s_settings: Any,
    phases_namespace: str,
) -> None:
    """Track all named phases while excluding warmup-kind records from results."""
    name = f"named-{uuid.uuid4().hex[:8]}"
    manifest = _named_phase_manifest(
        name=name,
        namespace=phases_namespace,
        image=k8s_settings.aiperf_image,
        image_pull_policy=k8s_settings.image_pull_policy,
        tolerations=_gpu_node_tolerations()
        if k8s_settings.tolerate_gpu_nodes
        else None,
    )

    await operator_ready.kubectl.apply(yaml.safe_dump(manifest, sort_keys=False))
    try:
        status = await operator_ready.wait_for_job_completion(
            name,
            phases_namespace,
            timeout=k8s_settings.benchmark_timeout,
        )

        assert status.is_completed, status.raw_status
        assert status.is_condition_true("Complete"), status.conditions
        assert status.is_condition_true("ResultsAvailable"), status.conditions
        assert status.raw_status.get("currentPhase") is None
        assert status.raw_status.get("subPhase") is None
        assert status.raw_status.get("observedGeneration") == manifest["metadata"].get(
            "generation", 1
        )

        expected_requests = {"cache_prime": 2, "baseline": 5, "cooldown": 2}
        for phase_name, request_count in expected_requests.items():
            phase = status.phases.get(phase_name)
            assert phase is not None, status.phases
            assert phase.get("requestsTotal") == request_count, phase
            assert phase.get("requestsCompleted") == request_count, phase
            assert phase.get("sendingComplete") is True, phase
            assert phase.get("isRequestsComplete") is True, phase
            assert phase.get("isRecordsComplete") is True, phase

        assert status.results is not None
        metrics = status.results.get("metrics", status.results)
        request_count = metrics.get("request_count", {})
        request_count_avg = (
            request_count.get("avg")
            if isinstance(request_count, dict)
            else request_count
        )
        assert request_count_avg == 5, status.results
    finally:
        await operator_ready.delete_job(name, phases_namespace)
