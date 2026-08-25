# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live acceptance coverage for named-phase operator status tracking."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
import yaml

from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer


def _named_phase_manifest(*, name: str, namespace: str, image: str) -> dict[str, Any]:
    config = AIPerfJobConfig(
        request_count=5,
        warmup_request_count=0,
        concurrency=1,
        image=image,
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


@pytest.mark.timeout(600)
@pytest.mark.asyncio
async def test_named_phases_are_complete_and_profiling_results_are_filtered(
    operator_ready: OperatorDeployer,
    k8s_settings: Any,
    operator_job_namespace: str,
) -> None:
    """Track all named phases while excluding warmup-kind records from results."""
    name = f"named-{uuid.uuid4().hex[:8]}"
    manifest = _named_phase_manifest(
        name=name,
        namespace=operator_job_namespace,
        image=k8s_settings.aiperf_image,
    )

    await operator_ready.kubectl.apply(yaml.safe_dump(manifest, sort_keys=False))
    try:
        status = await operator_ready.wait_for_job_completion(
            name,
            operator_job_namespace,
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
        await operator_ready.delete_job(name, operator_job_namespace)
