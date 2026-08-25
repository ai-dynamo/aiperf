# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live admission coverage for the Kubernetes workload update contract."""

from __future__ import annotations

import asyncio
import subprocess
import uuid
from copy import deepcopy
from typing import Any

import orjson
import pytest
import yaml

from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer


def _patch_body(body: dict[str, Any]) -> str:
    return orjson.dumps(body).decode()


async def _read_cr(
    kubectl: KubectlClient, kind: str, name: str, namespace: str
) -> dict[str, Any]:
    result = await kubectl.run("get", kind, name, "-o", "json", namespace=namespace)
    return orjson.loads(result.stdout)


async def _wait_for_observed_generation(
    kubectl: KubectlClient,
    kind: str,
    name: str,
    namespace: str,
    generation: int,
    *,
    timeout: float = 90.0,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout
    last: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        last = await _read_cr(kubectl, kind, name, namespace)
        if (last.get("status") or {}).get("observedGeneration") == generation:
            return last
        await asyncio.sleep(1)
    raise TimeoutError(
        f"{kind}/{namespace}/{name} did not observe generation {generation}; "
        f"last status={last.get('status')}"
    )


async def _patch(
    kubectl: KubectlClient,
    kind: str,
    name: str,
    namespace: str,
    body: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    return await kubectl.run(
        "patch",
        kind,
        name,
        "--type=merge",
        "-p",
        _patch_body(body),
        namespace=namespace,
        check=False,
    )


def _job_manifest(
    *, name: str, namespace: str, image: str, priority_class_name: str
) -> dict[str, Any]:
    config = AIPerfJobConfig(
        concurrency=1,
        request_count=100,
        warmup_request_count=0,
        image=image,
    )
    manifest = yaml.safe_load(config.to_cr_manifest(name, namespace))
    manifest["spec"]["skipEndpointCheck"] = True
    manifest["spec"]["ttlSecondsAfterFinished"] = 600
    # Keep this admission-focused test from consuming worker capacity. Kubernetes
    # leaves pods Pending when their native PriorityClass does not exist, while
    # the operator can still create and acknowledge the workload resources.
    manifest["spec"]["podTemplate"] = {
        "priorityClassName": priority_class_name,
    }
    return manifest


@pytest.mark.timeout(300)
@pytest.mark.asyncio
async def test_create_time_fields_are_immutable_but_live_controls_reconcile(
    operator_ready: OperatorDeployer,
    k8s_settings: Any,
    operator_job_namespace: str,
) -> None:
    """Reject ignored edits while accepting and acknowledging live controls."""
    suffix = uuid.uuid4().hex[:8]
    job_name = f"immutable-job-{suffix}"
    sweep_name = f"immutable-sweep-{suffix}"
    missing_priority_class = f"aiperf-admission-only-{suffix}"
    kubectl = operator_ready.kubectl

    job = _job_manifest(
        name=job_name,
        namespace=operator_job_namespace,
        image=k8s_settings.aiperf_image,
        priority_class_name=missing_priority_class,
    )
    sweep = deepcopy(job)
    sweep["kind"] = "AIPerfSweep"
    sweep["metadata"]["name"] = sweep_name
    sweep["spec"]["sweep"] = {
        "type": "grid",
        "parameters": {"phases.profiling.concurrency": [1, 2]},
    }

    priority_class = await kubectl.run(
        "get", "priorityclass", missing_priority_class, check=False
    )
    assert priority_class.returncode != 0

    await kubectl.apply(yaml.safe_dump(job, sort_keys=False))
    await kubectl.apply(yaml.safe_dump(sweep, sort_keys=False))
    try:
        job_cr = await _read_cr(kubectl, "aiperfjob", job_name, operator_job_namespace)
        sweep_cr = await _read_cr(
            kubectl, "aiperfsweep", sweep_name, operator_job_namespace
        )
        await _wait_for_observed_generation(
            kubectl,
            "aiperfjob",
            job_name,
            operator_job_namespace,
            job_cr["metadata"]["generation"],
        )
        await _wait_for_observed_generation(
            kubectl,
            "aiperfsweep",
            sweep_name,
            operator_job_namespace,
            sweep_cr["metadata"]["generation"],
        )

        phases = deepcopy(job["spec"]["benchmark"]["phases"])
        phases[0]["concurrency"] = 2
        rejected_job = await _patch(
            kubectl,
            "aiperfjob",
            job_name,
            operator_job_namespace,
            {"spec": {"benchmark": {"phases": phases}}},
        )
        assert rejected_job.returncode != 0
        assert "spec.benchmark is immutable after creation" in (
            rejected_job.stdout + rejected_job.stderr
        )
        unchanged_job = await _read_cr(
            kubectl, "aiperfjob", job_name, operator_job_namespace
        )
        assert (
            unchanged_job["metadata"]["generation"] == job_cr["metadata"]["generation"]
        )

        rejected_sweep = await _patch(
            kubectl,
            "aiperfsweep",
            sweep_name,
            operator_job_namespace,
            {"spec": {"childMetadata": {"labels": {"late": "true"}}}},
        )
        assert rejected_sweep.returncode != 0
        assert "spec.childMetadata is immutable after creation" in (
            rejected_sweep.stdout + rejected_sweep.stderr
        )
        unchanged_sweep = await _read_cr(
            kubectl, "aiperfsweep", sweep_name, operator_job_namespace
        )
        assert (
            unchanged_sweep["metadata"]["generation"]
            == sweep_cr["metadata"]["generation"]
        )

        accepted_job = await _patch(
            kubectl,
            "aiperfjob",
            job_name,
            operator_job_namespace,
            {"spec": {"timeoutSeconds": 900}},
        )
        assert accepted_job.returncode == 0, accepted_job.stderr
        job_cr = await _read_cr(kubectl, "aiperfjob", job_name, operator_job_namespace)
        assert job_cr["spec"]["timeoutSeconds"] == 900
        await _wait_for_observed_generation(
            kubectl,
            "aiperfjob",
            job_name,
            operator_job_namespace,
            job_cr["metadata"]["generation"],
        )

        accepted_sweep = await _patch(
            kubectl,
            "aiperfsweep",
            sweep_name,
            operator_job_namespace,
            {"spec": {"ttlSecondsAfterFinished": 601}},
        )
        assert accepted_sweep.returncode == 0, accepted_sweep.stderr
        sweep_cr = await _read_cr(
            kubectl, "aiperfsweep", sweep_name, operator_job_namespace
        )
        assert sweep_cr["spec"]["ttlSecondsAfterFinished"] == 601
        await _wait_for_observed_generation(
            kubectl,
            "aiperfsweep",
            sweep_name,
            operator_job_namespace,
            sweep_cr["metadata"]["generation"],
        )
    finally:
        for kind, name in (
            ("aiperfjob", job_name),
            ("aiperfsweep", sweep_name),
        ):
            await kubectl.run(
                "delete",
                kind,
                name,
                "--ignore-not-found",
                "--wait=false",
                namespace=operator_job_namespace,
                check=False,
            )
