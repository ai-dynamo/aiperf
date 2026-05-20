# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D320 -- restart KVBM consolidator between AddRequest and Free."""

from __future__ import annotations

import asyncio

import pytest

from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.chaos_dynamo.test_chaos_d317_kvbm_zmq_publisher_pause import (
    assert_successful_completion,
    discover_isolated_kvbm_process,
    discover_kvbm_prefill_target,
    post_completion,
    wait_for_pod_ready,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


async def test_d320_kvbm_consolidator_restart_between_add_and_free(
    request: pytest.FixtureRequest,
) -> None:
    """Restart an isolated KVBM consolidator while a request is active."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")
    faults: InjectorRegistry = request.getfixturevalue("faults")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D320")
    consolidator = await discover_isolated_kvbm_process(
        kubectl,
        kvbm_pod,
        role="consolidator",
        role_patterns=("consolidator", "consolidat", "free.*block", "block.*free"),
        scenario_id="D320",
    )

    in_flight = asyncio.create_task(
        post_completion(
            endpoint_url,
            content="D320 long request held across consolidator restart.",
            max_tokens=96,
        )
    )
    await asyncio.sleep(1.0)

    async with faults.inject(
        "process.signal",
        target={
            "kind": "pod",
            "ns": consolidator.pod_target.namespace,
            "pod": consolidator.pod_target.pod,
            "container": consolidator.pod_target.container,
            "pid": consolidator.pid,
        },
        signal="SIGTERM",
    ) as applied:
        assert applied.metadata.get("pid") == consolidator.pid

    result = await in_flight
    assert_successful_completion("D320 in-flight", result)
    await wait_for_pod_ready(kubectl, namespace, consolidator.pod_target.pod)

    recovery = await post_completion(
        endpoint_url,
        content="D320 recovery request after consolidator restart.",
        max_tokens=8,
    )
    assert_successful_completion("D320 recovery", recovery)
