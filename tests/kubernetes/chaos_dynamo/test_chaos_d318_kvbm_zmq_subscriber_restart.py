# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D318 -- restart a KVBM ZMQ subscriber during block-free traffic."""

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


async def test_d318_kvbm_zmq_subscriber_restart_during_frees(
    request: pytest.FixtureRequest,
) -> None:
    """Restart an isolated KVBM ZMQ subscriber and assert traffic recovers."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")
    faults: InjectorRegistry = request.getfixturevalue("faults")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D318")
    subscriber = await discover_isolated_kvbm_process(
        kubectl,
        kvbm_pod,
        role="subscriber",
        role_patterns=("subscriber", "sub", "zmq.*sub"),
        scenario_id="D318",
    )

    stream_task = asyncio.create_task(
        post_completion(
            endpoint_url,
            content="D318 long request generating KVBM add/free events.",
            max_tokens=64,
        )
    )
    await asyncio.sleep(1.0)

    async with faults.inject(
        "process.signal",
        target={
            "kind": "pod",
            "ns": subscriber.pod_target.namespace,
            "pod": subscriber.pod_target.pod,
            "container": subscriber.pod_target.container,
            "pid": subscriber.pid,
        },
        signal="SIGTERM",
    ) as applied:
        assert applied.metadata.get("signal") == "SIGTERM"

    stream_result = await stream_task
    assert_successful_completion("D318 in-flight request", stream_result)
    await wait_for_pod_ready(kubectl, namespace, subscriber.pod_target.pod)

    recovery = await post_completion(
        endpoint_url,
        content="D318 recovery request after KVBM subscriber restart.",
        max_tokens=8,
    )
    assert_successful_completion("D318 recovery", recovery)
