# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D322 -- inject reordered KVBM Add/Free events."""

from __future__ import annotations

import pytest

from tests.kubernetes.chaos_dynamo.test_chaos_d317_kvbm_zmq_publisher_pause import (
    assert_successful_completion,
    discover_kvbm_prefill_target,
    post_completion,
)
from tests.kubernetes.chaos_dynamo.test_chaos_d321_kvbm_duplicate_free import (
    discover_kvbm_chaos_hook,
    run_kvbm_chaos_hook,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


async def test_d322_reordered_kvbm_add_free_events_are_bounded(
    request: pytest.FixtureRequest,
) -> None:
    """Use the KVBM hook to publish Free-before-Add and assert recovery."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D322")
    hook = await discover_kvbm_chaos_hook(kubectl, kvbm_pod, "D322")

    await run_kvbm_chaos_hook(
        kubectl,
        hook,
        "reorder-add-free --request-id d322-synthetic-reorder",
        "D322",
    )

    recovery = await post_completion(
        endpoint_url,
        content="D322 recovery after reordered KVBM Add/Free events.",
        max_tokens=8,
    )
    assert_successful_completion("D322 recovery", recovery)
