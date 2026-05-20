# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D104 -- invalid DGD spec surfaces as status.state=failed with Ready=False."""

from __future__ import annotations

import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


async def test_d104_invalid_dgd_replicas_negative(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
) -> None:
    """Apply DGD with replicas=-1; assert status.state=failed within 60s.

    Targets: webhook validation + reconciler error path.

    The scaffold lays out the full happy-path assertion sequence so the
    body can be enabled once a real cluster + dynamo-operator are wired
    into CI. ``wait_for_dgd_state`` is imported from the package conftest
    as a plain async helper (not a fixture); it takes ``kubectl`` as its
    first positional argument.
    """
    # 1. Build an invalid DGD manifest:
    #    apiVersion: nvidia.com/v1beta1
    #    kind: DynamoGraphDeployment
    #    metadata: {name: d104-test, namespace: dynamo-server}
    #    spec:
    #      components:
    #        - name: Frontend
    #          type: frontend
    #          replicas: -1   # <-- INVALID
    #          podTemplate:
    #            spec:
    #              containers:
    #                - name: main
    #                  image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:v0.9.0
    #
    # 2. await faults.inject(
    #        "crd.apply_invalid",
    #        target={"ns": "dynamo-server", "name": "d104-test"},
    #        manifest=<the above dict>,
    #    )
    #
    # 3. from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
    #    await wait_for_dgd_state(
    #        kubectl, "d104-test", "dynamo-server", "failed", timeout=60,
    #    )
    #
    # 4. result = await kubectl.run(
    #        "get", "dynamographdeployment", "d104-test",
    #        "-n", "dynamo-server", "-o", "json", check=True,
    #    )
    #    cr = orjson.loads(result.stdout)
    #
    # 5. Assertions:
    #    - cr["status"]["state"] == "failed"
    #    - ready = next(c for c in cr["status"]["conditions"] if c["type"] == "Ready")
    #    - ready["status"] == "False"
    #    - "replicas" in (ready.get("reason", "") + ready.get("message", "")).lower()
    #      or "validation" in (...)
    #
    # 6. Cleanup: ``crd.apply_invalid``'s registered restore deletes the CR;
    #    the InjectorRegistry's LIFO teardown in the ``faults`` fixture
    #    handles it on test exit, no explicit ``finally`` needed.

    pytest.skip("scaffold landed; assertion-body pending real-cluster validation")
