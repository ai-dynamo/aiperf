# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D104 -- invalid DGD spec surfaces as status.state=failed with Ready=False."""

from __future__ import annotations

import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from dev.versions import DYNAMO_VERSION
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


async def test_d104_invalid_dgd_replicas_negative(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
) -> None:
    """Apply DGD with replicas=-1; assert status.state=failed within 60s.

    Targets: webhook validation + reconciler error path.

    The assertion body is materialized in :py:func:`_run_d104_assertion` so the
    outer test stays a one-line ``pytest.skip`` + helper call; flip the skip to
    enable the test once a real cluster with the dynamo CRDs is wired into CI.
    """
    pytest.skip(
        "scaffold landed; assertion body materialized but awaiting cluster with dynamo CRDs"
    )
    await _run_d104_assertion(faults, kubectl)


async def _run_d104_assertion(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
) -> None:
    """Full D104 assertion body; one-line unskip flip to run.

    Applies a ``DynamoGraphDeployment`` with ``replicas=-1`` via the
    ``crd.apply_invalid`` injector, then asserts the operator drives the CR
    to ``status.state=failed`` with ``Ready=False`` and a reason or message
    that names the invalid field. The ``faults.inject`` context manager
    handles teardown on exit; the ``finally`` block is a belt-and-braces
    delete in case the injector restore is bypassed.
    """
    name = "d104-test"
    ns = "dynamo-server"
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name, "namespace": ns},
        "spec": {
            "components": [
                {
                    "name": "Frontend",
                    "type": "frontend",
                    "replicas": -1,  # INVALID
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "image": f"nvcr.io/nvidia/ai-dynamo/vllm-runtime:{DYNAMO_VERSION}",
                                }
                            ]
                        }
                    },
                }
            ]
        },
    }

    try:
        async with faults.inject(
            "crd.apply_invalid",
            target={"ns": ns, "name": name},
            manifest=manifest,
        ):
            await wait_for_dgd_state(kubectl, name, ns, "failed", timeout=60)

            result = await kubectl.run(
                "get",
                "dynamographdeployment",
                name,
                "-n",
                ns,
                "-o",
                "json",
                check=True,
            )
            dgd = orjson.loads(result.stdout)

            assert dgd["status"]["state"] == "failed", (
                f"expected state=failed, got {dgd['status']['state']!r}"
            )

            conditions = dgd["status"].get("conditions", [])
            ready_condition = next(
                (c for c in conditions if c.get("type") == "Ready"), None
            )
            assert ready_condition is not None, "no Ready condition in status"
            assert ready_condition["status"] == "False", (
                f"expected Ready=False, got Ready={ready_condition['status']!r}"
            )

            reason = ready_condition.get("reason", "")
            message = ready_condition.get("message", "")
            haystack = f"{reason} {message}".lower()
            assert "replicas" in haystack or "validation" in haystack, (
                "expected reason/message to mention replicas or validation, "
                f"got reason={reason!r} message={message!r}"
            )
    finally:
        await kubectl.run(
            "delete",
            "dynamographdeployment",
            name,
            "-n",
            ns,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )
