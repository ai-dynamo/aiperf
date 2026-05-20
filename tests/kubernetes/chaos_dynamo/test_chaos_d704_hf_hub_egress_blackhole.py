# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D704 - HF Hub egress blackhole; weight download fails cleanly, not opaquely.

Wave-0 #10 (Cilium-gated). Targets the cache-miss weight-download path in
``components/src/dynamo/vllm/main.py`` (``await fetch_model(config.model)``)
to confirm the worker surfaces a clear failure status when external egress
is severed, rather than hanging or emitting an opaque error.

The fault is a ``NetworkPolicy`` denying egress to ``0.0.0.0/0`` except the
cluster CIDR (``allow_cluster_egress=True``). NetworkPolicy enforcement
requires a NetworkPolicy-aware CNI -- kindnet silently ignores the policy,
so the scenario is gated on :py:data:`KIND_HAS_CILIUM` via the
:py:func:`cilium_on_kind_required` mark. Without Cilium the test is
xfail-skipped; with Cilium the strict xfail flips and a failure surfaces
loudly (see ``tests/kubernetes/chaos_common/README.md``).
"""

from __future__ import annotations

import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from dev.versions import DYNAMO_VERSION
from tests.kubernetes.chaos_common.marks import cilium_on_kind_required
from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


@cilium_on_kind_required
async def test_d704_hf_hub_egress_blackhole(faults, kubectl):
    """Block egress; assert worker fails weight-download cleanly, DGD reports failure.

    Requires a NetworkPolicy-aware CNI (Cilium or Calico). The
    ``cilium_on_kind_required`` mark gates this test on the
    ``KIND_HAS_CILIUM`` env var; without Cilium the test is xfail-skipped.
    With Cilium the test must PASS or the strict xfail flips to a loud
    failure (see ``chaos_common/README.md`` "Verifying the flip").
    """
    pytest.skip(
        "scaffold landed; awaiting Cilium-equipped cluster + fresh DGD with "
        "cache-miss model"
    )
    await _run_d704_assertion(faults, kubectl)


async def _run_d704_assertion(faults, kubectl) -> None:
    """Full D704 assertion body; one-line unskip flip to run.

    Materialized separately so the public test stays a thin gated entrypoint:
    deleting the ``pytest.skip`` above is the only edit needed to exercise
    the path against a Cilium-equipped cluster. Keeps the assertion shape
    diff-reviewable without entangling it with the gating contract.
    """
    name = "d704-test"
    ns = "dynamo-server"
    policy_name = "d704-blackhole"

    # Qwen3-0.6B is the project default; a brand-new cluster has no cache, so
    # the worker must reach out to HF Hub during fetch_model and that egress
    # is the leg the NetworkPolicy severs.
    dgd_manifest = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name, "namespace": ns},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": f"nvcr.io/nvidia/ai-dynamo/vllm-runtime:{DYNAMO_VERSION}",
                            "args": ["--model", "Qwen/Qwen3-0.6B"],
                        }
                    },
                }
            }
        },
    }

    try:
        async with faults.inject(
            "cluster.network_policy.deny_egress",
            target={"ns": ns},
            name=policy_name,
            allow_cluster_egress=True,
        ):
            await kubectl.apply(manifest=orjson.dumps(dgd_manifest).decode())

            await wait_for_dgd_state(kubectl, name, ns, "failed", timeout=300)

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
            assert dgd["status"]["state"] == "failed"

            conditions = dgd["status"].get("conditions", [])
            ready_condition = next(
                (c for c in conditions if c.get("type") == "Ready"), None
            )
            assert ready_condition is not None, "no Ready condition in status"
            assert ready_condition["status"] == "False"

            reason = ready_condition.get("reason", "").lower()
            message = ready_condition.get("message", "").lower()
            haystack = reason + " " + message
            keywords = (
                "weight",
                "hub",
                "huggingface",
                "network",
                "egress",
                "dns",
                "connect",
                "download",
            )
            assert any(kw in haystack for kw in keywords), (
                f"expected reason/message to name network/weight failure; "
                f"got reason={reason!r} message={message!r}"
            )
    finally:
        # NetworkPolicy is cleaned up by the faults.inject context's restore;
        # the DGD is ours to delete.
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
