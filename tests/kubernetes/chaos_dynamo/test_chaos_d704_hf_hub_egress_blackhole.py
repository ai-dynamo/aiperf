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

import os

import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.gpu.conftest import GPUTestSettings
from tests.kubernetes.gpu.dynamo.helpers import (
    DynamoBackend,
    DynamoConfig,
    DynamoDeployer,
    DynamoMode,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

_D704_MODEL_ENV = "D704_HF_CACHE_MISS_MODEL"
_D704_DEFAULT_MODEL = "hf-internal-testing/tiny-random-gpt2"
_D704_NAME = "dynamo-agg"
_D704_NAMESPACE = "d704-hf-egress"
_D704_POLICY = "d704-blackhole"
_NETWORK_POLICY_CNI_NEEDLES = (
    "cilium",
    "calico",
    "tigera",
    "canal",
    "antrea",
)


async def test_d704_hf_hub_egress_blackhole(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
) -> None:
    """Block egress; assert worker fails weight-download cleanly, DGD reports failure.

    Requires a NetworkPolicy-enforcing CNI and a model name that is absent from
    the node/container HF cache. Without those prerequisites the test skips
    before applying the DGD so a green run cannot be a false positive from
    kindnet ignoring NetworkPolicy or from a cached model bypassing HF Hub.
    """
    await _skip_unless_network_policy_enforced(kubectl)
    model = _cache_miss_model_or_skip()
    request.getfixturevalue("dynamo_operator")
    faults = request.getfixturevalue("faults")
    await _run_d704_assertion(faults, kubectl, gpu_settings, model=model)


async def _run_d704_assertion(
    faults,
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
    *,
    model: str,
) -> None:
    """Apply DGD under an egress-deny policy and assert actionable failure.

    The DGD uses the aggregated vLLM worker shape because the HF download path
    lives in ``dynamo.vllm`` worker startup, not in the frontend component.
    """
    manifest = _build_d704_manifest(kubectl, gpu_settings, model=model)

    await kubectl.create_namespace(_D704_NAMESPACE)
    try:
        async with faults.inject(
            "cluster.network_policy.deny_egress",
            target={"ns": _D704_NAMESPACE},
            name=_D704_POLICY,
            allow_cluster_egress=True,
        ):
            await kubectl.apply(manifest=manifest, namespace=_D704_NAMESPACE)

            await wait_for_dgd_state(
                kubectl, _D704_NAME, _D704_NAMESPACE, "failed", timeout=300
            )

            result = await kubectl.run(
                "get",
                "dynamographdeployment",
                _D704_NAME,
                "-n",
                _D704_NAMESPACE,
                "-o",
                "json",
                check=True,
            )
            dgd = orjson.loads(result.stdout)
            assert dgd["status"]["state"] == "failed"

            status_text = orjson.dumps(dgd.get("status", {})).decode().lower()
            keywords = (
                "weight",
                "hub",
                "huggingface",
                "network",
                "egress",
                "dns",
                "connect",
                "download",
                "model",
            )
            assert any(kw in status_text for kw in keywords), (
                "expected DGD status to name network/model-download failure; "
                f"got status={status_text!r}"
            )
    finally:
        # NetworkPolicy is cleaned up by the faults.inject context's restore;
        # the DGD namespace is ours to delete.
        await kubectl.run(
            "delete",
            "namespace",
            _D704_NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


def _cache_miss_model_or_skip() -> str:
    """Return the requested cache-miss HF model or skip with setup guidance."""
    model = os.environ.get(_D704_MODEL_ENV, _D704_DEFAULT_MODEL).strip()
    if not model:
        pytest.skip(
            f"D704 requires {_D704_MODEL_ENV}=<uncached Hugging Face model id> "
            "so the worker must reach HF Hub during startup"
        )
    return model


def _build_d704_manifest(
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
    *,
    model: str,
) -> str:
    """Build a minimal v1alpha1 aggregated DGD that starts a vLLM worker."""
    config = DynamoConfig(
        model_name=model,
        namespace=_D704_NAMESPACE,
        backend=DynamoBackend.VLLM,
        mode=DynamoMode.AGGREGATED,
        gpu_count=0,
        max_model_len=gpu_settings.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=0.12,
        runtime_class_name=gpu_settings.runtime_class,
        hf_token_secret=gpu_settings.hf_token_secret,
        image=gpu_settings.dynamo_image,
        image_pull_secrets=gpu_settings.image_pull_secrets,
    )
    return DynamoDeployer(kubectl=kubectl, config=config).generate_manifest()


async def _skip_unless_network_policy_enforced(kubectl: KubectlClient) -> None:
    """Skip unless the cluster has NetworkPolicy API plus an enforcing CNI."""
    api_result = await kubectl.run(
        "api-resources",
        "--api-group=networking.k8s.io",
        "-o",
        "name",
        check=False,
    )
    if api_result.returncode != 0 or "networkpolicies" not in api_result.stdout:
        pytest.skip("D704 requires networking.k8s.io NetworkPolicy support")

    pods_result = await kubectl.run("get", "pods", "-A", "-o", "json", check=False)
    if pods_result.returncode != 0:
        pytest.skip("D704 could not inspect cluster CNI pods before NetworkPolicy test")

    pod_data = orjson.loads(pods_result.stdout or b"{}")
    cni_text = " ".join(
        f"{item.get('metadata', {}).get('namespace', '')}/"
        f"{item.get('metadata', {}).get('name', '')} "
        f"{item.get('metadata', {}).get('labels', {})}"
        for item in pod_data.get("items", [])
    ).lower()
    if not any(needle in cni_text for needle in _NETWORK_POLICY_CNI_NEEDLES):
        pytest.skip(
            "D704 requires a NetworkPolicy-enforcing CNI such as Cilium or Calico; "
            "kindnet applies NetworkPolicy objects but does not block egress"
        )
