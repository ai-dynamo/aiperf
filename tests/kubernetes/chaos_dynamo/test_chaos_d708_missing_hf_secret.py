# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D708 -- missing Hugging Face token secret status propagation."""

from __future__ import annotations

import asyncio

import orjson
import pytest

from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_DGD_NAMESPACE = "d708-hf-secret"
_MISSING_SECRET = "d708-missing-hf-token"
_EVENT_TIMEOUT_S = 90.0
_FAILED_TIMEOUT_S = 120.0
_STATUS_TERMS = (
    "secret",
    _MISSING_SECRET,
    "envfrom",
    "configerror",
    "createcontainerconfigerror",
    "huggingface",
    "hf",
)


async def test_d708_missing_hf_secret_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Worker envFromSecret points at a missing HF secret and DGD reports it."""
    config = DynamoConfig.single_gpu_disagg(
        namespace=_DGD_NAMESPACE,
        hf_token_secret=_MISSING_SECRET,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    dgd_name = deployer._deployment_name()

    await kubectl.delete_namespace(_DGD_NAMESPACE, wait=True)
    await kubectl.create_namespace(_DGD_NAMESPACE)
    try:
        await kubectl.apply(deployer.generate_manifest(), namespace=_DGD_NAMESPACE)
        event_text = await _wait_for_event(kubectl, _DGD_NAMESPACE)
        assert event_text, (
            f"D708: no missing-secret event appeared within {_EVENT_TIMEOUT_S}s"
        )

        observed_state = await wait_for_dgd_state(
            kubectl, dgd_name, _DGD_NAMESPACE, "failed", timeout=_FAILED_TIMEOUT_S
        )
        assert observed_state == "failed"

        status_text = await _read_status(kubectl, _DGD_NAMESPACE, dgd_name)
        assert any(term in status_text.lower() for term in _STATUS_TERMS), (
            "D708: DGD failed status did not name the missing HF token secret. "
            f"status={status_text!r}; event={event_text!r}"
        )
    finally:
        await kubectl.delete_namespace(_DGD_NAMESPACE, wait=False)


async def _wait_for_event(kubectl: KubectlClient, namespace: str) -> str:
    deadline = asyncio.get_event_loop().time() + _EVENT_TIMEOUT_S
    while asyncio.get_event_loop().time() < deadline:
        events = await _read_events(kubectl, namespace)
        if any(term in events.lower() for term in _STATUS_TERMS):
            return events
        await asyncio.sleep(1.0)
    return ""


async def _read_events(kubectl: KubectlClient, namespace: str) -> str:
    result = await kubectl.run(
        "get", "events", "-n", namespace, "-o", "json", check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    data = orjson.loads(result.stdout)
    return "\n".join(
        f"{item.get('reason', '')}: {item.get('message', '')}"
        for item in data.get("items", [])
    )


async def _read_status(kubectl: KubectlClient, namespace: str, name: str) -> str:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        namespace,
        "-o",
        "jsonpath={.status}",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""
