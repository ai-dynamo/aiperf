# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D104 -- invalid DGD spec is rejected with an actionable validation error."""

from __future__ import annotations

import orjson
import pytest

from dev.versions import DYNAMO_VERSION
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


async def test_d104_invalid_dgd_replicas_negative(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD CRD/webhook exists
) -> None:
    """Apply DGD with replicas=-1; assert admission rejects the invalid spec.

    The v1alpha1 CRD now rejects negative replicas at apiserver admission time,
    before a CR exists for the operator to drive into ``status.state=failed``.
    This keeps D104 runnable against its intended validation signal without
    polling for a resource the apiserver correctly never creates.
    """
    name = "d104-test"
    ns = "d104-invalid"
    manifest = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name, "namespace": ns},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": -1,  # INVALID
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": f"nvcr.io/nvidia/ai-dynamo/vllm-runtime:{DYNAMO_VERSION}",
                        }
                    },
                }
            }
        },
    }

    await kubectl.create_namespace(ns)
    try:
        try:
            await kubectl.apply(orjson.dumps(manifest).decode(), namespace=ns)
        except RuntimeError as exc:
            message = str(exc).lower()
        else:
            pytest.fail("expected replicas=-1 manifest to fail admission")

        assert "replicas" in message, (
            f"expected admission error to mention replicas, got {message!r}"
        )
        assert "greater than or equal to 0" in message or "minimum" in message, (
            "expected admission error to name the non-negative constraint, "
            f"got {message!r}"
        )

        get_result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            ns,
            check=False,
        )
        assert get_result.returncode != 0, (
            "invalid DGD should be rejected at admission, not created for "
            "operator reconciliation"
        )
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            ns,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )
