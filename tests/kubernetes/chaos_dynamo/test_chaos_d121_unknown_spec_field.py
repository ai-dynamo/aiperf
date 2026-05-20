# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D121 -- unknown DGD spec fields are rejected at admission."""

from __future__ import annotations

import orjson
import pytest

from dev.versions import DYNAMO_VERSION
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_NAMESPACE = "d121-unknown-spec-field"
_NAME = "d121-unknown-field"
_UNKNOWN_FIELD = "definitelyUnknownD121Field"


async def test_d121_unknown_dgd_spec_field_is_rejected(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures CRD/webhook exist
) -> None:
    """Apply a DGD with an unknown spec key and assert strict admission failure."""
    await kubectl.create_namespace(_NAMESPACE)
    try:
        try:
            await kubectl.apply(
                orjson.dumps(_manifest()).decode(), namespace=_NAMESPACE
            )
        except RuntimeError as exc:
            message = str(exc).lower()
        else:
            pytest.fail("D121 expected unknown DGD spec field to fail admission")

        assert _UNKNOWN_FIELD.lower() in message or "unknown field" in message, (
            "D121 expected admission error to name the unknown field; "
            f"message={message!r}"
        )
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            _NAME,
            "-n",
            _NAMESPACE,
            check=False,
        )
        assert result.returncode != 0, "D121: DGD with unknown spec field was persisted"
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


def _manifest() -> dict[str, object]:
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": _NAME, "namespace": _NAMESPACE},
        "spec": {
            _UNKNOWN_FIELD: True,
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": f"nvcr.io/nvidia/ai-dynamo/vllm-runtime:{DYNAMO_VERSION}",
                        }
                    },
                }
            },
        },
    }
