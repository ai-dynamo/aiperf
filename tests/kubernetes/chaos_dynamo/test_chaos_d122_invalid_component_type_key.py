# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D122 -- invalid DGD component type/key combinations are rejected."""

from __future__ import annotations

from typing import Any

import orjson
import pytest
from pytest import param

from dev.versions import DYNAMO_VERSION
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_NAMESPACE = "d122-invalid-component"


def _service(component_type: str) -> dict[str, Any]:
    return {
        "componentType": component_type,
        "replicas": 1,
        "extraPodSpec": {
            "mainContainer": {
                "image": f"nvcr.io/nvidia/ai-dynamo/vllm-runtime:{DYNAMO_VERSION}",
            }
        },
    }


@pytest.mark.parametrize(
    ("name", "services", "expected_terms"),
    [
        param(
            "d122-invalid-type",
            {
                "Frontend": _service("frontend"),
                "BadWorker": _service("not-a-dynamo-component"),
            },
            ("componenttype", "component type", "not-a-dynamo-component", "unsupported"),
            id="invalid-component-type",
        ),
        param(
            "d122-invalid-key",
            {
                "Frontend": _service("frontend"),
                "bad key with spaces": _service("worker"),
            },
            ("bad key with spaces", "service", "key", "metadata.name"),
            id="invalid-service-key",
        ),
    ],
)  # fmt: skip
async def test_d122_invalid_component_type_or_key_rejected(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures CRD/webhook exist
    name: str,
    services: dict[str, Any],
    expected_terms: tuple[str, ...],
) -> None:
    """Apply invalid component definitions and assert they do not persist."""
    await kubectl.create_namespace(_NAMESPACE)
    try:
        try:
            await kubectl.apply(
                orjson.dumps(_manifest(name=name, services=services)).decode(),
                namespace=_NAMESPACE,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
        else:
            pytest.fail(f"D122 expected invalid component manifest {name!r} to fail")

        assert any(term in message for term in expected_terms), (
            "D122 expected admission error to identify invalid component type/key; "
            f"terms={expected_terms!r}, message={message!r}"
        )
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            _NAMESPACE,
            check=False,
        )
        assert result.returncode != 0, (
            f"D122: invalid DGD {_NAMESPACE}/{name} persisted"
        )
    finally:
        await kubectl.run(
            "delete",
            "dynamographdeployment",
            name,
            "-n",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )
        await kubectl.run(
            "delete",
            "namespace",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


def _manifest(*, name: str, services: dict[str, Any]) -> dict[str, Any]:
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name, "namespace": _NAMESPACE},
        "spec": {"services": services},
    }
