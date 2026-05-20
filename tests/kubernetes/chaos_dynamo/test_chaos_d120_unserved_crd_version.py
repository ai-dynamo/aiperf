# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D120 -- unserved DynamoGraphDeployment CRD versions are rejected."""

from __future__ import annotations

import orjson
import pytest

from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_NAMESPACE = "d120-unserved-version"
_NAME = "d120-unserved"
_UNSERVED_VERSION = "v99alpha99"


async def test_d120_unserved_dgd_api_version_is_rejected(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures DGD CRD exists
) -> None:
    """Apply a DGD with a version not served by the CRD and assert rejection."""
    served_versions = await _served_dgd_versions(kubectl)
    if _UNSERVED_VERSION in served_versions:
        pytest.skip(
            f"D120 requires {_UNSERVED_VERSION!r} to be unserved; served={served_versions!r}"
        )

    await kubectl.create_namespace(_NAMESPACE)
    try:
        try:
            await kubectl.apply(
                orjson.dumps(_manifest()).decode(), namespace=_NAMESPACE
            )
        except RuntimeError as exc:
            message = str(exc).lower()
        else:
            pytest.fail(
                f"D120 expected nvidia.com/{_UNSERVED_VERSION} DGD to fail admission"
            )

        assert any(
            term in message
            for term in (
                "no matches for kind",
                "no kind",
                "not registered",
                "could not find",
                "unable to recognize",
            )
        ), f"D120 expected unserved-version error, got {message!r}"
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            _NAME,
            "-n",
            _NAMESPACE,
            check=False,
        )
        assert result.returncode != 0, "D120: unserved-version DGD was persisted"
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _served_dgd_versions(kubectl: KubectlClient) -> list[str]:
    result = await kubectl.run(
        "get",
        "crd",
        "dynamographdeployments.nvidia.com",
        "-o",
        "json",
        check=True,
    )
    crd = orjson.loads(result.stdout)
    return [
        version.get("name", "")
        for version in crd.get("spec", {}).get("versions", [])
        if version.get("served") is True
    ]


def _manifest() -> dict[str, object]:
    return {
        "apiVersion": f"nvidia.com/{_UNSERVED_VERSION}",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": _NAME, "namespace": _NAMESPACE},
        "spec": {"services": {}},
    }
