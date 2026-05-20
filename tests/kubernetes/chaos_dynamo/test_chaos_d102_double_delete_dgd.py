# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D102 -- rapid double-delete of a DGD is idempotent."""

from __future__ import annotations

import asyncio

import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

_DGD_NAME = "dynamo-agg"
_DGD_NAMESPACE = "d102-double-delete"
_DGD_ESTABLISHED_TIMEOUT_S = 60.0
_DGD_DELETE_TIMEOUT_S = 90.0


async def test_d102_rapid_double_delete_dgd_is_idempotent(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD CRD/operator exist
) -> None:
    """Delete a newly-created DGD twice quickly and assert no finalizer wedge.

    The first delete should mark the DGD for removal; the second delete, issued
    within one second, must be harmless even if it races apiserver deletion. The
    observable contract is that the CR disappears rather than remaining in a
    terminating state with stuck finalizers.
    """
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=_DGD_NAMESPACE,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)

    await kubectl.create_namespace(_DGD_NAMESPACE)
    try:
        await kubectl.apply(deployer.generate_manifest())
        established = await _wait_for_dgd_established(
            kubectl,
            name=_DGD_NAME,
            namespace=_DGD_NAMESPACE,
            timeout=_DGD_ESTABLISHED_TIMEOUT_S,
        )
        assert established, await _dgd_observed_status_text(
            kubectl,
            name=_DGD_NAME,
            namespace=_DGD_NAMESPACE,
            prefix=(
                f"D102: DGD {_DGD_NAMESPACE}/{_DGD_NAME} never became readable "
                f"within {_DGD_ESTABLISHED_TIMEOUT_S}s after apply"
            ),
        )

        first_delete = await kubectl.run(
            "delete",
            "dynamographdeployment",
            _DGD_NAME,
            "-n",
            _DGD_NAMESPACE,
            "--wait=false",
            check=False,
        )
        assert first_delete.returncode == 0, (
            "D102: first DGD delete failed; "
            f"stdout={first_delete.stdout!r} stderr={first_delete.stderr!r}"
        )

        second_delete = await kubectl.run(
            "delete",
            "dynamographdeployment",
            _DGD_NAME,
            "-n",
            _DGD_NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )
        assert second_delete.returncode == 0, (
            "D102: second DGD delete was not idempotent; "
            f"stdout={second_delete.stdout!r} stderr={second_delete.stderr!r}"
        )

        disappeared = await _wait_for_dgd_absent(
            kubectl,
            name=_DGD_NAME,
            namespace=_DGD_NAMESPACE,
            timeout=_DGD_DELETE_TIMEOUT_S,
        )
        assert disappeared, await _dgd_observed_status_text(
            kubectl,
            name=_DGD_NAME,
            namespace=_DGD_NAMESPACE,
            prefix=(
                f"D102: DGD {_DGD_NAMESPACE}/{_DGD_NAME} still existed "
                f"{_DGD_DELETE_TIMEOUT_S}s after rapid double-delete"
            ),
        )
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _DGD_NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _wait_for_dgd_established(
    kubectl: KubectlClient,
    *,
    name: str,
    namespace: str,
    timeout: float,
) -> bool:
    """Return True once the apiserver can read the DGD JSON document."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                dgd = orjson.loads(result.stdout)
            except orjson.JSONDecodeError as exc:
                logger.debug(lambda exc=exc: f"D102 DGD JSON parse failed: {exc!r}")
            else:
                metadata = dgd.get("metadata", {})
                if metadata.get("uid") and metadata.get("resourceVersion"):
                    return True
        await asyncio.sleep(0.5)
    return False


async def _wait_for_dgd_absent(
    kubectl: KubectlClient,
    *,
    name: str,
    namespace: str,
    timeout: float,
) -> bool:
    """Return True once ``kubectl get`` reports the DGD is gone."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            check=False,
        )
        if result.returncode != 0:
            return True
        await asyncio.sleep(1.0)
    return False


async def _dgd_observed_status_text(
    kubectl: KubectlClient,
    *,
    name: str,
    namespace: str,
    prefix: str,
) -> str:
    """Build a failure message with current status and finalizer context."""
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return (
            f"{prefix}; current read failed with "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    try:
        dgd = orjson.loads(result.stdout)
    except orjson.JSONDecodeError as exc:
        return f"{prefix}; current DGD JSON could not be parsed: {exc!r}"

    metadata = dgd.get("metadata", {})
    status = dgd.get("status", {})
    return (
        f"{prefix}; status={status!r}; "
        f"finalizers={metadata.get('finalizers', [])!r}; "
        f"deletionTimestamp={metadata.get('deletionTimestamp')!r}"
    )
