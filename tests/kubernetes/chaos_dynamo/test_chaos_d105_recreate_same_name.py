# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D105 -- Rapid create-delete-create with the same DGD name.

Exercises DynamoGraphDeployment finalizer tombstone handling by proving a
fresh CR with the same name can reconcile successfully immediately after the
prior CR has been deleted.
"""

from __future__ import annotations

import asyncio
from typing import Any

import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

_DGD_NAMESPACE = "d105-recreate-same-name"
_ABSENT_TIMEOUT_S = 5.0
_SUCCESS_TIMEOUT_S = 600.0


async def test_d105_recreate_same_dgd_name_reconciles_successfully(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the Dynamo operator is installed
) -> None:
    """Delete a successful DGD and re-create the same name within 5 seconds.

    The first apply is a prerequisite probe: if a clean minimal DGD cannot
    reach ``state=successful``, the cluster cannot start Dynamo workloads and
    this scenario cannot distinguish tombstone handling from baseline startup
    failure. Once that baseline succeeds, the second same-name CR must also
    reach ``state=successful`` and must have a different UID from the deleted
    predecessor.
    """
    config = DynamoConfig.single_gpu_disagg(
        namespace=_DGD_NAMESPACE,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    manifest = deployer.generate_manifest()
    name = deployer._deployment_name()

    try:
        await kubectl.apply(manifest, namespace=_DGD_NAMESPACE)
        try:
            await wait_for_dgd_state(
                kubectl,
                name,
                _DGD_NAMESPACE,
                "successful",
                timeout=_SUCCESS_TIMEOUT_S,
            )
        except TimeoutError as exc:
            status = await _dgd_status_snapshot(kubectl, name=name)
            pytest.skip(
                "D105 requires the cluster to start a minimal Dynamo workload; "
                f"baseline DGD {_DGD_NAMESPACE}/{name} did not reach "
                f"state='successful' within {_SUCCESS_TIMEOUT_S}s. "
                f"Status: {status!r}. Error: {exc}"
            )

        first_dgd = await _read_dgd_json(kubectl, name=name)
        first_uid = first_dgd["metadata"]["uid"]
        logger.info(f"D105: baseline DGD {_DGD_NAMESPACE}/{name} uid={first_uid}")

        await kubectl.run(
            "delete",
            "dynamographdeployment",
            name,
            "-n",
            _DGD_NAMESPACE,
            "--wait=false",
            check=True,
        )
        await _wait_for_dgd_absent(kubectl, name=name, timeout=_ABSENT_TIMEOUT_S)

        await kubectl.apply(manifest, namespace=_DGD_NAMESPACE)
        second_dgd = await _read_dgd_json(kubectl, name=name)
        second_uid = second_dgd["metadata"]["uid"]
        assert second_uid != first_uid, (
            f"D105 expected a fresh same-name DGD after deletion, but "
            f"{_DGD_NAMESPACE}/{name} still has uid={second_uid!r}"
        )

        observed_state = await wait_for_dgd_state(
            kubectl,
            name,
            _DGD_NAMESPACE,
            "successful",
            timeout=_SUCCESS_TIMEOUT_S,
        )
        assert observed_state == "successful"

        final_dgd = await _read_dgd_json(kubectl, name=name)
        metadata_generation = final_dgd["metadata"]["generation"]
        observed_generation = final_dgd["status"].get("observedGeneration")
        assert observed_generation == metadata_generation, (
            f"D105 same-name DGD observedGeneration={observed_generation} "
            f"!= generation={metadata_generation}"
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


async def _wait_for_dgd_absent(
    kubectl: KubectlClient,
    *,
    name: str,
    timeout: float,
) -> None:
    """Wait until the first same-name DGD is gone so the re-apply is a new CR."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            _DGD_NAMESPACE,
            "-o",
            "jsonpath={.metadata.uid}",
            check=False,
        )
        if result.returncode != 0:
            return
        await asyncio.sleep(0.25)

    status = await _dgd_status_snapshot(kubectl, name=name)
    raise AssertionError(
        f"D105 expected {_DGD_NAMESPACE}/{name} to be deleted within "
        f"{timeout}s before same-name re-create; last status: {status!r}"
    )


async def _read_dgd_json(kubectl: KubectlClient, *, name: str) -> dict[str, Any]:
    """Read a DynamoGraphDeployment as a parsed JSON object."""
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        _DGD_NAMESPACE,
        "-o",
        "json",
        check=True,
    )
    return orjson.loads(result.stdout)


async def _dgd_status_snapshot(kubectl: KubectlClient, *, name: str) -> str:
    """Return status/finalizers text for D105 failure and skip messages."""
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        _DGD_NAMESPACE,
        "-o",
        "jsonpath={.status}{' finalizers='}{.metadata.finalizers}",
        check=False,
    )
    if result.returncode != 0:
        return result.stderr.strip() or result.stdout.strip()
    return result.stdout.strip()
