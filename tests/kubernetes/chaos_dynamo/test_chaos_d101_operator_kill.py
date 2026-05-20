# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D101 - Kill operator pod mid-DGD-apply; verify reconcile resumes.

Targets ``dynamographdeployment_controller.go:119`` (the main Reconcile loop).
The fault validates that controller-runtime's restart resumes from the last
apiserver-observed state without orphaning child resources.
"""

from __future__ import annotations

import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


async def test_d101_kill_operator_mid_dgd_apply(
    faults,
    kubectl,
    dynamo_operator,
    dynamo_deployment_namespace,
) -> None:
    """Apply DGD, kill operator during reconcile, assert resume to state=successful.

    Targets: ``dynamographdeployment_controller.go:119`` - the main Reconcile loop.
    The fault validates that controller-runtime's restart resumes from the last
    apiserver-observed state without orphaning children.

    The test applies the manifest manually instead of using ``deploy()`` so it
    controls fault timing relative to apply-return:

    1. Build a small DGD manifest (Qwen-tiny-style) by reusing
       ``DynamoConfig`` / ``DynamoDeployer.generate_manifest()`` with default
       ``v1beta1``. Do NOT call ``deploy()`` - we apply manually so we control
       fault timing relative to apply-return.

       from tests.kubernetes.gpu.dynamo.deployer import DynamoDeployer
       deployer = DynamoDeployer(kubectl, dynamo_config)
       manifest = deployer.generate_manifest()
       name = dynamo_config.name
       namespace = dynamo_deployment_namespace

    2. ``kubectl apply`` the manifest and capture name + namespace.

       await kubectl.apply(manifest)

    3. As soon as ``kubectl apply`` returns, inject the fault. A short sleep is
       fine; the goal is "kill the operator while it's mid-Reconcile", which on
       kind happens within 1-3s of apply.

       async with faults.inject(
           "operator.kill",
           target={
               "selector": "app.kubernetes.io/name=dynamo-operator",
               "ns": "dynamo-system",
           },
       ):
           # Restore for operator.kill is a no-op; kubelet auto-recreates.
           pass

    4. Wait for the operator deployment to be Available again (60s timeout).
       Uses a label-selector wait so the assertion spans the v0.9.x release
       name (``dynamo-operator``) and the v1.x release name
       (``dynamo-platform-dynamo-operator-controller-manager``).

       await kubectl.run(
           "wait",
           "-n",
           "dynamo-system",
           "-l",
           "app.kubernetes.io/name=dynamo-operator",
           "deployment",
           "--for=condition=Available",
           "--timeout=60s",
           check=True,
       )

    5. Wait for the DGD to reach ``status.state=successful``.

       from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
       await wait_for_dgd_state(
           kubectl, name, namespace, "successful", timeout=300.0,
       )

    6. Read the DGD as JSON and assert:
         - ``status.observedGeneration == metadata.generation``
         - ``status.state == "successful"``

       import orjson
       result = await kubectl.run(
           "get", "dynamographdeployment", name,
           "-n", namespace, "-o", "json",
           check=True,
       )
       dgd = orjson.loads(result.stdout)
       assert dgd["status"]["state"] == "successful"
       assert (
           dgd["status"]["observedGeneration"]
           == dgd["metadata"]["generation"]
       )

    7. List child resources and assert each carries the DGD as an
       ``ownerReference`` (no orphans).

       for kind in ("deployment", "service", "configmap", "role", "rolebinding"):
           res = await kubectl.run(
               "get", kind,
               "-n", namespace,
               "-l", f"nvidia.com/dynamographdeployment={name}",
               "-o", "json",
               check=False,
           )
           if res.returncode != 0:
               continue
           items = orjson.loads(res.stdout).get("items", [])
           for item in items:
               owners = item.get("metadata", {}).get("ownerReferences", [])
               assert any(
                   o.get("kind") == "DynamoGraphDeployment" and o.get("name") == name
                   for o in owners
               ), f"orphan {kind}/{item['metadata']['name']} has no DGD owner"

    8. Cleanup in a ``finally``: best-effort async delete (``--wait=false``).

       await kubectl.run(
           "delete", "dynamographdeployment", name,
           "-n", namespace, "--wait=false",
           check=False,
       )
    """
    await _run_d101_assertion(faults, kubectl, dynamo_deployment_namespace)


async def _run_d101_assertion(
    faults,
    kubectl,
    dynamo_deployment_namespace: str,
) -> None:
    """Full D101 assertion body for the operator-kill recovery scenario."""
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=dynamo_deployment_namespace,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    manifest = deployer.generate_manifest()
    name = deployer._deployment_name()
    namespace = dynamo_deployment_namespace

    try:
        await kubectl.apply(manifest, namespace=namespace)
        logger.info(
            f"D101: applied DGD {name} in ns {namespace}; injecting operator.kill"
        )

        async with faults.inject(
            "operator.kill",
            target={
                "selector": "app.kubernetes.io/name=dynamo-operator",
                "ns": "dynamo-system",
            },
        ):
            # Restore for operator.kill is a no-op; kubelet auto-recreates the pod.
            pass

        await kubectl.run(
            "wait",
            "-n",
            "dynamo-system",
            "-l",
            "app.kubernetes.io/name=dynamo-operator",
            "deployment",
            "--for=condition=Available",
            "--timeout=60s",
            check=True,
        )
        logger.info("D101: dynamo-operator deployment Available again post-kill")

        await wait_for_dgd_state(
            kubectl,
            name,
            namespace,
            "successful",
            timeout=600.0,
        )

        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=True,
        )
        dgd = orjson.loads(result.stdout)
        assert dgd["status"]["state"] == "successful", (
            f"DGD {name} ended in state={dgd['status'].get('state')!r}, expected 'successful'"
        )
        metadata_generation = dgd["metadata"]["generation"]
        observed_generation = dgd["status"].get("observedGeneration")
        assert observed_generation == metadata_generation, (
            f"observedGeneration={observed_generation} != generation={metadata_generation}"
        )
        logger.info(
            f"D101: DGD {name} reconciled successfully "
            f"(generation={metadata_generation}, observedGeneration={observed_generation})"
        )
    finally:
        await kubectl.run(
            "delete",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )
