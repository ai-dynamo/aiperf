# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fault injection for the k8s mock-server deployment.

The k8s test harness runs a single-replica ``aiperf-mock-server`` Deployment
(see ``tests/kubernetes/conftest.py``); this module patches / restarts /
scales / env-overrides it to exercise benchmark-runtime chaos (B1-B3 in
``docs/superpowers/specs/2026-04-23-chaos-expansion-design.md``).

Every fault method records the mutation in ``self._applied_ops`` so
``restore()`` can reverse them in LIFO order between tests.

Usage::

    async def test_mock_server_500s(
        mock_server_injector: MockServerInjector,
    ) -> None:
        await mock_server_injector.patch_env(
            "default", "AIPERF_MOCK_FORCE_STATUS", "500"
        )
        # ... drive benchmark, assert error metrics ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import orjson

from tests.kubernetes.helpers.kubectl import KubectlClient

logger = logging.getLogger(__name__)

DEFAULT_MOCK_SERVER_DEPLOYMENT = "aiperf-mock-server"
"""Deployment name used by ``tests/kubernetes/conftest.py::mock_server``."""

DEFAULT_MOCK_SERVER_NAMESPACE = "default"
"""Namespace the mock server runs in for k8s tests."""


@dataclass
class _AppliedOp:
    """Record of a single fault applied to the mock server.

    Enables ``restore()`` to reverse mutations in LIFO order without the
    caller having to remember which APIs to call.
    """

    kind: str
    """One of ``env``, ``scale``, ``restart-annotation``."""

    payload: dict[str, str | int] = field(default_factory=dict)
    """Op-specific metadata (env var name, prior replica count, ...)."""


class MockServerInjector:
    """Drive chaos against the single-replica mock-server Deployment.

    All methods are async. Each mutation records an entry in
    ``self._applied_ops`` so ``restore()`` can roll back every fault
    injected during a single test without the caller tracking state.
    """

    def __init__(self, kubectl: KubectlClient) -> None:
        """Initialize the injector.

        Args:
            kubectl: Async kubectl wrapper pinned to the test cluster.
        """
        self.kubectl = kubectl
        self._applied_ops: list[_AppliedOp] = []

    async def restart(
        self,
        namespace: str = DEFAULT_MOCK_SERVER_NAMESPACE,
        deployment: str = DEFAULT_MOCK_SERVER_DEPLOYMENT,
    ) -> None:
        """Trigger a rolling restart via ``kubectl rollout restart``.

        Records the operation so ``restore()`` knows the deployment was
        perturbed even though no state mutation needs to be undone.
        """
        await self.kubectl.run(
            "rollout",
            "restart",
            f"deployment/{deployment}",
            "-n",
            namespace,
            check=True,
        )
        self._applied_ops.append(
            _AppliedOp(
                kind="restart-annotation",
                payload={"namespace": namespace, "deployment": deployment},
            )
        )

    async def delete_pod(
        self,
        namespace: str = DEFAULT_MOCK_SERVER_NAMESPACE,
        deployment: str = DEFAULT_MOCK_SERVER_DEPLOYMENT,
    ) -> None:
        """Force-delete the mock-server pod; the Deployment will respawn it.

        Uses ``--grace-period=0 --force`` to simulate an ungraceful crash
        (no SIGTERM window). The Deployment controller replaces the pod
        within the usual reconcile budget.
        """
        await self.kubectl.run(
            "delete",
            "pod",
            "-n",
            namespace,
            "-l",
            f"app={deployment}",
            "--grace-period=0",
            "--force",
            "--ignore-not-found",
            check=False,
        )

    async def scale(
        self,
        namespace: str,
        replicas: int,
        deployment: str = DEFAULT_MOCK_SERVER_DEPLOYMENT,
    ) -> None:
        """Scale the deployment to ``replicas``.

        Records the prior replica count so ``restore()`` can revert to
        exactly one replica (the harness default).
        """
        current = await self.kubectl.run(
            "get",
            "deployment",
            deployment,
            "-n",
            namespace,
            "-o",
            "jsonpath={.spec.replicas}",
            check=True,
        )
        prior_replicas = int(current.stdout.strip() or "1")
        await self.kubectl.run(
            "scale",
            f"deployment/{deployment}",
            f"--replicas={replicas}",
            "-n",
            namespace,
            check=True,
        )
        self._applied_ops.append(
            _AppliedOp(
                kind="scale",
                payload={
                    "namespace": namespace,
                    "deployment": deployment,
                    "prior_replicas": prior_replicas,
                },
            )
        )

    async def patch_env(
        self,
        namespace: str,
        env_var: str,
        value: str,
        deployment: str = DEFAULT_MOCK_SERVER_DEPLOYMENT,
    ) -> None:
        """Add or update an env var on the mock-server container.

        Uses ``kubectl set env`` so the Deployment rolls forward with the
        new value; kubelet restarts the pod automatically.

        Args:
            namespace: Deployment namespace.
            env_var: Env var name, e.g. ``AIPERF_MOCK_FORCE_STATUS``.
            value: Value to set.
            deployment: Deployment name.
        """
        await self.kubectl.run(
            "set",
            "env",
            f"deployment/{deployment}",
            f"{env_var}={value}",
            "-n",
            namespace,
            check=True,
        )
        self._applied_ops.append(
            _AppliedOp(
                kind="env",
                payload={
                    "namespace": namespace,
                    "deployment": deployment,
                    "env_var": env_var,
                },
            )
        )

    async def restore(self) -> None:
        """Reverse every fault applied during the current test.

        Called from the ``mock_server_injector`` fixture teardown. Safe
        to call even when no mutations were applied.
        """
        while self._applied_ops:
            op = self._applied_ops.pop()
            try:
                await self._undo(op)
            except Exception as exc:
                logger.warning(
                    "MockServerInjector.restore: undo %s failed: %s; remaining ops=%d",
                    op.kind,
                    exc,
                    len(self._applied_ops),
                )

    async def _undo(self, op: _AppliedOp) -> None:
        """Reverse a single applied mutation."""
        if op.kind == "env":
            namespace = str(op.payload["namespace"])
            deployment = str(op.payload["deployment"])
            env_var = str(op.payload["env_var"])
            await self.kubectl.run(
                "set",
                "env",
                f"deployment/{deployment}",
                f"{env_var}-",
                "-n",
                namespace,
                check=False,
            )
        elif op.kind == "scale":
            namespace = str(op.payload["namespace"])
            deployment = str(op.payload["deployment"])
            prior = int(op.payload["prior_replicas"])
            await self.kubectl.run(
                "scale",
                f"deployment/{deployment}",
                f"--replicas={prior}",
                "-n",
                namespace,
                check=False,
            )
        elif op.kind == "restart-annotation":
            namespace = str(op.payload["namespace"])
            deployment = str(op.payload["deployment"])
            # Strip kubectl's restartedAt annotation so the deployment
            # reverts to its original podTemplate hash.
            patch = orjson.dumps(
                {
                    "spec": {
                        "template": {
                            "metadata": {
                                "annotations": {
                                    "kubectl.kubernetes.io/restartedAt": None
                                }
                            }
                        }
                    }
                }
            ).decode()
            await self.kubectl.run(
                "patch",
                "deployment",
                deployment,
                "-n",
                namespace,
                "--type=strategic",
                "-p",
                patch,
                check=False,
            )
