# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cluster-scoped :py:class:`FaultInjector` for the unified-chaos interface.

Handles fault ids under the ``cluster.`` namespace (see spec §3.4):

* ``cluster.resource_quota`` -- apply / delete a ``ResourceQuota`` on a
  namespace. Restore deletes the quota; the inject site also records a
  :py:class:`recovery.ClusterScopedMutation` so a crashed session can be
  swept via ``pytest --chaos-sweep`` (Phase 1 plumbing).
* ``cluster.network_policy.deny_egress`` -- Phase 3 stub. Dispatch case
  exists so the fault-domain tree compiles; ``inject()`` raises
  :py:class:`NotImplementedError`.
* ``cluster.rbac.revoke`` -- Phase 3 stub, same shape.
"""

from __future__ import annotations

from typing import Any, ClassVar

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.chaos_common import recovery
from tests.kubernetes.chaos_common.base import (
    AppliedFault,
    FaultInjector,
    FaultMechanismError,
    FaultPreconditionError,
    FaultSpec,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


class _ResourceQuotaAppliedFault(AppliedFault):
    """Restore handle for a ``cluster.resource_quota`` injection.

    ``metadata`` keys:

    * ``namespace`` -- target namespace.
    * ``name`` -- ResourceQuota resource name.
    * ``hard_limits`` -- the limits dict that was applied (for diagnostics).
    """

    def __init__(
        self,
        spec: FaultSpec,
        chaos: ChaosInjector,
        namespace: str,
        name: str,
        hard_limits: dict[str, str],
    ) -> None:
        super().__init__(
            spec=spec,
            metadata={
                "namespace": namespace,
                "name": name,
                "hard_limits": dict(hard_limits),
            },
        )
        self._chaos = chaos
        self._namespace = namespace
        self._name = name

    async def restore(self) -> None:
        # `delete_resource_quota` already swallows NotFound via
        # `--ignore-not-found`, so this is idempotent by construction.
        try:
            await self._chaos.delete_resource_quota(self._namespace, self._name)
        except Exception as exc:
            raise FaultMechanismError(
                f"failed to delete ResourceQuota {self._name!r} in "
                f"namespace {self._namespace!r}: {exc!r}"
            ) from exc


class ClusterInjector(FaultInjector):
    """Injector for cluster-scoped fault primitives.

    Currently implements ``cluster.resource_quota``; ``cluster.network_policy.*``
    and ``cluster.rbac.*`` are Phase 3 stubs that raise
    :py:class:`NotImplementedError` so the fault-domain tree (spec §3.4) has
    matching code shape without committing to half-finished behaviour.
    """

    HANDLES: ClassVar[tuple[str, ...]] = ("cluster",)

    def __init__(self, kubectl: KubectlClient) -> None:
        self._kubectl = kubectl

    async def inject(self, spec: FaultSpec) -> AppliedFault:
        if spec.fault_id == "cluster.resource_quota":
            return await self._inject_resource_quota(spec)
        if spec.fault_id == "cluster.network_policy.deny_egress":
            raise NotImplementedError(
                "cluster.network_policy.deny_egress is a Phase 3 stub; "
                "see chaos_common/README.md"
            )
        if spec.fault_id == "cluster.rbac.revoke":
            raise NotImplementedError("cluster.rbac.revoke is a Phase 3 stub")
        raise FaultPreconditionError(
            f"ClusterInjector does not implement fault_id={spec.fault_id!r}"
        )

    async def _inject_resource_quota(self, spec: FaultSpec) -> AppliedFault:
        namespace = self._require(spec.target, "ns", where="spec.target")
        name = self._require(spec.params, "name", where="spec.params")
        hard_limits_raw = self._require(spec.params, "hard_limits", where="spec.params")
        if not isinstance(hard_limits_raw, dict):
            raise FaultPreconditionError(
                "cluster.resource_quota requires spec.params['hard_limits'] "
                f"to be a dict; got {type(hard_limits_raw).__name__}"
            )
        hard_limits: dict[str, str] = dict(hard_limits_raw)

        # Record BEFORE the apply so a crash between record + apply leaves
        # a sweep entry the recovery cache can no-op on (delete with
        # --ignore-not-found). Recovery cache write failures are logged
        # and swallowed so a flaky disk cannot block the test itself.
        try:
            recovery.record_mutation(
                recovery.ClusterScopedMutation(
                    kind="resourcequota",
                    api_version="v1",
                    name=name,
                    op="create",
                    namespace=namespace,
                )
            )
        except Exception as exc:
            logger.warning(
                lambda exc=exc, n=name, ns=namespace: (
                    f"failed to record chaos-sweep mutation for "
                    f"ResourceQuota {n!r} in ns {ns!r}: {exc!r}"
                )
            )

        chaos = ChaosInjector(self._kubectl)
        try:
            await chaos.apply_resource_quota(namespace, name, hard_limits)
        except Exception as exc:
            raise FaultMechanismError(
                f"failed to apply ResourceQuota {name!r} in namespace "
                f"{namespace!r} with hard={hard_limits!r}: {exc!r}"
            ) from exc

        return _ResourceQuotaAppliedFault(
            spec=spec,
            chaos=chaos,
            namespace=namespace,
            name=name,
            hard_limits=hard_limits,
        )

    @staticmethod
    def _require(source: dict[str, Any], key: str, *, where: str) -> Any:
        if key not in source or source[key] is None:
            raise FaultPreconditionError(
                f"missing required field {where}[{key!r}] for cluster.resource_quota"
            )
        return source[key]
