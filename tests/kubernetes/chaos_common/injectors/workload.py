# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workload-level :py:class:`FaultInjector` for Deployment-targeted faults.

Adapts :py:class:`tests.kubernetes.chaos.mock_server_injector.MockServerInjector`
(the legacy Deployment chaos driver) onto the unified
:py:class:`FaultInjector` ABC.

Each :py:meth:`WorkloadInjector.inject` call constructs a per-spec
:py:class:`MockServerInjector` so the LIFO restore stack stays scoped to a
single :py:class:`AppliedFault` -- the legacy class accumulates ops in an
instance list, which is exactly the lifetime we want when restore lives on
the returned handle.

Handled ``fault_id`` values (target shape ``{"ns": str, "deployment": str}``):

* ``workload.restart`` -- ``kubectl rollout restart deploy/<name>``. Restore
  strips the ``kubectl.kubernetes.io/restartedAt`` annotation.
* ``workload.rolling_upgrade`` -- same kubectl mechanism as ``restart``, but
  the distinct ``fault_id`` lets a test assert "this models a rolling upgrade,
  not a kill" (mirrors dynamo's ``RollingUpgradeFailure`` scenario).
* ``workload.scale`` -- ``kubectl scale --replicas=<n>``. Params:
  ``{"replicas": int}``. Restore returns to the prior replica count.
* ``workload.set_env`` -- ``kubectl set env``. Params:
  ``{"env_var": str, "value": str}``. Restore strips the env var.
"""

from __future__ import annotations

from typing import Any, ClassVar

from tests.kubernetes.chaos.mock_server_injector import MockServerInjector
from tests.kubernetes.chaos_common.base import (
    AppliedFault,
    FaultInjector,
    FaultMechanismError,
    FaultPreconditionError,
    FaultSpec,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

_RESTART_FAULT_IDS: frozenset[str] = frozenset(
    {"workload.restart", "workload.rolling_upgrade"}
)


class _AppliedWorkloadFault(AppliedFault):
    """Restore handle that delegates LIFO undo to a per-spec MockServerInjector."""

    def __init__(
        self,
        spec: FaultSpec,
        legacy: MockServerInjector,
    ) -> None:
        super().__init__(spec=spec, metadata={"legacy": legacy})
        self._legacy = legacy

    async def restore(self) -> None:
        # MockServerInjector.restore drains its own _applied_ops list in LIFO
        # order and already swallows per-op exceptions, so we don't need to
        # re-wrap. The base class _restored guard prevents double invocation.
        await self._legacy.restore()


class WorkloadInjector(FaultInjector):
    """Dispatch Deployment-targeted faults onto :py:class:`MockServerInjector`.

    The ``kubectl`` client is supplied once at construction and reused for
    every spec; each :py:meth:`inject` call gets its own
    :py:class:`MockServerInjector` so the LIFO op-stack belongs to the returned
    :py:class:`AppliedFault` (one handle == one restore scope).
    """

    HANDLES: ClassVar[tuple[str, ...]] = ("workload",)

    def __init__(self, kubectl: KubectlClient) -> None:
        self._kubectl = kubectl

    async def inject(self, spec: FaultSpec) -> AppliedFault:
        namespace, deployment = _require_target(spec)
        legacy = MockServerInjector(self._kubectl)

        try:
            if spec.fault_id in _RESTART_FAULT_IDS:
                await legacy.restart(namespace=namespace, deployment=deployment)
            elif spec.fault_id == "workload.scale":
                replicas = _require_int_param(spec, "replicas")
                await legacy.scale(
                    namespace=namespace,
                    replicas=replicas,
                    deployment=deployment,
                )
            elif spec.fault_id == "workload.set_env":
                env_var = _require_str_param(spec, "env_var")
                value = _require_str_param(spec, "value")
                await legacy.patch_env(
                    namespace=namespace,
                    env_var=env_var,
                    value=value,
                    deployment=deployment,
                )
            else:
                raise FaultPreconditionError(
                    f"WorkloadInjector does not handle fault_id={spec.fault_id!r}; "
                    f"known: workload.restart, workload.rolling_upgrade, "
                    f"workload.scale, workload.set_env"
                )
        except (FaultPreconditionError, FaultMechanismError):
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as FaultMechanismError below
            raise FaultMechanismError(
                f"WorkloadInjector.inject({spec.fault_id!r}) failed on "
                f"ns={namespace!r} deployment={deployment!r}: {exc!r}"
            ) from exc

        return _AppliedWorkloadFault(spec=spec, legacy=legacy)


def _require_target(spec: FaultSpec) -> tuple[str, str]:
    """Pull ``ns`` and ``deployment`` out of ``spec.target`` or raise."""
    namespace = spec.target.get("ns")
    deployment = spec.target.get("deployment")
    if not isinstance(namespace, str) or not namespace:
        raise FaultPreconditionError(
            f"WorkloadInjector requires spec.target['ns'] (got {namespace!r}) "
            f"for fault_id={spec.fault_id!r}"
        )
    if not isinstance(deployment, str) or not deployment:
        raise FaultPreconditionError(
            f"WorkloadInjector requires spec.target['deployment'] "
            f"(got {deployment!r}) for fault_id={spec.fault_id!r}"
        )
    return namespace, deployment


def _require_str_param(spec: FaultSpec, key: str) -> str:
    value: Any = spec.params.get(key)
    if not isinstance(value, str) or not value:
        raise FaultPreconditionError(
            f"WorkloadInjector fault_id={spec.fault_id!r} requires "
            f"spec.params[{key!r}] as non-empty str (got {value!r})"
        )
    return value


def _require_int_param(spec: FaultSpec, key: str) -> int:
    value: Any = spec.params.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise FaultPreconditionError(
            f"WorkloadInjector fault_id={spec.fault_id!r} requires "
            f"spec.params[{key!r}] as int (got {value!r})"
        )
    return value
