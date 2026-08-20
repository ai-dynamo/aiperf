# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Store-domain :py:class:`FaultInjector` for the unified chaos interface.

Owns the ``store.*`` fault namespace, covering the two backing stores AIPerf
test runs depend on when deployed alongside Dynamo: etcd (StatefulSet) and
NATS (single-replica Deployment) from the ``dynamo-platform`` Helm chart.

Recognized ``fault_id`` values:

* ``store.etcd.kill`` -- ``kubectl delete pod -l <selector> --force
  --grace-period=0`` against the etcd selector. The StatefulSet controller
  re-creates the pod. Non-restorable (kubelet owns recovery).
* ``store.etcd.timeout`` -- Toxiproxy ``timeout`` toxic on the etcd Service
  proxy. Delegated to :py:class:`NetworkInjector`.
* ``store.etcd.bandwidth`` -- Toxiproxy ``bandwidth`` toxic. Delegated.
  **Known-flake risk:** etcd's gRPC channel multiplexes streams over HTTP/2,
  and a bandwidth toxic operating at the byte layer can produce surprising
  cascading failures (stream head-of-line blocking, client-side timeouts that
  do not match the configured rate). D802 should prefer ``store.etcd.timeout``
  if ``store.etcd.bandwidth`` proves flaky in CI.
* ``store.etcd.partition`` -- full proxy disable (delegated).
* ``store.nats.kill`` -- pod force-delete against the NATS selector.
* ``store.nats.partition`` -- full proxy disable (delegated).
* ``store.nats.slow_close`` -- Toxiproxy ``slow_close`` toxic (delegated).

The Toxiproxy faults delegate to an internal :py:class:`NetworkInjector` so
toxic-add / proxy-disable mechanics stay in one place; this injector only
rewrites the ``fault_id`` from ``store.<svc>.<verb>`` to ``network.<verb>``
and forwards the original ``target`` and ``params`` unchanged.

The ctor selectors and namespaces default to dynamo-platform conventions
(``dynamo-system`` namespace, ``app.kubernetes.io/name=etcd`` /
``app=nats``) but accept overrides so AIPerf can reuse this injector if it
ever bundles its own etcd/NATS.
"""

from __future__ import annotations

import subprocess
from typing import ClassVar

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos.toxiproxy import ToxiproxyInjector
from tests.kubernetes.chaos_common.base import (
    AppliedFault,
    FaultInjector,
    FaultMechanismError,
    FaultPreconditionError,
    FaultSpec,
)
from tests.kubernetes.chaos_common.injectors.network import NetworkInjector
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


_STORE_TO_NETWORK_FAULT_ID: dict[str, str] = {
    "store.etcd.timeout": "network.timeout",
    "store.etcd.bandwidth": "network.bandwidth",
    "store.etcd.partition": "network.partition",
    "store.nats.partition": "network.partition",
    "store.nats.slow_close": "network.slow_close",
}
"""Map ``store.*`` Toxiproxy-backed fault_ids to their ``network.*`` equivalents."""


class _AppliedStoreKill(AppliedFault):
    """Restore handle for ``store.{etcd,nats}.kill``. ``restore()`` is a no-op.

    The store StatefulSet / Deployment controller is responsible for re-creating
    the pod; the injector has no LIFO-reversible state to unwind. We log the
    no-op so a misconfigured test reading ``metadata['restored']`` still sees
    a truthy marker.
    """

    async def restore(self) -> None:
        if self.metadata.get("restored"):
            return
        self.metadata["restored"] = True
        logger.debug(
            lambda fid=self.spec.fault_id: (
                f"_AppliedStoreKill.restore: no-op for fault_id={fid!r} "
                "(StatefulSet/Deployment controller owns pod re-creation)"
            )
        )


class StoreInjector(FaultInjector):
    """Store-domain :py:class:`FaultInjector` -- etcd + NATS faults.

    Pod-kill faults issue a labeled ``kubectl delete`` directly. Network-shape
    faults (timeout, bandwidth, partition, slow_close) are delegated to an
    internal :py:class:`NetworkInjector` instance, which keeps the Toxiproxy
    REST surface in exactly one place.
    """

    HANDLES: ClassVar[tuple[str, ...]] = ("store",)

    def __init__(
        self,
        kubectl: KubectlClient,
        toxiproxy: ToxiproxyInjector,
        *,
        etcd_namespace: str = "dynamo-system",
        etcd_selector: str = "app.kubernetes.io/name=etcd",
        nats_namespace: str = "dynamo-system",
        nats_selector: str = "app=nats",
    ) -> None:
        self._kubectl = kubectl
        self._toxiproxy = toxiproxy
        self._etcd_namespace = etcd_namespace
        self._etcd_selector = etcd_selector
        self._nats_namespace = nats_namespace
        self._nats_selector = nats_selector
        # Single shared NetworkInjector keeps Toxiproxy mechanics centralized.
        self._network = NetworkInjector(toxiproxy)

    async def inject(self, spec: FaultSpec) -> AppliedFault:
        if spec.fault_id == "store.etcd.kill":
            return await self._inject_pod_kill(
                spec,
                default_namespace=self._etcd_namespace,
                default_selector=self._etcd_selector,
            )
        if spec.fault_id == "store.nats.kill":
            return await self._inject_pod_kill(
                spec,
                default_namespace=self._nats_namespace,
                default_selector=self._nats_selector,
            )
        if spec.fault_id in _STORE_TO_NETWORK_FAULT_ID:
            return await self._delegate_to_network(spec)
        raise FaultPreconditionError(
            f"StoreInjector does not recognize fault_id={spec.fault_id!r}; "
            f"expected one of store.etcd.{{kill,timeout,bandwidth,partition}}, "
            "store.nats.{kill,partition,slow_close}"
        )

    async def _inject_pod_kill(
        self,
        spec: FaultSpec,
        *,
        default_namespace: str,
        default_selector: str,
    ) -> AppliedFault:
        namespace = spec.target.get("ns", default_namespace)
        selector = spec.target.get("selector", default_selector)
        try:
            await self._kubectl.run(
                "delete",
                "pod",
                "-l",
                selector,
                "-n",
                namespace,
                "--force",
                "--grace-period=0",
                check=True,
            )
        except (RuntimeError, subprocess.SubprocessError) as exc:
            raise FaultMechanismError(
                f"{spec.fault_id}: kubectl delete pod -l {selector} -n "
                f"{namespace} failed: {exc!r}"
            ) from exc
        return _AppliedStoreKill(
            spec=spec,
            metadata={"namespace": namespace, "selector": selector},
        )

    async def _delegate_to_network(self, spec: FaultSpec) -> AppliedFault:
        network_fault_id = _STORE_TO_NETWORK_FAULT_ID[spec.fault_id]
        synthetic = FaultSpec(
            fault_id=network_fault_id,
            params=spec.params,
            target=spec.target,
        )
        # NetworkInjector returns an AppliedFault whose restore() already
        # reverses the Toxiproxy mutation; we return it directly so the
        # registry's LIFO unwind unwinds the underlying toxic / proxy state.
        return await self._network.inject(synthetic)
