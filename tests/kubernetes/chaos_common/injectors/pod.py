# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pod-domain :py:class:`FaultInjector` for the unified chaos interface.

Wraps the existing legacy pod-killing surface
(:py:class:`tests.kubernetes.chaos.chaos_injector.ChaosInjector`) behind the
new :py:class:`FaultInjector` ABC so callers can request pod faults through
the same :py:class:`InjectorRegistry` dispatch path as every other domain.

Three dotted ``fault_id`` values are recognized:

* ``pod.kill`` -- force-delete the pod
  (``kubectl delete pod --force --grace-period=0``). Issued directly through
  the :py:class:`KubectlClient`; legacy ``MockServerInjector.delete_pod`` is
  Deployment-coupled and therefore unsuitable here.
* ``pod.kill_container`` -- kill PID 1 inside a single container of the pod
  via ``kubectl exec ... -- sh -c 'kill -KILL 1'``. Delegates to
  :py:meth:`ChaosInjector.kill_container_in_pod`.
* ``pod.kill_pid`` -- kill a sibling container by PID via shared-PID-namespace
  ``kubectl exec``. Requires ``spec.shareProcessNamespace: true`` on the target
  pod (production default is False -- see ``AIPERF_K8S_SHARE_PROCESS_NAMESPACE``
  and the chaos chart value). Delegates to
  :py:meth:`ChaosInjector.kill_container_by_pid`.

All three faults are **non-restorable**: killed pods are kubelet-recreated
(or not, depending on JobSet policy), but the original PID/container state is
gone. :py:meth:`_AppliedPodFault.restore` is therefore a logged no-op.
"""

from __future__ import annotations

import subprocess
from typing import Any, ClassVar

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.chaos_common.base import (
    AppliedFault,
    FaultInjector,
    FaultMechanismError,
    FaultPreconditionError,
    FaultSpec,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


class _AppliedPodFault(AppliedFault):
    """Handle for a pod-domain fault. ``restore()`` is intentionally a no-op.

    Pod kills are not LIFO-reversible: once the kubelet has SIGKILL'd PID 1
    or the apiserver has accepted a force-delete, the only meaningful
    "restore" is to wait for the kubelet (or JobSet controller) to re-create
    the pod -- which is the system-under-test's responsibility, not the
    injector's. We log the no-op so a misconfigured test reading
    ``metadata['restored']`` still sees a truthy marker.
    """

    async def restore(self) -> None:
        """No-op. Documented in the class docstring."""
        if self.metadata.get("restored"):
            return
        self.metadata["restored"] = True
        logger.debug(
            lambda fid=self.spec.fault_id: (
                f"_AppliedPodFault.restore: no-op for fault_id={fid!r} "
                "(kubelet owns pod re-creation)"
            )
        )


class PodInjector(FaultInjector):
    """Pod-domain :py:class:`FaultInjector` -- delete or kill containers.

    Construct with a shared :py:class:`KubectlClient`; the injector reuses
    legacy :py:class:`ChaosInjector` for ``exec``-based kills so the kubectl
    command shape stays in one place. The legacy helper is instantiated
    lazily per :py:meth:`inject` call -- it is a thin wrapper with no state.
    """

    HANDLES: ClassVar[tuple[str, ...]] = ("pod",)

    def __init__(self, kubectl: KubectlClient) -> None:
        self._kubectl = kubectl

    async def inject(self, spec: FaultSpec) -> AppliedFault:
        if spec.fault_id == "pod.kill":
            return await self._inject_pod_kill(spec)
        if spec.fault_id == "pod.kill_container":
            return await self._inject_pod_kill_container(spec)
        if spec.fault_id == "pod.kill_pid":
            return await self._inject_pod_kill_pid(spec)
        raise FaultPreconditionError(
            f"PodInjector does not recognize fault_id={spec.fault_id!r}; "
            "expected one of pod.kill, pod.kill_container, pod.kill_pid"
        )

    async def _inject_pod_kill(self, spec: FaultSpec) -> AppliedFault:
        pod, namespace = _require_target(spec, "pod", "ns")
        try:
            await self._kubectl.run(
                "delete",
                "pod",
                pod,
                "-n",
                namespace,
                "--force",
                "--grace-period=0",
                check=True,
            )
        except (RuntimeError, subprocess.SubprocessError) as exc:
            # "already gone" is benign at restore time but a precondition
            # failure at inject time -- the test asked us to kill something
            # that does not exist.
            raise FaultMechanismError(
                f"pod.kill: kubectl delete pod {namespace}/{pod} failed: {exc!r}"
            ) from exc
        return _AppliedPodFault(
            spec=spec, metadata={"pod": pod, "namespace": namespace}
        )

    async def _inject_pod_kill_container(self, spec: FaultSpec) -> AppliedFault:
        pod, namespace, container = _require_target(spec, "pod", "ns", "container")
        try:
            await ChaosInjector(self._kubectl).kill_container_in_pod(
                namespace=namespace, pod=pod, container=container
            )
        except (RuntimeError, subprocess.SubprocessError) as exc:
            raise FaultMechanismError(
                f"pod.kill_container: kill -KILL 1 in {namespace}/{pod}/{container} "
                f"failed: {exc!r}"
            ) from exc
        return _AppliedPodFault(
            spec=spec,
            metadata={"pod": pod, "namespace": namespace, "container": container},
        )

    async def _inject_pod_kill_pid(self, spec: FaultSpec) -> AppliedFault:
        pod, namespace, exec_container = _require_target(
            spec, "pod", "ns", "exec_container"
        )
        container_pid = spec.params.get("container_pid")
        if not isinstance(container_pid, int):
            raise FaultPreconditionError(
                "pod.kill_pid: params must include integer 'container_pid' "
                f"(got {container_pid!r})"
            )
        try:
            await ChaosInjector(self._kubectl).kill_container_by_pid(
                pod=pod,
                container_pid=container_pid,
                namespace=namespace,
                exec_container=exec_container,
            )
        except (RuntimeError, subprocess.SubprocessError) as exc:
            raise FaultMechanismError(
                f"pod.kill_pid: kill -9 {container_pid} from "
                f"{namespace}/{pod}/{exec_container} failed: {exc!r}"
            ) from exc
        return _AppliedPodFault(
            spec=spec,
            metadata={
                "pod": pod,
                "namespace": namespace,
                "exec_container": exec_container,
                "container_pid": container_pid,
            },
        )


def _require_target(spec: FaultSpec, *keys: str) -> tuple[Any, ...]:
    """Pull required keys out of ``spec.target`` or raise a precondition error.

    Returns the values in the same order as ``keys`` so callers can unpack.
    """
    missing = [k for k in keys if k not in spec.target]
    if missing:
        raise FaultPreconditionError(
            f"{spec.fault_id}: spec.target is missing required key(s) "
            f"{missing!r}; got target={spec.target!r}"
        )
    return tuple(spec.target[k] for k in keys)
