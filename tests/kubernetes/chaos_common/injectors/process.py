# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-domain :py:class:`FaultInjector` for the unified chaos interface.

Signals a specific PID inside a target container via ``kubectl exec``. This is
the Kubernetes-side analog of dynamo's ``os.kill(rank_pid, signal.SIGSTOP)``
pattern in ``tests/fault_tolerance/test_canary_rank_pause.py``: the engine PID
lives inside a worker pod and we discover + signal it via kubectl exec.

One dotted ``fault_id`` is recognized:

* ``process.signal`` -- send the named signal to ``spec.target['pid']`` inside
  ``spec.target['container']`` of ``spec.target['pod']``.

Spec shape:

* ``spec.target['kind']`` -- must be ``"pod"`` (only supported addressing kind
  in this phase). ``"managed_process"`` is reserved for the deferred Phase per
  the unified chaos design spec §7.
* ``spec.target['ns']`` -- namespace.
* ``spec.target['pod']`` -- pod name.
* ``spec.target['container']`` -- container name inside the pod. Required; no
  default because different SUTs use different container names.
* ``spec.target['pid']`` -- integer PID to signal (visible inside the container
  PID namespace).
* ``spec.params['signal']`` -- signal name like ``"SIGSTOP"``, ``"SIGCONT"``,
  ``"SIGTERM"``, ``"SIGKILL"``, ``"SIGUSR1"``. Validated against the shape
  ``^SIG[A-Z][A-Z0-9]*$``; the exact Linux signal set is not hard-coded since
  available signals vary by kernel / libc.

Restore semantics depend on the signal sent:

* ``SIGSTOP`` -- restore sends ``SIGCONT`` to the same PID. Leaving the process
  stopped permanently would break the cluster, so restore IS meaningful here.
* ``SIGTERM`` / ``SIGKILL`` -- process is gone; restore is a logged no-op
  (kubelet owns container re-creation, same as :py:mod:`pod` faults).
* Any other signal (``SIGUSR1``, ``SIGUSR2``, ``SIGHUP``, ...) -- no-op
  restore. We do not attempt to "undo" an arbitrary userspace signal.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, ClassVar

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.base import (
    AppliedFault,
    FaultInjector,
    FaultMechanismError,
    FaultPreconditionError,
    FaultSpec,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)

_SIGNAL_NAME_RE = re.compile(r"^SIG[A-Z][A-Z0-9]*$")

# Maps the signal we sent at inject() to the signal that reverses it at
# restore(). ``None`` means "no-op restore"; signals not present in the map
# also fall through to no-op (e.g. SIGUSR1, SIGHUP).
_RESTORE_SIGNAL_MAP: dict[str, str | None] = {
    "SIGSTOP": "SIGCONT",
    "SIGTERM": None,
    "SIGKILL": None,
}


class _AppliedProcessFault(AppliedFault):
    """Restore handle for ``process.signal``.

    Sends the mapped restore signal (e.g. ``SIGCONT`` for a prior ``SIGSTOP``)
    via the same ``kubectl exec ... -- kill`` mechanism used at inject time.
    Idempotent: a second call is a no-op once ``metadata['restored']`` is set.
    """

    def __init__(
        self,
        spec: FaultSpec,
        kubectl: KubectlClient,
        ns: str,
        pod: str,
        container: str,
        pid: int,
        signal_name: str,
    ) -> None:
        super().__init__(
            spec=spec,
            metadata={
                "namespace": ns,
                "pod": pod,
                "container": container,
                "pid": pid,
                "signal": signal_name,
            },
        )
        self._kubectl = kubectl
        self._ns = ns
        self._pod = pod
        self._container = container
        self._pid = pid
        self._signal_name = signal_name

    async def restore(self) -> None:
        if self.metadata.get("restored"):
            return
        self.metadata["restored"] = True

        restore_signal = _RESTORE_SIGNAL_MAP.get(self._signal_name, None)
        if restore_signal is None:
            logger.debug(
                lambda sig=self._signal_name, pid=self._pid: (
                    f"_AppliedProcessFault.restore: no-op for signal={sig!r} "
                    f"pid={pid} (no reverse signal defined)"
                )
            )
            return

        try:
            await self._kubectl.run(
                "exec",
                self._pod,
                "-c",
                self._container,
                "-n",
                self._ns,
                "--",
                "kill",
                f"-{restore_signal}",
                str(self._pid),
                check=True,
            )
        except (RuntimeError, subprocess.SubprocessError) as exc:
            raise FaultMechanismError(
                f"process.signal restore: kubectl exec kill -{restore_signal} "
                f"{self._pid} in {self._ns}/{self._pod}/{self._container} "
                f"failed: {exc!r}"
            ) from exc


class ProcessInjector(FaultInjector):
    """Process-domain :py:class:`FaultInjector` -- signal a PID inside a pod.

    See module docstring for spec shape and restore semantics.
    """

    HANDLES: ClassVar[tuple[str, ...]] = ("process",)

    def __init__(self, kubectl: KubectlClient) -> None:
        self._kubectl = kubectl

    async def inject(self, spec: FaultSpec) -> AppliedFault:
        if spec.fault_id != "process.signal":
            raise FaultPreconditionError(
                f"ProcessInjector does not recognize fault_id={spec.fault_id!r}; "
                "expected process.signal"
            )

        kind, ns, pod, container, pid = _require_target(
            spec, "kind", "ns", "pod", "container", "pid"
        )

        if kind != "pod":
            raise FaultPreconditionError(
                f"process.signal: spec.target['kind']={kind!r} is unsupported. "
                "Only 'pod' is implemented in this phase; 'managed_process' is "
                "deferred per unified chaos spec §7 (Phase: ManagedProcess "
                "integration)."
            )

        if not isinstance(pid, int):
            raise FaultPreconditionError(
                f"process.signal: spec.target['pid'] must be int (got {pid!r})"
            )

        signal_name = spec.params.get("signal")
        if signal_name is None:
            raise FaultPreconditionError(
                "process.signal: spec.params['signal'] is required "
                "(e.g. 'SIGSTOP', 'SIGCONT', 'SIGTERM')"
            )
        if not isinstance(signal_name, str) or not _SIGNAL_NAME_RE.match(signal_name):
            raise FaultPreconditionError(
                f"process.signal: spec.params['signal']={signal_name!r} does not "
                "match expected shape ^SIG[A-Z][A-Z0-9]*$"
            )

        try:
            await self._kubectl.run(
                "exec",
                pod,
                "-c",
                container,
                "-n",
                ns,
                "--",
                "kill",
                f"-{signal_name}",
                str(pid),
                check=True,
            )
        except (RuntimeError, subprocess.SubprocessError) as exc:
            raise FaultMechanismError(
                f"process.signal: kubectl exec kill -{signal_name} {pid} in "
                f"{ns}/{pod}/{container} failed: {exc!r}"
            ) from exc

        return _AppliedProcessFault(
            spec=spec,
            kubectl=self._kubectl,
            ns=ns,
            pod=pod,
            container=container,
            pid=pid,
            signal_name=signal_name,
        )


def _require_target(spec: FaultSpec, *keys: str) -> tuple[Any, ...]:
    """Pull required keys out of ``spec.target`` or raise a precondition error."""
    missing = [k for k in keys if k not in spec.target]
    if missing:
        raise FaultPreconditionError(
            f"{spec.fault_id}: spec.target is missing required key(s) "
            f"{missing!r}; got target={spec.target!r}"
        )
    return tuple(spec.target[k] for k in keys)
