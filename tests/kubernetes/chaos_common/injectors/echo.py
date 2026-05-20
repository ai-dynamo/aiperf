# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No-op :py:class:`FaultInjector` used to validate registry plumbing.

EchoInjector records every inject/restore call into ``AppliedFault.metadata``
without touching any external resource. It exists so the registry's contract
(dispatch, LIFO compose, restore-on-exception, idempotent restore) can be
covered by fast in-process unit tests before any Kubernetes-backed injector
ships.
"""

from __future__ import annotations

from typing import ClassVar

from tests.kubernetes.chaos_common.base import (
    AppliedFault,
    FaultInjector,
    FaultPreconditionError,
    FaultSpec,
)


class _EchoAppliedFault(AppliedFault):
    """Restore-handle that records its restore in ``metadata['restored_at']``."""

    def __init__(
        self,
        spec: FaultSpec,
        order_log: list[str],
        token: str,
    ) -> None:
        super().__init__(spec=spec, metadata={"order_log": order_log, "token": token})
        self._order_log = order_log
        self._token = token

    async def restore(self) -> None:
        # Tolerate "already gone" by being idempotent: even if called twice
        # (which the base class guards against via _restored), record only
        # the first invocation. The metadata flag lets tests assert exactly
        # one restore happened.
        if self.metadata.get("restored"):
            return
        self.metadata["restored"] = True
        self._order_log.append(f"restore:{self._token}")


class EchoInjector(FaultInjector):
    """No-op injector that records call order into a shared log.

    The shared ``order_log`` list is passed at construction so a test can
    inspect inject/restore interleaving across multiple :py:meth:`inject`
    calls (e.g. to assert LIFO restore in :py:meth:`InjectorRegistry.compose`).

    Pass ``raise_precondition_for`` to make :py:meth:`inject` raise
    :py:class:`FaultPreconditionError` whenever the spec's ``fault_id``
    matches; covers the registry's error-propagation contract.
    """

    HANDLES: ClassVar[tuple[str, ...]] = ("echo",)

    def __init__(
        self,
        order_log: list[str] | None = None,
        raise_precondition_for: str | None = None,
    ) -> None:
        self.order_log: list[str] = order_log if order_log is not None else []
        self._raise_precondition_for = raise_precondition_for

    async def inject(self, spec: FaultSpec) -> AppliedFault:
        if (
            self._raise_precondition_for is not None
            and spec.fault_id == self._raise_precondition_for
        ):
            raise FaultPreconditionError(
                f"EchoInjector configured to fail-fast on fault_id={spec.fault_id!r}"
            )
        # The token lets a test distinguish multiple inject calls on the
        # same injector instance in compose() scenarios.
        token = spec.params.get("token", spec.fault_id)
        self.order_log.append(f"inject:{token}")
        return _EchoAppliedFault(spec=spec, order_log=self.order_log, token=token)
