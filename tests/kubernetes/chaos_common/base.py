# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ABC + supporting types for the unified chaos-injection interface.

Concrete injectors MAY expose `wait_for_*` helpers; they are not part of this
ABC (open question Q1 resolved as "concrete-only" so the surface stays minimal).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, ClassVar


class FaultPreconditionError(RuntimeError):
    """The target was not in a state where the fault could be applied.

    Examples: CR missing, pod not ready, Toxiproxy unreachable. Tests SHOULD
    surface this as a clear failure rather than retry blindly.
    """


class FaultMechanismError(RuntimeError):
    """The underlying mutation mechanism itself failed.

    Examples: kubectl returned non-zero, a Toxiproxy REST POST returned 5xx,
    a process kill signal could not be delivered. Distinct from
    `FaultPreconditionError` so callers can decide whether to retry vs. abort.
    """


@dataclass(frozen=True)
class FaultSpec:
    """Identity + parameters for one fault application.

    Attributes:
        fault_id: Dotted name from the fault-domain tree (spec §3.4).
        params: Keyword arguments passed verbatim to the resolving injector.
        target: Opaque addressing payload (pod name, proxy name, deployment
            ref, ...). Each concrete injector parses its own shape.
    """

    fault_id: str
    params: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)


class AppliedFault(AbstractAsyncContextManager["AppliedFault"], ABC):
    """Handle to a single fault that has been (or will be) injected.

    Async context manager: ``__aenter__`` is a no-op (the inject already
    happened in :py:meth:`FaultInjector.inject`); ``__aexit__`` calls
    :py:meth:`restore` regardless of exception state. The registry's
    :py:meth:`InjectorRegistry.compose` builder relies on
    :py:class:`contextlib.AsyncExitStack` for LIFO ordering across multiple
    faults.

    Subclasses must populate ``metadata`` with enough state to restore the
    mutation -- mirrors AIPerf's ``_AppliedOp`` pattern in
    ``chaos/mock_server_injector.py`` but lifted to the ABC.

    Per spec open-question Q2, ``metadata`` is ``dict[str, Any]``; concrete
    injectors MAY store their own ``@dataclass`` op-tracker under a known key.
    """

    def __init__(
        self,
        spec: FaultSpec,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}
        # `_restored` guards against the registry calling restore() twice on
        # the same handle (e.g. if a test code path enters __aexit__ manually
        # and the AsyncExitStack also unwinds it).
        self._restored: bool = False

    async def __aenter__(self) -> AppliedFault:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._restored:
            self._restored = True
            await self.restore()

    @abstractmethod
    async def restore(self) -> None:
        """Reverse the mutation. Idempotent. Tolerates 'already gone'.

        Raises:
            FaultMechanismError: if the restore mechanism fails in a way
                that suggests cluster damage. Callers in the registry should
                log + swallow so the original test exception is not masked.
        """


class FaultInjector(ABC):
    """Resolve a :py:class:`FaultSpec` and produce an :py:class:`AppliedFault`.

    Subclasses declare which ``fault_id`` namespace prefixes they own via the
    class-level :py:attr:`HANDLES` tuple. The registry uses this for dispatch.
    """

    HANDLES: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    async def inject(self, spec: FaultSpec) -> AppliedFault:
        """Perform the mutation and return the restore handle.

        Raises:
            FaultPreconditionError: when the target is in an unexpected state
                (CR missing, pod not ready, Toxiproxy not reachable).
            FaultMechanismError: when the underlying mechanism fails
                (kubectl returned non-zero, REST POST returned 5xx).
        """

    @classmethod
    def handles(cls, fault_id: str) -> bool:
        """True when ``fault_id`` equals or is a dotted sub-path of any prefix."""
        return any(
            fault_id == prefix or fault_id.startswith(prefix + ".")
            for prefix in cls.HANDLES
        )
