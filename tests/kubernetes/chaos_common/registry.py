# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""InjectorRegistry: dispatch by `fault_id`, compose with LIFO restore.

Discovery is explicit (`register()` calls inside fixtures). Decorator-based
auto-discovery was rejected -- too much magic for a test harness.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.base import (
    AppliedFault,
    FaultInjector,
    FaultSpec,
)

logger = AIPerfLogger(__name__)


class InjectorRegistry:
    """Holds the set of :py:class:`FaultInjector` instances available to a test.

    A new registry is constructed per-test by the `faults` fixture so that
    state (registered injectors, in-flight handles) does not bleed between
    cases.
    """

    def __init__(self) -> None:
        self._injectors: list[FaultInjector] = []

    def register(self, injector: FaultInjector) -> None:
        """Add an injector to the dispatch table.

        Order matters: the first injector whose :py:meth:`FaultInjector.handles`
        returns True wins. Tests that need overrides should register the more
        specific injector first.
        """
        self._injectors.append(injector)

    def resolve(self, fault_id: str) -> FaultInjector:
        """Return the first registered injector that handles ``fault_id``.

        Raises:
            LookupError: when no registered injector claims the id; the
                message names every registered injector + its HANDLES tuple
                to short-circuit "which test imported the wrong fixture?".
        """
        for inj in self._injectors:
            if inj.handles(fault_id):
                return inj
        raise LookupError(
            f"no FaultInjector registered for {fault_id!r}; registered: "
            f"{[i.__class__.__name__ + str(i.HANDLES) for i in self._injectors]}"
        )

    @asynccontextmanager
    async def inject(
        self,
        fault_id: str,
        **params: Any,
    ) -> AsyncIterator[AppliedFault]:
        """Apply one fault for the lifetime of the ``async with`` block.

        Pops ``target`` out of ``params`` (so callers can pass it as a kwarg
        instead of having to construct a :py:class:`FaultSpec` manually).
        """
        target = params.pop("target", {})
        spec = FaultSpec(fault_id=fault_id, params=params, target=target)
        injector = self.resolve(fault_id)
        applied = await injector.inject(spec)
        try:
            yield applied
        finally:
            # AppliedFault.__aexit__ owns the idempotency guard via its
            # internal _restored flag; calling it here keeps the registry
            # off the private attribute and lets a caller who wrapped
            # `applied` in their own `async with` block run restore exactly
            # once. Per spec §5, restore failures must not mask the
            # original test exception -- log loudly and swallow.
            try:
                await applied.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning(
                    lambda exc=exc, fid=fault_id: (
                        f"restore failed for fault {fid!r}: {exc!r}"
                    )
                )

    @asynccontextmanager
    async def compose(
        self,
        *fault_specs: tuple[str, dict[str, Any]],
    ) -> AsyncIterator[list[AppliedFault]]:
        """Apply N faults atomically; restore in LIFO order on exit.

        Args:
            *fault_specs: A sequence of ``(fault_id, params_dict)`` tuples.

        Yields:
            A list of :py:class:`AppliedFault` handles, in inject order. The
            underlying :py:class:`contextlib.AsyncExitStack` guarantees that
            ``restore()`` is invoked in reverse order on block exit, even when
            the body raises.
        """
        async with AsyncExitStack() as stack:
            handles: list[AppliedFault] = []
            for fault_id, params in fault_specs:
                applied = await stack.enter_async_context(
                    self.inject(fault_id, **params)
                )
                handles.append(applied)
            yield handles
