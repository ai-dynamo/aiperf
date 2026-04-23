# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from typing import Any

from aiperf.common.exceptions import (
    ServiceProcessDiedError,
    ServiceRegistrationTimeoutError,
)
from aiperf.common.mixins.aiperf_logger_mixin import AIPerfLoggerMixin
from aiperf.common.models import ServiceRunInfo
from aiperf.common.types import ServiceTypeT


class _ServiceRegistryWaitMixin(AIPerfLoggerMixin):
    """Async-waiting and failure-reporting behavior for the service registry.

    Split out of ``_ServiceRegistry`` purely to keep file size within the
    ergonomics limit. All attributes are owned by the concrete registry; this
    mixin only provides async wait and raise/log helpers.
    """

    # -- Attributes (owned by _ServiceRegistry; declared here for type readers) --
    services: dict[str, ServiceRunInfo]
    expected_by_type: dict[ServiceTypeT, int]
    expected_ids: set[str]
    _first_expected_at: float | None
    _all_event: asyncio.Event | None
    _type_events: dict[ServiceTypeT, asyncio.Event]
    _id_events: dict[frozenset[str], asyncio.Event]
    _failure_errors: list[ServiceProcessDiedError]

    _PROGRESS_LOG_INTERVAL: float = 5.0

    # -- Async waiting --

    async def wait_for_all(self, timeout: float | None = None) -> None:
        """Wait until all expected services are registered.

        Raises:
            ServiceProcessDiedError: If a required service process dies while waiting.
            ServiceRegistrationTimeoutError: If services don't register within timeout.
        """
        if self.all_registered():
            self._log_all_registered()
            return
        self._raise_on_failure()

        # Always create a fresh event to avoid stale state from prior waits
        self._all_event = asyncio.Event()

        # Re-check after creating the event to close the race window where
        # a service registers between the check above and event creation
        if self.all_registered():
            self._log_all_registered()
            return

        self._log_waiting_for()
        await self._wait_with_progress(
            self._all_event, timeout, "all services to register"
        )

    async def wait_for_type(
        self, service_type: ServiceTypeT, timeout: float | None = None
    ) -> None:
        """Wait until all services of a specific type are registered.

        Raises:
            ServiceProcessDiedError: If a required service process dies while waiting.
            ServiceRegistrationTimeoutError: If services don't register within timeout.
        """
        if self.all_types_registered(service_type):
            return
        self._raise_on_failure()

        event = self._type_events.setdefault(service_type, asyncio.Event())
        expected = self.expected_by_type.get(service_type, 0)
        registered = self._num_registered_of_type(service_type)
        self.info(
            f"Waiting for {service_type.title()} services to be registered ({registered}/{expected})..."
        )
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            self._raise_timeout(
                f"Timed out waiting for {service_type.title()} services to register"
            )
        self._raise_on_failure()
        if not self.all_types_registered(service_type):
            self._raise_timeout(
                f"Not all {service_type.title()} services registered after waking"
            )

    async def wait_for_ids(
        self, service_ids: list[str], timeout: float | None = None
    ) -> None:
        """Wait until all specified service IDs are registered.

        Raises:
            ServiceProcessDiedError: If a required service process dies while waiting.
            ServiceRegistrationTimeoutError: If services don't register within timeout.
        """
        if self.all_ids_registered(service_ids):
            return
        self._raise_on_failure()

        ids = frozenset(service_ids)
        event = self._id_events.setdefault(ids, asyncio.Event())
        self.info(f"Waiting for {len(service_ids)} services to be registered...")
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            missing_ids = [sid for sid in service_ids if not self.is_registered(sid)]
            self._raise_timeout(
                f"Timed out waiting for service IDs to register: {missing_ids}"
            )
        self._raise_on_failure()
        if not self.all_ids_registered(service_ids):
            missing_ids = [sid for sid in service_ids if not self.is_registered(sid)]
            self._raise_timeout(
                f"Not all service IDs registered after waking: {missing_ids}"
            )

    async def _wait_with_progress(
        self,
        event: asyncio.Event,
        timeout: float | None,
        description: str,
    ) -> None:
        """Wait on an event with periodic progress logging.

        Logs registration progress every _PROGRESS_LOG_INTERVAL seconds while
        waiting, then checks for failures and completeness after waking.
        """
        elapsed = 0.0
        interval = self._PROGRESS_LOG_INTERVAL

        while not event.is_set():
            remaining = None if timeout is None else max(0, timeout - elapsed)
            wait_time = interval if remaining is None else min(interval, remaining)

            try:
                await asyncio.wait_for(event.wait(), wait_time)
                break
            except asyncio.TimeoutError:
                elapsed += wait_time
                self._raise_on_failure()
                if timeout is not None and elapsed >= timeout:
                    self._raise_timeout(f"Timed out waiting for {description}")
                self._log_waiting_for()

        self._raise_on_failure()
        if not self.all_registered():
            self._raise_timeout(
                f"Not all services registered after waking ({description})"
            )

    # -- Failure/timeout raising and diagnostics --

    def _raise_on_failure(self) -> None:
        """Raise the stored failure error if one exists.

        Logs all recorded failures before raising the first one so that
        operators can see the full picture in the logs.
        """
        if self._failure_errors:
            if len(self._failure_errors) > 1:
                self.error(
                    f"{len(self._failure_errors)} service(s) failed: "
                    + ", ".join(
                        f"'{e.service_id}' ({e.service_type})"
                        for e in self._failure_errors
                    )
                )
            raise self._failure_errors[0]

    def _raise_timeout(self, message: str) -> None:
        """Raise a ServiceRegistrationTimeoutError with missing service diagnostics."""
        missing = self._get_missing_services()
        details = ", ".join(
            f"{st}: {registered}/{expected}"
            for st, (registered, expected) in missing.items()
        )
        raise ServiceRegistrationTimeoutError(
            f"{message}. Missing: {details}" if details else message,
            missing={
                st: expected - registered
                for st, (registered, expected) in missing.items()
            },
        )

    def _get_missing_services(self) -> dict[ServiceTypeT, tuple[int, int]]:
        """Return service types that have fewer registrations than expected.

        Returns a dict of {service_type: (registered_count, expected_count)}.
        """
        missing: dict[ServiceTypeT, tuple[int, int]] = {}
        for service_type, expected in self.expected_by_type.items():
            registered = self._num_registered_of_type(service_type)
            if registered < expected:
                missing[service_type] = (registered, expected)
        return missing

    # -- Logging helpers --

    def _log_all_registered(self) -> None:
        """Log a summary when all expected services have registered."""
        total = sum(self.expected_by_type.values())
        by_type = ", ".join(
            f"{st}: {self._num_registered_of_type(st)}"
            for st in sorted(self.expected_by_type, key=str)
        )
        if self._first_expected_at is not None:
            elapsed = time.perf_counter() - self._first_expected_at
            self.info(
                f"All {total} expected services registered in {elapsed:.2f}s ({by_type})"
            )
        else:
            self.info(f"All {total} expected services registered ({by_type})")

    def _log_waiting_for(self) -> None:
        """Log which services we're still waiting for, with elapsed time."""
        missing = self._get_missing_services()
        parts = [
            f"{st} ({registered}/{expected})"
            for st, (registered, expected) in missing.items()
        ]
        elapsed_str = ""
        if self._first_expected_at is not None:
            elapsed = time.perf_counter() - self._first_expected_at
            elapsed_str = f" ({elapsed:.1f}s elapsed)"
        self.info(f"Waiting for services: {', '.join(parts)}{elapsed_str}")

    # -- Abstract hooks provided by the concrete registry --

    def all_registered(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def all_types_registered(
        self, service_type: ServiceTypeT
    ) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def all_ids_registered(
        self, service_ids: Any
    ) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def is_registered(self, service_id: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def _num_registered_of_type(
        self, service_type: ServiceTypeT
    ) -> int:  # pragma: no cover - interface
        raise NotImplementedError
