# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from collections.abc import Callable, Iterable
from typing import Any

from aiperf.common.enums import LifecycleState, ServiceRegistrationStatus
from aiperf.common.exceptions import (
    ServiceProcessDiedError,
    ServiceRegistrationTimeoutError,
)
from aiperf.common.mixins.aiperf_logger_mixin import AIPerfLoggerMixin
from aiperf.common.models import ServiceRunInfo
from aiperf.common.types import ServiceTypeT


class _ServiceRegistry(AIPerfLoggerMixin):
    """Centralized service registry for tracking service registration and state.

    All mutation and query methods are synchronous. In asyncio's single-threaded
    cooperative model, code between await points runs atomically — no locks needed.
    Only the wait_for_* methods are async (they suspend on asyncio.Event).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Expected services
        self.expected_by_type: dict[ServiceTypeT, int] = {}
        self.expected_ids: set[str] = set()
        self._total_expected: int = 0
        self._first_expected_at: float | None = None

        # Registered services
        self.services: dict[str, ServiceRunInfo] = {}
        self.by_type: dict[ServiceTypeT, set[str]] = {}

        # Events for waiting
        self._all_event: asyncio.Event | None = None
        self._type_events: dict[ServiceTypeT, asyncio.Event] = {}
        self._id_events: dict[frozenset[str], asyncio.Event] = {}

        # Failure tracking — set when a required service process dies
        self._failure_errors: list[ServiceProcessDiedError] = []
        self._failure_event: asyncio.Event | None = None

    def reset(self) -> None:
        """Reset all state. Used between tests to prevent leakage from the global singleton."""
        self._wake_all_waiters()
        self.expected_by_type = {}
        self.expected_ids = set()
        self._total_expected = 0
        self._first_expected_at = None
        self.services = {}
        self.by_type = {}
        self._all_event = None
        self._type_events = {}
        self._id_events = {}
        self._failure_errors = []
        self._failure_event = None

    # -- Expectation management --

    def expect_services(self, services: dict[ServiceTypeT, int]) -> None:
        """Add to expected service counts by type."""
        if self._first_expected_at is None:
            self._first_expected_at = time.perf_counter()
        for service_type, count in services.items():
            self.expected_by_type[service_type] = (
                self.expected_by_type.get(service_type, 0) + count
            )
            self._total_expected += count
        self.debug(lambda: f"Expecting services: {services}")

    def expect_service(self, service_id: str, service_type: ServiceTypeT) -> None:
        """Set an expected service by ID and type. Idempotent for a given service_id."""
        if service_id in self.expected_ids:
            return
        if self._first_expected_at is None:
            self._first_expected_at = time.perf_counter()
        self.expected_ids.add(service_id)
        self.expected_by_type[service_type] = (
            self.expected_by_type.get(service_type, 0) + 1
        )
        self._total_expected += 1
        if service_id not in self.services:
            self.services[service_id] = ServiceRunInfo(
                service_id=service_id,
                service_type=service_type,
                first_seen_ns=None,
                last_seen_ns=None,
                registration_status=ServiceRegistrationStatus.UNREGISTERED,
                state=LifecycleState.CREATED,
            )
        self.debug(lambda: f"Expecting service: {service_id} ({service_type})")
        self._check_events()

    def unexpect_service(self, service_id: str, service_type: ServiceTypeT) -> None:
        """Reverse an expect_service call. Used when a service fails to start."""
        if service_id not in self.expected_ids:
            return
        self.expected_ids.discard(service_id)
        self._decrement_expected(service_type)
        self.debug(lambda: f"Unexpecting service: {service_id} ({service_type})")
        self._check_events()

    # -- Registration lifecycle --

    def register(
        self,
        service_id: str,
        service_type: ServiceTypeT,
        *,
        first_seen_ns: int,
        state: LifecycleState,
        pod_name: str | None = None,
        pod_index: str | None = None,
    ) -> None:
        """Register a service and trigger any waiting events.

        Idempotent: re-registering an already-registered service updates its
        state and timestamp without producing warnings or side-effects.
        """
        info = self.services.get(service_id)
        if info:
            if info.registration_status == ServiceRegistrationStatus.REGISTERED:
                # Already registered — update state/timestamp only
                if info.last_seen_ns is None or first_seen_ns > info.last_seen_ns:
                    info.last_seen_ns = first_seen_ns
                    info.state = state
                return

            # Pre-expected service (from expect_service): handle type mismatch
            if info.service_type != service_type:
                self.warning(
                    f"Service '{service_id}' registered as {service_type} "
                    f"but was expected as {info.service_type}"
                )
                self.by_type.setdefault(info.service_type, set()).discard(service_id)
                self._decrement_expected(info.service_type)
                self.expected_by_type[service_type] = (
                    self.expected_by_type.get(service_type, 0) + 1
                )
                info.service_type = service_type
            info.registration_status = ServiceRegistrationStatus.REGISTERED
            info.first_seen_ns = first_seen_ns
            info.last_seen_ns = max(first_seen_ns, info.last_seen_ns or 0)
            info.state = state
            if pod_name is not None:
                info.pod_name = pod_name
            if pod_index is not None:
                info.pod_index = pod_index
        else:
            self.services[service_id] = ServiceRunInfo(
                service_id=service_id,
                service_type=service_type,
                first_seen_ns=first_seen_ns,
                last_seen_ns=first_seen_ns,
                registration_status=ServiceRegistrationStatus.REGISTERED,
                state=state,
                pod_name=pod_name,
                pod_index=pod_index,
            )
        self.by_type.setdefault(service_type, set()).add(service_id)

        total_registered = sum(
            self._num_registered_of_type(st) for st in self.expected_by_type
        )
        elapsed_str = ""
        if self._first_expected_at is not None:
            elapsed = time.perf_counter() - self._first_expected_at
            elapsed_str = f" +{elapsed:.2f}s"
        self.info(
            f"Registered: {service_type.title()} ('{service_id}') "
            f"[{total_registered}/{self._total_expected}]{elapsed_str}"
        )
        self._check_events()

    def update_service(
        self,
        service_id: str,
        service_type: ServiceTypeT,
        last_seen_ns: int,
        state: LifecycleState,
    ) -> None:
        """Update a service's last-seen timestamp and state.

        Ignores updates for services that haven't formally registered yet.
        StatusUpdate and Heartbeat messages can arrive before Registration
        due to message ordering across ZMQ sockets.
        """
        if service_id not in self.services:
            return

        info = self.services[service_id]
        if info.last_seen_ns is not None and info.last_seen_ns >= last_seen_ns:
            return
        info.last_seen_ns = last_seen_ns
        info.state = state

    def unregister(self, service_id: str) -> None:
        """Unregister a service."""
        if service_id not in self.services:
            self.warning(
                f"Attempting to unregister a service that is not registered: {service_id}"
            )
            return

        info = self.services[service_id]
        info.registration_status = ServiceRegistrationStatus.UNREGISTERED
        info.state = LifecycleState.STOPPED
        self.by_type.setdefault(info.service_type, set()).discard(service_id)
        self.debug(
            lambda: f"Unregistered: {info.service_type.title()} ('{service_id}')"
        )

    def forget(self, service_id: str) -> None:
        """Forget a service entirely, removing it from all tracking."""
        if service_id not in self.services:
            self.warning(
                f"Attempted to forget a service that is not registered: {service_id}"
            )
            return
        service_type = self.services[service_id].service_type
        self.by_type.setdefault(service_type, set()).discard(service_id)
        self.expected_ids.discard(service_id)
        self._decrement_expected(service_type)
        del self.services[service_id]
        self.debug(lambda: f"Forgot service: '{service_id}'")
        self._check_events()

    # -- Failure reporting --

    def fail_service(self, service_id: str, service_type: ServiceTypeT) -> None:
        """Record a fatal service failure and wake all waiters.

        Called when a required service process dies unexpectedly. Updates the
        service state to FAILED and stores a ServiceProcessDiedError that will
        be raised by any active or future wait_for_* call.

        Idempotent: calling again for the same service_id is a no-op.
        """
        if any(e.service_id == service_id for e in self._failure_errors):
            return

        error = ServiceProcessDiedError(service_id, service_type)
        self.error(str(error))

        info = self.services.get(service_id)
        if info:
            info.registration_status = ServiceRegistrationStatus.UNREGISTERED
            info.state = LifecycleState.FAILED
            self.by_type.setdefault(service_type, set()).discard(service_id)

        self._failure_errors.append(error)
        if self._failure_event is None:
            self._failure_event = asyncio.Event()
        self._failure_event.set()

        # Wake ALL pending waiters so they re-check and see the failure
        self._wake_all_waiters()

    def _wake_all_waiters(self) -> None:
        """Force-set every pending wait event so blocked callers unblock and re-check."""
        if self._all_event:
            self._all_event.set()
        for event in self._type_events.values():
            event.set()
        for event in self._id_events.values():
            event.set()

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

    # -- Queries --

    def get_services(
        self, service_type: ServiceTypeT | None = None
    ) -> list[ServiceRunInfo]:
        """Get registered services, optionally filtered by type."""
        ids = (
            self.by_type.get(service_type, ())
            if service_type is not None
            else self.services
        )
        return [self.services[sid] for sid in ids if self.is_registered(sid)]

    def get_service(self, service_id: str) -> ServiceRunInfo | None:
        """Get a specific service by ID, regardless of registration status."""
        return self.services.get(service_id)

    def get_all_registered_ids(self) -> set[str]:
        """Get all registered service IDs."""
        return {sid for sid in self.services if self.is_registered(sid)}

    def get_services_by_pod(self, pod_index: str) -> list[ServiceRunInfo]:
        """Get all registered services belonging to a specific pod index."""
        return [
            info
            for info in self.services.values()
            if info.pod_index == pod_index and self.is_registered(info.service_id)
        ]

    def get_stale_services(self, threshold_sec: float) -> list[ServiceRunInfo]:
        """Get registered services whose last heartbeat exceeds the threshold.

        Args:
            threshold_sec: Seconds since last heartbeat before a service is stale.

        Returns:
            List of ServiceRunInfo for stale services.
        """
        now_ns = time.time_ns()
        threshold_ns = int(threshold_sec * 1_000_000_000)
        stale: list[ServiceRunInfo] = []
        for sid, info in self.services.items():
            if not self.is_registered(sid):
                continue
            if info.last_seen_ns is None:
                continue
            if (now_ns - info.last_seen_ns) > threshold_ns:
                stale.append(info)
        return stale

    def is_registered(self, service_id: str) -> bool:
        """Check if a service is registered."""
        return (
            service_id in self.services
            and self.services[service_id].registration_status
            == ServiceRegistrationStatus.REGISTERED
        )

    def all_types_registered(self, service_type: ServiceTypeT) -> bool:
        """Check if all services of a type are registered."""
        expected = self.expected_by_type.get(service_type, 0)
        return expected == 0 or self._num_registered_of_type(service_type) >= expected

    def all_ids_registered(self, service_ids: Iterable[str]) -> bool:
        """Check if all specified service IDs are registered."""
        return all(self.is_registered(sid) for sid in service_ids)

    def all_registered(self) -> bool:
        """Check if all expected services are registered."""
        return all(
            self.all_types_registered(st) for st in self.expected_by_type
        ) and self.all_ids_registered(self.expected_ids)

    # -- Private helpers --

    def _num_registered_of_type(self, service_type: ServiceTypeT) -> int:
        return sum(
            1 for sid in self.by_type.get(service_type, ()) if self.is_registered(sid)
        )

    def _decrement_expected(self, service_type: ServiceTypeT) -> None:
        if self.expected_by_type.get(service_type, 0) > 0:
            self.expected_by_type[service_type] -= 1
            self._total_expected = max(0, self._total_expected - 1)

    def _check_events(self) -> None:
        """Check and trigger any satisfied events."""
        if self._all_event and not self._all_event.is_set() and self.all_registered():
            self._log_all_registered()
            self._all_event.set()
        self._fire_satisfied(self._type_events, self.all_types_registered)
        self._fire_satisfied(self._id_events, self.all_ids_registered)

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

    _PROGRESS_LOG_INTERVAL: float = 5.0

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

    @staticmethod
    def _fire_satisfied(
        events: dict[Any, asyncio.Event], predicate: Callable[[Any], bool]
    ) -> None:
        satisfied = [key for key in events if predicate(key)]
        for key in satisfied:
            events.pop(key).set()


# Global singleton
ServiceRegistry = _ServiceRegistry()
