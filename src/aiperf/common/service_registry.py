# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from collections.abc import Callable, Iterable
from typing import Any

from aiperf.common.enums import LifecycleState, ServiceRegistrationStatus
from aiperf.common.environment import Environment
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

    _PROGRESS_LOG_INTERVAL: float = (
        Environment.SERVICE.REGISTRATION_PROGRESS_LOG_INTERVAL
    )

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
        self._dead_services: dict[str, ServiceTypeT] = {}

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
        self._dead_services = {}

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
        self._clear_recorded_death(service_id)
        info = self.services.get(service_id)
        if info:
            if info.registration_status == ServiceRegistrationStatus.REGISTERED:
                if info.last_seen_ns is None or first_seen_ns > info.last_seen_ns:
                    info.last_seen_ns = first_seen_ns
                    info.state = state
                if pod_name is not None:
                    info.pod_name = pod_name
                if pod_index is not None:
                    info.pod_index = pod_index
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

        A strictly older ``last_seen_ns`` is a genuinely out-of-order update and
        is dropped whole. An *equal* one is not out-of-order, though: callers
        stamp on receipt from the controller's own clock, so two messages
        processed within one clock tick collide here -- and ``time.time_ns()``
        has ~15.6ms granularity on Windows, easily coarser than a startup state
        sequence. Treating that collision as stale would silently drop the newer
        state, so equal timestamps still apply their state.
        """
        if service_id not in self.services:
            return

        info = self.services[service_id]
        if info.last_seen_ns is not None and info.last_seen_ns > last_seen_ns:
            return
        info.state = state
        info.last_seen_ns = last_seen_ns

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

    def fail_service(
        self, service_id: str, service_type: ServiceTypeT, *, fatal: bool = True
    ) -> None:
        """Mark a service dead, optionally recording a fatal failure.

        Recoverable Kubernetes pod deaths are meant to use ``fatal=False`` so a
        replacement can register under the same deterministic service ID, with
        an accumulated-death threshold later promoting them to fatal via
        ``escalate_dead_services``. That call site does not exist yet:
        ``KubernetesServiceManager`` (``controller/kubernetes_service_manager.py``)
        is still a stub that raises ``NotImplementedError`` on every method, so
        every current caller of ``fail_service`` (e.g. the heartbeat watchdog in
        ``BaseServiceManager._judge_stale_service``) uses the default
        ``fatal=True`` terminal path. ``fatal=False`` and ``escalate_dead_services``
        are groundwork for the Kubernetes manager and are only exercised by unit
        tests today.
        """
        info = self.services.get(service_id)
        if info:
            info.registration_status = ServiceRegistrationStatus.UNREGISTERED
            info.state = LifecycleState.FAILED
            self.by_type.setdefault(service_type, set()).discard(service_id)

        if not fatal:
            self._dead_services[service_id] = service_type
            self.warning(
                f"Service '{service_id}' ({service_type}) is gone; awaiting a "
                "replacement"
            )
            return

        self._dead_services.pop(service_id, None)
        if any(e.service_id == service_id for e in self._failure_errors):
            return

        error = ServiceProcessDiedError(
            service_id=service_id, service_type=service_type
        )
        self.error(str(error))
        self._failure_errors.append(error)
        if self._failure_event is None:
            self._failure_event = asyncio.Event()
        self._failure_event.set()

        # Wake ALL pending waiters so they re-check and see the failure
        self._wake_all_waiters()

    def escalate_dead_services(self) -> None:
        """Promote every recoverable death to a fatal failure.

        Not yet invoked in production: it exists to be driven by a Kubernetes
        pod-failure threshold (e.g. polled from ``KubernetesServiceManager``
        once implemented) that decides a recoverable death has gone on too
        long without a replacement registering. See ``fail_service`` for the
        current wiring status.
        """
        for service_id, service_type in list(self._dead_services.items()):
            self.fail_service(service_id, service_type, fatal=True)

    def get_dead_services(self) -> dict[str, ServiceTypeT]:
        """Return services currently marked as recoverably dead.

        Not yet read by any production code path; see ``fail_service`` for the
        current wiring status. Intended for the future Kubernetes threshold
        check that drives ``escalate_dead_services``.
        """
        return dict(self._dead_services)

    def _clear_recorded_death(self, service_id: str) -> None:
        """Clear a previous death when a replacement proves the service is live."""
        self._dead_services.pop(service_id, None)
        remaining = [
            error for error in self._failure_errors if error.service_id != service_id
        ]
        if len(remaining) == len(self._failure_errors):
            return
        self._failure_errors = remaining
        self.info(
            f"Service '{service_id}' re-registered; clearing its recorded failure"
        )
        if not remaining and self._failure_event is not None:
            self._failure_event.clear()
        if not remaining:
            self._disarm_stale_waiters()

    def _disarm_stale_waiters(self) -> None:
        """Clear wait events force-set by a failure that has since been retracted.

        ``_wake_all_waiters`` sets every registration event so blocked callers
        re-check and see the failure. Those events stay armed afterwards, so
        once the failure is cleared the next ``wait_for_*`` returns immediately
        on a stale set() and then reports the still-unregistered services as a
        registration timeout. Only events whose predicate is not yet satisfied
        are cleared; ``_check_events`` re-fires the rest.
        """
        if self._all_event is not None and not self.all_registered():
            self._all_event.clear()
        for service_type, event in self._type_events.items():
            if not self.all_types_registered(service_type):
                event.clear()
        for service_ids, event in self._id_events.items():
            if not self.all_ids_registered(service_ids):
                event.clear()

    def _wake_all_waiters(self) -> None:
        """Force-set every pending wait event so blocked callers unblock and re-check."""
        if self._all_event:
            self._all_event.set()
        for event in self._type_events.values():
            event.set()
        for event in self._id_events.values():
            event.set()

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
        started = time.perf_counter()
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except TimeoutError:
            self._raise_timeout(
                f"Timed out waiting for {service_type.title()} services to register",
                timeout,
            )
        self._raise_on_failure()
        if not self.all_types_registered(service_type):
            # Report the time actually spent waiting, not the nominal window:
            # this path fires on a premature wake, and claiming the full
            # timeout sends an operator hunting a slow start that never was.
            self._raise_timeout(
                f"Not all {service_type.title()} services registered after waking",
                time.perf_counter() - started,
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
        started = time.perf_counter()
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except TimeoutError:
            missing_ids = [sid for sid in service_ids if not self.is_registered(sid)]
            self._raise_timeout(
                f"Timed out waiting for service IDs to register: {missing_ids}",
                timeout,
            )
        self._raise_on_failure()
        if not self.all_ids_registered(service_ids):
            missing_ids = [sid for sid in service_ids if not self.is_registered(sid)]
            # Actual elapsed, not the nominal window -- see wait_for_type.
            self._raise_timeout(
                f"Not all service IDs registered after waking: {missing_ids}",
                time.perf_counter() - started,
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
        started = time.perf_counter()

        while not event.is_set():
            remaining = None if timeout is None else max(0, timeout - elapsed)
            wait_time = interval if remaining is None else min(interval, remaining)

            try:
                await asyncio.wait_for(event.wait(), wait_time)
                break
            except TimeoutError:
                elapsed += wait_time
                self._raise_on_failure()
                if timeout is not None and elapsed >= timeout:
                    self._raise_timeout(f"Timed out waiting for {description}", timeout)
                self._log_waiting_for()

        self._raise_on_failure()
        if not self.all_registered():
            # Actual elapsed, not the nominal window -- see wait_for_type.
            self._raise_timeout(
                f"Not all services registered after waking ({description})",
                time.perf_counter() - started,
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

    def _raise_timeout(self, message: str, timeout: float | None) -> None:
        """Raise a ServiceRegistrationTimeoutError with missing-service diagnostics.

        ``message`` is passed through as the exception's ``context`` because it
        names which wait call gave up and the specific IDs it asked for --
        neither of which the aggregate counts can express.

        The registered/expected totals span every expected type, not just the
        short ones: reporting "0 of 1" when two of three services are up sends
        an operator hunting the wrong pod.
        """
        registered_total = sum(
            self._num_registered_of_type(st) for st in self.expected_by_type
        )
        expected_total = sum(self.expected_by_type.values())
        missing = self._get_missing_services()
        raise ServiceRegistrationTimeoutError(
            context=message,
            registered=registered_total,
            expected=expected_total,
            timeout_sec=timeout,
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

    @staticmethod
    def _fire_satisfied(
        events: dict[Any, asyncio.Event], predicate: Callable[[Any], bool]
    ) -> None:
        satisfied = [key for key in events if predicate(key)]
        for key in satisfied:
            events.pop(key).set()


# Global singleton. Mutable state lives on the instance, not at module level.
# noqa rationale: the ServiceRegistry IS the process-wide registration singleton;
# importers reference this name directly. Guarded by reset() between tests.
ServiceRegistry = _ServiceRegistry()
