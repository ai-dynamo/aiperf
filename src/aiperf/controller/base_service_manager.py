# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_start, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.service_registry import ServiceRegistry
from aiperf.common.types import ServiceTypeT

if TYPE_CHECKING:
    from aiperf.common.models.service_models import ServiceRunInfo
    from aiperf.config import BenchmarkRun


class BaseServiceManager(AIPerfLifecycleMixin, ABC):
    """
    Base class for service managers. It provides a common interface for managing services.
    """

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        run: BenchmarkRun,
        **kwargs,
    ):
        super().__init__(run=run, **kwargs)
        self.required_services = required_services
        self.run = run
        self.kwargs = kwargs
        # Maps to track service information
        self.service_map: dict[ServiceTypeT, list[ServiceRunInfo]] = {}
        self._shutdown_complete = False
        self._heartbeat_monitoring_active = False
        self._pod_monitoring_active = False
        self.pod_failure_abort_event = asyncio.Event()
        self.pod_failure_abort_reason: str = ""
        # Heartbeat watchdog state: two-strike verification + catch-up detection.
        # A service is only failed after being stale on TWO consecutive ticks,
        # and decisions are skipped entirely if the watchdog itself was delayed
        # (see `_monitor_heartbeats` for rationale).
        self._suspected_stale: dict[str, int] = {}
        self._last_heartbeat_tick_ns: int | None = None

    def notify_shutdown(self) -> None:
        """Signal that shutdown has been initiated.

        Suppresses heartbeat and process monitors from reporting expected
        process exits as errors. Called by the system controller before
        broadcasting the shutdown command.
        """
        self._shutdown_complete = True

    def activate_pod_monitoring(self) -> None:
        """Enable Kubernetes pod health monitoring.

        Called by the system controller after spawning services, before waiting
        for registration/configuration. Unlike heartbeat monitoring, pod phase
        checks are safe during startup — a pod in Failed/Unknown state is always
        an error, regardless of whether services have registered yet. This allows
        fast failure detection during the registration/configuration phase.
        """
        self._pod_monitoring_active = True

    def activate_heartbeat_monitoring(self) -> None:
        """Enable heartbeat-based stale service detection.

        Called by the system controller after all services have registered.
        Prevents false positives during the startup/registration phase when
        services may not yet be sending regular heartbeats.
        """
        self._heartbeat_monitoring_active = True

    @on_start
    async def _start_service_manager(self) -> None:
        await self.run_required_services()

    @on_stop
    async def _stop_service_manager(self) -> None:
        await self.shutdown_all_services()

    async def run_services(
        self, service_types: dict[ServiceTypeT, int]
    ) -> list[BaseException | None]:
        return await asyncio.gather(
            *[
                self.run_service(service_type, num_replicas)
                for service_type, num_replicas in service_types.items()
            ],
            return_exceptions=True,
        )

    @abstractmethod
    async def stop_service(
        self, service_type: ServiceTypeT, service_id: str | None = None
    ) -> list[BaseException | None]: ...

    async def stop_services_by_type(
        self, service_types: list[ServiceTypeT]
    ) -> list[BaseException | None]:
        """Stop a set of services."""
        results = await asyncio.gather(
            *[self.stop_service(service_type) for service_type in service_types],
            return_exceptions=True,
        )
        output: list[BaseException | None] = []
        for result in results:
            if isinstance(result, list):
                output.extend(result)
            else:
                output.append(result)
        return output

    async def run_required_services(self) -> None:
        results = await self.run_services(self.required_services)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            for error in errors:
                self.exception(f"Error starting required service: {error!r}")
            raise errors[0]

    @abstractmethod
    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        pass

    @abstractmethod
    async def shutdown_all_services(self) -> list[BaseException | None]:
        pass

    @abstractmethod
    async def kill_all_services(self) -> list[BaseException | None]:
        pass

    @abstractmethod
    async def wait_for_all_services_registration(
        self,
        timeout_seconds: float = Environment.SERVICE.REGISTRATION_TIMEOUT,
    ) -> None:
        pass

    @background_task(
        interval=lambda self: Environment.SERVICE.HEARTBEAT_INTERVAL,
        immediate=False,
    )
    async def _monitor_heartbeats(self) -> None:
        """Detect registered services that have stopped sending heartbeats.

        Two protections against false-positive batch expiry (observed in
        production at 285 WGMs where 141 were flagged dead in the same
        millisecond after a controller stall):

        1. Catch-up detection: if the gap between consecutive ticks exceeds
           `HEARTBEAT_INTERVAL * 2`, the watchdog itself was delayed — every
           registered service looks stale through no fault of its own. Skip
           death decisions this tick; the next tick will see fresh heartbeats.

        2. Two-strike verification: a service must appear stale on two
           consecutive ticks before being failed. `_suspected_stale` tracks
           strike counts; a service that heartbeats between ticks drops back
           off the suspect list. Worst-case detection latency for a genuinely
           dead service becomes `HEARTBEAT_INTERVAL * (MISSED_THRESHOLD + 1)`
           (default 20s), still well under the 60s goal.

        Marks confirmed-stale services as failed via ServiceRegistry.fail_service,
        which wakes all pending waiters.
        """
        if (
            self._shutdown_complete
            or self.stop_requested
            or not self._heartbeat_monitoring_active
        ):
            # Reset state so a later activation starts clean.
            self._suspected_stale.clear()
            self._last_heartbeat_tick_ns = None
            return

        now_ns = time.time_ns()
        last_tick_ns = self._last_heartbeat_tick_ns
        self._last_heartbeat_tick_ns = now_ns

        interval_sec = Environment.SERVICE.HEARTBEAT_INTERVAL
        threshold_sec = interval_sec * Environment.SERVICE.HEARTBEAT_MISSED_THRESHOLD
        stale = ServiceRegistry.get_stale_services(threshold_sec)
        stale_ids = {info.service_id for info in stale}

        # Drop strike counts for services that are no longer stale (they sent
        # a heartbeat since the previous tick).
        for sid in list(self._suspected_stale):
            if sid not in stale_ids:
                del self._suspected_stale[sid]

        # Catch-up detection: if our own tick was delayed, don't blame services.
        if last_tick_ns is not None:
            gap_sec = (now_ns - last_tick_ns) / 1_000_000_000
            if gap_sec > interval_sec * 2:
                self.warning(
                    f"Heartbeat watchdog tick delayed {gap_sec:.1f}s "
                    f"(expected ~{interval_sec:.1f}s); skipping stale checks "
                    f"for {len(stale_ids)} apparently-stale service(s) this tick"
                )
                # Clear strikes: everything looked stale due to our delay,
                # not due to actual missed heartbeats.
                self._suspected_stale.clear()
                return

        for info in stale:
            strikes = self._suspected_stale.get(info.service_id, 0) + 1
            if strikes < 2:
                self._suspected_stale[info.service_id] = strikes
                self.debug(
                    lambda i=info: f"Service '{i.service_id}' ({i.service_type}) "
                    f"appears stale; awaiting second-tick confirmation"
                )
                continue

            self.warning(
                f"Service '{info.service_id}' ({info.service_type}) "
                f"missed heartbeats on {strikes} consecutive ticks — marking as failed"
            )
            ServiceRegistry.fail_service(info.service_id, info.service_type)
            self._suspected_stale.pop(info.service_id, None)

    async def wait_for_api_subprocess(self) -> None:
        """Block until the API service runtime terminates (Kubernetes mode only).

        Default implementation is a no-op. Override in Kubernetes service
        managers that keep a local API runtime alive after benchmarking.
        """
        pass

    def get_pod_summary(self) -> dict[str, str]:
        """Get pod state summary for diagnostics (Kubernetes mode only).

        Default implementation returns an empty dict. Override in
        KubernetesServiceManager to return actual pod states.
        """
        return {}

    async def check_pods_healthy(self) -> None:
        """Verify all tracked pods are healthy before starting profiling.

        Default implementation is a no-op. Override in KubernetesServiceManager
        to check pod phases and fail fast if any pods are in a terminal state.
        """
