# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_start, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ServiceRunInfo
from aiperf.common.service_registry import ServiceRegistry
from aiperf.common.types import ServiceTypeT

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class BaseServiceManager(AIPerfLifecycleMixin, ABC):
    """
    Base class for service managers. It provides a common interface for managing services.
    """

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        run: "BenchmarkRun",
        **kwargs,
    ):
        super().__init__(run=run, **kwargs)
        self.required_services = required_services
        self.run = run
        self.kwargs = kwargs
        # Maps to track service information
        self.service_map: dict[ServiceTypeT, list[ServiceRunInfo]] = {}

        # Create service ID map for component lookups
        self.service_id_map: dict[str, ServiceRunInfo] = {}

        # Heartbeat watchdog state: two-strike verification + catch-up
        # detection. A service is only failed after appearing stale on TWO
        # consecutive ticks, and decisions are skipped entirely when the
        # watchdog itself was delayed -- see _monitor_heartbeats.
        self._suspected_stale: dict[str, int] = {}
        self._last_heartbeat_tick_ns: int | None = None
        self._heartbeat_monitoring_active = False
        self._shutdown_complete = False

        # Services reaped this tick, awaiting result-join barrier eviction.
        # Reaping happens from both async (_monitor_heartbeats) and sync
        # (_fail_pod_services) contexts, so reapers only record here and the
        # heartbeat loop drains it. That bounds eviction to one heartbeat
        # interval without spawning tasks from synchronous code.
        self._pending_reaped: dict[tuple[str, int | None], str] = {}
        self.on_service_reaped: (
            Callable[[str, str, int | None], Awaitable[None]] | None
        ) = None

    def get_service_liveness(self, service_id: str) -> bool | None:
        """Report authoritative liveness for a service, when the manager knows it.

        Returns ``True``/``False`` only when this manager owns a real handle on
        the running service. ``None`` means "unknown", which is the correct and
        only possible answer under Kubernetes: pods are owned by the apiserver,
        so heartbeat silence genuinely is the sole liveness signal there.

        The heartbeat watchdog infers death from silence. Where ground truth is
        available it must win over that inference, otherwise a service that
        merely blocks its event loop past the stale threshold is declared dead
        and its buffered results are dropped from the run.
        """
        return None

    def record_reaped_service(
        self, service_id: str, reason: str, first_seen_ns: int | None
    ) -> None:
        """Queue one registered service instance for controller reaping."""
        self._pending_reaped.setdefault((service_id, first_seen_ns), reason)

    async def _drain_reaped_services(self) -> None:
        """Hand reaped service instances to the controller."""
        if not self._pending_reaped or self.on_service_reaped is None:
            return
        pending, self._pending_reaped = self._pending_reaped, {}
        for (service_id, first_seen_ns), reason in pending.items():
            try:
                await self.on_service_reaped(service_id, reason, first_seen_ns)
            except Exception as exc:  # noqa: BLE001 - a failed eviction must not kill the watchdog
                self.warning(
                    f"Failed to evict reaped service '{service_id}' from the "
                    f"result-join barrier: {exc!r}"
                )

    def activate_heartbeat_monitoring(self) -> None:
        """Begin failing services that stop heartbeating.

        Called once every service has registered. Before that, services are
        legitimately silent while starting up, so judging them would fail the
        run during its own startup.
        """
        self._heartbeat_monitoring_active = True

    def _judge_stale_service(self, info: ServiceRunInfo) -> None:
        """Decide what a single stale service means, and act on it.

        Split out of ``_monitor_heartbeats`` so the tick-level protections
        (catch-up detection, strike bookkeeping) stay readable next to the
        per-service verdict.
        """
        if self.get_service_liveness(info.service_id) is True:
            # Ground truth beats inference. The process is demonstrably
            # running, so the silence is a stalled event loop (a long
            # to_thread summarize, a tokenizer load, a GC pause), not a death.
            # Reaping here would evict a live result producer and compute the
            # final metrics from the survivors only.
            if self._suspected_stale.pop(info.service_id, None):
                self.debug(
                    lambda i=info: f"Service '{i.service_id}' ({i.service_type}) "
                    "is silent but its process is alive; clearing strike"
                )
            return

        strikes = self._suspected_stale.get(info.service_id, 0) + 1
        if strikes < 2:
            self._suspected_stale[info.service_id] = strikes
            self.debug(
                lambda i=info: f"Service '{i.service_id}' ({i.service_type}) "
                "appears stale; awaiting second-tick confirmation"
            )
            return

        self._suspected_stale.pop(info.service_id, None)

        if info.service_type not in self.required_services:
            # Same contract the startup path already applies in
            # MultiProcessServiceManager._reap_dead_processes_during_registration:
            # an optional service (GPU telemetry with no DCGM endpoint, server
            # metrics with no scrape target) going away is a degraded run, not
            # a failed one. Unregister rather than fail_service so no
            # ServiceProcessDiedError is recorded and the next tick stops
            # re-reporting it.
            self.warning(
                f"Optional service '{info.service_id}' ({info.service_type}) "
                f"missed heartbeats on {strikes} consecutive ticks; dropping it "
                "and continuing the benchmark without it."
            )
            ServiceRegistry.unregister(info.service_id)
            # Still release the result-join barrier: eviction is a no-op unless
            # the service really was an awaited result producer.
            self.record_reaped_service(
                info.service_id,
                f"optional service missed heartbeats on {strikes} consecutive ticks",
                info.first_seen_ns,
            )
            return

        self.warning(
            f"Service '{info.service_id}' ({info.service_type}) missed "
            f"heartbeats on {strikes} consecutive ticks - marking as failed"
        )
        ServiceRegistry.fail_service(info.service_id, info.service_type)
        # fail_service only flips registry state and wakes startup-time
        # waiters; without this the service stays in the result-join barrier
        # forever and the controller hangs after profiling.
        self.record_reaped_service(
            info.service_id,
            f"missed heartbeats on {strikes} consecutive ticks",
            info.first_seen_ns,
        )

    @background_task(
        interval=lambda self: Environment.SERVICE.HEARTBEAT_INTERVAL,
        immediate=False,
    )
    async def _monitor_heartbeats(self) -> None:
        """Fail registered services that have stopped sending heartbeats.

        Heartbeats already reach the registry; without this loop nothing acts
        on staleness, so a genuinely dead service produces an indefinite hang
        rather than a fail-fast.

        Two protections against false-positive batch expiry, both earned in
        production at 285 worker-group managers where a controller stall
        flagged 141 of them dead in the same millisecond:

        1. Catch-up detection -- if the gap between consecutive ticks exceeds
           twice ``HEARTBEAT_INTERVAL``, the watchdog itself was delayed and
           every service looks stale through no fault of its own. Skip the
           tick; the next one sees fresh heartbeats.
        2. Two-strike verification -- a service must appear stale on two
           consecutive ticks before being failed. Worst-case detection for a
           genuinely dead service is ``HEARTBEAT_INTERVAL * (threshold + 1)``,
           20 s at the defaults.
        3. Liveness ground truth -- ``get_service_liveness`` is consulted before
           reaping. Where the manager owns a real process handle (local
           multiprocessing) a demonstrably-alive service is never reaped, so a
           blocked event loop cannot be mistaken for a death. Under Kubernetes
           it returns ``None`` and heartbeat inference stands unchanged.

        Optional services (those absent from ``required_services``) are dropped
        with a warning instead of being failed, matching the startup-time
        contract in ``_reap_dead_processes_during_registration``.

        Every exit path drains the reaped-service queue. A service reaped on a
        previous tick is only handed to the controller by that drain, so an
        early return would strand it in the result-join barrier for as long as
        the skip condition holds -- indefinitely, in the shutdown case.
        """
        try:
            await self._monitor_heartbeats_tick()
        finally:
            await self._drain_reaped_services()

    async def _monitor_heartbeats_tick(self) -> None:
        """One watchdog tick: strike bookkeeping plus per-service verdicts.

        Split from ``_monitor_heartbeats`` so its early returns cannot skip the
        reaped-service drain that the caller runs in a ``finally``.
        """
        if (
            self._shutdown_complete
            or self.stop_requested
            or not self._heartbeat_monitoring_active
        ):
            # Reset so a later activation starts clean.
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

        # Services that heartbeated since the previous tick lose their strike.
        for sid in list(self._suspected_stale):
            if sid not in stale_ids:
                del self._suspected_stale[sid]

        if last_tick_ns is not None:
            gap_sec = (now_ns - last_tick_ns) / 1_000_000_000
            if gap_sec > interval_sec * 2:
                self.warning(
                    f"Heartbeat watchdog tick delayed {gap_sec:.1f}s "
                    f"(expected ~{interval_sec:.1f}s); skipping stale checks for "
                    f"{len(stale_ids)} apparently-stale service(s) this tick"
                )
                self._suspected_stale.clear()
                return

        for info in stale:
            self._judge_stale_service(info)

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

    # TODO: This stuff needs some major cleanup

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
        # Log any exceptions that occurred during service startup
        for result in results:
            if isinstance(result, Exception):
                self.exception(f"Error starting required service: {result!r}")

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
        stop_event: asyncio.Event,
        timeout_seconds: float = Environment.SERVICE.REGISTRATION_TIMEOUT,
    ) -> None:
        pass

    @abstractmethod
    async def wait_for_all_services_start(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = Environment.SERVICE.START_TIMEOUT,
    ) -> None:
        pass
