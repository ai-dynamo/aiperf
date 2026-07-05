# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.enums import CommAddress, CommandType
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_command, on_stop
from aiperf.common.messages import (
    NetworkLatencyRecordMessage,
    ProfileCancelCommand,
    ProfileCompleteCommand,
    ProfileStartCommand,
)
from aiperf.common.models import ErrorDetails, NetworkLatencySample
from aiperf.common.protocols import PushClientProtocol
from aiperf.common.redact import redact_url
from aiperf.network_latency.probe import NetworkLatencyProbeCollector

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

# Default ports for schemes when a URL omits an explicit port.
_DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}


class NetworkLatencyManager(BaseComponentService):
    """Coordinates TCP-handshake RTT probes against the endpoint targets.

    Mirrors ServerMetricsManager: one probe collector per unique (host, port)
    target derived from the endpoint URLs. Probing spans the PROFILING phase —
    started on PROFILE_START and stopped on PROFILE_COMPLETE. On completion, if
    fewer than ``Environment.NETWORK_LATENCY.MIN_SAMPLES`` successful samples
    were collected, extra back-to-back probes are fired synchronously until the
    floor is reached (mirrors the server-metrics final-scrape pattern).

    This service is only spawned when ``run.cfg.network_latency.should_probe`` is
    True (i.e. enabled and no manual mean_ms); the manual-mean path
    needs no service.

    Args:
        run: BenchmarkRun carrying the BenchmarkConfig + per-run state.
        service_id: Optional unique identifier for this service instance.
    """

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            run=run,
            service_id=service_id,
            **kwargs,
        )

        self.records_push_client: PushClientProtocol = self.comms.create_push_client(
            CommAddress.RECORDS,
        )

        self._collectors: dict[str, NetworkLatencyProbeCollector] = {}
        self._ping_interval = self.run.cfg.network_latency.ping_interval
        self._min_samples = Environment.NETWORK_LATENCY.MIN_SAMPLES

        # One probe target per unique (host, port) derived from the endpoint URLs.
        self._targets: dict[str, tuple[str, str, int]] = {}
        for url in self.run.cfg.endpoint.urls:
            target = self._derive_target(url)
            if target is None:
                continue
            host, port = target
            key = f"{host}:{port}"
            if key not in self._targets:
                self._targets[key] = (redact_url(url), host, port)
        self.info(
            f"Network Latency: Discovered {len(self._targets)} probe target(s): "
            f"{list(self._targets.keys())}"
        )

        self._shutdown_task: asyncio.Task[None] | None = None

    @staticmethod
    def _derive_target(url: str) -> tuple[str, int] | None:
        """Parse host:port from an endpoint URL, applying scheme default ports."""
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = parsed.hostname
        if not host:
            return None
        port = parsed.port or _DEFAULT_PORTS.get(parsed.scheme, 80)
        return host, port

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message: ProfileStartCommand) -> None:
        """Create, resolve, and start a probe collector per unique target.

        Resolves each target host once (so probes time pure TCP connect) and
        starts the background probe loop. Partial failures are tolerated: the
        run continues as long as at least one collector starts.
        """
        if not self._targets:
            self.warning("Network Latency: No probe targets discovered; nothing to do")
            self._shutdown_task = self.execute_async(self._delayed_shutdown())
            return

        self._collectors.clear()
        started_count = 0
        for key, (display_url, host, port) in self._targets.items():
            collector = NetworkLatencyProbeCollector(
                target_url=display_url,
                target_host=host,
                target_port=port,
                ping_interval=self._ping_interval,
                connect_timeout=Environment.NETWORK_LATENCY.CONNECT_TIMEOUT,
                record_callback=self._on_network_latency_samples,
                error_callback=self._on_network_latency_error,
                collector_id=key,
            )
            # why: per-collector startup is isolated so one bad target can't
            # abort the @on_command handler; the run proceeds with the rest.
            try:
                await collector.initialize()
                await collector.resolve()
                await collector.start()
                self._collectors[key] = collector
                started_count += 1
            except Exception as e:
                self.error(f"Failed to start probe collector for {key}: {e!r}")

        if started_count == 0:
            self.warning("Network Latency: No probe collectors successfully started")
            self._shutdown_task = self.execute_async(self._delayed_shutdown())
            return
        self.info(
            f"Network Latency: Started {started_count} probe collector(s) at "
            f"{self._ping_interval}s interval"
        )

    @on_command(CommandType.PROFILE_COMPLETE)
    async def _handle_profile_complete_command(
        self, message: ProfileCompleteCommand
    ) -> None:
        """Top up to MIN_SAMPLES with synchronous probes, then stop all collectors.

        Idempotent: a no-op once collectors have been stopped.
        """
        if not self._collectors:
            self.debug("Network Latency: Already stopped, skipping final probes")
            return

        self.info("Network Latency: Profiling complete, topping up RTT samples...")
        # why: the top-up runs inside the PROFILE_COMPLETE command handler, which the
        # caller awaits on a fixed response budget. Bound the whole top-up by a
        # wall-clock deadline (across all collectors) so a slow/unreachable endpoint
        # cannot stall completion; whatever samples we have are used as-is.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + Environment.NETWORK_LATENCY.COMPLETE_TOPUP_TIMEOUT
        for key, collector in list(self._collectors.items()):
            attempts = 0
            max_attempts = self._min_samples * 2
            while (
                collector.successful_samples < self._min_samples
                and attempts < max_attempts
                and loop.time() < deadline
            ):
                attempts += 1
                # why: a top-up probe failure must not crash the completion path,
                # just stop topping up this collector.
                try:
                    await collector.probe_once()
                except Exception as e:
                    self.warning(
                        f"Network Latency: Final top-up probe failed for {key}: {e!r}"
                    )
                    break
            if loop.time() >= deadline:
                self.warning(
                    "Network Latency: top-up budget exhausted; proceeding to stop with "
                    "the samples collected so far"
                )
                break

        await self._stop_all_collectors()

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        """Stop all probe collectors when profiling is cancelled."""
        await self._stop_all_collectors()

    @on_stop
    async def _network_latency_manager_stop(self) -> None:
        """Stop all probe collectors during service shutdown."""
        await self._stop_all_collectors()

    async def _stop_all_collectors(self) -> None:
        """Stop all probe collectors, tolerating individual shutdown errors."""
        if not self._collectors:
            return
        collectors = list(self._collectors.items())
        self._collectors.clear()
        for key, collector in collectors:
            # why: shutdown isolation — one collector's stop() error must not
            # leave the remaining collectors running or fail service teardown.
            try:
                await collector.stop()
            except Exception as e:
                self.error(f"Failed to stop probe collector for {key}: {e!r}")

    async def _delayed_shutdown(self) -> None:
        """Shutdown the service after a delay so the command response can be sent."""
        await asyncio.sleep(Environment.SERVER_METRICS.SHUTDOWN_DELAY)
        await asyncio.shield(self.stop())

    async def _on_network_latency_samples(
        self, samples: list[NetworkLatencySample], collector_id: str
    ) -> None:
        """Forward probe samples to RecordsManager via the RECORDS push socket."""
        if not samples:
            return
        for sample in samples:
            # why: sample-forwarding callback is an isolation boundary; a push
            # failure must be logged, not crash the probe collector's loop.
            try:
                message = NetworkLatencyRecordMessage(
                    service_id=self.service_id,
                    collector_id=collector_id,
                    sample=sample,
                    error=None,
                )
                await self.records_push_client.push(message)
            except Exception as e:
                self.error(
                    f"Failed to send network latency sample from {collector_id}: {e!r}"
                )
                # why: error-forwarding fallback is also an isolation boundary.
                try:
                    error_message = NetworkLatencyRecordMessage(
                        service_id=self.service_id,
                        collector_id=collector_id,
                        sample=None,
                        error=ErrorDetails.from_exception(e),
                    )
                    await self.records_push_client.push(error_message)
                except Exception as nested_error:
                    self.error(
                        f"Failed to send error message after sample send failure: {nested_error!r}"
                    )

    async def _on_network_latency_error(
        self, error: ErrorDetails, collector_id: str
    ) -> None:
        """Forward a transport-level probe error to RecordsManager."""
        # why: error-forwarding callback is an isolation boundary; a push
        # failure must be logged, not crash the probe collector's loop.
        try:
            error_message = NetworkLatencyRecordMessage(
                service_id=self.service_id,
                collector_id=collector_id,
                sample=None,
                error=error,
            )
            await self.records_push_client.push(error_message)
        except Exception as e:
            self.error(f"Failed to send network latency error message: {e!r}")


def main() -> None:
    """Main entry point for the network latency manager."""
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.NETWORK_LATENCY_MANAGER)


if __name__ == "__main__":
    main()
