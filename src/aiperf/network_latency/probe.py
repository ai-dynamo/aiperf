# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Awaitable, Callable

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ErrorDetails, NetworkLatencySample

__all__ = ["NetworkLatencyProbeCollector"]


class NetworkLatencyProbeCollector(AIPerfLifecycleMixin):
    """Probes a single (host, port) target's TCP-handshake RTT on an interval.

    Each probe opens a fresh, unpooled TCP connection via
    ``asyncio.open_connection`` and times the handshake with
    ``time.perf_counter_ns``. A pooled HTTP client (aiohttp) would measure ~0 on
    connection reuse, so raw ``open_connection`` is used to force a real
    handshake every probe. For ``https://`` targets a plain TCP connect to the
    port is performed (no TLS) so ``rtt_ns`` is one uniform network round-trip
    across http/https.

    The host is resolved once at construction time (cached resolved address)
    so probes time pure TCP connect, not DNS resolution.

    Probe failures (timeout/refused) are recorded as a failed sample
    (``success=False``, ``rtt_ns=None``, ``error=...``) and never crash the run:
    the background loop wraps the probe in ``probe_once`` which isolates errors
    and routes them to ``record_callback`` as a failed sample.

    Args:
        target_url: Endpoint URL the target was derived from (credential-free).
        target_host: Host to connect to.
        target_port: TCP port to connect to.
        ping_interval: Seconds between probes.
        connect_timeout: Per-probe TCP-handshake timeout in seconds.
        record_callback: Async callback receiving probe samples.
            Signature: async (samples: list[NetworkLatencySample], collector_id: str) -> None
        error_callback: Async callback receiving transport-level errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector (typically host:port).
    """

    def __init__(
        self,
        target_url: str,
        target_host: str,
        target_port: int,
        *,
        ping_interval: float,
        connect_timeout: float | None = None,
        record_callback: Callable[[list[NetworkLatencySample], str], Awaitable[None]] | None = None,
        error_callback: Callable[[ErrorDetails, str], Awaitable[None]] | None = None,
        collector_id: str = "network_latency_probe",
    ) -> None:  # fmt: skip
        super().__init__(id=collector_id)
        self._target_url = target_url
        self._target_host = target_host
        self._target_port = target_port
        self._ping_interval = ping_interval
        self._connect_timeout = (
            connect_timeout
            if connect_timeout is not None
            else Environment.NETWORK_LATENCY.CONNECT_TIMEOUT
        )
        self._record_callback = record_callback
        self._error_callback = error_callback
        # Resolved connect target cached so probes time pure TCP connect, not DNS.
        # None until resolve() succeeds; falls back to the parsed host:port.
        self._resolved_host: str = target_host
        self._resolved_family: int = socket.AF_UNSPEC
        self._successful_samples: int = 0

    @property
    def ping_interval(self) -> float:
        """Seconds between probes."""
        return self._ping_interval

    @property
    def successful_samples(self) -> int:
        """Number of successful probes recorded by this collector."""
        return self._successful_samples

    async def resolve(self) -> None:
        """Resolve the target host once and cache the address for pure-connect probes.

        Resolution failure is non-fatal: the loop falls back to passing the
        original host to ``open_connection`` (which will resolve per-probe), so
        a transient DNS hiccup at configure time does not disable probing.
        """
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(
                self._target_host,
                self._target_port,
                type=socket.SOCK_STREAM,
            )
            if infos:
                family, _, _, _, sockaddr = infos[0]
                self._resolved_family = family
                self._resolved_host = sockaddr[0]
                self.debug(
                    lambda: f"Resolved {self._target_host}:{self._target_port} -> {self._resolved_host}"
                )
        except OSError as e:
            self.warning(
                f"Failed to resolve {self._target_host}:{self._target_port} "
                f"({e!r}); probes will resolve per-connect."
            )

    @background_task(immediate=True, interval=lambda self: self.ping_interval)
    async def _probe_loop(self) -> None:
        """Background task that fires a probe every ping_interval seconds.

        Uses execute_async (fire-and-forget) so a slow handshake near the
        connect timeout does not delay the next scheduled probe.
        """
        self.execute_async(self.probe_once())

    async def probe_once(self) -> None:
        """Issue one TCP-handshake RTT probe with full error isolation.

        Always produces exactly one NetworkLatencySample (success or failure)
        and delivers it via the record callback. Never raises — a probe failure
        becomes a failed sample so the run is never crashed by a refused/timed-out
        endpoint (mirrors base_metrics_collector_mixin error isolation).
        """
        timestamp_ns = time.time_ns()
        rtt_ns: int | None = None
        error: ErrorDetails | None = None
        success = False

        start_perf_ns = time.perf_counter_ns()
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(self._resolved_host, self._target_port),
                timeout=self._connect_timeout,
            )
            rtt_ns = time.perf_counter_ns() - start_perf_ns
            success = True
            try:
                writer.close()
                await writer.wait_closed()
            except OSError as close_error:
                self.debug(
                    lambda err=close_error: f"Error closing probe connection to "
                    f"{self._target_host}:{self._target_port}: {err!r}"
                )
        except (OSError, asyncio.TimeoutError) as e:
            error = ErrorDetails.from_exception(e)

        if success:
            self._successful_samples += 1

        sample = NetworkLatencySample(
            timestamp_ns=timestamp_ns,
            target_url=self._target_url,
            target_host=self._target_host,
            target_port=self._target_port,
            probe_type="tcp_connect",
            rtt_ns=rtt_ns,
            success=success,
            error=error,
        )
        await self._send_sample(sample)

    async def _send_sample(self, sample: NetworkLatencySample) -> None:
        """Deliver a probe sample via the record callback, isolating callback errors."""
        if self._record_callback is None:
            return
        # why: callback dispatch is the probe-loop isolation boundary; any user
        # callback failure must be logged, not crash the benchmark run.
        try:
            await self._record_callback([sample], self.id)
        except Exception as e:  # noqa: BLE001
            self.error(f"Failed to send probe sample via callback: {e!r}")
            if self._error_callback:
                # why: error-callback dispatch is also an isolation boundary.
                try:
                    await self._error_callback(ErrorDetails.from_exception(e), self.id)
                except Exception as callback_error:  # noqa: BLE001
                    self.error(f"Failed to send error via callback: {callback_error!r}")
