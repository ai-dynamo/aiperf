# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base mixin for async HTTP metrics data collectors.

This mixin provides common functionality for collecting metrics from HTTP endpoints,
used by both GPU telemetry and server metrics systems.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

import aiohttp

from aiperf.common.hooks import background_task, on_init, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ErrorDetails

# Type variable for records returned by collectors
TRecord = TypeVar("TRecord")


class BaseMetricsCollectorMixin(AIPerfLifecycleMixin, ABC, Generic[TRecord]):
    """Mixin providing async HTTP collection for metrics endpoints.

    This mixin encapsulates the pattern of periodically fetching metrics from
    HTTP endpoints, parsing them, and delivering them via callbacks.

    Common patterns:
        - aiohttp session management
        - Reachability testing
        - Background collection task with error handling
        - Callback-based delivery

    Used by:
        - TelemetryDataCollector (DCGM metrics)
        - ServerMetricsDataCollector (Prometheus metrics)
    """

    def __init__(
        self,
        endpoint_url: str,
        collection_interval: float,
        reachability_timeout: float,
        record_callback: Callable[[list[TRecord], str], Awaitable[None]] | None = None,
        error_callback: Callable[[ErrorDetails, str], Awaitable[None]] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            endpoint_url: URL of the metrics endpoint
            collection_interval: Interval in seconds between collections
            record_callback: Optional callback to receive collected records
            error_callback: Optional callback to receive collection errors
            **kwargs: Additional arguments passed to super().__init__()
        """
        self._endpoint_url = endpoint_url
        self._collection_interval = collection_interval
        self._reachability_timeout = reachability_timeout
        self._record_callback = record_callback
        self._error_callback = error_callback
        self._session: aiohttp.ClientSession | None = None
        super().__init__(**kwargs)

    @property
    def endpoint_url(self) -> str:
        """Get the metrics endpoint URL."""
        return self._endpoint_url

    @property
    def collection_interval(self) -> float:
        """Get the collection interval in seconds."""
        return self._collection_interval

    @on_init
    async def _initialize_http_client(self) -> None:
        """Initialize the aiohttp client session.

        Called automatically during initialization phase.
        Creates an aiohttp ClientSession with appropriate timeout settings.
        """
        timeout = aiohttp.ClientTimeout(total=self._reachability_timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)

    @on_stop
    async def _cleanup_http_client(self) -> None:
        """Clean up the aiohttp client session.

        Called automatically during shutdown phase.
        """
        if self._session:
            await self._session.close()
            self._session = None

    async def is_url_reachable(self) -> bool:
        """Check if metrics endpoint is accessible.

        Attempts HEAD request first for efficiency, falls back to GET if HEAD is not supported.
        Uses existing session if available, otherwise creates a temporary session.

        Returns:
            True if endpoint responds with HTTP 200, False otherwise
        """
        if not self._endpoint_url:
            return False

        # Use existing session if available, otherwise create a temporary one
        if self._session:
            return await self._check_reachability_with_session(self._session)
        else:
            # Create a temporary session for reachability check
            timeout = aiohttp.ClientTimeout(total=self._reachability_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as temp_session:
                return await self._check_reachability_with_session(temp_session)

    async def _check_reachability_with_session(
        self, session: aiohttp.ClientSession
    ) -> bool:
        """Check reachability using a specific session.

        Args:
            session: aiohttp session to use for the check

        Returns:
            True if endpoint is reachable with HTTP 200
        """
        try:
            # Try HEAD first for efficiency
            async with session.head(
                self._endpoint_url, allow_redirects=False
            ) as response:
                if response.status == 200:
                    return True
            # Fall back to GET if HEAD is not supported
            async with session.get(self._endpoint_url) as response:
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    @background_task(immediate=True, interval=lambda self: self._collection_interval)
    async def _collect_metrics_task(self) -> None:
        """Background task for collecting metrics at regular intervals.

        This uses the @background_task decorator which automatically handles
        lifecycle management and stopping when the collector is stopped.

        Errors during collection are caught and sent via error_callback if configured.
        CancelledError is propagated to allow graceful shutdown.
        """
        try:
            await self._collect_and_process_metrics()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self._error_callback:
                try:
                    await self._error_callback(
                        ErrorDetails.from_exception(e),
                        self.id,
                    )
                except Exception as callback_error:
                    self.error(f"Failed to send error via callback: {callback_error}")
            else:
                self.error(f"Metrics collection error: {e}")

    @abstractmethod
    async def _collect_and_process_metrics(self) -> None:
        """Collect metrics from endpoint and process them into records.

        Subclasses must implement this to:
        1. Fetch raw metrics data from the endpoint
        2. Parse data into record objects
        3. Send records via callback (if configured)
        """
        pass

    async def _fetch_metrics_text(self) -> str:
        """Fetch raw metrics text from the HTTP endpoint.

        Performs safety checks before making HTTP request:
        - Verifies stop_requested flag to allow graceful shutdown
        - Checks session is initialized and not closed

        Returns:
            Raw metrics text from the endpoint

        Raises:
            RuntimeError: If HTTP session is not initialized
            aiohttp.ClientError: If HTTP request fails
            asyncio.CancelledError: If collector is being stopped or session is closed
        """
        if self.stop_requested:
            raise asyncio.CancelledError

        if not self._session:
            raise RuntimeError("HTTP session not initialized")

        if self._session.closed:
            raise asyncio.CancelledError

        async with self._session.get(self._endpoint_url) as response:
            response.raise_for_status()
            return await response.text()

    async def _send_records_via_callback(self, records: list[TRecord]) -> None:
        """Send records to the callback if configured.

        Args:
            records: List of records to send
        """
        if records and self._record_callback:
            try:
                await self._record_callback(records, self.id)
            except Exception as e:
                self.warning(f"Failed to send records via callback: {e}")
