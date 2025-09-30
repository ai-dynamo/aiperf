# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections.abc import Callable

import aiohttp
from prometheus_client.parser import text_string_to_metric_families

from aiperf.common.hooks import background_task, on_init, on_stop
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.models import ErrorDetails, TelemetryRecord
from aiperf.gpu_telemetry.constants import (
    DCGM_TO_FIELD_MAPPING,
    DEFAULT_COLLECTION_INTERVAL,
    SCALING_FACTORS,
    URL_REACHABILITY_TIMEOUT,
)

__all__ = ["TelemetryDataCollector"]


class TelemetryDataCollector(AIPerfLifecycleMixin):
    """Collects telemetry metrics from DCGM metrics endpoint using async architecture.

    Modern async collector that fetches GPU metrics from DCGM exporter and converts them to
    TelemetryRecord objects. Uses AIPerf lifecycle management and background tasks.
    - Extends AIPerfLifecycleMixin for proper lifecycle management
    - Uses aiohttp for async HTTP requests
    - Uses prometheus_client for robust metric parsing
    - Uses @background_task for periodic collection
    - Sends TelemetryRecord list via callback function
    - No local storage (follows centralized architecture)
    """

    def __init__(
        self,
        dcgm_url: str,
        collection_interval: float = DEFAULT_COLLECTION_INTERVAL,
        record_callback: Callable[[list[TelemetryRecord]], None] | None = None,
        error_callback: Callable[[Exception], None] | None = None,
        collector_id: str = "telemetry_collector",
    ) -> None:
        """
        Initialize the TelemetryDataCollector with its DCGM endpoint, collection interval, and optional callbacks.
        
        Parameters:
            dcgm_url (str): HTTP URL of the DCGM Prometheus-formatted metrics endpoint to poll.
            collection_interval (float): Seconds between periodic collections.
            record_callback (Callable[[list[TelemetryRecord]], None] | None): Optional callable (sync or async) invoked with a list of TelemetryRecord objects when metrics are successfully collected.
            error_callback (Callable[[Exception], None] | None): Optional callable (sync or async) invoked with an Exception or error details when collection or processing fails.
            collector_id (str): Identifier used for the collector instance.
        """
        self._dcgm_url = dcgm_url
        self._collection_interval = collection_interval
        self._record_callback = record_callback
        self._error_callback = error_callback
        self._scaling_factors = SCALING_FACTORS
        self._session: aiohttp.ClientSession | None = None

        super().__init__(id=collector_id)

    @on_init
    async def _initialize_http_client(self) -> None:
        """
        Create and store an aiohttp ClientSession configured with the module's URL reachability timeout.
        
        This initializes the collector's internal HTTP session so it can perform requests to the configured DCGM endpoint.
        """
        timeout = aiohttp.ClientTimeout(total=URL_REACHABILITY_TIMEOUT)
        self._session = aiohttp.ClientSession(timeout=timeout)

    @on_stop
    async def _cleanup_http_client(self) -> None:
        """
        Close and clear the internal aiohttp ClientSession if one exists.
        
        If a session is present, it is closed and the internal reference is set to None. Background tasks are expected to coordinate using the collector's stop request mechanism to avoid races.
        """
        if self._session:
            await self._session.close()
            self._session = None

    async def is_url_reachable(self) -> bool:
        """Check if DCGM metrics endpoint is accessible.

        Returns:
            True if endpoint responds with 200, False otherwise
        """
        if not self._dcgm_url:
            return False

        # Use existing session if available, otherwise create a temporary one
        if self._session:
            try:
                async with self._session.get(self._dcgm_url) as response:
                    return response.status == 200
            except Exception:
                return False
        else:
            # Create a temporary session for reachability check
            timeout = aiohttp.ClientTimeout(total=URL_REACHABILITY_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as temp_session:
                try:
                    async with temp_session.get(self._dcgm_url) as response:
                        return response.status == 200
                except Exception:
                    return False

    @background_task(immediate=True, interval=lambda self: self._collection_interval)
    async def _collect_telemetry_task(self) -> None:
        """
        Run one collection-and-processing iteration and handle any errors.
        
        Attempts to collect and process telemetry; if a CancelledError occurs it is re-raised. On other exceptions, invokes the configured error callback with an ErrorDetails and collector id if provided, logging a warning if the callback itself fails; otherwise logs the error.
        """
        try:
            await self._collect_and_process_metrics()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self._error_callback:
                try:
                    self._error_callback(ErrorDetails.from_exception(e), self.id)
                except Exception as callback_error:
                    self.error(f"Failed to send error via callback: {callback_error}")
            else:
                self.error(f"Telemetry collection error: {e}")

    async def _collect_and_process_metrics(self) -> None:
        """
        Collect metrics from the DCGM endpoint, convert them to TelemetryRecord objects, and deliver them to the configured record callback.
        
        If one or more records are produced and a record callback is configured, the callback is invoked with (records, self.id). The callback may be a coroutine or a regular callable; any exceptions raised by the callback are logged and not propagated by this method. Exceptions raised while fetching or parsing metrics are logged and re-raised.
        """
        try:
            metrics_data = await self._fetch_metrics()
            records = self._parse_metrics_to_records(metrics_data)

            if records and self._record_callback:
                try:
                    if asyncio.iscoroutinefunction(self._record_callback):
                        await self._record_callback(records, self.id)
                    else:
                        self._record_callback(records, self.id)
                except Exception as e:
                    self.warning(f"Failed to send telemetry records via callback: {e}")

        except Exception as e:
            self.error(f"Error collecting and processing metrics: {e}")
            raise

    async def _fetch_metrics(self) -> str:
        """
        Retrieve raw Prometheus-formatted metrics from the configured DCGM endpoint.
        
        Returns:
            Raw metrics text in Prometheus format.
        
        Raises:
            aiohttp.ClientError: If the HTTP request fails or the response status is not successful.
            asyncio.CancelledError: If collection is being stopped or the HTTP session is closed during shutdown.
            RuntimeError: If the HTTP session has not been initialized.
        """
        if self.stop_requested:
            raise asyncio.CancelledError("Telemetry collector is being stopped")

        if not self._session:
            raise RuntimeError("HTTP session not initialized. Call initialize() first.")

        if self._session.closed:
            raise asyncio.CancelledError("HTTP session is closed during shutdown")

        async with self._session.get(self._dcgm_url) as response:
            response.raise_for_status()
            text = await response.text()
            return text

    def _parse_metrics_to_records(self, metrics_data: str) -> list[TelemetryRecord]:
        """
        Convert Prometheus-formatted DCGM metrics text into TelemetryRecord objects grouped by GPU.
        
        Parameters:
            metrics_data (str): Raw Prometheus-formatted metrics text from the DCGM exporter.
        
        Returns:
            list[TelemetryRecord]: List of TelemetryRecord objects, one for each GPU that has parsed metrics.
        """
        if not metrics_data.strip():
            self.warning("Response from DCGM metrics endpoint is empty")
            return []

        current_timestamp = time.time_ns()
        gpu_data = {}
        gpu_metadata = {}

        try:
            for family in text_string_to_metric_families(metrics_data):
                for sample in family.samples:
                    metric_name = sample.name
                    labels = sample.labels
                    value = sample.value
                    gpu_index = labels.get("gpu")
                    if gpu_index is not None:
                        try:
                            gpu_index = int(gpu_index)
                        except ValueError:
                            continue
                    else:
                        continue

                    gpu_metadata[gpu_index] = {
                        "model_name": labels.get("modelName"),
                        "uuid": labels.get("UUID"),
                        "pci_bus_id": labels.get("pci_bus_id"),
                        "device": labels.get("device"),
                        "hostname": labels.get("Hostname"),
                    }

                    base_metric_name = metric_name.removesuffix("_total")
                    if base_metric_name in DCGM_TO_FIELD_MAPPING:
                        field_name = DCGM_TO_FIELD_MAPPING[base_metric_name]
                        gpu_data.setdefault(gpu_index, {})[field_name] = value
        except ValueError:
            self.warning("Failed to parse Prometheus metrics - invalid format")
            return []

        records = []
        for gpu_index, metrics in gpu_data.items():
            metadata = gpu_metadata.get(gpu_index, {})
            scaled_metrics = self._apply_scaling_factors(metrics)

            record = TelemetryRecord(
                timestamp_ns=current_timestamp,
                dcgm_url=self._dcgm_url,
                gpu_index=gpu_index,
                gpu_uuid=metadata.get("uuid", f"unknown-gpu-{gpu_index}"),
                gpu_model_name=metadata.get("model_name", f"GPU {gpu_index}"),
                pci_bus_id=metadata.get("pci_bus_id"),
                device=metadata.get("device"),
                hostname=metadata.get("hostname"),
                gpu_power_usage=scaled_metrics.get("gpu_power_usage"),
                energy_consumption=scaled_metrics.get("energy_consumption"),
                gpu_utilization=scaled_metrics.get("gpu_utilization"),
                gpu_memory_used=scaled_metrics.get("gpu_memory_used"),
                sm_clock_frequency=scaled_metrics.get("sm_clock_frequency"),
                memory_clock_frequency=scaled_metrics.get("memory_clock_frequency"),
                memory_temperature=scaled_metrics.get("memory_temperature"),
                gpu_temperature=scaled_metrics.get("gpu_temperature"),
            )
            records.append(record)

        return records

    def _apply_scaling_factors(self, metrics: dict) -> dict:
        """
        Apply configured scaling factors to metric values.
        
        Parameters:
            metrics (dict): Mapping from metric names to raw numeric values.
        
        Returns:
            dict: Copy of `metrics` where any value for a metric present in the collector's scaling factors has been multiplied by that factor; other entries are unchanged.
        """
        scaled_metrics = metrics.copy()
        for metric, factor in self._scaling_factors.items():
            if metric in scaled_metrics and scaled_metrics[metric] is not None:
                scaled_metrics[metric] *= factor
        return scaled_metrics
