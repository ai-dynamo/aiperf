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
    DEFAULT_DCGM_ENDPOINT,
    SCALING_FACTORS,
    URL_REACHABILITY_TIMEOUT,
)


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
        self._dcgm_url = dcgm_url
        self._collection_interval = collection_interval
        self._record_callback = record_callback
        self._error_callback = error_callback
        self._scaling_factors = SCALING_FACTORS
        self._session: aiohttp.ClientSession | None = None

        super().__init__(id=collector_id)

    @on_init
    async def _initialize_http_client(self) -> None:
        """Initialize the aiohttp client session."""
        timeout = aiohttp.ClientTimeout(total=URL_REACHABILITY_TIMEOUT)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self.debug(f"Initialized HTTP client for DCGM endpoint: {self._dcgm_url}")

    @on_stop
    async def _cleanup_http_client(self) -> None:
        """Clean up the aiohttp client session.

        Race conditions with background tasks are handled by checking
        self.stop_requested in the background task itself.
        """
        if self._session:
            await self._session.close()
            self._session = None
            self.debug("Cleaned up HTTP client session")

    async def is_url_reachable(self) -> bool:
        """Check if DCGM metrics endpoint is accessible.

        Returns:
            True if endpoint responds with 200, False otherwise
        """
        if not self._dcgm_url or not self._session:
            return False

        try:
            async with self._session.get(self._dcgm_url) as response:
                return response.status == 200
        except Exception:
            return False

    @background_task(immediate=True, interval=lambda self: self._collection_interval)
    async def _collect_telemetry_task(self) -> None:
        """Background task for collecting telemetry data at regular intervals.

        This uses the @background_task decorator which automatically handles
        lifecycle management and stopping when the collector is stopped.
        The interval is set to the collection_interval so this runs periodically.
        """
        self.debug(
            f"Starting telemetry collection task (interval: {self._collection_interval}s)"
        )
        try:
            await self._collect_and_process_metrics()
        except asyncio.CancelledError:
            self.debug("Telemetry collection task cancelled during shutdown")
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
        """Collect metrics from DCGM endpoint and process them into TelemetryRecord objects.

        This method fetches metrics, parses them, and sends them via callback.
        """
        try:
            metrics_data = await self._fetch_metrics()
            records = self._parse_metrics_to_records(metrics_data)

            if records and self._record_callback:
                self.debug(f"Sending {len(records)} records via callback")
                try:
                    if asyncio.iscoroutinefunction(self._record_callback):
                        await self._record_callback(records, self.id)
                    else:
                        self._record_callback(records, self.id)
                    self.debug("Successfully sent records via callback")
                except Exception as e:
                    self.warning(f"Failed to send telemetry records via callback: {e}")

        except Exception as e:
            self.error(f"Error collecting and processing metrics: {e}")
            raise

    async def _fetch_metrics(self) -> str:
        """Fetch raw metrics data from DCGM endpoint using aiohttp.

        Returns:
            Raw metrics text in Prometheus format

        Raises:
            aiohttp.ClientError: If HTTP request fails
            asyncio.CancelledError: If collector is being stopped
        """
        if self.stop_requested:
            raise asyncio.CancelledError("Telemetry collector is being stopped")

        if not self._session:
            raise RuntimeError("HTTP session not initialized. Call initialize() first.")

        if self._session.closed:
            raise asyncio.CancelledError("HTTP session is closed during shutdown")

        self.debug(f"Fetching metrics from {self._dcgm_url}")
        async with self._session.get(self._dcgm_url) as response:
            response.raise_for_status()
            text = await response.text()
            self.debug(f"Received {len(text)} characters from DCGM")
            return text

    def _parse_metrics_to_records(self, metrics_data: str) -> list[TelemetryRecord]:
        """Parse DCGM metrics text into TelemetryRecord objects using prometheus_client.

        Args:
            metrics_data: Raw metrics text from DCGM exporter

        Returns:
            List of TelemetryRecord objects, one per GPU with valid data
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
                            self.warning(f"Invalid GPU index: {gpu_index}")
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
                gpu_power_limit=scaled_metrics.get("gpu_power_limit"),
                energy_consumption=scaled_metrics.get("energy_consumption"),
                gpu_utilization=scaled_metrics.get("gpu_utilization"),
                gpu_memory_used=scaled_metrics.get("gpu_memory_used"),
                total_gpu_memory=scaled_metrics.get("total_gpu_memory"),
                sm_clock_frequency=scaled_metrics.get("sm_clock_frequency"),
                memory_clock_frequency=scaled_metrics.get("memory_clock_frequency"),
                memory_temperature=scaled_metrics.get("memory_temperature"),
                gpu_temperature=scaled_metrics.get("gpu_temperature"),
            )
            records.append(record)

        return records

    def _apply_scaling_factors(self, metrics: dict) -> dict:
        """Apply scaling factors to convert raw DCGM units to display units.

        Args:
            metrics: Dict of metric_name -> raw_value

        Returns:
            Dict with scaled values for display
        """
        scaled_metrics = metrics.copy()
        for metric, factor in self._scaling_factors.items():
            if metric in scaled_metrics and scaled_metrics[metric] is not None:
                scaled_metrics[metric] *= factor
        return scaled_metrics


# [AIP-331] TODO: @ilana-n Remove when CLI argument is added to AI perf (–server-metrics-url) and telemetry_data_collector.py no longer requires local testing via main().
async def main() -> None:
    """Main entry point for locally testing the telemetry collector."""

    import statistics
    from collections import defaultdict

    all_records = []
    metrics_data = defaultdict(lambda: defaultdict(list))

    def record_callback(records, collector_id):
        print(
            f"\n=== Received {len(records)} telemetry records from {collector_id} ==="
        )
        for record in records:
            all_records.append(record)
            print(f"\nGPU {record.gpu_index} ({record.gpu_model_name}):")
            print(f"  UUID: {record.gpu_uuid}")
            print(f"  Host: {record.hostname or 'Unknown'}")
            print(f"  PCI Bus: {record.pci_bus_id or 'Unknown'}")

            metric_values = {
                "Power Usage": (record.gpu_power_usage, "W"),
                "Power Limit": (record.gpu_power_limit, "W"),
                "Energy Consumption": (record.energy_consumption, "MJ"),
                "GPU Utilization": (record.gpu_utilization, "%"),
                "Memory Used": (record.gpu_memory_used, "GB"),
                "Total Memory": (record.total_gpu_memory, "GB"),
                "SM Clock Frequency": (record.sm_clock_frequency, "MHz"),
                "Memory Clock Frequency": (record.memory_clock_frequency, "MHz"),
                "Memory Temperature": (record.memory_temperature, "°C"),
                "GPU Temperature": (record.gpu_temperature, "°C"),
            }

            print("  Metrics:")
            for metric_name, (value, unit) in metric_values.items():
                if value is not None:
                    print(f"    {metric_name}: {value:.2f} {unit}")
                    metrics_data[record.gpu_uuid][metric_name].append(value)
                else:
                    print(f"    {metric_name}: N/A")

    def error_callback(error, collector_id):
        print(f"\nTelemetry error from {collector_id}: {error}")

    def print_final_summary():
        if not all_records:
            print("\nNo data collected for summary")
            return

        print("\nFINAL SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total records collected: {len(all_records)}")
        print(
            f"Collection period: {all_records[-1].timestamp_ns - all_records[0].timestamp_ns} ns"
        )
        print(f"Unique GPUs detected: {len(metrics_data)}")

        for gpu_uuid, gpu_metrics in metrics_data.items():
            gpu_record = next(r for r in all_records if r.gpu_uuid == gpu_uuid)

            print(
                f"\n GPU {gpu_record.gpu_index} ({gpu_record.gpu_model_name[:20]}...)"
            )
            print(f"   UUID: {gpu_uuid}")
            print(
                f"   Samples: {len([r for r in all_records if r.gpu_uuid == gpu_uuid])}"
            )

            for metric_name, values in gpu_metrics.items():
                if values:
                    avg_val = statistics.mean(values)
                    min_val = min(values)
                    max_val = max(values)
                    std_val = 0 if len(values) <= 1 else statistics.stdev(values)

                    unit = next(
                        (
                            unit
                            for name, (_, unit) in [
                                ("Power Usage", (None, "W")),
                                ("Power Limit", (None, "W")),
                                ("Energy Consumption", (None, "MJ")),
                                ("GPU Utilization", (None, "%")),
                                ("Memory Used", (None, "GB")),
                                ("Total Memory", (None, "GB")),
                                ("SM Clock Frequency", (None, "MHz")),
                                ("Memory Clock Frequency", (None, "MHz")),
                                ("Memory Temperature", (None, "°C")),
                                ("GPU Temperature", (None, "°C")),
                            ]
                            if name == metric_name
                        ),
                        "",
                    )

                    print(f"    {metric_name}:")
                    print(f"      Avg: {avg_val:.2f} {unit}")
                    print(f"      Min: {min_val:.2f} {unit}")
                    print(f"      Max: {max_val:.2f} {unit}")
                    print(f"      Std: {std_val:.2f} {unit}")

    print("AIPerf GPU Telemetry Collector Test")
    print("=====================================")

    collector = TelemetryDataCollector(
        dcgm_url=DEFAULT_DCGM_ENDPOINT,
        record_callback=record_callback,
        error_callback=error_callback,
        collector_id="test_collector",
    )

    await collector.initialize_and_start()

    try:
        if await collector.is_url_reachable():
            print("DCGM endpoint reachable - starting collection...")
            print("Collection interval: 1.0 seconds")
            print("Press Ctrl+C to stop and see summary")

            try:
                await asyncio.sleep(10)
            except KeyboardInterrupt:
                print("\nStopping collection...")
            finally:
                print_final_summary()
        else:
            print("DCGM endpoint not reachable at http://localhost:9401/metrics")
            print("Make sure DCGM is running and accessible")
    finally:
        await collector.stop()


if __name__ == "__main__":
    asyncio.run(main())
