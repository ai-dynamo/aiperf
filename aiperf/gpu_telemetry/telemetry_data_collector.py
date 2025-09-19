# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import requests
import time
from threading import Thread, Event
from typing import Optional, Callable

from aiperf.common.models import ErrorDetails
from aiperf.common.models.telemetry_models import TelemetryRecord
from aiperf.gpu_telemetry.constants import (
    DEFAULT_COLLECTION_INTERVAL,
    DEFAULT_DCGM_ENDPOINT,
    DCGM_TO_FIELD_MAPPING,
    SCALING_FACTORS,
    THREAD_JOIN_TIMEOUT,
    URL_REACHABILITY_TIMEOUT,
)


class TelemetryDataCollector:
    """Collects telemetry metrics from DCGM metrics endpoint.

    Simple collector that fetches GPU metrics from DCGM exporter and converts them to
    TelemetryRecord objects. Uses callback pattern to send data to parent service.
    - No service dependencies (avoids circular imports)
    - Sends TelemetryRecord list via callback function
    - Handles errors gracefully with ErrorDetails
    - No local storage (follows centralized architecture)
    """

    def __init__(
        self, 
        dcgm_url: str,
        collection_interval: float = DEFAULT_COLLECTION_INTERVAL,
        record_callback: Optional[Callable[[list[TelemetryRecord]], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None,
        collector_id: str = "telemetry_collector"
    ) -> None:
        self._dcgm_url = dcgm_url
        self._collection_interval = collection_interval
        self._record_callback = record_callback
        self._error_callback = error_callback
        self._collector_id = collector_id
        
        self._stop_event = Event()
        self._collection_thread: Optional[Thread] = None
        
        self._scaling_factors = SCALING_FACTORS
        
        self._logger = logging.getLogger(f"telemetry.{collector_id}")

    def is_url_reachable(self) -> bool:
        """Check if DCGM metrics endpoint is accessible.

        Returns:
            True if endpoint responds with 200, False otherwise
        """

        timeout_seconds = URL_REACHABILITY_TIMEOUT
        if self._dcgm_url:
            try:
                response = requests.get(self._dcgm_url, timeout=timeout_seconds)
                return response.status_code == requests.codes.ok
            except requests.RequestException:
                return False
        return False

    def start(self) -> None:
        """Start telemetry collection in background thread.

        Note: Uses thread instead of async task because requests library
        is synchronous and we don't want to block the main event loop.
        """

        if self._collection_thread is None or not self._collection_thread.is_alive():
            self._logger.info(f"Starting telemetry collection from {self._dcgm_url}")
            self._stop_event.clear()
            self._collection_thread = Thread(target=self._collection_loop, daemon=True)
            self._collection_thread.start()

    def stop(self) -> None:
        """Stop telemetry collection and wait for thread to finish."""

        if self._collection_thread is not None and self._collection_thread.is_alive():
            self._logger.info("Stopping telemetry collection")
            self._stop_event.set()
            self._collection_thread.join(timeout=THREAD_JOIN_TIMEOUT)
            if self._collection_thread.is_alive():
                self._logger.warning("Telemetry collection thread did not stop gracefully")

    def _collection_loop(self) -> None:
        """Main collection loop running in background thread.
        
        Continuously collects telemetry and sends TelemetryRecord list
        via callback until stop is requested.
        """
        while not self._stop_event.is_set():
            try:
                metrics_data = self._fetch_metrics()
                
                records = self._parse_metrics_to_records(metrics_data)
                
                if records and self._record_callback:
                    try:
                        self._record_callback(records)
                    except Exception as e:
                        self._logger.warning(f"Failed to send telemetry records via callback: {e}")
                
            except Exception as e:
                if self._error_callback:
                    try:
                        self._error_callback(e)
                    except Exception as callback_error:
                        self._logger.error(f"Failed to send error via callback: {callback_error}")
                else:
                    self._logger.error(f"Telemetry collection error: {e}")
                
            if not self._stop_event.wait(self._collection_interval):
                continue

    def _fetch_metrics(self) -> str:
        """Fetch raw metrics data from DCGM endpoint.

        Returns:
            Raw metrics text in Prometheus format

        Raises:
            requests.RequestException: If HTTP request fails
        """

        response = requests.get(self._dcgm_url, timeout=URL_REACHABILITY_TIMEOUT)
        response.raise_for_status()
        return response.text

    def _parse_metrics_to_records(self, metrics_data: str) -> list[TelemetryRecord]:
        """Parse DCGM metrics text into TelemetryRecord objects.

        Args:
            metrics_data: Raw metrics text from DCGM exporter

        Returns:
            List of TelemetryRecord objects, one per GPU with valid data
        """

        if not metrics_data.strip():
            self._logger.warning("Response from DCGM metrics endpoint is empty")
            return []

        current_timestamp = time.perf_counter_ns()
        gpu_data = {}
        gpu_metadata = {}

        for line in metrics_data.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parsed = self._parse_metric_line(line)
            if not parsed:
                continue

            metric_name, gpu_index, value, model_name, uuid, pci_bus_id, device, hostname = parsed
            
            gpu_metadata[gpu_index] = {
                'model_name': model_name,
                'uuid': uuid,
                'pci_bus_id': pci_bus_id,
                'device': device,
                'hostname': hostname
            }
            
            if metric_name in DCGM_TO_FIELD_MAPPING:
                field_name = DCGM_TO_FIELD_MAPPING[metric_name]
                gpu_data.setdefault(gpu_index, {})[field_name] = value

        records = []
        for gpu_index, metrics in gpu_data.items():
            metadata = gpu_metadata.get(gpu_index, {})
            
            scaled_metrics = self._apply_scaling_factors(metrics)

            record = TelemetryRecord(
                timestamp_ns=current_timestamp,
                dcgm_url=self._dcgm_url,
                gpu_index=gpu_index,
                gpu_uuid=metadata.get('uuid', f"unknown-gpu-{gpu_index}"),
                gpu_model_name=metadata.get('model_name', f"GPU {gpu_index}"),
                pci_bus_id=metadata.get('pci_bus_id'),
                device=metadata.get('device'),
                hostname=metadata.get('hostname'),
                gpu_power_usage=scaled_metrics.get("gpu_power_usage"),
                gpu_power_limit=scaled_metrics.get("gpu_power_limit"),
                energy_consumption=scaled_metrics.get("energy_consumption"),
                gpu_utilization=scaled_metrics.get("gpu_utilization"),
                gpu_memory_used=scaled_metrics.get("gpu_memory_used"),
                total_gpu_memory=scaled_metrics.get("total_gpu_memory")
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
    
    def _parse_metric_line(self, line: str) -> Optional[tuple[str, int, float, str, str, str, str, str]]:
        """Parse a single metric line from DCGM output.

        Expected format: metric_name{gpu="0",UUID="...",pci_bus_id="...",device="...",modelName="...",Hostname="..."} value

        Returns:
            Tuple of (metric_name, gpu_index, value, model_name, uuid, pci_bus_id, device, hostname)
            or None if parsing fails
        """

        try:
            metric_full_name, value_str = line.rsplit(" ", 1)
            value = float(value_str.strip())
            
            metric_name = metric_full_name.split("{")[0]
            gpu_index = self._extract_gpu_index(metric_full_name)
            model_name = self._extract_model_name(metric_full_name)
            uuid = self._extract_uuid(metric_full_name)
            pci_bus_id = self._extract_pci_bus_id(metric_full_name)
            device = self._extract_device(metric_full_name)
            hostname = self._extract_hostname(metric_full_name)
            
            if gpu_index is not None and model_name:
                return metric_name, gpu_index, value, model_name, uuid, pci_bus_id, device, hostname
        except (ValueError, IndexError) as e:
            self._logger.warning(f"Could not parse metric line: {line} - {e}")
        return None
    
    def _extract_gpu_index(self, metric_full_name: str) -> Optional[int]:
        """Extract GPU index from metric labels."""

        try:
            match = re.search(r'(?:^|[,{])gpu="(\d+)"', metric_full_name)
            if match:
                return int(match.group(1))
        except (ValueError, AttributeError) as e:
            self._logger.warning(f"Failed to extract GPU index from: {metric_full_name} - {e}")
        return None
    
    def _extract_model_name(self, metric_full_name: str) -> Optional[str]:
        """Extract GPU model name from metric labels."""

        try:
            match = re.search(r'modelName="([^"]+)"', metric_full_name)
            if match:
                return match.group(1)
        except AttributeError as e:
            self._logger.warning(f"Failed to extract model name from: {metric_full_name} - {e}")
        return None
    
    def _extract_uuid(self, metric_full_name: str) -> Optional[str]:
        """Extract GPU UUID from metric labels."""

        try:
            match = re.search(r'UUID="([^"]+)"', metric_full_name)
            if match:
                return match.group(1)
        except AttributeError as e:
            self._logger.warning(f"Failed to extract UUID from: {metric_full_name} - {e}")
        return None
    
    def _extract_pci_bus_id(self, metric_full_name: str) -> Optional[str]:
        """Extract PCI Bus ID from metric labels."""

        try:
            match = re.search(r'pci_bus_id="([^"]+)"', metric_full_name)
            if match:
                return match.group(1)
        except AttributeError as e:
            self._logger.warning(f"Failed to extract PCI Bus ID from: {metric_full_name} - {e}")
        return None
    
    def _extract_device(self, metric_full_name: str) -> Optional[str]:
        """Extract device identifier from metric labels."""

        try:
            match = re.search(r'device="([^"]+)"', metric_full_name)
            if match:
                return match.group(1)
        except AttributeError as e:
            self._logger.warning(f"Failed to extract device from: {metric_full_name} - {e}")
        return None
    
    def _extract_hostname(self, metric_full_name: str) -> Optional[str]:
        """Extract hostname from metric labels."""

        try:
            match = re.search(r'Hostname="([^"]+)"', metric_full_name)
            if match:
                return match.group(1)
        except AttributeError as e:
            self._logger.warning(f"Failed to extract hostname from: {metric_full_name} - {e}")
        return None

def main() -> None:
    """Main entry point for locally testing the telemetry collector."""

    import time
    from collections import defaultdict
    import statistics
    
    all_records = []
    metrics_data = defaultdict(lambda: defaultdict(list))  # gpu_uuid -> metric_name -> values
    
    def record_callback(records):
        print(f"\n=== Received {len(records)} telemetry records ===")
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
            }
            
            print("  Metrics:")
            for metric_name, (value, unit) in metric_values.items():
                if value is not None:
                    print(f"    {metric_name}: {value:.2f} {unit}")
                    metrics_data[record.gpu_uuid][metric_name].append(value)
                else:
                    print(f"    {metric_name}: N/A")
    
    def error_callback(error):
        print(f"\nTelemetry error: {error}")
    
    def print_final_summary():
        if not all_records:
            print("\nNo data collected for summary")
            return
            
        print(f"\nFINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total records collected: {len(all_records)}")
        print(f"Collection period: {all_records[-1].timestamp_ns - all_records[0].timestamp_ns} ns")
        print(f"Unique GPUs detected: {len(metrics_data)}")
        
        for gpu_uuid, gpu_metrics in metrics_data.items():
            gpu_record = next(r for r in all_records if r.gpu_uuid == gpu_uuid)
            
            print(f"\n GPU {gpu_record.gpu_index} ({gpu_record.gpu_model_name[:20]}...)")
            print(f"   UUID: {gpu_uuid}")
            print(f"   Samples: {len([r for r in all_records if r.gpu_uuid == gpu_uuid])}")
            
            for metric_name, values in gpu_metrics.items():
                if values:
                    avg_val = statistics.mean(values)
                    min_val = min(values)
                    max_val = max(values)
                    if len(values) <= 1:
                        std_val = 0
                    else:
                        std_val = statistics.stdev(values)
                    
                    unit = next((unit for name, (_, unit) in [
                        ("Power Usage", (None, "W")),
                        ("Power Limit", (None, "W")),
                        ("Energy Consumption", (None, "mJ")),
                        ("GPU Utilization", (None, "%")),
                        ("Memory Used", (None, "GB")),
                        ("Total Memory", (None, "GB")),
                    ] if name == metric_name), "")
                    
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
        collector_id="test_collector"
    )
    
    if collector.is_url_reachable():
        print("DCGM endpoint reachable - starting collection...")
        print("Collection interval: 1.0 seconds")
        print("Press Ctrl+C to stop and see summary")
        
        collector.start()
        try:
            time.sleep(10)  # Collect for 1s or until interrupted
        except KeyboardInterrupt:
            print("\nStopping collection...")
        finally:
            collector.stop()
            print_final_summary()
    else:
        print("DCGM endpoint not reachable at http://localhost:9401/metrics")
        print("Make sure DCGM is running and accessible")


if __name__ == "__main__":
    main()