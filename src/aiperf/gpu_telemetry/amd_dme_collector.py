import time

from prometheus_client.parser import text_string_to_metric_families

from aiperf.common.environment import Environment
from aiperf.common.mixins import (
    BaseMetricsCollectorMixin,
    TErrorCallback,
    TRecordCallback,
)
from aiperf.common.models import GpuMetadata, TelemetryMetrics, TelemetryRecord

__all__ = ["AMDDMETelemetryCollector"]

SCALING_FACTORS = {
    "amd_energy_consumption": 1e-12,
    "amd_memory_used": 1.0 / 1024.0,
    "amd_memory_free": 1.0 / 1024.0,
    "amd_memory_total": 1.0 / 1024.0,
}


class AMDDMETelemetryCollector(BaseMetricsCollectorMixin[TelemetryRecord]):
    """Collects AMD GPU telemetry metrics from AMD Device Metrics Exporter (DME) HTTP endpoints.

    Async collector that fetches AMD GPU metrics from Prometheus-compatible AMD DME
    (gpu-operator-metrics-exporter) and converts them to TelemetryRecord objects.
    This enables remote AMD GPU monitoring via AMD's Device Metrics Exporter.

    Features:
        - Async HTTP collection with aiohttp
        - AMD Prometheus format parsing
        - GPU metadata extraction (serial_number, model, hostname)
        - Automatic unit scaling (e.g., MB to GB, mJ to MJ)
        - Callback-based record delivery
        - Maps AMD-specific metrics to standard telemetry fields

    Args:
        dcgm_url: URL of the AMD DME endpoint (e.g., "http://<amd-exporter-ip>:5000/metrics")
        collection_interval: Interval in seconds between metric collections (default: from Environment)
        reachability_timeout: Timeout in seconds for reachability checks (default: from Environment)
        record_callback: Optional async callback to receive collected records.
            Signature: async (records: list[TelemetryRecord], collector_id: str) -> None
        error_callback: Optional async callback to receive collection errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector instance
    """

    @classmethod
    def validate_environment(cls) -> None:
        """Remote HTTP collector GÇö no local environment to validate."""

    def __init__(
        self,
        dcgm_url: str,
        *,
        collection_interval: float = Environment.GPU.COLLECTION_INTERVAL,
        reachability_timeout: float = Environment.GPU.REACHABILITY_TIMEOUT,
        record_callback: TRecordCallback | None = None,
        error_callback: TErrorCallback | None = None,
        collector_id: str = "telemetry_collector",
    ) -> None:
        self._scaling_factors = SCALING_FACTORS
        super().__init__(
            endpoint_url=dcgm_url,
            collection_interval=collection_interval,
            reachability_timeout=reachability_timeout,
            record_callback=record_callback,
            error_callback=error_callback,
            id=collector_id,
        )

    async def _collect_and_process_metrics(self) -> None:
        """Collect metrics from AMD exporter endpoint and process them into TelemetryRecord objects.

        Implements the abstract method from BaseMetricsCollectorMixin.

        Orchestrates the full collection flow:
        1. Fetches raw metrics data from AMD exporter endpoint (via mixin's _fetch_metrics_text)
        2. Parses Prometheus-format data into TelemetryRecord objects
        3. Sends records via callback (via mixin's _send_records_via_callback)

        Raises:
            Exception: Any exception from fetch or parse is logged and re-raised
        """
        fetch_result = await self._fetch_metrics_text()
        if fetch_result.is_duplicate:
            return
        records = self._parse_metrics_to_records(fetch_result.text)
        await self._send_records_via_callback(records)

    # AMD DME metric name -> TelemetryRecord field. A table rather than an
    # if/elif chain: adding an exporter metric is a one-line edit here, and the
    # per-sample handler stays flat enough to read.
    _METRIC_FIELDS: dict[str, str] = {
        "gpu_package_power": "amd_power",
        "gpu_energy_consumed": "amd_energy_consumption",
        "gpu_gfx_activity": "amd_gfx_activity",
        "gpu_umc_activity": "amd_umc_activity",
        "gpu_memory_activity": "amd_umc_activity",
        "gpu_memory_used": "amd_memory_used",
        "gpu_used_vram": "amd_memory_used",
        "gpu_free_vram": "amd_memory_free",
        "gpu_total_vram": "amd_memory_total",
        "gpu_junction_temperature": "amd_temperature",
        "gpu_memory_temperature": "amd_memory_temperature",
        "gpu_ecc_uncorrect_total": "amd_ecc_uncorrectable",
    }

    # gpu_clock carries the clock in labels rather than in the metric name, so
    # (clock_type, clock_index) selects which field it lands in.
    _CLOCK_FIELDS: dict[tuple[str, str], str] = {
        ("GPU_CLOCK_TYPE_SYSTEM", "0"): "amd_sm_clock",
        ("GPU_CLOCK_TYPE_MEMORY", "8"): "amd_mem_clock",
    }

    @staticmethod
    def _gpu_index_from(labels: dict) -> int | None:
        """The integer GPU index from a sample's labels, or None if unusable."""
        gpu_id = labels.get("gpu_id")
        if gpu_id is None:
            return None
        try:
            return int(gpu_id)
        except ValueError:
            return None

    def _ingest_sample(
        self,
        sample,
        gpu_data: dict[int, dict[str, float]],
        gpu_metadata: dict[int, GpuMetadata],
    ) -> None:
        """Fold one Prometheus sample into the per-GPU accumulators.

        Non-finite values and samples without a usable gpu_id are skipped, since
        a NaN would otherwise reach the record and serialize as null.
        """
        value = sample.value
        if isinstance(value, float) and (
            value != value or value in (float("inf"), float("-inf"))
        ):
            return

        labels = sample.labels
        gpu_index = self._gpu_index_from(labels)
        if gpu_index is None:
            return

        if gpu_index not in gpu_metadata:
            gpu_metadata[gpu_index] = GpuMetadata(
                gpu_index=gpu_index,
                gpu_model_name=labels.get("card_model", "Unknown AMD GPU"),
                gpu_uuid=labels.get("serial_number", f"amd-gpu-{gpu_index}"),
                pci_bus_id=None,
                device=None,
                hostname=labels.get("hostname"),
                namespace=labels.get("namespace"),
                pod_name=labels.get("pod"),
            )

        metrics = gpu_data.setdefault(gpu_index, {})

        field = self._METRIC_FIELDS.get(sample.name)
        if field is not None:
            metrics[field] = value
            return

        if sample.name == "gpu_clock":
            clock = (labels.get("clock_type", ""), labels.get("clock_index", ""))
            clock_field = self._CLOCK_FIELDS.get(clock)
            if clock_field is not None:
                metrics[clock_field] = value

    def _parse_metrics_to_records(self, metrics_data: str) -> list[TelemetryRecord]:
        """Parse AMD metrics text into TelemetryRecord objects using prometheus_client.

        Processes Prometheus exposition format metrics from AMD gpu-operator-metrics-exporter:
        1. Parses metric families using prometheus_client parser
        2. Extracts GPU metadata (serial_number, card_model, hostname, etc.) from labels
        3. Maps AMD metric names to TelemetryRecord field names
        4. Applies scaling factors to convert units (e.g., MB to GB, mJ to MJ)
        5. Aggregates metrics by GPU ID into TelemetryRecord objects

        AMD-specific metric mappings:
        - gpu_package_power -> amd_power (W)
        - gpu_energy_consumed -> amd_energy_consumption (uJ -> MJ)
        - gpu_gfx_activity -> amd_gfx_activity (%)
        - gpu_memory_activity -> amd_umc_activity (%)
        - gpu_used_vram -> amd_memory_used (MB -> GB)
        - gpu_clock{clock_type="GPU_CLOCK_TYPE_SYSTEM",clock_index="0"} -> amd_sm_clock (MHz)
        - gpu_clock{clock_type="GPU_CLOCK_TYPE_MEMORY",clock_index="8"} -> amd_mem_clock (MHz)

        Skips non-finite values (NaN, inf) and metrics without valid GPU ID.

        Args:
            metrics_data: Raw metrics text from AMD exporter in Prometheus format

        Returns:
            list[TelemetryRecord]: List of TelemetryRecord objects, one per GPU with valid data.
                Returns empty list if metrics_data is empty or parsing fails.
        """
        if not metrics_data.strip():
            return []

        current_timestamp = time.time_ns()
        gpu_data: dict[int, dict[str, float]] = {}
        gpu_metadata: dict[int, GpuMetadata] = {}

        try:
            for family in text_string_to_metric_families(metrics_data):
                for sample in family.samples:
                    self._ingest_sample(sample, gpu_data, gpu_metadata)

        except ValueError as e:
            self.warning(f"Failed to parse Prometheus metrics - invalid format: {e}")
            return []

        records = []
        for gpu_index, metrics in gpu_data.items():
            metadata = gpu_metadata.get(gpu_index)
            if metadata is None:
                self.warning(f"No metadata found for GPU {gpu_index}")
                continue
            scaled_metrics = self._apply_scaling_factors(metrics)

            record = TelemetryRecord(
                timestamp_ns=current_timestamp,
                telemetry_source_url=self.endpoint_url,
                **metadata.model_dump(),
                telemetry_data=TelemetryMetrics(**scaled_metrics),
            )
            records.append(record)

        return records

    def _apply_scaling_factors(self, metrics: dict) -> dict:
        """Apply scaling factors to convert raw AMD units to display units.

        Converts metrics from AMD's native units to human-readable units:
        - Energy: microjoules (uJ) -> megajoules (MJ)
        - Memory: MB -> GB

        Only applies scaling to metrics present in the input dict. None values are preserved.

        Args:
            metrics: Dict of metric_name -> raw_value from AMD exporter

        Returns:
            dict: New dict with scaled values ready for display. Unscaled metrics are copied as-is.
        """
        scaled_metrics = metrics.copy()
        for metric, factor in self._scaling_factors.items():
            if metric in scaled_metrics and scaled_metrics[metric] is not None:
                scaled_metrics[metric] *= factor
        return scaled_metrics
