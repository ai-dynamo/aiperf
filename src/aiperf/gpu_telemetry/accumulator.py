# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import datetime
from typing import Any

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import GPUTelemetryMode
from aiperf.common.environment import Environment
from aiperf.common.exceptions import NoMetricValue, PostProcessorDisabled
from aiperf.common.hooks import background_task
from aiperf.common.messages import RealtimeTelemetryMetricsMessage
from aiperf.common.models import (
    EndpointData,
    ErrorDetailsCount,
    GpuSummary,
    MetricResult,
    TelemetryExportData,
    TelemetrySummary,
)
from aiperf.common.models.server_metrics_models import TimeRangeFilter
from aiperf.common.models.telemetry_models import TelemetryHierarchy, TelemetryRecord
from aiperf.common.protocols import PubClientProtocol
from aiperf.exporters.display_units_utils import normalize_endpoint_display
from aiperf.gpu_telemetry.constants import (
    GPU_TELEMETRY_COUNTER_METRICS,
    get_gpu_telemetry_metrics_config,
)
from aiperf.plugin.enums import UIType
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


class GPUTelemetryAccumulator(BaseMetricsProcessor):
    """Accumulate GPU telemetry records and compute metrics in a hierarchical structure.

    Processes TelemetryRecord objects from GPU monitoring into hierarchical storage
    organized by endpoint, hostname, GPU device, and metric. Computes summary statistics
    and supports realtime telemetry updates for dashboard display.

    Features:
        - Hierarchical storage (endpoint -> hostname -> device -> metric)
        - Summary statistics computation with time filtering
        - Realtime metric publishing for dashboard UI
        - Background task for periodic metric updates

    Args:
        user_config: User configuration including GPU telemetry settings
        service_config: Service configuration for communication and UI settings
        pub_client: Publish client for sending realtime metric updates
        **kwargs: Additional arguments passed to base class

    Raises:
        PostProcessorDisabled: If GPU telemetry is disabled via --no-gpu-telemetry
    """

    def __init__(
        self,
        user_config: UserConfig,
        service_config: ServiceConfig,
        pub_client: PubClientProtocol,
        **kwargs: Any,
    ):
        if user_config.gpu_telemetry_disabled:
            raise PostProcessorDisabled(
                "GPU telemetry accumulator is disabled via --no-gpu-telemetry"
            )
        self.pub_client = pub_client
        self.service_config = service_config
        super().__init__(
            user_config=user_config, service_config=service_config, **kwargs
        )

        self._hierarchy = TelemetryHierarchy()
        self._realtime_enable_event = asyncio.Event()
        self._last_metric_values: dict[str, float | None] | None = None
        self._total_metrics_generated = 0

    async def process_telemetry_record(self, record: TelemetryRecord) -> None:
        """Process individual GPU telemetry record into hierarchical storage.

        Args:
            record: GPU TelemetryRecord containing GPU metrics and hierarchical metadata
        """
        self._hierarchy.add_record(record)

    def start_realtime_telemetry(self) -> None:
        """Start the realtime telemetry background task.

        This is called when the user dynamically enables the telemetry dashboard
        by pressing the telemetry option in the UI without having passed the 'dashboard' parameter
        at startup.
        """
        self.info("Received START_REALTIME_TELEMETRY command")

        self.user_config.gpu_telemetry_mode = GPUTelemetryMode.REALTIME_DASHBOARD

        # Wake up the sleeping telemetry task
        self._realtime_enable_event.set()

    @background_task(interval=None, immediate=True)
    async def _report_realtime_telemetry_metrics_task(self) -> None:
        """Report GPU telemetry metrics - sleeps when disabled, resumes on command."""
        if self.service_config.ui_type != UIType.DASHBOARD:
            return

        while not self.stop_requested:
            if (
                self.user_config.gpu_telemetry_mode
                != GPUTelemetryMode.REALTIME_DASHBOARD
            ):
                # Disabled - sleep until command wakes us
                await self._realtime_enable_event.wait()
                self._realtime_enable_event.clear()
                continue

            await self._report_realtime_metrics()
            await asyncio.sleep(Environment.UI.REALTIME_METRICS_INTERVAL)

    async def _report_realtime_metrics(self) -> None:
        """Report real-time GPU telemetry metrics."""

        # TODO: This can keep track of the last update time and only publish
        # if the time has elapsed. (and avoid summarizing the metrics again)

        telemetry_metrics = await self.summarize()
        self._total_metrics_generated += len(telemetry_metrics)

        if telemetry_metrics:
            # Only publish if values have changed - extract once for efficiency
            new_values = {m.tag: m.current for m in telemetry_metrics}
            if (
                self._last_metric_values is None
                or new_values != self._last_metric_values
            ):
                await self.pub_client.publish(
                    RealtimeTelemetryMetricsMessage(
                        service_id=self.id,
                        metrics=telemetry_metrics,
                    )
                )
                self._last_metric_values = new_values

    async def summarize(self) -> list[MetricResult]:
        """Generate GPU MetricResult list for real-time display and final export.

        This method is called by RecordsManager for:
        1. Final results generation when profiling completes
        2. Real-time dashboard updates when --gpu-telemetry dashboard is enabled

        Returns:
            List of MetricResult objects, one per GPU per metric type.
            Tags follow hierarchical naming pattern for dashboard filtering.
        """
        results: list[MetricResult] = []

        for dcgm_url, gpu_data in self._hierarchy.dcgm_endpoints.items():
            endpoint_display = normalize_endpoint_display(dcgm_url)

            for gpu_uuid, telemetry_data in gpu_data.items():
                gpu_index = telemetry_data.metadata.gpu_index
                model_name = telemetry_data.metadata.gpu_model_name

                for (
                    metric_display,
                    metric_name,
                    unit_enum,
                ) in get_gpu_telemetry_metrics_config():
                    try:
                        dcgm_tag = (
                            dcgm_url.replace(":", "_")
                            .replace("/", "_")
                            .replace(".", "_")
                        )
                        tag = f"{metric_name}_dcgm_{dcgm_tag}_gpu{gpu_index}_{gpu_uuid[:12]}"

                        header = f"{metric_display} | {endpoint_display} | GPU {gpu_index} | {model_name}"

                        result = telemetry_data.get_metric_result(
                            metric_name, tag, header, unit_enum
                        )
                        results.append(result)
                    except NoMetricValue:
                        self.debug(
                            f"No data available for metric '{metric_name}' on GPU {gpu_uuid[:12]} from {dcgm_url}"
                        )
                        continue
                    except Exception as e:
                        self.exception(
                            f"Unexpected error generating metric result for '{metric_name}' on GPU {gpu_uuid[:12]} from {dcgm_url}: {e}"
                        )
                        continue

        return results

    def export_results(
        self,
        start_ns: int | None = None,
        end_ns: int | None = None,
        error_summary: list[ErrorDetailsCount] | None = None,
    ) -> "TelemetryExportData | None":
        """Export accumulated telemetry data as a TelemetryExportData object.

        Transforms the internal numpy-backed telemetry hierarchy into a serializable
        format with pre-computed metric statistics for each GPU.

        Time filtering is applied to exclude warmup periods from statistics:
        - Gauge metrics (power, utilization, etc.): Stats computed on filtered data only
        - Counter metrics (energy, errors): Delta computed from baseline before start_ns

        Args:
            start_ns: Start time of profiling phase in nanoseconds (excludes warmup).
                     If None, includes all data from beginning.
            end_ns: End time of profiling phase in nanoseconds. If None, includes all
                   data after start_ns (including final scrape after profiling completes).
            error_summary: Optional list of error counts

        Returns:
            TelemetryExportData object with pre-computed metrics for each GPU
        """
        # Create time filter for warmup exclusion
        # Note: end_ns is typically None to include the final telemetry scrape
        # that occurs after PROFILE_COMPLETE but before export
        time_filter = TimeRangeFilter(start_ns=start_ns, end_ns=end_ns)

        # Build summary
        # When start_ns/end_ns is None, use current time as the timestamp
        start_time = (
            datetime.fromtimestamp(start_ns / NANOS_PER_SECOND)
            if start_ns is not None
            else datetime.now()
        )
        end_time = (
            datetime.fromtimestamp(end_ns / NANOS_PER_SECOND)
            if end_ns is not None
            else datetime.now()
        )
        summary = TelemetrySummary(
            endpoints_configured=list(self._hierarchy.dcgm_endpoints.keys()),
            endpoints_successful=list(self._hierarchy.dcgm_endpoints.keys()),
            start_time=start_time,
            end_time=end_time,
        )

        # Build endpoints dict with pre-computed metrics
        endpoints: dict[str, EndpointData] = {}

        if self._hierarchy.dcgm_endpoints:
            for (
                dcgm_url,
                gpus_data,
            ) in self._hierarchy.dcgm_endpoints.items():
                endpoint_display = normalize_endpoint_display(dcgm_url)
                gpus_dict: dict[str, GpuSummary] = {}

                for gpu_uuid, gpu_data in gpus_data.items():
                    metrics_dict = {}

                    for (
                        metric_display,
                        metric_key,
                        unit_enum,
                    ) in get_gpu_telemetry_metrics_config():
                        try:
                            is_counter = metric_key in GPU_TELEMETRY_COUNTER_METRICS
                            metric_result = gpu_data.get_metric_result(
                                metric_key,
                                metric_key,
                                metric_display,
                                unit_enum,
                                time_filter=time_filter,
                                is_counter=is_counter,
                            )
                            metrics_dict[metric_key] = metric_result.to_json_result()
                        except NoMetricValue:
                            continue
                        except Exception as e:
                            self.warning(
                                f"Failed to compute metric '{metric_key}' for GPU {gpu_uuid[:12]}: {e}"
                            )
                            continue

                    gpu_summary = GpuSummary(
                        gpu_index=gpu_data.metadata.gpu_index,
                        gpu_name=gpu_data.metadata.gpu_model_name,
                        gpu_uuid=gpu_uuid,
                        hostname=gpu_data.metadata.hostname,
                        namespace=gpu_data.metadata.namespace,
                        pod_name=gpu_data.metadata.pod_name,
                        metrics=metrics_dict,
                    )

                    gpus_dict[f"gpu_{gpu_data.metadata.gpu_index}"] = gpu_summary

                endpoints[endpoint_display] = EndpointData(gpus=gpus_dict)

        return TelemetryExportData(
            summary=summary, endpoints=endpoints, error_summary=error_summary
        )
