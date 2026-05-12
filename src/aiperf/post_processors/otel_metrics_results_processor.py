# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import multiprocessing as mp
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from queue import Empty, Full
from typing import Any

from aiperf.common.config import MLflowDefaults, UserConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.exceptions import (
    MetricTypeError,
    MetricUnitError,
    PostProcessorDisabled,
)
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models import CreditPhaseStats, MetricResult
from aiperf.common.optional_dependencies import (
    OTEL_METRICS_STREAMING_FEATURE,
    otel_dependency_message,
)
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor
from aiperf.post_processors.otel_streaming_fanout import (
    OTelStreamingFanoutConfig,
    run_otel_streaming_fanout,
)
from aiperf.post_processors.strategies import (
    MetricResultsStrategy,
    OTelResultData,
    OTelResultsStrategyProtocol,
    TimingResultsStrategy,
)


@dataclass
class _FanoutHistogramInstrument:
    """Proxy histogram instrument that emits events to the fanout queue."""

    metric_name: str
    unit: str
    description: str
    emit_event: Callable[[str, dict[str, Any]], None]

    def record(self, value: float, attributes: dict[str, Any]) -> None:
        self.emit_event(
            "histogram_record",
            {
                "metric_name": self.metric_name,
                "unit": self.unit,
                "description": self.description,
                "value": float(value),
                "attributes": attributes,
            },
        )


@dataclass
class _FanoutAddInstrument:
    """Proxy counter-like instrument that emits add events to the fanout queue."""

    event_type: str
    metric_name: str
    unit: str
    description: str
    emit_event: Callable[[str, dict[str, Any]], None]

    def add(self, value: float, attributes: dict[str, Any]) -> None:
        self.emit_event(
            self.event_type,
            {
                "metric_name": self.metric_name,
                "unit": self.unit,
                "description": self.description,
                "value": float(value),
                "attributes": attributes,
            },
        )


class OTelMetricsResultsProcessor(BaseMetricsProcessor):
    """Streams record and timing telemetry to configured live sinks."""

    def __init__(
        self,
        service_id: str | None,
        user_config: UserConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(user_config=user_config, **kwargs)
        self.service_id = service_id or "records-manager"
        self._otel_metrics_url = user_config.otel_metrics_url
        self._mlflow_live_enabled = user_config.mlflow_enabled

        if not self._otel_metrics_url and not self._mlflow_live_enabled:
            self.info(
                "Telemetry streaming is disabled "
                "(set --otel-url and/or --mlflow-tracking-uri to enable)"
            )
            raise PostProcessorDisabled(
                "Telemetry streaming is disabled "
                "(set --otel-url and/or --mlflow-tracking-uri to enable)"
            )
        if self._otel_metrics_url:
            try:
                import opentelemetry.exporter.otlp.proto.http.metric_exporter  # noqa: F401
                import opentelemetry.sdk.metrics  # noqa: F401
            except ImportError as exc:
                message = otel_dependency_message(OTEL_METRICS_STREAMING_FEATURE)
                self.warning(
                    f"{message} ImportError={exc!r}. python_executable={sys.executable}"
                )
                raise PostProcessorDisabled(message) from exc

        self._meter_provider: Any | None = None
        self._meter: Any | None = None
        self._histogram_instruments: dict[str, Any] = {}
        self._counter_instruments: dict[str, Any] = {}
        self._up_down_counter_instruments: dict[str, Any] = {}
        self._timing_counter_state: dict[tuple[CreditPhase, str], int] = {}
        self._timing_gauge_state: dict[tuple[CreditPhase, str], float] = {}
        self._instrument_lock = asyncio.Lock()
        self._result_strategies: list[OTelResultsStrategyProtocol] = []
        if self.user_config.otel_stream_metrics_enabled:
            self._result_strategies.append(MetricResultsStrategy(self))
        if self.user_config.otel_stream_timing_enabled:
            self._result_strategies.append(TimingResultsStrategy(self))
        if not self._result_strategies:
            raise PostProcessorDisabled(
                "OTel telemetry selection disabled all stream domains."
            )
        self._export_timeout_millis = self._to_millis(
            Environment.OTEL.REQUEST_TIMEOUT_SECONDS,
            minimum=1000,
        )
        self._streaming_ready = False
        self._use_fanout_process = self._should_use_fanout_process()
        self._fanout_queue: Any | None = None
        self._fanout_process: mp.Process | None = None
        self._fanout_dropped_events = 0
        self._fanout_sent_events = 0
        self._fanout_queue_maxsize = Environment.OTEL.MAX_BUFFERED_RECORDS
        self.info("Initialized OTelMetricsResultsProcessor")

    @on_init
    async def _initialize_meter_provider(self) -> None:
        """Initialize telemetry streaming sinks."""
        self.info("Initializing telemetry streaming sinks")
        if self._use_fanout_process:
            await self._start_fanout_process()
            return

        await self._initialize_in_process_meter_provider()

    async def _initialize_in_process_meter_provider(self) -> None:
        """Initialize in-process OTel SDK metric export."""
        if not self._otel_metrics_url:
            raise RuntimeError(
                "In-process OTel meter provider requires --otel-url to be set."
            )
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import (
                PeriodicExportingMetricReader,
            )
            from opentelemetry.sdk.resources import Resource
        except ImportError as exc:
            message = otel_dependency_message(OTEL_METRICS_STREAMING_FEATURE)
            self.warning(
                f"{message} ImportError={exc!r}. python_executable={sys.executable}"
            )
            raise PostProcessorDisabled(message) from exc

        resource = Resource.create(self._build_resource_attributes())
        exporter = OTLPMetricExporter(
            endpoint=self._otel_metrics_url,
            timeout=Environment.OTEL.REQUEST_TIMEOUT_SECONDS,
        )

        reader = PeriodicExportingMetricReader(
            exporter,
            export_interval_millis=self._to_millis(
                Environment.OTEL.FLUSH_INTERVAL_SECONDS,
                minimum=100,
            ),
            export_timeout_millis=self._export_timeout_millis,
        )
        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[reader],
        )
        self._meter = self._meter_provider.get_meter("aiperf.records")
        self._streaming_ready = True
        self.info(f"OTel metrics streaming enabled: {self._otel_metrics_url}")

    async def _start_fanout_process(self) -> None:
        """Start a dedicated process that fans out streaming telemetry to OTel + MLflow."""
        config = OTelStreamingFanoutConfig(
            endpoint_url=self._otel_metrics_url,
            request_timeout_seconds=Environment.OTEL.REQUEST_TIMEOUT_SECONDS,
            export_interval_millis=self._to_millis(
                Environment.OTEL.FLUSH_INTERVAL_SECONDS,
                minimum=100,
            ),
            export_timeout_millis=self._export_timeout_millis,
            max_batch_records=Environment.OTEL.MAX_BATCH_RECORDS,
            resource_attributes=self._build_resource_attributes(),
            mlflow_tracking_uri=self.user_config.mlflow_tracking_uri,
            mlflow_experiment=self.user_config.mlflow_experiment,
            mlflow_run_name=self.user_config.mlflow_run_name,
            mlflow_tags=self.user_config.mlflow_tags_dict,
            benchmark_id=self.user_config.benchmark_id,
            metadata_file=(
                self.user_config.output.artifact_directory
                / MLflowDefaults.EXPORT_METADATA_FILE
            ),
        )
        was_daemon = mp.current_process().daemon
        if was_daemon:
            self._set_current_process_daemon(False)

        try:
            context = mp.get_context()
            queue = context.Queue(maxsize=self._fanout_queue_maxsize)
            process = context.Process(
                target=run_otel_streaming_fanout,
                args=(queue, config),
                name=f"aiperf-otel-fanout-{self.service_id}",
                daemon=True,
            )
            await asyncio.to_thread(process.start)
        except Exception as exc:
            self.warning(f"Failed to start telemetry fanout process. Error={exc!r}")
            with suppress(Exception):
                if "queue" in locals():
                    queue.close()
            if self._otel_metrics_url:
                self.warning(
                    "Falling back to in-process OTel streaming because --otel-url "
                    "is configured."
                )
                self._use_fanout_process = False
                await self._initialize_in_process_meter_provider()
            else:
                self.warning(
                    "Disabling live telemetry streaming for this run because fanout "
                    "startup failed and no OTel fallback is configured."
                )
            return
        finally:
            if was_daemon:
                self._set_current_process_daemon(True)

        self._fanout_queue = queue
        self._fanout_process = process
        self._streaming_ready = True
        self.info(
            "Telemetry streaming enabled with process fanout "
            f"(OTLP: {bool(self._otel_metrics_url)}, "
            f"MLflow live: {self._mlflow_live_enabled})"
        )

    async def process_result(self, record_data: OTelResultData) -> None:
        """Record metric data for export via the OpenTelemetry SDK."""
        if not self._streaming_ready:
            return

        for strategy in self._result_strategies:
            if strategy.supports(record_data):
                await strategy.process(record_data)
                return

        self.debug(
            lambda: (
                f"Skipping unsupported OTel result payload type: {type(record_data)}"
            )
        )

    async def process_realtime_metrics(self, metrics: list[MetricResult]) -> None:
        """Send realtime snapshot metrics to the MLflow live sink."""
        if (
            not self._streaming_ready
            or not self._use_fanout_process
            or not self._mlflow_live_enabled
        ):
            return

        for metric in metrics:
            value = self._metric_result_snapshot_value(metric)
            if value is None:
                continue
            self._queue_fanout_event(
                "mlflow_metric_snapshot",
                {
                    "metric_name": f"aiperf.{metric.tag}",
                    "value": value,
                },
            )

    async def flush(self, *, force: bool = False) -> None:
        """Force a flush of pending SDK metrics exports."""
        if self._use_fanout_process:
            self._queue_fanout_event("flush", {})
            return
        if self._meter_provider is None:
            return
        await asyncio.to_thread(
            self._meter_provider.force_flush,
            timeout_millis=self._export_timeout_millis,
        )

    async def summarize(self) -> list[MetricResult]:
        return []

    @on_stop
    async def _flush_and_shutdown(self) -> None:
        """Final flush before shutdown and close SDK resources."""
        try:
            await self.flush(force=True)
        except Exception as exc:
            self.warning(f"Failed to flush metrics: {exc}")
        finally:
            if self._use_fanout_process:
                await self._stop_fanout_process()
            elif self._meter_provider is not None:
                await asyncio.to_thread(self._meter_provider.shutdown)
                self._meter_provider = None
                self._meter = None
            self._streaming_ready = False

    async def get_or_create_histogram(self, metric_name: str) -> Any:
        """Create or reuse a histogram instrument for a metric name."""
        if metric_name in self._histogram_instruments:
            return self._histogram_instruments[metric_name]

        async with self._instrument_lock:
            if metric_name in self._histogram_instruments:
                return self._histogram_instruments[metric_name]
            instrument_name = f"aiperf.{metric_name}"
            unit = self._metric_unit(metric_name)
            description = f"AIPerf streaming metric: {metric_name}"
            if self._use_fanout_process:
                instrument = _FanoutHistogramInstrument(
                    metric_name=instrument_name,
                    unit=unit,
                    description=description,
                    emit_event=self._queue_fanout_event,
                )
            else:
                if self._meter is None:
                    raise RuntimeError("OTel meter is not initialized")
                instrument = self._meter.create_histogram(
                    name=instrument_name,
                    unit=unit,
                    description=description,
                )
            self._histogram_instruments[metric_name] = instrument
            return instrument

    async def get_or_create_counter(
        self, metric_name: str, unit: str, description: str
    ) -> Any:
        """Create or reuse a counter instrument."""
        if metric_name in self._counter_instruments:
            return self._counter_instruments[metric_name]

        async with self._instrument_lock:
            if metric_name in self._counter_instruments:
                return self._counter_instruments[metric_name]
            if self._use_fanout_process:
                instrument = _FanoutAddInstrument(
                    event_type="counter_add",
                    metric_name=metric_name,
                    unit=unit,
                    description=description,
                    emit_event=self._queue_fanout_event,
                )
            else:
                if self._meter is None:
                    raise RuntimeError("OTel meter is not initialized")
                instrument = self._meter.create_counter(
                    name=metric_name,
                    unit=unit,
                    description=description,
                )
            self._counter_instruments[metric_name] = instrument
            return instrument

    async def get_or_create_up_down_counter(
        self, metric_name: str, unit: str, description: str
    ) -> Any:
        """Create or reuse an up-down counter instrument."""
        if metric_name in self._up_down_counter_instruments:
            return self._up_down_counter_instruments[metric_name]

        async with self._instrument_lock:
            if metric_name in self._up_down_counter_instruments:
                return self._up_down_counter_instruments[metric_name]
            if self._use_fanout_process:
                instrument = _FanoutAddInstrument(
                    event_type="up_down_counter_add",
                    metric_name=metric_name,
                    unit=unit,
                    description=description,
                    emit_event=self._queue_fanout_event,
                )
            else:
                if self._meter is None:
                    raise RuntimeError("OTel meter is not initialized")
                instrument = self._meter.create_up_down_counter(
                    name=metric_name,
                    unit=unit,
                    description=description,
                )
            self._up_down_counter_instruments[metric_name] = instrument
            return instrument

    async def _stop_fanout_process(self) -> None:
        """Gracefully stop fanout process and drain final metrics."""
        if self._fanout_queue is not None:
            self._queue_fanout_event("shutdown", {})

        if self._fanout_process is not None:
            await asyncio.to_thread(self._fanout_process.join, 5.0)
            if self._fanout_process.is_alive():
                self.warning("OTel fanout process did not stop in time; terminating.")
                self._fanout_process.terminate()
                await asyncio.to_thread(self._fanout_process.join, 1.0)
            self._fanout_process = None

        if self._fanout_queue is not None:
            with suppress(Exception):
                self._fanout_queue.close()
            self._fanout_queue = None

        if self._fanout_dropped_events > 0:
            self.warning(
                "Dropped OTel fanout events due to backpressure: "
                f"{self._fanout_dropped_events}"
            )

    def _record_fanout_drop(self, message: str) -> None:
        self._fanout_dropped_events += 1
        if self._fanout_dropped_events in {1, 100, 1000}:
            self.warning(f"{message} (dropped={self._fanout_dropped_events}).")

    def _drop_oldest_fanout_event(self) -> bool:
        """Drop the oldest queued event to preserve fresher live telemetry."""
        if self._fanout_queue is None:
            return False

        try:
            self._fanout_queue.get_nowait()
        except Empty:
            return False
        except Exception as exc:
            self.warning(f"Failed to drop oldest OTel fanout event: {exc!r}")
            return False

        self._record_fanout_drop("OTel fanout queue is full; dropping oldest event")
        return True

    def _queue_fanout_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Enqueue streaming event for the fanout process without blocking the event loop."""
        if self._fanout_queue is None:
            return

        event = {"type": event_type, "payload": payload}
        try:
            self._fanout_queue.put_nowait(event)
            self._fanout_sent_events += 1
        except Full:
            if self._drop_oldest_fanout_event():
                try:
                    self._fanout_queue.put_nowait(event)
                    self._fanout_sent_events += 1
                    return
                except Full:
                    pass

            self._record_fanout_drop(
                "OTel fanout queue remained full; dropping newest event"
            )
        except Exception as exc:
            self.warning(f"Failed to enqueue OTel fanout event: {exc!r}")

    def _should_use_fanout_process(self) -> bool:
        """Fanout is the default streaming path for telemetry sinks."""
        return True

    @staticmethod
    def _set_current_process_daemon(daemon: bool) -> None:
        """Set daemon flag on current process, including fallback for strict assertions."""
        try:
            mp.current_process().daemon = daemon
        except AssertionError:
            mp.current_process()._config["daemon"] = daemon

    def _build_resource_attributes(self) -> dict[str, str]:
        """Build OTLP resource attributes shared across all streamed metrics."""
        attributes: dict[str, str] = {}
        attributes["service.name"] = "aiperf"
        attributes["service.instance.id"] = self.service_id
        if self.user_config.benchmark_id is not None:
            attributes["aiperf.benchmark.id"] = self.user_config.benchmark_id
        attributes["aiperf.endpoint.type"] = str(self.user_config.endpoint.type)
        attributes["aiperf.model.name"] = self.user_config.endpoint.model_names[0]
        return attributes

    def build_record_attributes(self, record: MetricRecordsData) -> dict[str, Any]:
        """Build OTLP metric attributes for an individual metric record."""
        metadata = record.metadata
        attributes: dict[str, Any] = {}
        attributes["aiperf.worker.id"] = metadata.worker_id
        attributes["aiperf.record_processor.id"] = metadata.record_processor_id
        attributes["aiperf.benchmark_phase"] = str(metadata.benchmark_phase)
        attributes["aiperf.was_cancelled"] = metadata.was_cancelled
        attributes["aiperf.has_error"] = record.error is not None
        return attributes

    def build_timing_attributes(self, stats: CreditPhaseStats) -> dict[str, Any]:
        """Build OTLP metric attributes for phase-level timing metrics."""
        attributes: dict[str, Any] = {}
        attributes["aiperf.benchmark_phase"] = str(stats.phase)
        if stats.total_expected_requests is not None:
            attributes["aiperf.total_expected_requests"] = stats.total_expected_requests
        if stats.expected_duration_sec is not None:
            attributes["aiperf.expected_duration_sec"] = stats.expected_duration_sec
        if stats.expected_num_sessions is not None:
            attributes["aiperf.expected_num_sessions"] = stats.expected_num_sessions
        return attributes

    def calculate_timing_counter_delta(
        self, *, metric_name: str, phase: CreditPhase, current_value: int
    ) -> int:
        """Calculate delta from cumulative timing counters."""
        key = (phase, metric_name)
        previous_value = self._timing_counter_state.get(key)
        self._timing_counter_state[key] = current_value
        if previous_value is None:
            return current_value
        if current_value < previous_value:
            self.warning(
                f"Timing counter reset detected for {metric_name} ({phase}). "
                f"current={current_value}, previous={previous_value}"
            )
            return current_value
        return current_value - previous_value

    def calculate_timing_gauge_delta(
        self, *, metric_name: str, phase: CreditPhase, current_value: float
    ) -> float:
        """Calculate delta required to represent the latest gauge-like snapshot."""
        key = (phase, metric_name)
        previous_value = self._timing_gauge_state.get(key)
        self._timing_gauge_state[key] = current_value
        if previous_value is None:
            return current_value
        return current_value - previous_value

    def coerce_metric_values(self, metric_name: str, metric_value: Any) -> list[float]:
        """Convert metric value to numeric values suitable for histograms."""
        if isinstance(metric_value, bool):
            return []
        if isinstance(metric_value, int | float):
            return [
                self._convert_metric_value_to_display_unit(metric_name, metric_value)
            ]
        if isinstance(metric_value, list):
            numeric_values = [
                self._convert_metric_value_to_display_unit(metric_name, value)
                for value in metric_value
                if isinstance(value, int | float) and not isinstance(value, bool)
            ]
            if not numeric_values:
                return []
            return numeric_values
        self.debug(
            lambda: (
                f"Skipping unsupported OTel metric value type for "
                f"'{metric_name}': {type(metric_value)}"
            )
        )
        return []

    def _metric_unit(self, metric_name: str) -> str:
        """Return a unit string for a metric name."""
        try:
            metric_class = MetricRegistry.get_class(metric_name)
        except MetricTypeError:
            metric_class = None
        if metric_class is not None:
            display_unit = metric_class.display_unit or metric_class.unit
            return str(display_unit)
        if metric_name.endswith("_ns"):
            return "ns"
        return "1"

    def _convert_metric_value_to_display_unit(
        self, metric_name: str, value: int | float
    ) -> float:
        """Convert record metric values to the same display unit used elsewhere."""
        try:
            metric_class = MetricRegistry.get_class(metric_name)
        except MetricTypeError:
            return float(value)

        display_unit = metric_class.display_unit or metric_class.unit
        try:
            return float(metric_class.unit.convert_to(display_unit, value))
        except MetricUnitError as exc:
            self.warning(
                f"Failed to convert OTel metric {metric_name} from "
                f"{metric_class.unit} to {display_unit}: {exc!r}"
            )
            return float(value)

    def _metric_result_snapshot_value(self, metric: MetricResult) -> float | None:
        """Pick the single snapshot value MLflow should plot for a realtime metric."""
        value = metric.current if metric.current is not None else metric.avg
        if value is None or isinstance(value, bool):
            return None
        if not isinstance(value, int | float):
            return None
        return float(value)

    def timing_unit(self, metric_name: str) -> str:
        """Return a unit string for timing metrics."""
        if metric_name.endswith("_sec"):
            return "s"
        return "1"

    def _to_millis(self, seconds: float, *, minimum: int) -> int:
        """Convert seconds to milliseconds with a minimum bound."""
        return max(minimum, int(seconds * 1000))
