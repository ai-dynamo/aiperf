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
from typing import Any, ClassVar

from aiperf.common.config import MLflowDefaults, UserConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models import CreditPhaseStats, MetricResult
from aiperf.common.optional_dependencies import (
    OTEL_METRICS_STREAMING_FEATURE,
    otel_dependency_message,
)
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor
from aiperf.post_processors.otel_streaming_fanout import (
    FanoutEvent,
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
    explicit_bucket_boundaries: tuple[float, ...] | None = None

    def record(self, value: float, attributes: dict[str, Any]) -> None:
        payload: dict[str, Any] = {
            "metric_name": self.metric_name,
            "unit": self.unit,
            "description": self.description,
            "value": float(value),
            "attributes": attributes,
        }
        if self.explicit_bucket_boundaries is not None:
            payload["explicit_bucket_boundaries"] = list(
                self.explicit_bucket_boundaries
            )
        self.emit_event("histogram_record", payload)


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
    """Producer side of the OTel/MLflow streaming pipeline.

    Receives ``MetricRecordsData`` and ``CreditPhaseStats`` from ``RecordsManager``,
    dispatches each to a registered ``OTelResultsStrategyProtocol`` (selected by
    ``--stream``), and emits the resulting events to a bounded
    ``multiprocessing.Queue``. The actual export to an OTel collector and/or MLflow
    tracking server happens in a dedicated child process — see
    ``run_otel_streaming_fanout``.

    Registered as the ``otel_metrics_streamer`` results processor in ``plugins.yaml``.
    Raises ``PostProcessorDisabled`` from ``__init__`` when neither ``--otel-url``
    nor ``--mlflow-tracking-uri`` is set, or when the optional ``aiperf[otel]`` extra
    is missing.

    See ``docs/dev/patterns.md#drop-oldest-fanout-queue`` for the queue back-pressure
    protocol.
    """

    # Telemetry failures must not crash the benchmark. The records manager
    # checks this attribute (see ``post_processors.protocols.BestEffortMarker``
    # and ``IS_BEST_EFFORT_ATTR``) and swallows the exception when True.
    is_best_effort: ClassVar[bool] = True

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
                # Disable only the OTel sink. MLflow live streaming runs in the
                # same fanout process but does not need opentelemetry — tearing
                # down the whole processor here would drop live MLflow metrics
                # whenever OTel imports fail. The fanout itself repeats the
                # import guard per sink (see ``otel_streaming_fanout.py``), so
                # ``endpoint_url=None`` in the fanout config is sufficient.
                if not self._mlflow_live_enabled:
                    raise PostProcessorDisabled(message) from exc
                self._otel_metrics_url = None

        self._histogram_instruments: dict[str, _FanoutHistogramInstrument] = {}
        self._counter_instruments: dict[str, _FanoutAddInstrument] = {}
        self._up_down_counter_instruments: dict[str, _FanoutAddInstrument] = {}
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
                f"--stream selection {user_config.stream!r} disabled all OTel "
                "stream domains. Set --stream metrics, --stream timing, or "
                "--stream default to enable streaming."
            )
        self._export_timeout_millis = self._to_millis(
            Environment.OTEL.REQUEST_TIMEOUT_SECONDS,
            minimum=1000,
        )
        self._streaming_ready = False
        self._fanout_queue: mp.Queue[FanoutEvent] | None = None
        self._fanout_process: mp.Process | None = None
        self._fanout_dropped_events = 0
        self._fanout_sent_events = 0
        self._fanout_queue_maxsize = Environment.OTEL.MAX_BUFFERED_RECORDS
        self.info("Initialized OTelMetricsResultsProcessor")

    @on_init
    async def _initialize_meter_provider(self) -> None:
        """Initialize telemetry streaming sinks."""
        self.info("Initializing telemetry streaming sinks")
        await self._start_fanout_process()

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
            mlflow_parent_run_id=self.user_config.mlflow_parent_run_id,
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
        except Exception as exc:  # noqa: BLE001 - multiprocessing startup can raise OS/resource errors
            self.warning(f"Failed to start telemetry fanout process. Error={exc!r}")
            with suppress(Exception):
                if "queue" in locals():
                    queue.close()
            self.warning(
                "Disabling live telemetry streaming for this run because fanout "
                "startup failed."
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

    async def flush(self, *, force: bool = False) -> None:
        """Force a flush of pending SDK metrics exports."""
        self._queue_fanout_event("flush", {})

    async def summarize(self) -> list[MetricResult]:
        return []

    @on_stop
    async def _flush_and_shutdown(self) -> None:
        """Final flush before shutdown and close SDK resources."""
        try:
            await self.flush(force=True)
        except Exception as exc:  # noqa: BLE001 - flush must not crash shutdown sequence
            self.warning(f"Failed to flush metrics: {exc!r}")
        finally:
            await self._stop_fanout_process()
            self._streaming_ready = False

    async def get_or_create_histogram(
        self,
        metric_name: str,
        *,
        unit: str | None = None,
        description: str | None = None,
        explicit_bucket_boundaries: tuple[float, ...] | None = None,
    ) -> _FanoutHistogramInstrument:
        """Create or reuse a histogram instrument for a metric name."""
        if metric_name in self._histogram_instruments:
            return self._histogram_instruments[metric_name]

        async with self._instrument_lock:
            if metric_name in self._histogram_instruments:
                return self._histogram_instruments[metric_name]
            # When unit is provided, the caller already passed a fully-qualified
            # metric name (e.g. from GenAI semconv); don't prepend "aiperf.".
            instrument_name = metric_name if unit else f"aiperf.{metric_name}"
            resolved_unit = unit or self._metric_unit(metric_name)
            resolved_description = (
                description or f"AIPerf streaming metric: {metric_name}"
            )
            instrument = _FanoutHistogramInstrument(
                metric_name=instrument_name,
                unit=resolved_unit,
                description=resolved_description,
                emit_event=self._queue_fanout_event,
                explicit_bucket_boundaries=explicit_bucket_boundaries,
            )
            self._histogram_instruments[metric_name] = instrument
            return instrument

    async def get_or_create_counter(
        self, metric_name: str, unit: str, description: str
    ) -> _FanoutAddInstrument:
        """Create or reuse a counter instrument."""
        if metric_name in self._counter_instruments:
            return self._counter_instruments[metric_name]

        async with self._instrument_lock:
            if metric_name in self._counter_instruments:
                return self._counter_instruments[metric_name]
            instrument = _FanoutAddInstrument(
                event_type="counter_add",
                metric_name=metric_name,
                unit=unit,
                description=description,
                emit_event=self._queue_fanout_event,
            )
            self._counter_instruments[metric_name] = instrument
            return instrument

    async def get_or_create_up_down_counter(
        self, metric_name: str, unit: str, description: str
    ) -> _FanoutAddInstrument:
        """Create or reuse an up-down counter instrument."""
        if metric_name in self._up_down_counter_instruments:
            return self._up_down_counter_instruments[metric_name]

        async with self._instrument_lock:
            if metric_name in self._up_down_counter_instruments:
                return self._up_down_counter_instruments[metric_name]
            instrument = _FanoutAddInstrument(
                event_type="up_down_counter_add",
                metric_name=metric_name,
                unit=unit,
                description=description,
                emit_event=self._queue_fanout_event,
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
                self._fanout_queue.cancel_join_thread()
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
        except Exception as exc:  # noqa: BLE001 - queue ops may raise OS errors; telemetry must not crash hot path
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
        except Exception as exc:  # noqa: BLE001 - queue/OS errors must not block the benchmarking event loop
            self.warning(f"Failed to enqueue OTel fanout event: {exc!r}")

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
        # Guard against empty model_names. EndpointConfig declares it Field(...),
        # so the CLI path always populates at least one, but the field has no
        # min_length=1 so a programmatic caller could construct an empty list
        # and an unguarded [0] would crash the fanout. Matches the guard
        # pattern in genai_semconv.py.
        model_names = self.user_config.endpoint.model_names
        if model_names:
            attributes["aiperf.model.name"] = model_names[0]
        attributes.update(self.user_config.otel_custom_resource_attributes)
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
                f"current={current_value}, previous={previous_value}. "
                "Skipping emission to avoid double-counting."
            )
            return 0
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
            return [float(metric_value)]
        if isinstance(metric_value, list):
            numeric_values = [
                float(value)
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
        """Return a unit string for an aiperf metric name on the fallback path.

        Consults the metric registry for the authoritative unit when available,
        falling back to tag-based heuristics.
        """
        from aiperf.metrics.metric_registry import MetricRegistry

        metric_cls = MetricRegistry.get_class_or_none(metric_name)
        if metric_cls is not None:
            unit = getattr(metric_cls, "unit", None)
            if unit is not None:
                return str(unit)

        if metric_name.endswith("_ns"):
            return "ns"
        return "1"

    def timing_unit(self, metric_name: str) -> str:
        """Return a unit string for timing metrics."""
        if metric_name.endswith("_sec"):
            return "s"
        return "1"

    def _to_millis(self, seconds: float, *, minimum: int) -> int:
        """Convert seconds to milliseconds with a minimum bound."""
        return max(minimum, int(seconds * 1000))
