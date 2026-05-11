# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty
from typing import Any, Literal, TypedDict

import orjson

from aiperf.common.optional_dependencies import (
    mlflow_dependency_message,
    otel_dependency_message,
)

# Hashable canonical form for attribute dicts used as dict keys in gauge snapshots.
AttributeKey = tuple[tuple[str, str], ...]

FanoutEventType = Literal[
    "histogram_record", "counter_add", "up_down_counter_add", "flush", "shutdown"
]


class FanoutEvent(TypedDict):
    """Wire format for events passed through the fanout queue."""

    type: FanoutEventType
    payload: dict[str, Any]


def _attribute_key(attrs: dict[str, object] | None) -> AttributeKey:
    """Convert an attribute dict to a hashable canonical key.

    Non-hashable values are coerced to str; matches existing resource-attribute
    construction in OTelMetricsResultsProcessor.
    """
    return tuple(sorted((str(k), str(v)) for k, v in (attrs or {}).items()))


@dataclass(slots=True)
class _MLflowFanoutState:
    """Typed state for the MLflow live-streaming fanout subprocess."""

    module: Any
    """The mlflow module handle."""

    client: Any
    """mlflow.tracking.MlflowClient instance."""

    metric_cls: type
    """mlflow.entities.Metric class reference."""

    run_id: str
    """Active MLflow run ID."""

    step: int
    """Monotonically increasing step counter for log_batch."""

    buffer: list[tuple[str, float]]
    """Pending (metric_name, value) pairs awaiting flush."""

    timing_gauge_snapshots: dict[str, dict[AttributeKey, float]]
    """Cumulative gauge snapshots keyed by metric name then attribute key."""

    counter_snapshots: dict[str, dict[AttributeKey, float]]
    """Cumulative counter snapshots keyed by metric name then attribute key."""


@dataclass(frozen=True)
class OTelStreamingFanoutConfig:
    """Configuration for the dedicated OTel/MLflow streaming fanout process."""

    endpoint_url: str | None
    request_timeout_seconds: float
    export_interval_millis: int
    export_timeout_millis: int
    max_batch_records: int
    resource_attributes: dict[str, str]
    mlflow_tracking_uri: str | None
    mlflow_experiment: str
    mlflow_run_name: str | None
    mlflow_tags: dict[str, str]
    mlflow_parent_run_id: str | None
    benchmark_id: str | None
    metadata_file: Path


def _write_live_mlflow_metadata(
    *,
    metadata_file: Path,
    tracking_uri: str,
    experiment: str,
    run_id: str,
    run_name: str | None,
    benchmark_id: str | None,
    parent_run_id: str | None,
) -> None:
    from aiperf.common.redact import redact_url
    from aiperf.exporters.mlflow_metadata import MLflowExportMetadata

    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    # mlflow_export.json is uploaded as a run artifact by the deferred
    # MLflowDataExporter, so strip userinfo (user:secret@) from the tracking
    # URI before persisting it. The exporter's reuse check redacts both sides
    # at compare time, so same-backend reuse still works.
    payload: MLflowExportMetadata = {
        "tracking_uri": redact_url(tracking_uri),
        "experiment": experiment,
        "run_id": run_id,
        "run_name": run_name,
        "benchmark_id": benchmark_id,
        "parent_run_id": parent_run_id,
        "live_streaming": True,
        "stream_started_at_ns": time.time_ns(),
    }
    data = orjson.dumps(payload, option=orjson.OPT_INDENT_2)
    # Atomic write to avoid corrupt metadata on crash.
    tmp_file = metadata_file.with_suffix(".json.tmp")
    tmp_file.write_bytes(data)
    tmp_file.replace(metadata_file)


def run_otel_streaming_fanout(
    event_queue: Queue[FanoutEvent],
    config: OTelStreamingFanoutConfig,
) -> None:
    """Consumer process for the OTel/MLflow streaming pipeline.

    Drains ``event_queue`` in a tight loop, exporting each event to up to two
    optional sinks:

    - **OTel Collector** (OTLP/HTTP) — when ``config.endpoint_url`` is set.
    - **MLflow Tracking** — when ``config.mlflow_tracking_uri`` is set.

    Event types consumed:

    - ``histogram_record``: records a value on a named OTel histogram; logs the
      value as a live MLflow scalar.
    - ``counter_add``: adds a delta to an OTel counter; logs the cumulative sum
      to MLflow.
    - ``up_down_counter_add``: adds a delta to an OTel UpDownCounter; logs the
      cumulative gauge snapshot to MLflow.
    - ``flush``: forces an immediate OTel SDK flush and MLflow batch write.
    - ``shutdown``: flushes both sinks and exits the loop.

    Lifetime: exits on a ``shutdown`` event or SIGTERM. On exit, the OTel
    MeterProvider is shut down and the active MLflow run is ended.

    Side effects: writes ``config.metadata_file`` (``mlflow_export.json``) after
    the MLflow run is successfully started. The file is written atomically.

    Failure contract: missing optional dependencies (``opentelemetry``, ``mlflow``)
    are logged as warnings and the corresponding sink is disabled — never raised.
    """
    import signal

    shutdown_requested = False

    def _handle_sigterm(signum: int, frame: object) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    logger = logging.getLogger(__name__)
    meter_provider: Any | None = None
    meter: Any | None = None
    max_batch_records = max(config.max_batch_records, 1)

    if config.endpoint_url:
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create(config.resource_attributes)
            exporter = OTLPMetricExporter(
                endpoint=config.endpoint_url,
                timeout=config.request_timeout_seconds,
            )
            reader = PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=config.export_interval_millis,
                export_timeout_millis=config.export_timeout_millis,
            )
            meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            meter = meter_provider.get_meter("aiperf.records")
        except ImportError as exc:
            logger.warning(
                "%s ImportError=%r",
                otel_dependency_message("OTel sink is enabled in the fanout process"),
                exc,
            )
        except Exception as exc:  # noqa: BLE001 - OTel SDK init failure must not crash fanout process
            logger.warning(f"OTel sink disabled in fanout process: {exc!r}")

    histograms: dict[str, Any] = {}
    counters: dict[str, Any] = {}
    up_down_counters: dict[str, Any] = {}

    mlflow_state: _MLflowFanoutState | None = None
    if config.mlflow_tracking_uri:
        try:
            import mlflow
            from mlflow.entities import Metric
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.mlflow_experiment)
            start_run_kwargs: dict[str, Any] = {}
            if config.mlflow_run_name:
                start_run_kwargs["run_name"] = config.mlflow_run_name
            if config.mlflow_parent_run_id:
                start_run_kwargs["parent_run_id"] = config.mlflow_parent_run_id
            run = mlflow.start_run(**start_run_kwargs)
            if config.mlflow_tags:
                mlflow.set_tags(config.mlflow_tags)
            if config.benchmark_id:
                mlflow.set_tag("benchmark_id", config.benchmark_id)
            run_id = run.info.run_id
            mlflow_state = _MLflowFanoutState(
                module=mlflow,
                client=MlflowClient(),
                metric_cls=Metric,
                run_id=run_id,
                step=0,
                buffer=[],
                timing_gauge_snapshots={},
                counter_snapshots={},
            )
            # Write metadata only after all setup succeeds to avoid
            # stale metadata pointing to a partially-initialized run.
            _write_live_mlflow_metadata(
                metadata_file=config.metadata_file,
                tracking_uri=config.mlflow_tracking_uri,
                experiment=config.mlflow_experiment,
                run_id=run_id,
                run_name=config.mlflow_run_name,
                benchmark_id=config.benchmark_id,
                parent_run_id=config.mlflow_parent_run_id,
            )
        except ImportError as exc:
            logger.warning(
                "%s ImportError=%r",
                mlflow_dependency_message(
                    "MLflow live streaming is enabled in the fanout process"
                ),
                exc,
            )
            mlflow_state = None
        except Exception as exc:  # noqa: BLE001 - MLflow client init failure must not crash fanout process
            logger.warning(f"MLflow live streaming disabled in fanout process: {exc!r}")
            mlflow_state = None

    def _append_mlflow_metric(metric_name: str, metric_value: float) -> None:
        if mlflow_state is None:
            return
        mlflow_state.buffer.append((f"live.{metric_name}", float(metric_value)))
        if len(mlflow_state.buffer) >= max_batch_records:
            _flush_mlflow_metrics()

    def _append_mlflow_timing_gauge_snapshot(
        metric_name: str,
        delta_value: float,
        attributes: dict[str, Any],
    ) -> None:
        if mlflow_state is None:
            return

        attr_key = _attribute_key(attributes)
        metric_snapshots = mlflow_state.timing_gauge_snapshots.setdefault(
            metric_name, {}
        )
        next_snapshot = metric_snapshots.get(attr_key, 0.0) + float(delta_value)
        if abs(next_snapshot) < 1e-9:
            metric_snapshots.pop(attr_key, None)
            if not metric_snapshots:
                mlflow_state.timing_gauge_snapshots.pop(metric_name, None)
            next_snapshot = 0.0
        else:
            metric_snapshots[attr_key] = next_snapshot

        # Emit per-attribute gauge snapshot with phase dimension in the name.
        phase = dict(attributes).get("aiperf.benchmark_phase")
        mlflow_name = f"{metric_name}.{phase}" if phase else metric_name
        _append_mlflow_metric(mlflow_name, next_snapshot)

    # Exponential backoff state for persistent MLflow flush failures.
    _flush_backoff_until: float = 0.0
    _flush_backoff_seconds: float = 1.0
    _MAX_FLUSH_BACKOFF: float = 60.0

    def _flush_mlflow_metrics(*, force: bool = False) -> None:
        nonlocal _flush_backoff_until, _flush_backoff_seconds
        if mlflow_state is None:
            return
        if not mlflow_state.buffer:
            return
        # Skip flush attempts during backoff unless forced (shutdown).
        if not force and time.monotonic() < _flush_backoff_until:
            return
        now_ms = int(time.time() * 1000)
        metrics = []
        step_start = mlflow_state.step
        for metric_name, metric_value in mlflow_state.buffer:
            metrics.append(
                mlflow_state.metric_cls(
                    metric_name,
                    metric_value,
                    now_ms,
                    step_start + len(metrics),
                )
            )
        try:
            mlflow_state.client.log_batch(
                run_id=mlflow_state.run_id,
                metrics=metrics,
                params=[],
                tags=[],
            )
            # Success: clear buffer, advance step, reset backoff.
            mlflow_state.step = step_start + len(metrics)
            mlflow_state.buffer = []
            _flush_backoff_seconds = 1.0
            _flush_backoff_until = 0.0
        except Exception as exc:  # noqa: BLE001 - MLflow log_batch failure must not lose the entire buffer
            # Apply exponential backoff to avoid log-spam on persistent failures.
            _flush_backoff_until = time.monotonic() + _flush_backoff_seconds
            _flush_backoff_seconds = min(_flush_backoff_seconds * 2, _MAX_FLUSH_BACKOFF)
            # Cap buffer to prevent unbounded growth on persistent failures.
            max_buffer = max_batch_records * 5
            if len(mlflow_state.buffer) > max_buffer:
                dropped = len(mlflow_state.buffer) - max_batch_records
                mlflow_state.buffer = mlflow_state.buffer[-max_batch_records:]
                # Advance the step counter past the dropped entries so later
                # successful writes keep a monotonically increasing step (MLflow
                # renders step as the chart x-axis). This produces a visible gap
                # in MLflow charts, which is the intended signal that data was
                # lost to backpressure.
                mlflow_state.step = step_start + dropped
                logger.warning(
                    f"MLflow flush failed and buffer exceeded {max_buffer} entries; "
                    f"dropped {dropped} oldest metrics. Error: {exc!r}"
                )
            else:
                logger.warning(
                    f"Failed to log live MLflow metrics batch ({len(metrics)} metrics): "
                    f"{exc!r}. Will retry on next flush."
                )

    poll_timeout_sec = max(config.export_interval_millis / 1000.0, 0.1)

    def _maybe_flush(
        *,
        now: float,
        force: bool,
        last_flush_monotonic: float,
    ) -> float:
        if mlflow_state is None:
            return last_flush_monotonic
        should_flush = (
            force
            or len(mlflow_state.buffer) >= config.max_batch_records
            or (now - last_flush_monotonic) >= config.export_interval_millis / 1000.0
        )
        if should_flush:
            _flush_mlflow_metrics(force=force)
            return now
        return last_flush_monotonic

    last_flush_monotonic: float = time.monotonic()

    try:
        while not shutdown_requested:
            try:
                event = event_queue.get(timeout=poll_timeout_sec)
            except Empty:
                last_flush_monotonic = _maybe_flush(
                    now=time.monotonic(),
                    force=False,
                    last_flush_monotonic=last_flush_monotonic,
                )
                continue

            event_type = event.get("type")
            payload = event.get("payload", {})

            if event_type == "histogram_record":
                try:
                    metric_name = payload["metric_name"]
                    if meter is not None and metric_name not in histograms:
                        kwargs: dict[str, Any] = {
                            "name": metric_name,
                            "unit": payload["unit"],
                            "description": payload["description"],
                        }
                        boundaries = payload.get("explicit_bucket_boundaries")
                        if boundaries is not None:
                            kwargs["explicit_bucket_boundaries_advisory"] = boundaries
                        histograms[metric_name] = meter.create_histogram(**kwargs)
                    if meter is not None:
                        histograms[metric_name].record(
                            payload["value"], payload["attributes"]
                        )
                    _append_mlflow_metric(metric_name, payload["value"])
                except Exception as exc:  # noqa: BLE001 - malformed payload must not crash fanout loop
                    logger.warning(
                        f"Invalid histogram fanout payload received: {exc!r}"
                    )
                last_flush_monotonic = _maybe_flush(
                    now=time.monotonic(),
                    force=False,
                    last_flush_monotonic=last_flush_monotonic,
                )
                continue

            if event_type == "counter_add":
                try:
                    metric_name = payload["metric_name"]
                    if meter is not None and metric_name not in counters:
                        counters[metric_name] = meter.create_counter(
                            name=metric_name,
                            unit=payload["unit"],
                            description=payload["description"],
                        )
                    if meter is not None:
                        counters[metric_name].add(
                            payload["value"], payload["attributes"]
                        )
                    # MLflow: accumulate delta into cumulative snapshot per phase.
                    if mlflow_state is not None:
                        attrs = payload.get("attributes") or {}
                        attr_key = _attribute_key(attrs)
                        metric_snaps = mlflow_state.counter_snapshots.setdefault(
                            metric_name, {}
                        )
                        metric_snaps[attr_key] = metric_snaps.get(
                            attr_key, 0.0
                        ) + float(payload["value"])
                        # Emit per-phase metric to preserve dimension separation.
                        phase = attrs.get("aiperf.benchmark_phase")
                        mlflow_name = f"{metric_name}.{phase}" if phase else metric_name
                        _append_mlflow_metric(mlflow_name, metric_snaps[attr_key])
                except Exception as exc:  # noqa: BLE001 - malformed payload must not crash fanout loop
                    logger.warning(f"Invalid counter fanout payload received: {exc!r}")
                last_flush_monotonic = _maybe_flush(
                    now=time.monotonic(),
                    force=False,
                    last_flush_monotonic=last_flush_monotonic,
                )
                continue

            if event_type == "up_down_counter_add":
                try:
                    metric_name = payload["metric_name"]
                    if meter is not None and metric_name not in up_down_counters:
                        up_down_counters[metric_name] = meter.create_up_down_counter(
                            name=metric_name,
                            unit=payload["unit"],
                            description=payload["description"],
                        )
                    if meter is not None:
                        up_down_counters[metric_name].add(
                            payload["value"], payload["attributes"]
                        )
                    _append_mlflow_timing_gauge_snapshot(
                        metric_name,
                        payload["value"],
                        payload["attributes"],
                    )
                except Exception as exc:  # noqa: BLE001 - malformed payload must not crash fanout loop
                    logger.warning(
                        f"Invalid up-down-counter fanout payload received: {exc!r}"
                    )
                last_flush_monotonic = _maybe_flush(
                    now=time.monotonic(),
                    force=False,
                    last_flush_monotonic=last_flush_monotonic,
                )
                continue

            if event_type == "flush":
                if meter_provider is not None:
                    meter_provider.force_flush(
                        timeout_millis=config.export_timeout_millis
                    )
                last_flush_monotonic = _maybe_flush(
                    now=time.monotonic(),
                    force=True,
                    last_flush_monotonic=last_flush_monotonic,
                )
                continue

            if event_type == "shutdown":
                if meter_provider is not None:
                    meter_provider.force_flush(
                        timeout_millis=config.export_timeout_millis
                    )
                last_flush_monotonic = _maybe_flush(
                    now=time.monotonic(),
                    force=True,
                    last_flush_monotonic=last_flush_monotonic,
                )
                break
    finally:
        if meter_provider is not None:
            meter_provider.shutdown()
        if mlflow_state is not None:
            try:
                _flush_mlflow_metrics(force=True)
                mlflow_state.module.end_run()
            except Exception as exc:  # noqa: BLE001 - MLflow shutdown failure must not crash fanout exit
                logger.warning(f"Failed to close live MLflow run cleanly: {exc!r}")
