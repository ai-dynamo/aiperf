# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty
from typing import Any

import orjson

from aiperf.common.optional_dependencies import (
    mlflow_dependency_message,
    otel_dependency_message,
)

# Hashable canonical form for attribute dicts used as dict keys in gauge snapshots.
AttributeKey = tuple[tuple[str, str], ...]


def _attribute_key(attrs: dict[str, object] | None) -> AttributeKey:
    """Convert an attribute dict to a hashable canonical key.

    Non-hashable values are coerced to str; matches existing resource-attribute
    construction in OTelMetricsResultsProcessor.
    """
    return tuple(sorted((str(k), str(v)) for k, v in (attrs or {}).items()))


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
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tracking_uri": tracking_uri,
        "experiment": experiment,
        "run_id": run_id,
        "run_name": run_name,
        "benchmark_id": benchmark_id,
        "parent_run_id": parent_run_id,
        "live_streaming": True,
        "stream_started_at_ns": time.time_ns(),
    }
    metadata_file.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def run_otel_streaming_fanout(
    event_queue: Queue[Any],
    config: OTelStreamingFanoutConfig,
) -> None:
    """Run OTel + MLflow live streaming fanout in a dedicated process."""
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
        except Exception as exc:
            logger.warning(f"OTel sink disabled in fanout process: {exc!r}")

    histograms: dict[str, Any] = {}
    counters: dict[str, Any] = {}
    up_down_counters: dict[str, Any] = {}

    mlflow_state: dict[str, Any] | None = None
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
            mlflow_gauge_snapshots: dict[str, dict[AttributeKey, float]] = {}
            mlflow_counter_snapshots: dict[str, dict[AttributeKey, float]] = {}
            mlflow_state = {
                "module": mlflow,
                "client": MlflowClient(),
                "metric_cls": Metric,
                "run_id": run_id,
                "step": 0,
                "buffer": [],
                "timing_gauge_snapshots": mlflow_gauge_snapshots,
                "counter_snapshots": mlflow_counter_snapshots,
            }
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
        except Exception as exc:
            logger.warning(f"MLflow live streaming disabled in fanout process: {exc!r}")
            mlflow_state = None

    def _append_mlflow_metric(metric_name: str, metric_value: float) -> None:
        if mlflow_state is None:
            return
        buffered: list[tuple[str, float]] = mlflow_state["buffer"]
        buffered.append((f"live.{metric_name}", float(metric_value)))
        if len(buffered) >= max_batch_records:
            _flush_mlflow_metrics()

    def _append_mlflow_timing_gauge_snapshot(
        metric_name: str,
        delta_value: float,
        attributes: dict[str, Any],
    ) -> None:
        if mlflow_state is None:
            return

        timing_gauge_snapshots: dict[str, dict[AttributeKey, float]] = mlflow_state[
            "timing_gauge_snapshots"
        ]
        attr_key = _attribute_key(attributes)
        metric_snapshots = timing_gauge_snapshots.setdefault(metric_name, {})
        next_snapshot = metric_snapshots.get(attr_key, 0.0) + float(delta_value)
        if abs(next_snapshot) < 1e-9:
            metric_snapshots.pop(attr_key, None)
            if not metric_snapshots:
                timing_gauge_snapshots.pop(metric_name, None)
        else:
            metric_snapshots[attr_key] = next_snapshot

        aggregate_snapshot = sum(metric_snapshots.values())
        if abs(aggregate_snapshot) < 1e-9:
            aggregate_snapshot = 0.0
        _append_mlflow_metric(metric_name, aggregate_snapshot)

    def _flush_mlflow_metrics() -> None:
        if mlflow_state is None:
            return
        buffered: list[tuple[str, float]] = mlflow_state["buffer"]
        if not buffered:
            return
        now_ms = int(time.time() * 1000)
        metrics = []
        metric_cls = mlflow_state["metric_cls"]
        step_start = mlflow_state["step"]
        for metric_name, metric_value in buffered:
            metrics.append(
                metric_cls(
                    metric_name,
                    metric_value,
                    now_ms,
                    step_start + len(metrics),
                )
            )
        try:
            mlflow_state["client"].log_batch(
                run_id=mlflow_state["run_id"],
                metrics=metrics,
                params=[],
                tags=[],
            )
            # Clear buffer and advance step counter only on success.
            mlflow_state["step"] = step_start + len(metrics)
            mlflow_state["buffer"] = []
        except Exception as exc:
            # Cap buffer to prevent unbounded growth on persistent failures.
            max_buffer = max_batch_records * 5
            if len(mlflow_state["buffer"]) > max_buffer:
                dropped = len(mlflow_state["buffer"]) - max_batch_records
                mlflow_state["buffer"] = mlflow_state["buffer"][-max_batch_records:]
                mlflow_state["step"] = step_start + dropped
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
            or len(mlflow_state["buffer"]) >= config.max_batch_records
            or (now - last_flush_monotonic) >= config.export_interval_millis / 1000.0
        )
        if should_flush:
            _flush_mlflow_metrics()
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
                except Exception as exc:
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
                    # MLflow: accumulate delta into cumulative snapshot
                    if mlflow_state is not None:
                        counter_snapshots: dict[str, dict[AttributeKey, float]] = (
                            mlflow_state["counter_snapshots"]
                        )
                        attr_key = _attribute_key(payload.get("attributes"))
                        metric_snaps = counter_snapshots.setdefault(metric_name, {})
                        metric_snaps[attr_key] = metric_snaps.get(
                            attr_key, 0.0
                        ) + float(payload["value"])
                        cumulative = sum(metric_snaps.values())
                        _append_mlflow_metric(metric_name, cumulative)
                except Exception as exc:
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
                except Exception as exc:
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
                _flush_mlflow_metrics()
                mlflow_state["module"].end_run()
            except Exception as exc:
                logger.warning(f"Failed to close live MLflow run cleanly: {exc!r}")
