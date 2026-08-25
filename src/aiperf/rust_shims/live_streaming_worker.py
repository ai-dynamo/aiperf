# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict stdio adapter from the Rust runner to Python telemetry extensions.

Rust owns request execution, timing, and metric facts. This worker validates
the versioned event stream, adapts those facts into the canonical
``OTelMetricsResultsProcessor``, and therefore preserves the existing OTel and
live-MLflow implementations without moving benchmark work back into Python.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import orjson
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from aiperf.common.models import CreditPhaseStats
from aiperf.common.models.record_models import MetricRecordInfo

PROTOCOL_VERSION = 1


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OTelWorkerConfig(_StrictModel):
    """OTel settings required by the canonical streaming processor."""

    metrics_url: str | None = None
    stream_metrics_enabled: bool = True
    stream_timing_enabled: bool = True
    custom_resource_attributes: dict[str, str] = Field(default_factory=dict)
    gen_ai_provider: str | None = None


class MLflowWorkerConfig(_StrictModel):
    """MLflow settings required by the canonical live fanout."""

    tracking_uri: str | None = None
    experiment: str = "aiperf"
    run_name: str | None = None
    tags: dict[str, str] | None = None
    parent_run_id: str | None = None
    artifact_globs: list[str] | None = None


class StreamingWorkerConfig(_StrictModel):
    """Resolved single-run identity and extension configuration."""

    models: list[str] = Field(min_length=1)
    endpoint_type: str = Field(min_length=1)
    endpoint_urls: list[str] = Field(min_length=1)
    streaming: bool
    artifact_dir: Path
    otel: OTelWorkerConfig
    mlflow: MLflowWorkerConfig


class InitializeEvent(_StrictModel):
    """Side-effect-free preparation event sent before artifact ownership."""

    protocol_version: Literal[PROTOCOL_VERSION]
    event: Literal["initialize"]
    benchmark_id: str = Field(min_length=1)
    config: StreamingWorkerConfig


class ActivateEvent(_StrictModel):
    """Post-artifact-commit barrier that permits exporter startup."""

    protocol_version: Literal[PROTOCOL_VERSION]
    event: Literal["activate"]


class MetricRecordEvent(_StrictModel):
    """One terminal request record computed by Rust."""

    protocol_version: Literal[PROTOCOL_VERSION]
    event: Literal["metric_record"]
    record: MetricRecordInfo


class NativeGracePeriod(_StrictModel):
    """Rust phase grace policy serialization."""

    kind: Literal["disabled", "finite", "infinite"]
    duration_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.kind == "finite" and self.duration_ns is None:
            raise ValueError("finite grace period omitted duration_ns")
        if self.kind != "finite" and self.duration_ns is not None:
            raise ValueError(f"{self.kind} grace period cannot define duration_ns")
        return self

    def seconds(self) -> float | None:
        if self.kind == "finite":
            assert self.duration_ns is not None
            return self.duration_ns / 1_000_000_000
        return 0.0 if self.kind == "disabled" else None


class NativePhaseStats(_StrictModel):
    """Exact ``aiperf_timing::PhaseStats`` wire shape."""

    phase_id: str = Field(min_length=1)
    kind: Literal["warmup", "profiling"]
    state: Literal["created", "started", "sending_complete", "complete"]
    start_ns: int | None = Field(default=None, ge=0)
    sent_end_ns: int | None = Field(default=None, ge=0)
    requests_end_ns: int | None = Field(default=None, ge=0)
    total_expected_requests: int | None = Field(default=None, ge=1)
    expected_num_sessions: int | None = Field(default=None, ge=1)
    expected_duration_ns: int | None = Field(default=None, ge=1)
    grace_period: NativeGracePeriod
    requests_sent: int = Field(ge=0)
    requests_completed: int = Field(ge=0)
    requests_cancelled: int = Field(ge=0)
    request_errors: int = Field(ge=0)
    sent_sessions: int = Field(ge=0)
    completed_sessions: int = Field(ge=0)
    cancelled_sessions: int = Field(ge=0)
    total_session_turns: int = Field(ge=0)
    in_flight_requests: int = Field(ge=0)
    in_flight_sessions: int = Field(ge=0)
    in_flight_prefills: int = Field(ge=0)
    pending_branch_work: int = Field(ge=0)
    stuck_session_slots_released: int = Field(ge=0)
    stuck_prefill_slots_released: int = Field(ge=0)
    final_requests_sent: int | None = Field(default=None, ge=0)
    final_requests_completed: int | None = Field(default=None, ge=0)
    final_requests_cancelled: int | None = Field(default=None, ge=0)
    final_request_errors: int | None = Field(default=None, ge=0)
    final_sent_sessions: int | None = Field(default=None, ge=0)
    final_completed_sessions: int | None = Field(default=None, ge=0)
    final_cancelled_sessions: int | None = Field(default=None, ge=0)
    timeout_triggered: bool
    grace_period_timeout_triggered: bool
    cancel_drain_timeout_triggered: bool
    forced_completion: bool
    was_cancelled: bool
    completion_reason: (
        Literal["completed", "grace_timeout", "cancelled", "force_completed", "failed"]
        | None
    ) = None


class PhaseStatsEvent(_StrictModel):
    """One native phase lifecycle/progress observation."""

    protocol_version: Literal[PROTOCOL_VERSION]
    event: Literal["phase_stats"]
    observed_at_ns: int = Field(ge=0)
    stats: NativePhaseStats


class ShutdownEvent(_StrictModel):
    """Ordered end-of-stream marker from Rust."""

    protocol_version: Literal[PROTOCOL_VERSION]
    event: Literal["shutdown"]
    dropped_events: int = Field(default=0, ge=0)


ActivationControl = Annotated[
    ActivateEvent | ShutdownEvent,
    Field(discriminator="event"),
]
_ACTIVATION_CONTROL_ADAPTER = TypeAdapter(ActivationControl)


WorkerEvent = Annotated[
    MetricRecordEvent | PhaseStatsEvent | ShutdownEvent,
    Field(discriminator="event"),
]
_WORKER_EVENT_ADAPTER = TypeAdapter(WorkerEvent)


class NativeCreditPhaseStats(CreditPhaseStats):
    """Credit snapshot whose elapsed time uses Rust's injected clock."""

    snapshot_ns: int = Field(ge=0)

    @property
    def requests_elapsed_time(self) -> float:
        if self.start_ns is None:
            return 0.0
        end_ns = self.requests_end_ns or self.snapshot_ns
        return max(0, end_ns - self.start_ns) / 1_000_000_000


def _build_run(event: InitializeEvent) -> Any:
    """Construct canonical Config-v2 models from the narrow extension ABI."""
    from aiperf.config import BenchmarkConfig
    from aiperf.config.resolution.plan import BenchmarkRun

    config = event.config
    cfg = BenchmarkConfig.model_validate(
        {
            "models": config.models,
            "endpoint": {
                "type": config.endpoint_type,
                "urls": config.endpoint_urls,
                "streaming": config.streaming,
            },
            "dataset": {
                "type": "synthetic",
                "entries": 1,
                "prompts": {"isl": 1, "osl": 1},
            },
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 1,
                    "concurrency": 1,
                }
            ],
            "artifacts": {"dir": config.artifact_dir},
            "otel": config.otel.model_dump(mode="json"),
            "mlflow": config.mlflow.model_dump(mode="json"),
            "gpu_telemetry": {"enabled": False},
            "server_metrics": {"enabled": False},
        }
    )
    return BenchmarkRun(
        benchmark_id=event.benchmark_id,
        cfg=cfg,
        artifact_dir=config.artifact_dir,
        label="native-streaming",
        cli_command=None,
    )


def _metric_data(event: MetricRecordEvent) -> Any:
    from aiperf.common.messages.inference_messages import MetricRecordsData

    record = event.record
    return MetricRecordsData(
        metadata=record.metadata,
        metrics={name: metric.value for name, metric in record.metrics.items()},
        trace_data=None,
        error=record.error,
    )


def _phase_data(event: PhaseStatsEvent) -> NativeCreditPhaseStats:
    stats = event.stats
    result = NativeCreditPhaseStats(
        phase=stats.kind,
        start_ns=stats.start_ns,
        sent_end_ns=stats.sent_end_ns,
        requests_end_ns=stats.requests_end_ns,
        snapshot_ns=event.observed_at_ns,
        total_expected_requests=stats.total_expected_requests,
        expected_duration_sec=(
            stats.expected_duration_ns / 1_000_000_000
            if stats.expected_duration_ns is not None
            else None
        ),
        expected_num_sessions=stats.expected_num_sessions,
        expected_grace_period_sec=stats.grace_period.seconds(),
        requests_sent=stats.requests_sent,
        requests_completed=stats.requests_completed,
        requests_cancelled=stats.requests_cancelled,
        request_errors=stats.request_errors,
        sent_sessions=stats.sent_sessions,
        completed_sessions=stats.completed_sessions,
        cancelled_sessions=stats.cancelled_sessions,
        total_session_turns=stats.total_session_turns,
        final_requests_sent=stats.final_requests_sent,
        final_requests_completed=stats.final_requests_completed,
        final_requests_cancelled=stats.final_requests_cancelled,
        final_request_errors=stats.final_request_errors,
        final_sent_sessions=stats.final_sent_sessions,
        final_completed_sessions=stats.final_completed_sessions,
        final_cancelled_sessions=stats.final_cancelled_sessions,
        timeout_triggered=stats.timeout_triggered,
        grace_period_timeout_triggered=stats.grace_period_timeout_triggered,
        was_cancelled=stats.was_cancelled,
    )
    if result.in_flight_requests != stats.in_flight_requests:
        raise ValueError("native phase in-flight request count is inconsistent")
    if result.in_flight_sessions != stats.in_flight_sessions:
        raise ValueError("native phase in-flight session count is inconsistent")
    return result


async def _readline() -> bytes:
    return await asyncio.to_thread(sys.stdin.buffer.readline)


def _write_protocol(value: dict[str, Any]) -> None:
    _PROTOCOL_STDOUT.write(orjson.dumps(value))
    _PROTOCOL_STDOUT.write(b"\n")
    _PROTOCOL_STDOUT.flush()


async def _serve() -> int:
    first_line = await _readline()
    if not first_line:
        raise ValueError("native streaming worker received EOF before initialize")
    initialize = InitializeEvent.model_validate_json(first_line)
    run = _build_run(initialize)

    from aiperf.common.exceptions import PostProcessorDisabled
    from aiperf.post_processors.otel_metrics_results_processor import (
        OTelMetricsResultsProcessor,
    )

    processor: OTelMetricsResultsProcessor | None
    disabled_reason: str | None = None
    try:
        processor = OTelMetricsResultsProcessor("aiperf", run)
    except PostProcessorDisabled as error:
        processor = None
        disabled_reason = str(error)

    _write_protocol(
        {
            "protocol_version": PROTOCOL_VERSION,
            "event": "prepared",
            "active": processor is not None,
            "disabled_reason": disabled_reason,
        }
    )

    activation_line = await _readline()
    if not activation_line:
        raise ValueError("native streaming worker received EOF before activation")
    activation = _ACTIVATION_CONTROL_ADAPTER.validate_json(activation_line)
    if isinstance(activation, ShutdownEvent):
        _write_protocol(
            {
                "protocol_version": PROTOCOL_VERSION,
                "event": "terminal",
                "success": True,
                "metric_records": 0,
                "phase_events": 0,
                "processing_errors": 0,
                "dropped_events": activation.dropped_events,
            }
        )
        return 0

    metric_records = 0
    phase_events = 0
    processing_errors = 0
    dropped_events = 0
    processor_started = False
    try:
        if processor is not None:
            processor_started = True
            await processor.initialize_and_start()
        _write_protocol(
            {
                "protocol_version": PROTOCOL_VERSION,
                "event": "ready",
                "active": processor is not None,
                "disabled_reason": disabled_reason,
            }
        )

        while True:
            line = await _readline()
            if not line:
                raise ValueError("native streaming worker received EOF before shutdown")
            event = _WORKER_EVENT_ADAPTER.validate_json(line)
            if isinstance(event, ShutdownEvent):
                dropped_events = event.dropped_events
                break
            if processor is None:
                continue
            try:
                if isinstance(event, MetricRecordEvent):
                    await processor.process_result(_metric_data(event))
                    metric_records += 1
                else:
                    await processor.process_result(_phase_data(event))
                    phase_events += 1
            except Exception as error:
                processing_errors += 1
                print(
                    f"native telemetry event was dropped after processing error: {error!r}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if processor is not None and processor_started:
            await processor.stop()

    _write_protocol(
        {
            "protocol_version": PROTOCOL_VERSION,
            "event": "terminal",
            "success": True,
            "metric_records": metric_records,
            "phase_events": phase_events,
            "processing_errors": processing_errors,
            "dropped_events": dropped_events,
        }
    )
    return 0


async def _main() -> int:
    try:
        return await _serve()
    except Exception as error:
        print(f"native streaming worker failed: {error!r}", file=sys.stderr, flush=True)
        _write_protocol(
            {
                "protocol_version": PROTOCOL_VERSION,
                "event": "terminal",
                "success": False,
                "error": str(error),
            }
        )
        return 1


_PROTOCOL_STDOUT = sys.stdout.buffer


def main(arguments: list[str] | None = None) -> int:
    """Run the strict stdio worker without accepting shim-specific arguments."""
    argv = sys.argv[1:] if arguments is None else arguments
    if argv:
        print("live-streaming shim does not accept arguments", file=sys.stderr)
        return 2

    # Reserve stdout exclusively for the machine protocol. Canonical AIPerf
    # logger output and extension diagnostics belong on stderr.
    stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        return asyncio.run(_main())
    finally:
        sys.stdout = stdout


if __name__ == "__main__":
    raise SystemExit(main())
