# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact emission helpers for adaptive scale timing."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.timing.strategies.adaptive_scale_sla import percentile_value
from aiperf.timing.strategies.adaptive_scale_types import WindowStats

_LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 2


def _iso_utc_from_ns(timestamp_ns: int) -> str:
    """Return an ISO-8601 UTC timestamp with nanosecond input precision."""
    dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


class AdaptiveScaleArtifactWriter:
    """Write adaptive scale JSONL events and JSON summary artifacts."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Callable[[], None]] | None = None
        self._task: asyncio.Task[None] | None = None
        self._write_error: Exception | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._drain())

    async def flush(self) -> None:
        if self._queue is None:
            return
        await self._queue.join()
        if self._write_error is not None:
            raise self._write_error

    async def close(self) -> None:
        if self._queue is None or self._task is None:
            return
        try:
            await self.flush()
        finally:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._queue = None
            self._task = None

    async def _drain(self) -> None:
        if self._queue is None:
            return
        while True:
            write = await self._queue.get()
            try:
                await asyncio.to_thread(write)
            except Exception as exc:
                if self._write_error is None:
                    self._write_error = exc
                _LOGGER.exception("adaptive scale artifact write failed")
            finally:
                self._queue.task_done()

    def _schedule_write(self, write: Callable[[], None]) -> None:
        if self._queue is None:
            raise RuntimeError("adaptive scale artifact writer was not started")
        self._queue.put_nowait(write)

    @staticmethod
    def resolve_path(artifact_dir: Path | None, filename: str) -> Path | None:
        if artifact_dir is None:
            return None
        return artifact_dir / filename

    @staticmethod
    def phase_scoped_path(
        artifact_dir: Path | None, phase_name: str | None, filename: str
    ) -> Path | None:
        if artifact_dir is None or phase_name is None:
            return None
        return artifact_dir / "phases" / phase_name / filename

    @staticmethod
    def correlation_payload(
        *,
        run_id: str | None,
        phase_id: str,
        phase_name: str | None,
        adaptive_iteration: int,
        candidate_value: float,
        accepted_value: float,
        phase_start_ts: str | None = None,
        phase_end_ts: str | None = None,
        fault_window_id: str | None = None,
        phase_index: int | None = None,
        profiling_index: int | None = None,
        phase_kind: str | None = None,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "phase_id": phase_id,
            "phase_index": phase_index,
            "profiling_index": profiling_index,
            "phase_name": phase_name,
            "phase_kind": phase_kind,
            "phase_start_ts": phase_start_ts,
            "phase_end_ts": phase_end_ts,
            "adaptive_iteration": adaptive_iteration,
            "candidate_value": candidate_value,
            "accepted_value": accepted_value,
            "fault_window_id": fault_window_id,
        }

    @staticmethod
    def candidate_payload(
        *,
        adaptive_iteration: int,
        candidate_value: float,
        stats: WindowStats,
        accepted: bool,
        rejection_reason: str,
    ) -> dict[str, Any]:
        samples_ms = [sample / 1_000_000 for sample in stats.samples]

        def percentile_ms(pct: float) -> float:
            if not stats.samples:
                return 0.0
            return percentile_value(stats.samples, pct) / 1_000_000

        ttft_samples = stats.ttfts if isinstance(stats.ttfts, list) else []
        itl_samples = stats.itls if isinstance(stats.itls, list) else []
        success_count = len(stats.samples)
        request_count = stats.total
        return {
            "adaptive_iteration": adaptive_iteration,
            "candidate_value": candidate_value,
            "control_value": candidate_value,
            "start_ts": _iso_utc_from_ns(stats.start_ns)
            if stats.start_ns is not None
            else None,
            "end_ts": _iso_utc_from_ns(stats.end_ns)
            if stats.end_ns is not None
            else None,
            "duration_s": stats.elapsed_sec,
            "request_count": request_count,
            "error_count": stats.errors,
            "cancelled": stats.cancelled,
            "success_count": success_count,
            "success_rate": success_count / request_count if request_count else 0.0,
            "throughput_rps": stats.throughput,
            "latency_p50_ms": percentile_ms(50),
            "latency_p95_ms": percentile_ms(95),
            "latency_p99_ms": percentile_ms(99),
            "ttft_p50_ms": percentile_value(ttft_samples, 50) / 1_000_000
            if ttft_samples
            else 0.0,
            "ttft_p95_ms": percentile_value(ttft_samples, 95) / 1_000_000
            if ttft_samples
            else 0.0,
            "ttft_p99_ms": percentile_value(ttft_samples, 99) / 1_000_000
            if ttft_samples
            else 0.0,
            "itl_p50_ms": percentile_value(itl_samples, 50) / 1_000_000
            if itl_samples
            else 0.0,
            "itl_p95_ms": percentile_value(itl_samples, 95) / 1_000_000
            if itl_samples
            else 0.0,
            "itl_p99_ms": percentile_value(itl_samples, 99) / 1_000_000
            if itl_samples
            else 0.0,
            "accepted": accepted,
            "rejection_reason": None if accepted else rejection_reason,
            "latency_avg_ms": sum(samples_ms) / len(samples_ms) if samples_ms else 0.0,
        }

    @staticmethod
    def event_payload(
        *,
        timestamp_ns: int,
        event: str,
        phase: str,
        control_value: float,
        control_snapshot: dict[str, Any],
        control_variable: str,
        boundary_value: float | None,
        last_passing_value: float | None,
        first_failing_value: float | None,
        primary_sla: SLAFilter,
        strategy_type: str,
        step_policy: str,
        reason: str,
        sla_value: float | None,
        throughput: float,
        sample_count: int,
        error_count: int,
        cancelled_count: int = 0,
        before: float | None = None,
        passed: bool | None = None,
        step_size: float | None = None,
        sla_values: dict[str, float] | None = None,
        binding_sla: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "timestamp": timestamp_ns,
            "timestamp_ns": timestamp_ns,
            "timestamp_utc": _iso_utc_from_ns(timestamp_ns),
            "event": event,
            "phase": phase,
            "control_variable": control_variable,
            "control_value_before": control_value if before is None else before,
            "control_value_after": control_value,
            "control_value": control_value,
            "control": control_snapshot,
            "boundary_value": boundary_value,
            "last_passing_value": last_passing_value,
            "first_failing_value": first_failing_value,
            "sla_metric": primary_sla.metric_tag,
            "sla_stat": primary_sla.stat,
            "sla_op": primary_sla.op,
            "sla_value": sla_value,
            "sla_bound": primary_sla.threshold,
            "sla_values": sla_values or {},
            "binding_sla": binding_sla,
            "throughput": throughput,
            "sample_count": sample_count,
            "completed": sample_count,
            "sent": sample_count + error_count + cancelled_count,
            "in_flight": None,
            "cancelled": cancelled_count,
            "errored": error_count,
            "error_count": error_count,
            "sla_passed": passed,
            "strategy_type": strategy_type,
            "step_policy": step_policy,
            "step_size": step_size,
            "reason": reason,
        }

    @staticmethod
    def summary_payload(
        *,
        control_variable: str,
        control_value: float,
        control_snapshot: dict[str, Any],
        boundary_value: float | None,
        last_passing_value: float | None,
        first_failing_value: float | None,
        sustain_started_at_ns: int | None,
        sustain_duration: float,
        completed_reason: str | None,
        status: str,
        sustain_windows: int,
        sustain_passed_windows: int,
        throughput: float,
        sample_count: int,
        error_count: int,
        cancelled_count: int = 0,
        candidates: list[dict[str, Any]] | None = None,
        primary_sla: SLAFilter,
        strategy_type: str,
        step_policy: str,
        base_step: int,
        max_step_multiplier: int,
        step_percent: float,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "control_variable": control_variable,
            "control_value": control_value,
            "control": control_snapshot,
            "boundary_value": boundary_value,
            "last_passing_value": last_passing_value,
            "first_failing_value": first_failing_value,
            "result": {
                "last_passing_value": last_passing_value,
                "first_failing_value": first_failing_value,
                "boundary_value": boundary_value,
            },
            "sustain_started_at": sustain_started_at_ns,
            "sustain_duration_seconds": sustain_duration,
            "completed_reason": completed_reason,
            "sla_passed_during_sustain": (
                sustain_windows > 0 and sustain_passed_windows == sustain_windows
            ),
            "sustain_windows": sustain_windows,
            "sustain_passed_windows": sustain_passed_windows,
            "sla_metric": primary_sla.metric_tag,
            "sla_stat": primary_sla.stat,
            "sla_op": primary_sla.op,
            "sla_bound": primary_sla.threshold,
            "sla": {
                "metric": primary_sla.metric_tag,
                "stat": primary_sla.stat,
                "op": primary_sla.op,
                "bound": primary_sla.threshold,
            },
            "totals": {
                "sent": sample_count + error_count + cancelled_count,
                "completed": sample_count,
                "errored": error_count,
                "cancelled": cancelled_count,
            },
            "throughput": throughput,
            "candidates": candidates or [],
            "strategy_type": strategy_type,
            "step_policy": step_policy,
            "base_step": base_step,
            "max_step_multiplier": max_step_multiplier,
            "step_percent": step_percent,
        }

    def emit_event(self, path: Path | None, payload: dict[str, Any]) -> None:
        if path is None:
            return

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as f:
                f.write(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n")

        self._schedule_write(write)

    def write_summary(self, path: Path | None, summary: dict[str, Any]) -> None:
        if path is None:
            return

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            encoded = orjson.dumps(
                summary, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
            )
            path.write_bytes(encoded + b"\n")

        self._schedule_write(write)

    def write_manifest_entry(
        self, artifact_dir: Path | None, entry: dict[str, Any]
    ) -> None:
        if artifact_dir is None:
            return
        manifest_path = artifact_dir / "adaptive_scale_manifest.json"

        def write() -> None:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            if manifest_path.exists():
                payload = orjson.loads(manifest_path.read_bytes())
            else:
                payload = {"schema_version": SCHEMA_VERSION, "adaptive_phases": []}
            phases = payload.setdefault("adaptive_phases", [])
            phases = [
                phase
                for phase in phases
                if phase.get("phase_name") != entry.get("phase_name")
                or phase.get("phase_index") != entry.get("phase_index")
            ]
            phases.append(entry)
            phases.sort(
                key=lambda item: (
                    item.get("phase_index")
                    if item.get("phase_index") is not None
                    else 10**9
                )
            )
            payload["adaptive_phases"] = phases
            encoded = orjson.dumps(
                payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
            )
            manifest_path.write_bytes(encoded + b"\n")

        self._schedule_write(write)
