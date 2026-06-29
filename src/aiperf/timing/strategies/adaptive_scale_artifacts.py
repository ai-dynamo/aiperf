# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact emission helpers for adaptive scale timing."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.timing.strategies.adaptive_scale_sla import percentile_value
from aiperf.timing.strategies.adaptive_scale_types import WindowStats

_LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1


def _iso_utc_from_ns(timestamp_ns: int) -> str:
    """Return an ISO-8601 UTC timestamp with nanosecond input precision."""
    dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


class AdaptiveScaleArtifactWriter:
    """Write adaptive scale JSONL events and JSON summary artifacts."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Callable[[], None]] | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._drain())

    async def flush(self) -> None:
        if self._queue is None:
            return
        await self._queue.join()

    async def close(self) -> None:
        if self._queue is None or self._task is None:
            return
        await self.flush()
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
            except Exception:
                _LOGGER.exception("adaptive scale artifact write failed")
            finally:
                self._queue.task_done()

    def _schedule_write(self, write: Callable[[], None]) -> None:
        if self._queue is None:
            write()
            return
        self._queue.put_nowait(write)

    @staticmethod
    def resolve_path(artifact_dir: Path | None, filename: str) -> Path | None:
        if artifact_dir is None:
            return None
        path = artifact_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def correlation_payload(
        *,
        run_id: str | None,
        phase_id: str,
        phase_name: str | None,
        adaptive_iteration: int,
        candidate_concurrency: int,
        accepted_concurrency: int,
        phase_start_ts: str | None = None,
        phase_end_ts: str | None = None,
        fault_window_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "phase_id": phase_id,
            "phase_name": phase_name,
            "phase_start_ts": phase_start_ts,
            "phase_end_ts": phase_end_ts,
            "adaptive_iteration": adaptive_iteration,
            "candidate_concurrency": candidate_concurrency,
            "accepted_concurrency": accepted_concurrency,
            "fault_window_id": fault_window_id,
        }

    @staticmethod
    def candidate_payload(
        *,
        adaptive_iteration: int,
        candidate_concurrency: int,
        stats: WindowStats,
        accepted: bool,
        rejection_reason: str,
    ) -> dict[str, Any]:
        samples_ms = [sample / 1_000_000 for sample in stats.samples]

        def percentile_ms(pct: float) -> float:
            if not stats.samples:
                return 0.0
            return percentile_value(stats.samples, pct) / 1_000_000

        success_count = len(stats.samples)
        request_count = stats.total
        return {
            "adaptive_iteration": adaptive_iteration,
            "candidate_concurrency": candidate_concurrency,
            "start_ts": _iso_utc_from_ns(stats.start_ns)
            if stats.start_ns is not None
            else None,
            "end_ts": _iso_utc_from_ns(stats.end_ns)
            if stats.end_ns is not None
            else None,
            "duration_s": stats.elapsed_sec,
            "request_count": request_count,
            "error_count": stats.errors,
            "success_count": success_count,
            "goodput_ratio": success_count / request_count if request_count else 0.0,
            "throughput_rps": stats.throughput,
            "latency_p50_ms": percentile_ms(50),
            "latency_p95_ms": percentile_ms(95),
            "latency_p99_ms": percentile_ms(99),
            "ttft_p50_ms": 0.0,
            "ttft_p95_ms": 0.0,
            "ttft_p99_ms": 0.0,
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
        current_concurrency: int,
        control_variable: str,
        boundary_concurrency: int | None,
        last_good_concurrency: int | None,
        first_failing_concurrency: int | None,
        primary_sla: SLAFilter,
        strategy_type: str,
        step_policy: str,
        reason: str,
        sla_value: float | None,
        throughput: float,
        sample_count: int,
        error_count: int,
        before: int | None = None,
        passed: bool | None = None,
        step_size: int | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "timestamp": timestamp_ns,
            "timestamp_ns": timestamp_ns,
            "timestamp_utc": _iso_utc_from_ns(timestamp_ns),
            "event": event,
            "phase": phase,
            "concurrency_before": current_concurrency if before is None else before,
            "concurrency_after": current_concurrency,
            "control_variable": control_variable,
            "control_value": current_concurrency,
            "active_concurrency": current_concurrency,
            "boundary_concurrency": boundary_concurrency,
            "last_passing_value": last_good_concurrency,
            "first_failing_value": first_failing_concurrency,
            "sla_metric": primary_sla.metric_tag,
            "sla_stat": primary_sla.stat,
            "sla_op": primary_sla.op,
            "sla_value": sla_value,
            "sla_bound": primary_sla.threshold,
            "throughput": throughput,
            "sample_count": sample_count,
            "completed": sample_count,
            "sent": sample_count + error_count,
            "in_flight": None,
            "cancelled": None,
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
        current_concurrency: int,
        boundary_concurrency: int | None,
        last_good_concurrency: int | None,
        first_failing_concurrency: int | None,
        sustain_started_at_ns: int | None,
        sustain_duration: float,
        completed_reason: str | None,
        status: str,
        sustain_windows: int,
        sustain_passed_windows: int,
        throughput: float,
        sample_count: int,
        error_count: int,
        candidates: list[dict[str, Any]] | None = None,
        primary_sla: SLAFilter,
        strategy_type: str,
        step_policy: str,
        base_step: int,
        max_step_multiplier: int,
        step_percent: float,
    ) -> dict[str, Any]:
        boundary_value = boundary_concurrency
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "control_variable": control_variable,
            "control_value": current_concurrency,
            "active_concurrency": current_concurrency,
            "boundary_concurrency": boundary_concurrency,
            "last_passing_value": last_good_concurrency,
            "first_failing_value": first_failing_concurrency,
            "last_good_concurrency": last_good_concurrency,
            "result": {
                "last_passing_value": last_good_concurrency,
                "first_failing_value": first_failing_concurrency,
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
                "sent": sample_count + error_count,
                "completed": sample_count,
                "errored": error_count,
                "cancelled": None,
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
            with path.open("ab") as f:
                f.write(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n")

        self._schedule_write(write)

    def write_summary(self, path: Path | None, summary: dict[str, Any]) -> None:
        if path is None:
            return

        def write() -> None:
            encoded = orjson.dumps(
                summary, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
            )
            path.write_bytes(encoded + b"\n")

        self._schedule_write(write)
