# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-run adaptive scale timing strategy."""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import orjson

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import CreditReturn
from aiperf.timing.strategies.request_rate import RequestRateStrategy

if TYPE_CHECKING:
    from aiperf.config.sweep.adaptive import SLAFilter
    from aiperf.credit.structs import Credit
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker


AdaptiveControllerPhase = Literal["discover", "sustain", "complete"]

MIN_ASSESSMENT_PERIOD_SEC = 1.0


@dataclass(slots=True)
class WindowStats:
    samples: list[int]
    errors: int
    elapsed_sec: float

    @property
    def total(self) -> int:
        return len(self.samples) + self.errors

    @property
    def throughput(self) -> float:
        if self.elapsed_sec <= 0:
            return 0.0
        return len(self.samples) / self.elapsed_sec


class AdaptiveScaleStrategy(RequestRateStrategy):
    """Adjust session concurrency during one profiling phase.

    The strategy keeps the existing request-rate/concurrency-burst issuance path
    and layers an assessment task over it. Each window evaluates the configured
    SLA filters, adjusts ``ConcurrencyManager``'s dynamic session limit, and
    appends a JSONL decision event.
    """

    EVENT_FILE = "adaptive_scale_events.jsonl"
    SUMMARY_FILE = "adaptive_scale_summary.json"

    def __init__(
        self,
        *,
        concurrency_manager: ConcurrencyManager,
        progress: PhaseProgressTracker,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._concurrency_manager = concurrency_manager
        self._progress = progress
        self._max_concurrency = self._require_positive(
            self._config.concurrency, "concurrency"
        )
        self._current_concurrency = self._require_positive(
            self._config.adaptive_scale_min_concurrency,
            "adaptive_scale_min_concurrency",
        )
        if self._config.adaptive_control_variable != "concurrency":
            raise ValueError(
                "adaptive scale currently supports only control.variable='concurrency'"
            )
        self._min_completed_requests = self._config.adaptive_min_completed_requests
        self._assessment_period = self._config.adaptive_assessment_period_sec
        if self._assessment_period < MIN_ASSESSMENT_PERIOD_SEC:
            raise ValueError(
                "adaptive_assessment_period_sec must be >= "
                f"{MIN_ASSESSMENT_PERIOD_SEC:g}"
            )
        self._sustain_duration = self._config.adaptive_sustain_duration_sec
        if self._sustain_duration is None:
            raise ValueError("adaptive_sustain_duration_sec is required")
        if self._config.adaptive_scale_strategy_type != "ramp_until_fail":
            raise ValueError("adaptive_scale strategy type must be 'ramp_until_fail'")
        if not self._config.adaptive_sla_filters:
            raise ValueError("adaptive_sla_filters is required")
        self._sla_filters = list(self._config.adaptive_sla_filters)
        self._primary_sla = self._sla_filters[0]
        self._validate_sla_filters()

        self._controller_phase: AdaptiveControllerPhase = "discover"
        self._boundary_concurrency: int | None = None
        self._last_good_concurrency: int | None = None
        self._first_failing_concurrency: int | None = None
        self._sustain_started_at: float | None = None
        self._assessment_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._window_latency_ns: list[int] = []
        self._window_errors = 0
        self._window_started_at = time.perf_counter()
        self._event_path = self._resolve_artifact_path(self.EVENT_FILE)
        self._summary_path = self._resolve_artifact_path(self.SUMMARY_FILE)
        self._sustain_started_at_ns: int | None = None
        self._sustain_windows = 0
        self._sustain_passed_windows = 0
        self._completed_reason: str | None = None
        self._summary_written = False

    @staticmethod
    def _require_positive(value: int | None, name: str) -> int:
        if value is None or value < 1:
            raise ValueError(f"{name} must be >= 1 for adaptive scale")
        return value

    @staticmethod
    def _request_latency_value(samples: list[int], stat: str) -> float:
        if not samples:
            raise ValueError("request_latency SLA requires completed request samples")
        values_ms = [sample / 1_000_000 for sample in samples]
        match stat:
            case "avg":
                return sum(values_ms) / len(values_ms)
            case "min":
                return min(values_ms)
            case "max":
                return max(values_ms)
            case "p1" | "p5" | "p10" | "p25" | "p50" | "p75" | "p90" | "p95" | "p99":
                percentile = float(stat[1:])
                return _percentile(samples, percentile) / 1_000_000
        raise ValueError(f"Unsupported request_latency SLA stat: {stat}")

    @staticmethod
    def _throughput_value(stats: WindowStats, stat: str) -> float:
        match stat:
            case "avg" | "min" | "max":
                return stats.throughput
        raise ValueError(f"Unsupported throughput SLA stat: {stat}")

    @staticmethod
    def _goodput_ratio_value(stats: WindowStats, stat: str) -> float:
        match stat:
            case "avg" | "min" | "max":
                if stats.total == 0:
                    return 0.0
                return len(stats.samples) / stats.total
        raise ValueError(f"Unsupported goodput_ratio SLA stat: {stat}")

    def _sla_value(self, sla: SLAFilter, stats: WindowStats) -> float:
        match sla.metric_tag:
            case "request_latency":
                return self._request_latency_value(stats.samples, sla.stat)
            case "throughput" | "request_throughput" | "completed_request_throughput":
                return self._throughput_value(stats, sla.stat)
            case "goodput_ratio" | "success_rate" | "request_success_rate":
                return self._goodput_ratio_value(stats, sla.stat)
        raise ValueError(
            "adaptive_scale supports request_latency, request throughput, "
            "and goodput_ratio SLA metrics in this release, got "
            f"{sla.metric_tag!r}"
        )

    def _validate_sla_filters(self) -> None:
        for sla in self._sla_filters:
            self._validate_single_sla_filter(sla)

    @staticmethod
    def _validate_single_sla_filter(sla: SLAFilter) -> None:
        if sla.op not in {"lt", "le", "gt", "ge"}:
            raise ValueError(f"Unsupported SLA operator: {sla.op}")
        match sla.metric_tag:
            case "request_latency":
                if sla.stat not in {
                    "avg",
                    "min",
                    "max",
                    "p1",
                    "p5",
                    "p10",
                    "p25",
                    "p50",
                    "p75",
                    "p90",
                    "p95",
                    "p99",
                }:
                    raise ValueError(
                        f"Unsupported request_latency SLA stat: {sla.stat}"
                    )
            case "throughput" | "request_throughput" | "completed_request_throughput":
                if sla.stat not in {"avg", "min", "max"}:
                    raise ValueError(f"Unsupported throughput SLA stat: {sla.stat}")
            case "goodput_ratio" | "success_rate" | "request_success_rate":
                if sla.stat not in {"avg", "min", "max"}:
                    raise ValueError(f"Unsupported goodput_ratio SLA stat: {sla.stat}")
            case _:
                raise ValueError(
                    "adaptive_scale supports request_latency, request throughput, "
                    "and goodput_ratio SLA metrics in this release, got "
                    f"{sla.metric_tag!r}"
                )

    def _sla_values(self, stats: WindowStats) -> dict[str, float]:
        return {
            self._sla_key(sla): self._sla_value(sla, stats) for sla in self._sla_filters
        }

    @staticmethod
    def _sla_key(sla: SLAFilter) -> str:
        return f"{sla.metric_tag}:{sla.stat}:{sla.op}:{sla.threshold:g}"

    def _resolve_artifact_path(self, filename: str) -> Path | None:
        if self._config.artifact_dir is None:
            return None
        path = self._config.artifact_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    async def setup_phase(self) -> None:
        await super().setup_phase()
        self._set_concurrency(self._current_concurrency)
        self._emit_event(
            event="adaptive_phase_started",
            reason="adaptive scale discover phase started",
            sla_value=None,
            throughput=0.0,
            sample_count=0,
            error_count=0,
        )

    async def execute_phase(self) -> None:
        self._assessment_task = asyncio.create_task(self._assessment_loop())
        try:
            await super().execute_phase()
        finally:
            if self._completed_reason is None:
                self._complete_controller(reason="phase_stopped")
            if self._assessment_task is not None:
                self._assessment_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._assessment_task

    async def handle_credit_return(self, credit: Credit) -> None:
        await super().handle_credit_return(credit)

    async def handle_credit_result(self, credit_return: CreditReturn) -> None:
        latency_ns = max(0, time.time_ns() - credit_return.credit.issued_at_ns)
        async with self._lock:
            if credit_return.error is not None or credit_return.cancelled:
                self._window_errors += 1
            else:
                self._window_latency_ns.append(latency_ns)

    async def _assessment_loop(self) -> None:
        try:
            while (
                self._controller_phase != "complete"
                and self._stop_checker.can_send_any_turn()
            ):
                await asyncio.sleep(self._assessment_period)
                await self._assess_window()
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            self.exception(f"Adaptive scale assessment failed: {exc}")
            self._complete_controller(
                reason=f"assessment_failed: {exc}",
                terminal_event="adaptive_failed",
            )
            self._lifecycle.cancel()

    async def _assess_window(self) -> None:
        stats = await self._take_window()
        if not stats.samples and stats.errors:
            self._emit_event(
                event="adaptive_window",
                reason="all requests failed in assessment window",
                sla_value=None,
                throughput=stats.throughput,
                sample_count=0,
                error_count=stats.errors,
                passed=False,
            )
            self._assess_failed_window(stats)
            return
        if len(stats.samples) < self._min_completed_requests:
            self._emit_event(
                event="adaptive_window",
                reason="inconclusive: completed request count below minimum",
                sla_value=None,
                throughput=stats.throughput,
                sample_count=len(stats.samples),
                error_count=stats.errors,
                passed=None,
            )
            return

        sla_values = self._sla_values(stats)
        primary_value = sla_values[self._sla_key(self._primary_sla)]
        passing = self._passes_sla(sla_values)
        self._emit_event(
            event="adaptive_window",
            reason="SLA window evaluated",
            sla_value=primary_value,
            throughput=stats.throughput,
            sample_count=len(stats.samples),
            error_count=stats.errors,
            passed=passing,
        )

        if self._controller_phase == "discover":
            self._assess_discover(primary_value, passing, stats, sla_values)
        elif self._controller_phase == "sustain":
            self._assess_sustain(primary_value, passing, stats, sla_values)

    def _assess_failed_window(self, stats: WindowStats) -> None:
        reason = "all requests failed in assessment window"
        if self._controller_phase == "discover":
            if self._last_good_concurrency is None:
                self._first_failing_concurrency = self._current_concurrency
                self._complete_controller(
                    reason="no_sustainable_concurrency_found",
                    terminal_event="adaptive_failed",
                    throughput=stats.throughput,
                    sample_count=0,
                    error_count=stats.errors,
                )
                self._lifecycle.cancel()
                return
            self._first_failing_concurrency = self._current_concurrency
            self._enter_sustain(None, stats, reason)
        elif self._controller_phase == "sustain":
            self._assess_sustain(None, False, stats, reason=reason)

    async def _take_window(self) -> WindowStats:
        async with self._lock:
            now = time.perf_counter()
            stats = WindowStats(
                samples=self._window_latency_ns,
                errors=self._window_errors,
                elapsed_sec=now - self._window_started_at,
            )
            self._window_latency_ns = []
            self._window_errors = 0
            self._window_started_at = now
            return stats

    def _assess_discover(
        self,
        sla_value: float,
        passing: bool,
        stats: WindowStats,
        sla_values: dict[str, float] | None = None,
    ) -> None:
        if passing:
            self._last_good_concurrency = self._current_concurrency
            if self._current_concurrency >= self._max_concurrency:
                self._complete_controller(
                    reason="max_concurrency_reached_without_saturation",
                    terminal_event="adaptive_incomplete",
                    sla_value=sla_value,
                    throughput=stats.throughput,
                    sample_count=len(stats.samples),
                    error_count=stats.errors,
                )
                self._lifecycle.cancel()
                return
            before = self._current_concurrency
            next_value = self._next_up(sla_values)
            step_size = next_value - before
            self._set_concurrency(next_value)
            self._emit_event(
                event="adaptive_decision",
                reason=f"SLA value {sla_value:.3f} passes configured filters",
                sla_value=sla_value,
                throughput=stats.throughput,
                sample_count=len(stats.samples),
                error_count=stats.errors,
                before=before,
                step_size=step_size,
            )
            return

        if self._last_good_concurrency is None:
            self._first_failing_concurrency = self._current_concurrency
            self._complete_controller(
                reason="no_sustainable_concurrency_found",
                terminal_event="adaptive_failed",
                sla_value=sla_value,
                throughput=stats.throughput,
                sample_count=len(stats.samples),
                error_count=stats.errors,
            )
            self._lifecycle.cancel()
            return
        self._first_failing_concurrency = self._current_concurrency
        self._enter_sustain(
            sla_value,
            stats,
            f"SLA value {sla_value:.3f} breaches configured filters",
        )

    def _assess_sustain(
        self,
        sla_value: float | None,
        passing: bool,
        stats: WindowStats,
        sla_values: dict[str, float] | None = None,
        *,
        reason: str | None = None,
    ) -> None:
        self._sustain_windows += 1
        if passing:
            self._sustain_passed_windows += 1
            self._last_good_concurrency = self._current_concurrency
            self._emit_event(
                event="adaptive_decision",
                reason=f"SLA value {sla_value:.3f} passes configured filters during sustain",
                sla_value=sla_value,
                throughput=stats.throughput,
                sample_count=len(stats.samples),
                error_count=stats.errors,
            )
        else:
            before = self._current_concurrency
            target = max(
                self._config.adaptive_scale_min_concurrency,
                self._last_good_concurrency
                or self._config.adaptive_scale_min_concurrency,
            )
            if target >= before:
                target = max(
                    self._config.adaptive_scale_min_concurrency,
                    before - self._step_size(before, sla_values),
                )
            if target == before == self._config.adaptive_scale_min_concurrency:
                self._complete_controller(
                    reason="sustain_failed_sla_unrecoverable",
                    terminal_event="adaptive_failed",
                    sla_value=sla_value,
                    throughput=stats.throughput,
                    sample_count=len(stats.samples),
                    error_count=stats.errors,
                )
                self._lifecycle.cancel()
                return
            self._set_concurrency(target)
            self._last_good_concurrency = target
            self._emit_event(
                event="adaptive_decision",
                reason=reason
                or f"SLA value {sla_value:.3f} breaches configured filters during sustain",
                sla_value=sla_value,
                throughput=stats.throughput,
                sample_count=len(stats.samples),
                error_count=stats.errors,
                before=before,
                step_size=abs(before - target),
            )

        if self._sustain_started_at is not None:
            elapsed = time.perf_counter() - self._sustain_started_at
            if elapsed >= self._sustain_duration:
                self._complete_controller(
                    reason="sustain_duration_completed",
                    terminal_event="adaptive_complete",
                    sla_value=sla_value,
                    throughput=stats.throughput,
                    sample_count=len(stats.samples),
                    error_count=stats.errors,
                )
                self._lifecycle.cancel()

    def _enter_sustain(
        self, sla_value: float | None, stats: WindowStats, reason: str
    ) -> None:
        if self._last_good_concurrency is None:
            raise RuntimeError("cannot enter sustain without a passing boundary")
        boundary = max(
            self._config.adaptive_scale_min_concurrency,
            self._last_good_concurrency,
        )
        before = self._current_concurrency
        self._boundary_concurrency = boundary
        self._set_concurrency(boundary)
        self._controller_phase = "sustain"
        self._sustain_started_at = time.perf_counter()
        self._sustain_started_at_ns = time.time_ns()
        self._emit_event(
            event="sustain_started",
            phase="sustain",
            reason=f"holding boundary_concurrency={boundary}",
            sla_value=sla_value,
            throughput=stats.throughput,
            sample_count=len(stats.samples),
            error_count=stats.errors,
            before=before,
        )
        self._emit_event(
            event="boundary_discovered",
            phase="sustain",
            reason=reason,
            sla_value=sla_value,
            throughput=stats.throughput,
            sample_count=len(stats.samples),
            error_count=stats.errors,
            before=boundary,
        )

    def _passes_sla(self, observed: dict[str, float]) -> bool:
        return all(
            self._passes_single_sla(sla, observed[self._sla_key(sla)])
            for sla in self._sla_filters
        )

    @staticmethod
    def _passes_single_sla(sla: SLAFilter, observed: float) -> bool:
        match sla.op:
            case "lt":
                return observed < sla.threshold
            case "le":
                return observed <= sla.threshold
            case "gt":
                return observed > sla.threshold
            case "ge":
                return observed >= sla.threshold
        raise ValueError(f"Unsupported SLA operator: {sla.op}")

    def _next_up(self, observed_sla_values: dict[str, float] | None) -> int:
        return min(
            self._max_concurrency,
            self._current_concurrency
            + self._step_size(self._current_concurrency, observed_sla_values),
        )

    def _step_size(
        self, current: int, observed_sla_values: dict[str, float] | float | None
    ) -> int:
        if self._config.adaptive_scale_step_policy == "fixed_percent_step":
            pct = self._config.adaptive_scale_step_percent / 100.0
            return max(1, math.ceil(current * pct))
        if isinstance(observed_sla_values, (int, float)):
            observed_sla_values = {
                self._sla_key(self._primary_sla): float(observed_sla_values)
            }
        return self._sla_margin_step_size(observed_sla_values)

    def _sla_margin_step_size(
        self, observed_sla_values: dict[str, float] | None
    ) -> int:
        base_step = self._config.adaptive_scale_base_step
        if not observed_sla_values:
            return base_step

        margins: list[float] = []
        for sla in self._sla_filters:
            observed = observed_sla_values.get(self._sla_key(sla))
            if observed is None or sla.threshold == 0:
                continue
            threshold = abs(sla.threshold)
            if threshold == 0:
                continue
            match sla.op:
                case "lt" | "le":
                    margins.append((sla.threshold - observed) / threshold)
                case "gt" | "ge":
                    margins.append((observed - sla.threshold) / threshold)
        if not margins:
            return base_step

        effective_margin = max(0.0, min(margins))
        multiplier = max(
            1,
            min(
                self._config.adaptive_scale_max_step_multiplier,
                int(effective_margin * self._config.adaptive_scale_max_step_multiplier),
            ),
        )
        return base_step * multiplier

    def _set_concurrency(self, value: int) -> None:
        self._current_concurrency = max(1, min(value, self._max_concurrency))
        self._concurrency_manager.set_session_limit(
            CreditPhase.PROFILING, self._current_concurrency
        )

    def _emit_event(
        self,
        *,
        event: str,
        reason: str,
        sla_value: float | None,
        throughput: float,
        sample_count: int,
        error_count: int,
        before: int | None = None,
        phase: AdaptiveControllerPhase | None = None,
        passed: bool | None = None,
        step_size: int | None = None,
    ) -> None:
        payload = {
            "timestamp": time.time_ns(),
            "event": event,
            "phase": phase or self._controller_phase,
            "concurrency_before": (
                self._current_concurrency if before is None else before
            ),
            "concurrency_after": self._current_concurrency,
            "control_variable": self._config.adaptive_control_variable,
            "control_value": self._current_concurrency,
            "active_concurrency": self._current_concurrency,
            "boundary_concurrency": self._boundary_concurrency,
            "last_passing_value": self._last_good_concurrency,
            "first_failing_value": self._first_failing_concurrency,
            "sla_metric": self._primary_sla.metric_tag,
            "sla_stat": self._primary_sla.stat,
            "sla_op": self._primary_sla.op,
            "sla_value": sla_value,
            "sla_bound": self._primary_sla.threshold,
            "throughput": throughput,
            "sample_count": sample_count,
            "completed": sample_count,
            "sent": sample_count + error_count,
            "in_flight": None,
            "cancelled": None,
            "errored": error_count,
            "error_count": error_count,
            "sla_passed": passed,
            "strategy_type": self._config.adaptive_scale_strategy_type,
            "step_policy": self._config.adaptive_scale_step_policy,
            "step_size": step_size,
            "reason": reason,
        }
        if self._event_path is not None:
            with self._event_path.open("a", encoding="utf-8") as f:
                encoded = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
                f.write(encoded.decode() + "\n")

    def _complete_controller(
        self,
        *,
        reason: str,
        terminal_event: str = "adaptive_complete",
        sla_value: float | None = None,
        throughput: float = 0.0,
        sample_count: int = 0,
        error_count: int = 0,
    ) -> None:
        if self._completed_reason is not None:
            return
        self._controller_phase = "complete"
        self._completed_reason = reason
        self._emit_event(
            event=terminal_event,
            phase="complete",
            reason=reason,
            sla_value=sla_value,
            throughput=throughput,
            sample_count=sample_count,
            error_count=error_count,
        )
        self._write_summary()

    def _write_summary(self) -> None:
        if self._summary_written:
            return
        self._summary_written = True
        if self._summary_path is None:
            return
        summary = {
            "control_variable": self._config.adaptive_control_variable,
            "control_value": self._current_concurrency,
            "active_concurrency": self._current_concurrency,
            "boundary_concurrency": self._boundary_concurrency,
            "last_passing_value": self._last_good_concurrency,
            "first_failing_value": self._first_failing_concurrency,
            "last_good_concurrency": self._last_good_concurrency,
            "sustain_started_at": self._sustain_started_at_ns,
            "sustain_duration_seconds": self._sustain_duration,
            "completed_reason": self._completed_reason,
            "sla_passed_during_sustain": (
                self._sustain_windows > 0
                and self._sustain_passed_windows == self._sustain_windows
            ),
            "sustain_windows": self._sustain_windows,
            "sustain_passed_windows": self._sustain_passed_windows,
            "sla_metric": self._primary_sla.metric_tag,
            "sla_stat": self._primary_sla.stat,
            "sla_op": self._primary_sla.op,
            "sla_bound": self._primary_sla.threshold,
            "strategy_type": self._config.adaptive_scale_strategy_type,
            "step_policy": self._config.adaptive_scale_step_policy,
            "base_step": self._config.adaptive_scale_base_step,
            "max_step_multiplier": self._config.adaptive_scale_max_step_multiplier,
            "step_percent": self._config.adaptive_scale_step_percent,
        }
        encoded = orjson.dumps(
            summary, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
        )
        self._summary_path.write_bytes(encoded + b"\n")


def _percentile(samples: list[int], percentile: float) -> float:
    if not samples:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percentile / 100) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction
