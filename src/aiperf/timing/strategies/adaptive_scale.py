# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-run adaptive scale timing strategy."""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import CreditReturn
from aiperf.timing.strategies.adaptive_scale_artifacts import (
    AdaptiveScaleArtifactWriter,
)
from aiperf.timing.strategies.adaptive_scale_controller import (
    AdaptiveScaleController,
)
from aiperf.timing.strategies.adaptive_scale_sla import (
    AdaptiveScaleSLAEvaluator,
    _percentile,
)
from aiperf.timing.strategies.adaptive_scale_types import (
    MIN_ASSESSMENT_PERIOD_SEC,
    AdaptiveControllerPhase,
    WindowStats,
)
from aiperf.timing.strategies.request_rate import RequestRateStrategy

__all__ = ["AdaptiveScaleStrategy", "WindowStats", "_percentile"]

if TYPE_CHECKING:
    from aiperf.config.sweep.adaptive import SLAFilter
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker


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
        self._controller = AdaptiveScaleController()
        self._sla = AdaptiveScaleSLAEvaluator()
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
        self._window_started_at_ns = time.time_ns()
        self._artifacts = AdaptiveScaleArtifactWriter()
        self._event_path = self._resolve_artifact_path(self.EVENT_FILE)
        self._summary_path = self._resolve_artifact_path(self.SUMMARY_FILE)
        self._adaptive_iteration = 0
        self._candidate_summaries: list[dict] = []
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

    _request_latency_value = staticmethod(
        AdaptiveScaleSLAEvaluator.request_latency_value
    )
    _throughput_value = staticmethod(AdaptiveScaleSLAEvaluator.throughput_value)
    _goodput_ratio_value = staticmethod(AdaptiveScaleSLAEvaluator.goodput_ratio_value)
    _validate_single_sla_filter = staticmethod(
        AdaptiveScaleSLAEvaluator.validate_single_filter
    )
    _passes_single_sla = staticmethod(AdaptiveScaleSLAEvaluator.passes_single)

    def _sla_value(self, sla: SLAFilter, stats: WindowStats) -> float:
        return self._sla.value(sla, stats)

    def _validate_sla_filters(self) -> None:
        self._sla.validate_filters(self._sla_filters)

    def _sla_values(self, stats: WindowStats) -> dict[str, float]:
        return self._sla.values(self._sla_filters, stats)

    @staticmethod
    def _sla_key(sla: SLAFilter) -> str:
        return AdaptiveScaleSLAEvaluator.key(sla)

    def _passes_sla(self, observed: dict[str, float]) -> bool:
        return self._sla.passes(self._sla_filters, observed)

    def _resolve_artifact_path(self, filename: str) -> Path | None:
        return self._artifacts.resolve_path(self._config.artifact_dir, filename)

    async def setup_phase(self) -> None:
        await super().setup_phase()
        await self._artifacts.start()
        self._set_concurrency(self._current_concurrency)
        self._emit_event(
            event="adaptive_phase_started",
            reason="adaptive scale discover phase started",
            sla_value=None,
            throughput=0.0,
            sample_count=0,
            error_count=0,
        )
        await self._artifacts.flush()

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
            await self._artifacts.close()

    async def handle_credit_result(self, credit_return: CreditReturn) -> None:
        async with self._lock:
            if (
                credit_return.error is not None
                or credit_return.cancelled
                or credit_return.request_latency_ns is None
            ):
                self._window_errors += 1
            else:
                self._window_latency_ns.append(credit_return.request_latency_ns)

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

    def _stop_sending(self) -> None:
        if not self._lifecycle.is_sending_complete:
            self._lifecycle.mark_sending_complete(timeout_triggered=False)
            self._progress.freeze_sent_counts()
        self._progress.all_credits_sent_event.set()

    async def _assess_window(self) -> None:
        await self._controller.assess_window(self)

    def _assess_failed_window(self, stats: WindowStats) -> None:
        self._controller.assess_failed_window(self, stats)

    async def _take_window(self) -> WindowStats:
        async with self._lock:
            now = time.perf_counter()
            end_ns = time.time_ns()
            stats = WindowStats(
                samples=self._window_latency_ns,
                errors=self._window_errors,
                elapsed_sec=now - self._window_started_at,
                start_ns=self._window_started_at_ns,
                end_ns=end_ns,
            )
            self._window_latency_ns = []
            self._window_errors = 0
            self._window_started_at = now
            self._window_started_at_ns = end_ns
            return stats

    def _assess_discover(
        self,
        sla_value: float,
        passing: bool,
        stats: WindowStats,
        sla_values: dict[str, float] | None = None,
    ) -> None:
        self._controller.assess_discover(
            self,
            sla_value=sla_value,
            passing=passing,
            stats=stats,
            sla_values=sla_values,
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
        self._controller.assess_sustain(
            self, sla_value, passing, stats, sla_values, reason=reason
        )

    def _enter_sustain(
        self, sla_value: float | None, stats: WindowStats, reason: str
    ) -> None:
        self._controller.enter_sustain(self, sla_value, stats, reason)

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
        phase_name = getattr(self._config, "name", None)
        phase_id = phase_name or CreditPhase.PROFILING.value
        run = getattr(self, "run", None)
        run_id = getattr(run, "benchmark_id", None)
        payload = self._artifacts.event_payload(
            timestamp_ns=time.time_ns(),
            event=event,
            phase=phase or self._controller_phase,
            current_concurrency=self._current_concurrency,
            control_variable=self._config.adaptive_control_variable,
            boundary_concurrency=self._boundary_concurrency,
            last_good_concurrency=self._last_good_concurrency,
            first_failing_concurrency=self._first_failing_concurrency,
            primary_sla=self._primary_sla,
            strategy_type=self._config.adaptive_scale_strategy_type,
            step_policy=self._config.adaptive_scale_step_policy,
            reason=reason,
            sla_value=sla_value,
            throughput=throughput,
            sample_count=sample_count,
            error_count=error_count,
            before=before,
            passed=passed,
            step_size=step_size,
        )
        payload.update(
            self._artifacts.correlation_payload(
                run_id=run_id,
                phase_id=phase_id,
                phase_name=phase_name,
                adaptive_iteration=self._adaptive_iteration,
                candidate_concurrency=before or self._current_concurrency,
                accepted_concurrency=self._current_concurrency,
            )
        )
        self._artifacts.emit_event(self._event_path, payload)

    def _record_candidate(
        self,
        *,
        stats: WindowStats,
        accepted: bool,
        rejection_reason: str,
    ) -> None:
        self._candidate_summaries.append(
            self._artifacts.candidate_payload(
                adaptive_iteration=self._adaptive_iteration,
                candidate_concurrency=self._current_concurrency,
                stats=stats,
                accepted=accepted,
                rejection_reason=rejection_reason,
            )
        )

    def _advance_adaptive_iteration(self) -> None:
        self._adaptive_iteration += 1

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
        status = self._status_for_terminal_reason(reason)
        self._emit_event(
            event=terminal_event,
            phase="complete",
            reason=reason,
            sla_value=sla_value,
            throughput=throughput,
            sample_count=sample_count,
            error_count=error_count,
        )
        self._write_summary(
            status=status,
            throughput=throughput,
            sample_count=sample_count,
            error_count=error_count,
        )

    @staticmethod
    def _status_for_terminal_reason(reason: str) -> str:
        if reason.startswith("assessment_failed:"):
            return "failed"
        return "completed"

    def _write_summary(
        self,
        *,
        status: str = "completed",
        throughput: float = 0.0,
        sample_count: int = 0,
        error_count: int = 0,
    ) -> None:
        if self._summary_written:
            return
        self._summary_written = True
        summary = self._artifacts.summary_payload(
            control_variable=self._config.adaptive_control_variable,
            current_concurrency=self._current_concurrency,
            boundary_concurrency=self._boundary_concurrency,
            last_good_concurrency=self._last_good_concurrency,
            first_failing_concurrency=self._first_failing_concurrency,
            sustain_started_at_ns=self._sustain_started_at_ns,
            sustain_duration=self._sustain_duration,
            completed_reason=self._completed_reason,
            status=status,
            sustain_windows=self._sustain_windows,
            sustain_passed_windows=self._sustain_passed_windows,
            throughput=throughput,
            sample_count=sample_count,
            error_count=error_count,
            candidates=self._candidate_summaries,
            primary_sla=self._primary_sla,
            strategy_type=self._config.adaptive_scale_strategy_type,
            step_policy=self._config.adaptive_scale_step_policy,
            base_step=self._config.adaptive_scale_base_step,
            max_step_multiplier=self._config.adaptive_scale_max_step_multiplier,
            step_percent=self._config.adaptive_scale_step_percent,
        )
        self._artifacts.write_summary(self._summary_path, summary)
