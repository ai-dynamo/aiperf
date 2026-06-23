# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller decisions for adaptive scale timing."""

from __future__ import annotations

import time

from aiperf.timing.strategies.adaptive_scale_types import WindowStats


class AdaptiveScaleController:
    """Evaluate windows and drive adaptive scale state transitions."""

    async def assess_window(self, strategy) -> None:
        stats = await strategy._take_window()
        try:
            if not stats.samples and stats.errors:
                strategy._emit_event(
                    event="adaptive_window",
                    reason="all requests failed in assessment window",
                    sla_value=None,
                    throughput=stats.throughput,
                    sample_count=0,
                    error_count=stats.errors,
                    passed=False,
                )
                strategy._record_candidate(
                    stats=stats,
                    accepted=False,
                    rejection_reason="error_threshold",
                )
                self.assess_failed_window(strategy, stats)
                return

            if len(stats.samples) < strategy._min_completed_requests:
                strategy._emit_event(
                    event="adaptive_window",
                    reason="inconclusive: completed request count below minimum",
                    sla_value=None,
                    throughput=stats.throughput,
                    sample_count=len(stats.samples),
                    error_count=stats.errors,
                    passed=None,
                )
                strategy._record_candidate(
                    stats=stats,
                    accepted=False,
                    rejection_reason="insufficient_samples",
                )
                return

            sla_values = strategy._sla_values(stats)
            primary_value = sla_values[strategy._sla_key(strategy._primary_sla)]
            passing = strategy._passes_sla(sla_values)
            strategy._emit_event(
                event="adaptive_window",
                reason="SLA window evaluated",
                sla_value=primary_value,
                throughput=stats.throughput,
                sample_count=len(stats.samples),
                error_count=stats.errors,
                passed=passing,
            )
            strategy._record_candidate(
                stats=stats,
                accepted=passing,
                rejection_reason="sla_miss",
            )

            if strategy._controller_phase == "discover":
                self.assess_discover(
                    strategy,
                    sla_value=primary_value,
                    passing=passing,
                    stats=stats,
                    sla_values=sla_values,
                )
            elif strategy._controller_phase == "sustain":
                self.assess_sustain(strategy, primary_value, passing, stats, sla_values)
        finally:
            strategy._advance_adaptive_iteration()

    def assess_failed_window(self, strategy, stats: WindowStats) -> None:
        reason = "all requests failed in assessment window"
        if strategy._controller_phase == "discover":
            if strategy._last_good_concurrency is None:
                strategy._first_failing_concurrency = strategy._current_concurrency
                strategy._complete_controller(
                    reason="no_sustainable_concurrency_found",
                    terminal_event="adaptive_failed",
                    throughput=stats.throughput,
                    sample_count=0,
                    error_count=stats.errors,
                )
                strategy._stop_sending()
                return
            strategy._first_failing_concurrency = strategy._current_concurrency
            self.enter_sustain(strategy, None, stats, reason)
        elif strategy._controller_phase == "sustain":
            self.assess_sustain(strategy, None, False, stats, reason=reason)

    def assess_discover(
        self,
        strategy,
        *,
        sla_value: float,
        passing: bool,
        stats: WindowStats,
        sla_values: dict[str, float] | None = None,
    ) -> None:
        if passing:
            strategy._last_good_concurrency = strategy._current_concurrency
            if strategy._current_concurrency >= strategy._max_concurrency:
                strategy._complete_controller(
                    reason="max_concurrency_reached_without_saturation",
                    terminal_event="adaptive_incomplete",
                    sla_value=sla_value,
                    throughput=stats.throughput,
                    sample_count=len(stats.samples),
                    error_count=stats.errors,
                )
                strategy._stop_sending()
                return
            before = strategy._current_concurrency
            next_value = strategy._next_up(sla_values)
            step_size = next_value - before
            strategy._set_concurrency(next_value)
            strategy._emit_event(
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

        if strategy._last_good_concurrency is None:
            strategy._first_failing_concurrency = strategy._current_concurrency
            strategy._complete_controller(
                reason="no_sustainable_concurrency_found",
                terminal_event="adaptive_failed",
                sla_value=sla_value,
                throughput=stats.throughput,
                sample_count=len(stats.samples),
                error_count=stats.errors,
            )
            strategy._stop_sending()
            return
        strategy._first_failing_concurrency = strategy._current_concurrency
        self.enter_sustain(
            strategy,
            sla_value,
            stats,
            f"SLA value {sla_value:.3f} breaches configured filters",
        )

    def assess_sustain(
        self,
        strategy,
        sla_value: float | None,
        passing: bool,
        stats: WindowStats,
        sla_values: dict[str, float] | None = None,
        *,
        reason: str | None = None,
    ) -> None:
        strategy._sustain_windows += 1
        if passing:
            strategy._sustain_passed_windows += 1
            strategy._last_good_concurrency = strategy._current_concurrency
            strategy._emit_event(
                event="adaptive_decision",
                reason=f"SLA value {sla_value:.3f} passes configured filters during sustain",
                sla_value=sla_value,
                throughput=stats.throughput,
                sample_count=len(stats.samples),
                error_count=stats.errors,
            )
        else:
            before = strategy._current_concurrency
            target = max(
                strategy._config.adaptive_scale_min_concurrency,
                strategy._last_good_concurrency
                or strategy._config.adaptive_scale_min_concurrency,
            )
            if target >= before:
                target = max(
                    strategy._config.adaptive_scale_min_concurrency,
                    before - strategy._step_size(before, sla_values),
                )
            if target == before == strategy._config.adaptive_scale_min_concurrency:
                strategy._complete_controller(
                    reason="sustain_failed_sla_unrecoverable",
                    terminal_event="adaptive_failed",
                    sla_value=sla_value,
                    throughput=stats.throughput,
                    sample_count=len(stats.samples),
                    error_count=stats.errors,
                )
                strategy._stop_sending()
                return
            strategy._set_concurrency(target)
            strategy._emit_event(
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

        if strategy._sustain_started_at is not None:
            elapsed = time.perf_counter() - strategy._sustain_started_at
            if elapsed >= strategy._sustain_duration:
                strategy._complete_controller(
                    reason="sustain_duration_completed",
                    terminal_event="adaptive_complete",
                    sla_value=sla_value,
                    throughput=stats.throughput,
                    sample_count=len(stats.samples),
                    error_count=stats.errors,
                )
                strategy._stop_sending()

    def enter_sustain(
        self, strategy, sla_value: float | None, stats: WindowStats, reason: str
    ) -> None:
        if strategy._last_good_concurrency is None:
            raise RuntimeError("cannot enter sustain without a passing boundary")
        boundary = max(
            strategy._config.adaptive_scale_min_concurrency,
            strategy._last_good_concurrency,
        )
        before = strategy._current_concurrency
        strategy._boundary_concurrency = boundary
        strategy._set_concurrency(boundary)
        strategy._controller_phase = "sustain"
        strategy._sustain_started_at = time.perf_counter()
        strategy._sustain_started_at_ns = time.time_ns()
        strategy._emit_event(
            event="sustain_started",
            phase="sustain",
            reason=f"holding boundary_concurrency={boundary}",
            sla_value=sla_value,
            throughput=stats.throughput,
            sample_count=len(stats.samples),
            error_count=stats.errors,
            before=before,
        )
        strategy._emit_event(
            event="boundary_discovered",
            phase="sustain",
            reason=reason,
            sla_value=sla_value,
            throughput=stats.throughput,
            sample_count=len(stats.samples),
            error_count=stats.errors,
            before=boundary,
        )
