# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime artifact helpers for adaptive scale timing."""

from __future__ import annotations

import time

from aiperf.common.enums import CreditPhase
from aiperf.timing.strategies.adaptive_scale_types import (
    AdaptiveControllerPhase,
    WindowStats,
)


class AdaptiveScaleRuntimeMixin:
    """Emit adaptive-scale events, candidates, and terminal summaries."""

    def _emit_event(
        self,
        *,
        event: str,
        reason: str,
        sla_value: float | None,
        throughput: float,
        sample_count: int,
        error_count: int,
        cancelled_count: int = 0,
        before: int | None = None,
        phase: AdaptiveControllerPhase | None = None,
        passed: bool | None = None,
        step_size: float | None = None,
        sla_values: dict[str, float] | None = None,
    ) -> None:
        phase_name = getattr(self._config, "name", None)
        phase_id = phase_name or CreditPhase.PROFILING
        run = getattr(self, "run", None)
        run_id = getattr(run, "benchmark_id", None)
        payload = self._artifacts.event_payload(
            timestamp_ns=time.time_ns(),
            event=event,
            phase=phase or self._controller_phase,
            control_value=self._control.current,
            control_snapshot=self._control.snapshot(),
            control_variable=self._config.adaptive_control_variable,
            boundary_value=self._boundary_concurrency,
            last_passing_value=self._last_good_concurrency,
            first_failing_value=self._first_failing_concurrency,
            primary_sla=self._primary_sla,
            strategy_type=self._config.adaptive_scale_strategy_type,
            step_policy=self._config.adaptive_scale_step_policy,
            reason=reason,
            sla_value=sla_value,
            throughput=throughput,
            sample_count=sample_count,
            error_count=error_count,
            cancelled_count=cancelled_count,
            before=before,
            passed=passed,
            step_size=step_size,
            sla_values=sla_values,
            binding_sla=self._binding_sla_key(sla_values),
        )
        payload.update(
            self._artifacts.correlation_payload(
                run_id=run_id,
                phase_id=phase_id,
                phase_name=phase_name,
                adaptive_iteration=self._adaptive_iteration,
                candidate_value=(self._control.current if before is None else before),
                accepted_value=self._control.current,
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
                candidate_value=self._control.current,
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
        cancelled_count: int = 0,
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
            cancelled_count=cancelled_count,
        )
        self._write_summary(
            status=status,
            throughput=throughput,
            sample_count=sample_count,
            error_count=error_count,
            cancelled_count=cancelled_count,
        )

    @staticmethod
    def _status_for_terminal_reason(reason: str) -> str:
        if reason == "max_control_value_reached_without_saturation":
            return "incomplete"
        if reason.startswith("assessment_failed:") or reason in {
            "no_sustainable_concurrency_found",
            "sustain_failed_sla_unrecoverable",
            "sustain_failed_after_recovery",
        }:
            return "failed"
        return "completed"

    def _write_summary(
        self,
        *,
        status: str = "completed",
        throughput: float = 0.0,
        sample_count: int = 0,
        error_count: int = 0,
        cancelled_count: int = 0,
    ) -> None:
        if self._summary_written:
            return
        self._summary_written = True
        summary = self._artifacts.summary_payload(
            control_variable=self._config.adaptive_control_variable,
            control_value=self._control.current,
            control_snapshot=self._control.snapshot(),
            boundary_value=self._boundary_concurrency,
            last_passing_value=self._last_good_concurrency,
            first_failing_value=self._first_failing_concurrency,
            sustain_started_at_ns=self._sustain_started_at_ns,
            sustain_duration=self._sustain_duration,
            completed_reason=self._completed_reason,
            status=status,
            sustain_windows=self._sustain_windows,
            sustain_passed_windows=self._sustain_passed_windows,
            throughput=throughput,
            sample_count=sample_count,
            error_count=error_count,
            cancelled_count=cancelled_count,
            candidates=self._candidate_summaries,
            primary_sla=self._primary_sla,
            strategy_type=self._config.adaptive_scale_strategy_type,
            step_policy=self._config.adaptive_scale_step_policy,
            base_step=self._config.adaptive_scale_base_step,
            max_step_multiplier=self._config.adaptive_scale_max_step_multiplier,
            step_percent=self._config.adaptive_scale_step_percent,
        )
        self._artifacts.write_summary(self._summary_path, summary)
