# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-run adaptive scale timing strategy."""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit
from aiperf.timing.strategies.adaptive_scale_artifacts import (
    AdaptiveScaleArtifactWriter,
)
from aiperf.timing.strategies.adaptive_scale_backends import (
    build_adaptive_control_backend,
)
from aiperf.timing.strategies.adaptive_scale_controller import (
    AdaptiveScaleController,
)
from aiperf.timing.strategies.adaptive_scale_runtime import AdaptiveScaleRuntimeMixin
from aiperf.timing.strategies.adaptive_scale_sla import (
    AdaptiveScaleSLAEvaluator,
    _percentile,
)
from aiperf.timing.strategies.adaptive_scale_types import (
    MIN_ASSESSMENT_PERIOD_SEC,
    AdaptiveControllerPhase,
    WindowRequestSample,
    WindowStats,
)
from aiperf.timing.strategies.request_rate import RequestRateStrategy
from aiperf.timing.strategies.user_centric_rate import UserCentricStrategy

__all__ = ["AdaptiveScaleStrategy", "WindowStats", "_percentile"]

if TYPE_CHECKING:
    from aiperf.config.sweep.adaptive import SLAFilter
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker


class AdaptiveScaleStrategy(AdaptiveScaleRuntimeMixin, RequestRateStrategy):
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
    ) -> None:
        super().__init__(**kwargs)
        self._concurrency_manager = concurrency_manager
        self._progress = progress
        self._init_user_control(kwargs)
        self._init_adaptive_control(concurrency_manager)
        self._init_controller_state()
        self._init_window_state()
        self._init_artifacts()

    def _init_user_control(self, kwargs: dict[str, Any]) -> None:
        self._target_users = self._config.num_users or 0
        self._retiring_users = 0
        self._retired_user_cancellations = 0
        self._user_strategy = (
            UserCentricStrategy(**kwargs)
            if self._config.adaptive_control_variable == "users"
            else None
        )

    def _init_adaptive_control(self, concurrency_manager: ConcurrencyManager) -> None:
        self._control = build_adaptive_control_backend(
            strategy=self,
            concurrency_manager=concurrency_manager,
            config=self._config,
        )
        self._max_concurrency = self._control.maximum
        self._min_completed_requests = self._config.adaptive_min_completed_requests
        self._assessment_period = self._config.adaptive_assessment_period_sec
        self._validate_adaptive_config()
        self._sustain_duration = self._config.adaptive_sustain_duration_sec
        self._controller = AdaptiveScaleController()
        self._sla = AdaptiveScaleSLAEvaluator()
        self._sla_filters = list(self._config.adaptive_sla_filters)
        self._primary_sla = self._sla_filters[0]
        self._validate_sla_filters()

    def _validate_adaptive_config(self) -> None:
        if self._assessment_period < MIN_ASSESSMENT_PERIOD_SEC:
            raise ValueError(
                "adaptive_assessment_period_sec must be >= "
                f"{MIN_ASSESSMENT_PERIOD_SEC:g}"
            )
        if self._config.adaptive_sustain_duration_sec is None:
            raise ValueError("adaptive_sustain_duration_sec is required")
        if self._config.adaptive_scale_strategy_type != "ramp_until_fail":
            raise ValueError("adaptive_scale strategy type must be 'ramp_until_fail'")
        if not self._config.adaptive_sla_filters:
            raise ValueError("adaptive_sla_filters is required")

    def _init_controller_state(self) -> None:
        self._controller_phase: AdaptiveControllerPhase = "discover"
        self._boundary_concurrency: float | None = None
        self._last_good_concurrency: float | None = None
        self._first_failing_concurrency: float | None = None
        self._sustain_started_at: float | None = None
        self._assessment_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._adaptive_iteration = 0
        self._candidate_summaries: list[dict] = []
        self._sustain_started_at_ns: int | None = None
        self._sustain_recovery_used = False
        self._sustain_windows = 0
        self._sustain_passed_windows = 0
        self._completed_reason: str | None = None
        self._summary_written = False

    def _init_window_state(self) -> None:
        self._window_latency_ns: list[int] = []
        self._window_itl_ns: list[float] = []
        self._window_ttft_by_credit_id: dict[int, int] = {}
        self._window_successful_requests: list[WindowRequestSample] = []
        self._window_errors = 0
        self._window_cancelled = 0
        self._window_started_at = time.perf_counter()
        self._window_started_at_ns = time.time_ns()

    def _init_artifacts(self) -> None:
        self._artifacts = AdaptiveScaleArtifactWriter()
        self._event_path = self._resolve_artifact_path(self.EVENT_FILE)
        self._summary_path = self._resolve_artifact_path(self.SUMMARY_FILE)

    @staticmethod
    def _require_positive(value: int | None, name: str) -> int:
        if value is None or value < 1:
            raise ValueError(f"{name} must be >= 1 for adaptive scale")
        return value

    _request_latency_value = staticmethod(
        AdaptiveScaleSLAEvaluator.request_latency_value
    )
    _throughput_value = staticmethod(AdaptiveScaleSLAEvaluator.throughput_value)
    _inter_token_latency_value = staticmethod(
        AdaptiveScaleSLAEvaluator.inter_token_latency_value
    )
    _goodput_value = staticmethod(AdaptiveScaleSLAEvaluator.goodput_value)
    _success_rate_value = staticmethod(AdaptiveScaleSLAEvaluator.success_rate_value)
    _validate_single_sla_filter = staticmethod(
        AdaptiveScaleSLAEvaluator.validate_single_filter
    )
    _passes_single_sla = staticmethod(AdaptiveScaleSLAEvaluator.passes_single)

    def _sla_value(self, sla: SLAFilter, stats: WindowStats) -> float:
        return self._sla.value(sla, stats, self._sla_filters)

    def _validate_sla_filters(self) -> None:
        self._sla.validate_filters(self._sla_filters)

    def _sla_values(self, stats: WindowStats) -> dict[str, float]:
        return self._sla.values(self._sla_filters, stats)

    @staticmethod
    def _sla_key(sla: SLAFilter) -> str:
        return AdaptiveScaleSLAEvaluator.key(sla)

    def _passes_sla(self, observed: dict[str, float]) -> bool:
        return self._sla.passes(self._sla_filters, observed)

    def _binding_sla_key(self, observed: dict[str, float] | None) -> str | None:
        if not observed:
            return None
        best_key: str | None = None
        best_margin: float | None = None
        for sla in self._sla_filters:
            key = self._sla_key(sla)
            margin = self._sla_margin(sla, observed.get(key))
            if margin is None:
                continue
            if best_margin is None or margin < best_margin:
                best_margin = margin
                best_key = key
        return best_key

    @staticmethod
    def _sla_margin(sla: SLAFilter, observed: float | None) -> float | None:
        if observed is None or sla.threshold == 0:
            return None
        threshold = abs(sla.threshold)
        if sla.op in {"lt", "le"}:
            return (sla.threshold - observed) / threshold
        return (observed - sla.threshold) / threshold

    def _resolve_artifact_path(self, filename: str) -> Path | None:
        return self._artifacts.phase_scoped_path(
            self._config.artifact_dir, self._config.phase_name, filename
        )

    def _write_adaptive_manifest_entry(self) -> None:
        if self._config.artifact_dir is None or self._config.phase_name is None:
            return
        phase_root = f"phases/{self._config.phase_name}"
        self._artifacts.write_manifest_entry(
            self._config.artifact_dir,
            {
                "phase_index": self._config.phase_index,
                "profiling_index": self._config.profiling_index,
                "phase_name": self._config.phase_name,
                "phase_kind": self._config.phase_kind,
                "events_path": f"{phase_root}/{self.EVENT_FILE}",
                "summary_path": f"{phase_root}/{self.SUMMARY_FILE}",
            },
        )

    async def setup_phase(self) -> None:
        await self._artifacts.start()
        setup_complete = False
        try:
            self._set_control(self._control.minimum)
            if self._user_strategy is not None:
                await self._user_strategy.setup_phase()
            else:
                await super().setup_phase()
            self._emit_event(
                event="adaptive_phase_started",
                reason="adaptive scale discover phase started",
                sla_value=None,
                throughput=0.0,
                sample_count=0,
                error_count=0,
            )
            self._write_adaptive_manifest_entry()
            await self._artifacts.flush()
            setup_complete = True
        finally:
            if not setup_complete:
                await self._artifacts.close()

    async def execute_phase(self) -> None:
        self._assessment_task = asyncio.create_task(self._assessment_loop())
        try:
            try:
                if self._user_strategy is not None:
                    await self._user_strategy.execute_phase()
                else:
                    await super().execute_phase()
            except asyncio.CancelledError:
                if self._completed_reason is None:
                    self._complete_controller(
                        reason="phase_cancelled",
                        terminal_event="adaptive_failed",
                    )
                raise
            except Exception as exc:
                if self._completed_reason is None:
                    self._complete_controller(
                        reason=f"phase_failed: {exc}",
                        terminal_event="adaptive_failed",
                    )
                raise
        finally:
            if self._completed_reason is None:
                self._complete_controller(reason="phase_stopped")
            if self._assessment_task is not None:
                self._assessment_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._assessment_task
            await self._artifacts.close()

    async def handle_credit_return(
        self, credit: Credit, *, error: str | None = None
    ) -> None:
        if self._user_strategy is not None:
            await self._user_strategy.handle_credit_return(credit, error=error)
            return
        await super().handle_credit_return(credit, error=error)

    def set_target_users(self, value: int) -> None:
        self._target_users = value
        if self._user_strategy is not None:
            self._user_strategy.set_target_users(value)

    def user_control_snapshot(self) -> dict[str, int]:
        if self._user_strategy is not None:
            return self._user_strategy.user_control_snapshot()
        active = self._target_users + self._retiring_users
        return {
            "target_value": self._target_users,
            "actual_value": active,
            "active_users": active,
            "retiring_users": self._retiring_users,
            "cancelled": self._retired_user_cancellations,
        }

    def _is_pre_sustain_credit(self, credit: Credit) -> bool:
        return (
            self._controller_phase == "sustain"
            and self._sustain_started_at_ns is not None
            and credit.issued_at_ns < self._sustain_started_at_ns
        )

    async def handle_credit_result(self, credit_return: CreditReturn) -> None:
        async with self._lock:
            ttft_ns = self._window_ttft_by_credit_id.pop(credit_return.credit.id, None)
            if self._is_pre_sustain_credit(credit_return.credit):
                return
            if (
                credit_return.error is not None
                or credit_return.request_latency_ns is None
            ):
                if credit_return.cancelled:
                    self._window_cancelled += 1
                else:
                    self._window_errors += 1
            else:
                self._window_latency_ns.append(credit_return.request_latency_ns)
                if credit_return.inter_token_latency_ns is not None:
                    self._window_itl_ns.append(credit_return.inter_token_latency_ns)
                self._window_successful_requests.append(
                    WindowRequestSample(
                        request_latency_ns=credit_return.request_latency_ns,
                        ttft_ns=ttft_ns,
                        inter_token_latency_ns=credit_return.inter_token_latency_ns,
                        output_sequence_length=credit_return.output_sequence_length,
                    )
                )

    async def handle_first_token(self, first_token: FirstToken) -> None:
        async with self._lock:
            self._window_ttft_by_credit_id[first_token.credit_id] = first_token.ttft_ns

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
        except Exception as exc:  # noqa: BLE001 - keep background task failures terminal
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
            successful_requests = self._window_successful_requests
            stats = WindowStats(
                samples=self._window_latency_ns,
                errors=self._window_errors,
                ttft_samples=[
                    sample.ttft_ns
                    for sample in successful_requests
                    if sample.ttft_ns is not None
                ],
                itl_samples=self._window_itl_ns,
                successful_requests=successful_requests,
                cancelled=self._window_cancelled,
                elapsed_sec=now - self._window_started_at,
                start_ns=self._window_started_at_ns,
                end_ns=end_ns,
            )
            self._window_latency_ns = []
            self._window_itl_ns = []
            self._window_successful_requests = []
            self._window_errors = 0
            self._window_cancelled = 0
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

    def _next_up(self, observed_sla_values: dict[str, float] | None) -> float:
        return min(
            self._control.maximum,
            self._control.current
            + self._step_size(self._control.current, observed_sla_values),
        )

    def _step_size(
        self, current: float, observed_sla_values: dict[str, float] | float | None
    ) -> float:
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
            margin = self._sla_margin(sla, observed)
            if margin is not None:
                margins.append(margin)
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

    def _set_control(self, value: float) -> None:
        self._control.set(value)

    def _set_concurrency(self, value: int) -> None:
        self._set_control(value)
