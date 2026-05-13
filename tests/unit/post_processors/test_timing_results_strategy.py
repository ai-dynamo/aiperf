# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import CreditPhaseStats
from aiperf.post_processors.strategies.timing_results import TimingResultsStrategy
from tests.unit.post_processors.conftest import create_metric_records_message


@dataclass
class _FakeAddInstrument:
    adds: list[tuple[float, dict[str, Any]]] = field(default_factory=list)

    def add(self, value: float, attributes: dict[str, Any]) -> None:
        self.adds.append((value, attributes))


class _TimingStrategyContext:
    def __init__(
        self,
        *,
        attributes: dict[str, Any],
        counter_deltas: dict[str, int] | None = None,
        gauge_deltas: dict[str, float] | None = None,
        timing_units: dict[str, str] | None = None,
    ) -> None:
        self.attributes = attributes
        self.counter_deltas = counter_deltas or {}
        self.gauge_deltas = gauge_deltas or {}
        self.timing_units = timing_units or {}
        self.counters: dict[str, _FakeAddInstrument] = {}
        self.up_down_counters: dict[str, _FakeAddInstrument] = {}
        self.build_timing_attributes_calls: list[CreditPhaseStats] = []
        self.counter_delta_calls: list[tuple[str, CreditPhase, int]] = []
        self.gauge_delta_calls: list[tuple[str, CreditPhase, float]] = []
        self.get_counter_calls: list[tuple[str, str, str]] = []
        self.get_up_down_counter_calls: list[tuple[str, str, str]] = []
        self._user_config: Any = None

    @property
    def user_config(self) -> Any:
        if self._user_config is None:
            from aiperf.common.config import EndpointConfig, UserConfig
            from aiperf.plugin.enums import EndpointType

            self._user_config = UserConfig(
                endpoint=EndpointConfig(
                    model_names=["test-model"],
                    type=EndpointType.CHAT,
                ),
            )
        return self._user_config

    def build_timing_attributes(self, stats: CreditPhaseStats) -> dict[str, Any]:
        self.build_timing_attributes_calls.append(stats)
        return self.attributes

    def calculate_timing_counter_delta(
        self, *, metric_name: str, phase: CreditPhase, current_value: int
    ) -> int:
        self.counter_delta_calls.append((metric_name, phase, current_value))
        return self.counter_deltas.get(metric_name, 0)

    def calculate_timing_gauge_delta(
        self, *, metric_name: str, phase: CreditPhase, current_value: float
    ) -> float:
        self.gauge_delta_calls.append((metric_name, phase, current_value))
        return self.gauge_deltas.get(metric_name, 0.0)

    async def get_or_create_counter(
        self, metric_name: str, unit: str, description: str
    ) -> _FakeAddInstrument:
        self.get_counter_calls.append((metric_name, unit, description))
        return self.counters.setdefault(metric_name, _FakeAddInstrument())

    async def get_or_create_up_down_counter(
        self, metric_name: str, unit: str, description: str
    ) -> _FakeAddInstrument:
        self.get_up_down_counter_calls.append((metric_name, unit, description))
        return self.up_down_counters.setdefault(metric_name, _FakeAddInstrument())

    def timing_unit(self, metric_name: str) -> str:
        return self.timing_units.get(metric_name, "1")


def _create_credit_phase_stats(**overrides: Any) -> CreditPhaseStats:
    values: dict[str, Any] = {
        "phase": CreditPhase.PROFILING,
        "start_ns": 1_000_000_000,
        "requests_end_ns": 3_000_000_000,
        "requests_sent": 10,
        "requests_completed": 7,
        "requests_cancelled": 1,
        "request_errors": 2,
        "sent_sessions": 4,
        "completed_sessions": 2,
        "cancelled_sessions": 1,
        "total_session_turns": 9,
        "timeout_triggered": True,
        "grace_period_timeout_triggered": False,
        "was_cancelled": False,
    }
    values.update(overrides)
    return CreditPhaseStats(**values)


class TestTimingResultsStrategy:
    def test_supports_only_credit_phase_stats(self) -> None:
        strategy = TimingResultsStrategy(_TimingStrategyContext(attributes={}))
        timing_stats = _create_credit_phase_stats()
        metric_record = create_metric_records_message(
            results=[{"request_latency_ns": 1}]
        ).to_data()

        assert strategy.supports(timing_stats) is True
        assert strategy.supports(metric_record) is False

    @pytest.mark.asyncio
    async def test_process_emits_positive_counter_and_gauge_deltas(self) -> None:
        context = _TimingStrategyContext(
            attributes={"aiperf.benchmark_phase": "profiling"},
            counter_deltas={
                "aiperf.timing.requests.sent": 3,
                "aiperf.timing.sessions.completed": 2,
            },
            gauge_deltas={
                "aiperf.timing.phase.timeout_triggered": 1.0,
                "aiperf.timing.phase.elapsed_sec": 2.0,
            },
            timing_units={"aiperf.timing.phase.elapsed_sec": "s"},
        )
        strategy = TimingResultsStrategy(context)
        timing_stats = _create_credit_phase_stats()

        await strategy.process(timing_stats)

        assert context.build_timing_attributes_calls == [timing_stats]
        assert (
            "aiperf.timing.requests.sent",
            CreditPhase.PROFILING,
            10,
        ) in context.counter_delta_calls
        assert (
            "aiperf.timing.sessions.completed",
            CreditPhase.PROFILING,
            2,
        ) in context.counter_delta_calls
        assert (
            "aiperf.timing.phase.timeout_triggered",
            CreditPhase.PROFILING,
            1.0,
        ) in context.gauge_delta_calls
        assert (
            "aiperf.timing.phase.elapsed_sec",
            CreditPhase.PROFILING,
            2.0,
        ) in context.gauge_delta_calls

        expected_attrs = {
            "aiperf.benchmark_phase": "profiling",
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "_OTHER",
            "gen_ai.request.model": "test-model",
        }
        assert context.counters["aiperf.timing.requests.sent"].adds == [
            (3, expected_attrs)
        ]
        assert context.counters["aiperf.timing.sessions.completed"].adds == [
            (2, expected_attrs)
        ]
        assert context.up_down_counters[
            "aiperf.timing.phase.timeout_triggered"
        ].adds == [(1.0, expected_attrs)]
        assert context.up_down_counters["aiperf.timing.phase.elapsed_sec"].adds == [
            (2.0, expected_attrs)
        ]
        assert (
            "aiperf.timing.phase.elapsed_sec",
            "s",
            "AIPerf timing gauge metric: requests_elapsed_time",
        ) in context.get_up_down_counter_calls

    @pytest.mark.asyncio
    async def test_process_skips_non_positive_or_zero_deltas(self) -> None:
        context = _TimingStrategyContext(
            attributes={"aiperf.benchmark_phase": "profiling"},
            counter_deltas={
                "aiperf.timing.requests.sent": 0,
                "aiperf.timing.requests.completed": -1,
            },
            gauge_deltas={
                "aiperf.timing.requests.in_flight": 0.0,
                "aiperf.timing.phase.elapsed_sec": 0.0,
            },
        )
        strategy = TimingResultsStrategy(context)

        await strategy.process(_create_credit_phase_stats())

        assert context.counters == {}
        assert context.up_down_counters == {}
        assert context.get_counter_calls == []
        assert context.get_up_down_counter_calls == []

    @pytest.mark.asyncio
    async def test_process_ignores_non_timing_inputs(self) -> None:
        context = _TimingStrategyContext(attributes={})
        strategy = TimingResultsStrategy(context)

        await strategy.process(create_metric_records_message(results=[]).to_data())

        assert context.build_timing_attributes_calls == []
        assert context.counter_delta_calls == []
        assert context.gauge_delta_calls == []
