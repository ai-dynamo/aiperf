# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.models import CreditPhaseStats
from aiperf.post_processors.strategies import genai_semconv
from aiperf.post_processors.strategies.core import (
    OTelResultData,
    OTelResultsStrategyProtocol,
    OTelStrategyContextProtocol,
)


class TimingResultsStrategy(OTelResultsStrategyProtocol):
    """Streams phase-level timing snapshots using counters and gauge-like metrics."""

    _COUNTER_FIELDS = {
        "aiperf.timing.requests.sent": "requests_sent",
        "aiperf.timing.requests.completed": "requests_completed",
        "aiperf.timing.requests.cancelled": "requests_cancelled",
        "aiperf.timing.requests.errors": "request_errors",
        "aiperf.timing.sessions.sent": "sent_sessions",
        "aiperf.timing.sessions.completed": "completed_sessions",
        "aiperf.timing.sessions.cancelled": "cancelled_sessions",
        "aiperf.timing.sessions.turns_total": "total_session_turns",
    }
    _GAUGE_FIELDS = {
        "aiperf.timing.requests.in_flight": "in_flight_requests",
        "aiperf.timing.sessions.in_flight": "in_flight_sessions",
        "aiperf.timing.phase.timeout_triggered": "timeout_triggered",
        "aiperf.timing.phase.grace_timeout_triggered": "grace_period_timeout_triggered",
        "aiperf.timing.phase.was_cancelled": "was_cancelled",
        "aiperf.timing.phase.elapsed_sec": "requests_elapsed_time",
    }

    def __init__(self, context: OTelStrategyContextProtocol) -> None:
        self._context = context

    def supports(self, record_data: OTelResultData) -> bool:
        return isinstance(record_data, CreditPhaseStats)

    async def process(self, record_data: OTelResultData) -> None:
        if not isinstance(record_data, CreditPhaseStats):
            return

        attributes = self._context.build_timing_attributes(record_data)
        attributes.update(
            genai_semconv.cross_metric_attributes(self._context.user_config)
        )

        # Instrument counter fields
        for metric_name, field_name in self._COUNTER_FIELDS.items():
            current_value = int(getattr(record_data, field_name))
            delta_value = self._context.calculate_timing_counter_delta(
                metric_name=metric_name,
                phase=record_data.phase,
                current_value=current_value,
            )
            if delta_value <= 0:
                continue

            instrument = await self._context.get_or_create_counter(
                metric_name=metric_name,
                unit="1",
                description=f"AIPerf timing counter: {field_name}",
            )
            instrument.add(delta_value, attributes)

        # Instrument gauge fields
        for metric_name, field_name in self._GAUGE_FIELDS.items():
            raw_value = getattr(record_data, field_name)
            if isinstance(raw_value, bool):
                current_value = 1.0 if raw_value else 0.0
            elif isinstance(raw_value, int | float):
                current_value = float(raw_value)
            else:
                continue

            delta_value = self._context.calculate_timing_gauge_delta(
                metric_name=metric_name,
                phase=record_data.phase,
                current_value=current_value,
            )
            if abs(delta_value) < 1e-9:
                continue

            instrument = await self._context.get_or_create_up_down_counter(
                metric_name=metric_name,
                unit=self._context.timing_unit(metric_name),
                description=f"AIPerf timing gauge metric: {field_name}",
            )
            instrument.add(delta_value, attributes)
