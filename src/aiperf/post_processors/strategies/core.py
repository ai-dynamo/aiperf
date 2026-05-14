# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from aiperf.common.enums import CreditPhase
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models import CreditPhaseStats

if TYPE_CHECKING:
    from aiperf.common.config.user_config import UserConfig

OTelResultData = MetricRecordsData | CreditPhaseStats


@runtime_checkable
class HistogramInstrument(Protocol):
    """Protocol for histogram-like instruments returned by the context."""

    def record(self, value: float, attributes: dict[str, Any]) -> None: ...


@runtime_checkable
class CounterInstrument(Protocol):
    """Protocol for counter-like instruments returned by the context."""

    def add(self, value: float, attributes: dict[str, Any]) -> None: ...


@runtime_checkable
class OTelResultsStrategyProtocol(Protocol):
    """Public extension point for new streamed OTel result domains.

    A strategy owns exactly one ``OTelResultData`` variant and emits its
    telemetry via ``OTelStrategyContextProtocol``. Strategies MUST NOT touch
    OTel instruments, the fanout queue, or the MLflow client directly — the
    context owns instrument lifecycle and cross-strategy state (e.g. cumulative
    counter/gauge snapshots) so fanout stays consistent across strategies.
    """

    def supports(self, record_data: OTelResultData) -> bool:
        """Return True iff ``record_data`` is the variant this strategy consumes.

        Implementations use ``isinstance`` against a single concrete type —
        strategies are mutually exclusive by record type, so a given record
        dispatches to exactly one strategy. The dispatcher in
        ``OTelMetricsResultsProcessor`` iterates strategies in registration
        order and calls ``process`` on the first match.
        """
        ...

    async def process(self, record_data: OTelResultData) -> None:
        """Emit telemetry for ``record_data`` without blocking the hot path.

        Must be cheap — the caller is on the benchmark's record-processing
        dispatch path. Instrument access goes through the context's
        ``get_or_create_*`` factories (which enqueue fanout events rather than
        constructing OTel SDK instruments inline). Raising is permitted;
        ``OTelMetricsResultsProcessor.is_best_effort = True`` means the
        records manager logs and swallows the failure.
        """
        ...


@runtime_checkable
class OTelStrategyContextProtocol(Protocol):
    """Protocol implemented by the OTel processor to support strategy execution."""

    @property
    def user_config(self) -> UserConfig: ...

    async def get_or_create_histogram(
        self,
        metric_name: str,
        *,
        unit: str | None = None,
        description: str | None = None,
        explicit_bucket_boundaries: tuple[float, ...] | None = None,
    ) -> HistogramInstrument: ...

    async def get_or_create_counter(
        self, metric_name: str, unit: str, description: str
    ) -> CounterInstrument: ...

    async def get_or_create_up_down_counter(
        self, metric_name: str, unit: str, description: str
    ) -> CounterInstrument: ...

    def build_record_attributes(self, record: MetricRecordsData) -> dict[str, Any]: ...

    def build_timing_attributes(self, stats: CreditPhaseStats) -> dict[str, Any]: ...

    def coerce_metric_values(
        self, metric_name: str, metric_value: Any
    ) -> list[float]: ...

    def calculate_timing_counter_delta(
        self, *, metric_name: str, phase: CreditPhase, current_value: int
    ) -> int: ...

    def calculate_timing_gauge_delta(
        self, *, metric_name: str, phase: CreditPhase, current_value: float
    ) -> float: ...

    def timing_unit(self, metric_name: str) -> str: ...
