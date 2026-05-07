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
class OTelResultsStrategyProtocol(Protocol):
    """Protocol for OTel result processing strategies."""

    def supports(self, record_data: OTelResultData) -> bool: ...

    async def process(self, record_data: OTelResultData) -> None: ...


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
    ) -> Any: ...

    async def get_or_create_counter(
        self, metric_name: str, unit: str, description: str
    ) -> Any: ...

    async def get_or_create_up_down_counter(
        self, metric_name: str, unit: str, description: str
    ) -> Any: ...

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
