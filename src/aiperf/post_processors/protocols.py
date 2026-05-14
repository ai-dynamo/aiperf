# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final, Protocol, runtime_checkable

from aiperf.common.models import ParsedResponseRecord
from aiperf.common.protocols import AIPerfLifecycleProtocol

# Class-level attribute name that a results processor can set to ``True`` to
# opt out of hard-failure propagation in the records manager dispatch loop.
# Used by telemetry processors (OTel / MLflow live streaming) whose failures
# must not crash the benchmark. Centralised here so future authors can find
# the convention and type-check their use.
IS_BEST_EFFORT_ATTR: Final[str] = "is_best_effort"


class BestEffortMarker(Protocol):
    """Marker protocol for results processors that tolerate dispatch failures.

    A processor that sets ``is_best_effort: ClassVar[bool] = True`` signals to
    the records manager that ``process_result`` exceptions should be logged but
    not re-raised. The records manager reads this attribute via ``getattr``
    (structural typing is advisory — runtime isinstance checks would force
    every processor to inherit from a shared base).
    """

    is_best_effort: ClassVar[bool]


if TYPE_CHECKING:
    from aiperf.common.messages.inference_messages import MetricRecordsData
    from aiperf.common.models import CreditPhaseStats, MetricResult
    from aiperf.common.models.record_models import MetricRecordMetadata
    from aiperf.metrics.metric_dicts import MetricRecordDict


@runtime_checkable
class RecordProcessorProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for a record processor that processes the incoming records and returns the results of the post processing."""

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> MetricRecordDict: ...


@runtime_checkable
class ResultsProcessorProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for a results processor that processes the results of multiple
    record processors, and provides the ability to summarize the results."""

    async def process_result(
        self, record_data: MetricRecordsData | CreditPhaseStats
    ) -> None: ...

    async def summarize(self) -> list[MetricResult]: ...


@runtime_checkable
class FlushableResultsProcessorProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for metric results processors that support explicit flushing."""

    async def flush(self, *, force: bool = False) -> None: ...
