# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final, Protocol, runtime_checkable

from aiperf.common.models import ParsedResponseRecord
from aiperf.common.protocols import AIPerfLifecycleProtocol

if TYPE_CHECKING:
    from aiperf.common.metric_records_wire import (
        MetricRecordMetadata,
        MetricRecordsData,
    )
    from aiperf.common.models import MetricResult
    from aiperf.metrics.metric_dicts import MetricRecordDict

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

    async def process_result(self, record_data: MetricRecordsData) -> None: ...

    async def summarize(self) -> list[MetricResult]: ...

    async def finalize(self) -> None:
        """Finalize at end-of-run, after the last summarize() call.

        Called once by the records-manager BEFORE publishing
        ``ProcessRecordsResultMessage`` so any per-record streaming files
        (profile_export.jsonl, profile_export_records.csv) are fully flushed
        before the controller writes the readiness marker. Without this,
        the operator's progress poll observes ``results_exported=True``
        and fetches a partial per-record file in the gap between marker
        write and the processor's @on_stop close.

        Default implementation is a no-op; processors that buffer to disk
        override to flush + close.
        """
        ...
