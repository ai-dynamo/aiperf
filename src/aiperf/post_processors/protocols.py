# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.models import ParsedResponseRecord
from aiperf.common.protocols import AIPerfLifecycleProtocol

if TYPE_CHECKING:
    from aiperf.common.metric_records_wire import (
        MetricRecordMetadata,
        MetricRecordsData,
    )
    from aiperf.common.models import MetricResult
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
