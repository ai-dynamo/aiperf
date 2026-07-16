# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.models import ParsedResponseRecord
from aiperf.common.protocols import AIPerfLifecycleProtocol

if TYPE_CHECKING:
    from aiperf.common.models.record_models import MetricRecordMetadata, RecordData
    from aiperf.post_processors.record_observer_context import RecordObserverContext


@runtime_checkable
class RecordProcessorProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for record PRODUCERS: parse a record and emit one typed result
    on a declared record-type channel."""

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> RecordData | None: ...


@runtime_checkable
class RecordObserverProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for record OBSERVERS: view the produced results + the record and
    act (e.g. write JSONL). Observers return nothing and emit no channel record."""

    async def observe(self, ctx: RecordObserverContext) -> None: ...
