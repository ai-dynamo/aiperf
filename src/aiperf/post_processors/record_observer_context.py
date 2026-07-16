# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only context object handed to record observers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aiperf.common.messages.inference_messages import MetricRecordsData

if TYPE_CHECKING:
    from aiperf.common.models import MetricRecordMetadata, ParsedResponseRecord


@dataclass(slots=True)
class RecordObserverContext:
    """What an observer sees: the original record + all producer outputs for it."""

    record: ParsedResponseRecord
    """The original parsed record."""

    metadata: MetricRecordMetadata
    """The metric record metadata for this record."""

    produced: dict[str, list[Any]]
    """Producer outputs keyed by declared record_type.

    e.g. ``{"metric_records": [MetricRecordDict], "accuracy": [AccuracyRecordsData]}``.
    """

    def get(self, record_type: str) -> list[Any]:
        """Return the producer outputs emitted on ``record_type`` (empty if none)."""
        return self.produced.get(record_type, [])

    @property
    def metrics(self) -> MetricRecordsData | None:
        """The first metric-records output, or None when no producer emitted one."""
        items = self.produced.get(MetricRecordsData.record_type) or []
        return items[0] if items else None
