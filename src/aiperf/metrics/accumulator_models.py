# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public data models for the metrics accumulator (summary + CSV row helper)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from aiperf.common.models import MetricResult, TimesliceWindow
from aiperf.common.types import MetricTagT, TimeSliceT


@dataclass
class AccumulatorMetricsSummary:
    """Typed result from MetricsAccumulator.summarize().

    Unified summary replacing both the old MetricsSummary (results only) and
    TimesliceSummary (timeslices only). When timeslicing is configured, both
    fields are populated from a single accumulator. ``multi_turn_ttft_trend``
    populates only when records have ``turn_index`` metadata and a non-empty
    set of distinct turn indices appeared.
    """

    results: dict[MetricTagT, MetricResult]
    timeslices: dict[TimeSliceT, dict[MetricTagT, MetricResult]] | None = field(
        default=None
    )
    timeslice_windows: dict[TimeSliceT, TimesliceWindow] | None = field(default=None)
    multi_turn_ttft_trend: dict[int, MetricResult] | None = field(default=None)

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "results": [asdict(r.to_json_result()) for r in self.results.values()],
        }
        if self.timeslices is not None:
            data["timeslices"] = {
                str(k): [asdict(r.to_json_result()) for r in v.values()]
                for k, v in self.timeslices.items()
            }
        if self.multi_turn_ttft_trend is not None:
            data["multi_turn_ttft_trend"] = {
                str(turn): asdict(r.to_json_result())
                for turn, r in self.multi_turn_ttft_trend.items()
            }
        return data

    def to_csv(self) -> list[dict[str, Any]]:
        rows = [_metric_result_to_csv_row(r) for r in self.results.values()]
        if self.timeslices is not None:
            for ts, results in self.timeslices.items():
                for r in results.values():
                    row = _metric_result_to_csv_row(r)
                    row["timeslice"] = ts
                    rows.append(row)
        if self.multi_turn_ttft_trend is not None:
            for turn, r in self.multi_turn_ttft_trend.items():
                row = _metric_result_to_csv_row(r)
                row["turn_index"] = turn
                rows.append(row)
        return rows


def _metric_result_to_csv_row(result: MetricResult) -> dict[str, Any]:
    """Serialize a MetricResult dataclass to a CSV-row dict, excluding ``current``."""
    row = asdict(result)
    row.pop("current", None)
    return row
