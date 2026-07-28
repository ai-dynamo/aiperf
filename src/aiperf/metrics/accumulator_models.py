# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public data models for the metrics accumulator (summary + CSV row helper)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aiperf.common.models import MetricResult, TimesliceResult
from aiperf.common.types import MetricTagT


@dataclass
class AccumulatorMetricsSummary:
    """Typed result from MetricsAccumulator.summarize().

    Unified summary replacing both the old MetricsSummary (results only) and
    TimesliceSummary (timeslices only). When timeslicing is configured,
    ``timeslices`` is populated as an ordered list of :class:`TimesliceResult`
    — each entry bundles window bounds (start_ns / end_ns / is_complete)
    with the slice's metric results. Position in the list is the slice's
    chronological index.
    """

    results: dict[MetricTagT, MetricResult]
    timeslices: list[TimesliceResult] | None = field(default=None)
    pooled_spec_decode_acceptance_histogram: dict[int, int] | None = field(default=None)
    """Run-level pooled accepted-draft histogram ({j: total_steps}) for the
    exported phase/window, or None when no request carried spec-decode stats.
    Dict aggregation lives outside the numpy columnar store, so it rides here
    rather than in ``results``."""

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "results": [_metric_result_to_json(r) for r in self.results.values()],
        }
        if self.timeslices is not None:
            data["timeslices"] = [
                [_metric_result_to_json(r) for r in ts.metric_results.values()]
                for ts in self.timeslices
            ]
        if self.pooled_spec_decode_acceptance_histogram is not None:
            data["pooled_spec_decode_acceptance_histogram"] = (
                self.pooled_spec_decode_acceptance_histogram
            )
        return data

    def to_csv(self) -> list[dict[str, Any]]:
        rows = [_metric_result_to_csv_row(r) for r in self.results.values()]
        if self.timeslices is not None:
            for ts_idx, ts in enumerate(self.timeslices):
                for r in ts.metric_results.values():
                    row = _metric_result_to_csv_row(r)
                    row["timeslice"] = ts_idx
                    rows.append(row)
        return rows


def _metric_result_to_json(result: MetricResult) -> dict[str, Any]:
    """Serialize the MetricResult's JSON-export shape (no ``sum`` field).

    ``MetricResult`` is a Pydantic model; ``to_json_result()`` returns a
    ``JsonMetricResult`` Pydantic model. ``model_dump(mode="json")`` keeps
    None-valued fields so the export schema stays consistent.
    """
    return result.to_json_result().model_dump(mode="json")


def _metric_result_to_csv_row(result: MetricResult) -> dict[str, Any]:
    """Serialize a MetricResult to a CSV-row dict, excluding ``current``."""
    row = result.model_dump(mode="json")
    row.pop("current", None)
    return row
