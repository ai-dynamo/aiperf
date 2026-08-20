# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from aiperf.operator.routers.sweeps_models import (
    CellAggregatesResponse,
    CellEntry,
    DimensionInfo,
    SpecSummary,
    SweepDetailResponse,
    SweepListResponse,
    SweepSummary,
)
from pydantic import ValidationError


def test_sweep_summary_required_fields() -> None:
    s = SweepSummary(
        namespace="bench",
        name="saturation-sweep",
        source="live",
        phase="Running",
        total_variations=12,
        completed_runs=8,
        failed_runs=0,
        age_seconds=120,
        model="meta-llama/Llama-3-8B",
    )
    assert s.namespace == "bench"
    assert s.source == "live"


def test_sweep_summary_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        SweepSummary(
            namespace="bench",
            name="s",
            source="ghost",  # invalid
            phase="Running",
            total_variations=0,
            completed_runs=0,
            failed_runs=0,
            age_seconds=0,
            model=None,
        )


def test_dimension_info_values_preserved() -> None:
    d = DimensionInfo(name="concurrency", values=[8, 32, 128])
    assert d.values == [8, 32, 128]


def test_cell_entry_metrics_open_dict() -> None:
    cell = CellEntry(
        variation_index=7,
        variation_label="concurrency-128-rate-50",
        values={"concurrency": 128, "rate": 50},
        trials_completed=3,
        trials_failed=0,
        metrics={"request_throughput": {"avg": 1234.5, "p99": 1500.0}},
        children=[
            {
                "namespace": "bench",
                "name": "ch-7-0",
                "trial_index": 0,
                "phase": "Succeeded",
            }
        ],
    )
    assert cell.metrics["request_throughput"]["avg"] == 1234.5


def test_sweep_list_response_default_empty() -> None:
    assert SweepListResponse(sweeps=[]).sweeps == []


def test_sweep_detail_response_required() -> None:
    sd = SweepDetailResponse(
        sweep=SweepSummary(
            namespace="bench",
            name="s",
            source="archived",
            phase="Succeeded",
            total_variations=4,
            completed_runs=4,
            failed_runs=0,
            age_seconds=999,
            model="m",
        ),
        status={"phase": "Succeeded"},
        spec_summary=SpecSummary(
            sweep_type="grid",
            dimensions=[DimensionInfo(name="concurrency", values=[1, 2])],
            multi_run=None,
            convergence=None,
        ),
        children=[],
    )
    assert sd.spec_summary.sweep_type == "grid"


def test_cell_aggregates_response_source_literal() -> None:
    with pytest.raises(ValidationError):
        CellAggregatesResponse(dimensions=[], cells=[], source="oops")
