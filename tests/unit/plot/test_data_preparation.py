# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for aiperf.plot.core.data_preparation.aggregate_gpu_telemetry."""

from pathlib import Path

import pandas as pd

from aiperf.plot.core.data_loader import RunData, RunMetadata
from aiperf.plot.core.data_preparation import (
    aggregate_gpu_telemetry,
    resolve_gpu_activity_column,
)


def _run(gpu_telemetry: pd.DataFrame) -> RunData:
    return RunData(
        metadata=RunMetadata(
            run_name="test", run_path=Path("."), duration_seconds=None
        ),
        requests=None,
        aggregated={},
        gpu_telemetry=gpu_telemetry,
    )


class TestAggregateGpuTelemetry:
    def test_legacy_output_col_with_both_columns_present_no_duplicate(self) -> None:
        """Renaming to a legacy output_col must not create a duplicate column.

        When the telemetry frame carries both the namespaced
        `nvidia_gpu_utilization` and a legacy `gpu_utilization` column (and has no
        `gpu_index` to collapse them), renaming the namespaced column onto the
        legacy `output_col` name would otherwise yield two `gpu_utilization`
        columns, so `df["gpu_utilization"]` returns a DataFrame instead of a
        Series and downstream `df[y2_metric]` lookups break.
        """
        gpu_telemetry = pd.DataFrame(
            {
                "timestamp_s": [1, 2],
                "nvidia_gpu_utilization": [80.0, 85.0],
                "gpu_utilization": [10.0, 15.0],  # stale legacy column
            }
        )

        result = aggregate_gpu_telemetry(
            _run(gpu_telemetry), output_col="gpu_utilization"
        )

        # Exactly one output column, accessible as a Series.
        assert list(result.columns).count("gpu_utilization") == 1
        assert isinstance(result["gpu_utilization"], pd.Series)
        # The namespaced source wins (legacy stale column is dropped).
        assert result["gpu_utilization"].tolist() == [80.0, 85.0]

    def test_namespaced_output_col_passthrough(self) -> None:
        """With only the namespaced column and matching output_col, it passes through."""
        gpu_telemetry = pd.DataFrame(
            {"timestamp_s": [1, 2], "nvidia_gpu_utilization": [80.0, 85.0]}
        )

        result = aggregate_gpu_telemetry(_run(gpu_telemetry))

        assert list(result.columns).count("nvidia_gpu_utilization") == 1
        assert result["nvidia_gpu_utilization"].tolist() == [80.0, 85.0]

    def test_amd_gfx_activity_fallback(self) -> None:
        """AMD-only telemetry falls back to amd_gfx_activity as the activity series."""
        gpu_telemetry = pd.DataFrame(
            {"timestamp_s": [1, 2], "amd_gfx_activity": [60.0, 70.0]}
        )

        result = aggregate_gpu_telemetry(
            _run(gpu_telemetry), output_col="amd_gfx_activity"
        )

        assert "amd_gfx_activity" in result.columns
        assert result["amd_gfx_activity"].tolist() == [60.0, 70.0]


class TestResolveGpuActivityColumn:
    def test_prefers_nvidia(self) -> None:
        df = pd.DataFrame(
            columns=["nvidia_gpu_utilization", "amd_gfx_activity", "gpu_utilization"]
        )
        assert resolve_gpu_activity_column(df) == "nvidia_gpu_utilization"

    def test_amd_when_no_nvidia(self) -> None:
        df = pd.DataFrame(columns=["amd_gfx_activity", "gpu_utilization"])
        assert resolve_gpu_activity_column(df) == "amd_gfx_activity"

    def test_legacy_when_only_legacy(self) -> None:
        df = pd.DataFrame(columns=["gpu_utilization"])
        assert resolve_gpu_activity_column(df) == "gpu_utilization"

    def test_none_when_no_activity_column(self) -> None:
        df = pd.DataFrame(columns=["nvidia_temperature"])
        assert resolve_gpu_activity_column(df) is None
