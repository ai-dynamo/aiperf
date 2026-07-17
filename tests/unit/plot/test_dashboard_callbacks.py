# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dashboard single-run figure generation."""

import pandas as pd

from aiperf.plot.constants import PlotTheme
from aiperf.plot.core.data_loader import RunData, RunMetadata
from aiperf.plot.core.plot_specs import DataSource, MetricSpec, PlotSpec, PlotType
from aiperf.plot.dashboard.callbacks import _generate_singlerun_figure


def _non_streaming_run(tmp_path) -> RunData:
    """Run with a non-empty requests table that lacks streaming-only columns
    (no time_to_first_token / inter_token_latency), as in non-streaming mode."""
    return RunData(
        metadata=RunMetadata(
            run_name="r", run_path=tmp_path / "r", model="m", concurrency=1
        ),
        requests=pd.DataFrame(
            {
                "request_end_ns": pd.to_datetime(
                    [1_000_000_000_000 + i * 500_000_000 for i in range(5)],
                    unit="ns",
                    utc=True,
                ),
                "request_latency": [900.0 + i * 10 for i in range(5)],
            }
        ),
        aggregated={},
        timeslices=None,
    )


def _spec(name: str, y_metric: str) -> PlotSpec:
    return PlotSpec(
        name=name,
        plot_type=PlotType.SCATTER,
        metrics=[
            MetricSpec(name="request_number", source=DataSource.REQUESTS, axis="x"),
            MetricSpec(name=y_metric, source=DataSource.REQUESTS, axis="y"),
        ],
        title=name,
        filename=f"{name}.png",
    )


class TestGenerateSingleRunFigure:
    def test_missing_column_returns_none_at_debug(self, tmp_path, caplog):
        """A streaming-only plot on non-streaming data is skipped (None) and logged
        at DEBUG - not rendered as an error tile and not logged at ERROR."""
        run = _non_streaming_run(tmp_path)
        spec = _spec("ttft_over_time", "time_to_first_token")

        with caplog.at_level("DEBUG"):
            fig = _generate_singlerun_figure(
                "ttft_over_time", {"is_default": True}, run, [spec], PlotTheme.DARK
            )

        assert fig is None
        assert not [r for r in caplog.records if r.levelname == "ERROR"]
        assert [
            r
            for r in caplog.records
            if r.levelname == "DEBUG"
            and "Skipping" in r.message
            and "ttft_over_time" in r.message
        ]

    def test_present_column_returns_figure(self, tmp_path):
        """A plot whose column is present still renders normally."""
        run = _non_streaming_run(tmp_path)
        spec = _spec("latency_over_time", "request_latency")

        fig = _generate_singlerun_figure(
            "latency_over_time", {"is_default": True}, run, [spec], PlotTheme.DARK
        )

        assert fig is not None

    def test_unexpected_error_still_returns_error_figure(self, tmp_path, caplog):
        """Genuine, unexpected errors (not missing-data) keep ERROR logging and an
        error placeholder figure - they are not silently swallowed as a skip."""
        run = _non_streaming_run(tmp_path)
        # is_default=False with a config that the custom-plot path cannot build
        # raises a non-(DataUnavailableError/KeyError) error.
        with caplog.at_level("DEBUG"):
            fig = _generate_singlerun_figure(
                "broken", {"is_default": False}, run, [], PlotTheme.DARK
            )

        assert fig is not None  # error placeholder figure, not None
        assert [r for r in caplog.records if r.levelname == "ERROR"]
