# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dashboard single-run figure generation."""

import base64
import io
import zipfile
from pathlib import Path
from unittest.mock import Mock

import dash
import pandas as pd
import plotly.graph_objects as go
import pytest
from pytest import param

from aiperf.plot import browser
from aiperf.plot.constants import PlotTheme
from aiperf.plot.core.data_loader import RunData, RunMetadata
from aiperf.plot.core.mode_detector import VisualizationMode
from aiperf.plot.core.plot_specs import DataSource, MetricSpec, PlotSpec, PlotType
from aiperf.plot.dashboard import callbacks
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


@pytest.mark.parametrize(
    "linux_arm64,expected,absent",
    [
        param(True, "BROWSER_PATH", "plotly_get_chrome", id="linux-arm64"),
        param(False, "plotly_get_chrome", "BROWSER_PATH", id="other-platform"),
    ],
)  # fmt: skip
def test_export_bundle_includes_fallback_readme(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    linux_arm64: bool,
    expected: str,
    absent: str,
) -> None:
    monkeypatch.setattr(browser, "IS_LINUX_ARM64", linux_arm64)
    monkeypatch.setattr(callbacks, "_drill_down_run_cache", {})
    fig = go.Figure()
    monkeypatch.setattr(
        go.Figure, "to_image", Mock(side_effect=RuntimeError("no browser"))
    )
    cache = Mock()
    cache.get.return_value = fig
    monkeypatch.setattr(callbacks, "get_plot_cache", lambda: cache)
    app = dash.Dash(__name__)
    config = Mock()
    config.get_single_run_plot_specs.return_value = []
    callbacks.register_export_png_callback(
        app,
        [_non_streaming_run(tmp_path)],
        VisualizationMode.SINGLE_RUN,
        PlotTheme.DARK,
        config,
    )
    export = app.callback_map["download-png-bundle.data"]["callback"].__wrapped__
    result = export(
        1,
        "png",
        "small",
        {
            "visible_plots": ["latency"],
            "plot_configs": {"latency": {"is_default": True}},
        },
        [],
        {},
    )
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(result["content"]))) as bundle:
        assert set(bundle.namelist()) == {"latency.html", "README.txt"}
        assert b"<html>" in bundle.read("latency.html")
        readme = bundle.read("README.txt").decode()
    assert expected in readme
    assert absent not in readme
