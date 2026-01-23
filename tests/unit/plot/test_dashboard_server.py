# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Dashboard Server."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aiperf.plot.config import PlotConfig
from aiperf.plot.constants import PlotTheme
from aiperf.plot.core.data_loader import DataLoader, RunData
from aiperf.plot.core.mode_detector import VisualizationMode
from aiperf.plot.dashboard.server import DashboardServer


class TestDashboardServer:
    """Tests for DashboardServer class."""

    @pytest.fixture
    def mock_run_data(self):
        """Mock RunData Object"""
        mock_run = Mock(spec=RunData)
        return mock_run

    @pytest.fixture
    def mock_plot_config(self):
        """Mock PlotConfig Object"""
        mock_config = Mock(spec=PlotConfig)
        return mock_config

    @pytest.fixture
    def mock_data_loader(self):
        """Mock DataLoader Object"""
        return Mock(spec=DataLoader)

    @pytest.fixture
    def dashboard_server(self, mock_run_data, mock_plot_config, mock_data_loader):
        """Create DashboardServer Instance"""

        mock_run_dir = Mock(spec=Path)
        mock_run_dir.__str__ = Mock(return_value="/path/to/run")

        return DashboardServer(
            runs=[mock_run_data],
            run_dirs=[mock_run_dir],
            mode=VisualizationMode.SINGLE_RUN,
            theme=PlotTheme.LIGHT,
            plot_config=mock_plot_config,
            loader=mock_data_loader,
            host="127.0.0.1",
            port=8050,
        )

    def test_run_opens_browser(self, dashboard_server, capsys):
        """Test whether the run() method attempts to open a browser"""

        with (
            patch.object(dashboard_server, "build_layout"),
            patch.object(dashboard_server, "register_callbacks"),
            patch.object(dashboard_server.app, "run"),
            patch("webbrowser.open"),
            patch("builtins.open", Mock()),
            patch("os.dup", return_value=999),
            patch("os.dup2"),
            patch("os.close"),
        ):
            dashboard_server.run()

            captured = capsys.readouterr()
            output = captured.out
            assert (
                "Please open http://127.0.0.1:8050 manually in your browser" in output
            )
