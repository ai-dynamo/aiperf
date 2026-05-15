# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for plot CLI command."""

from aiperf.cli_commands.plot import plot


def test_plot_passes_paths_and_output_by_keyword(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_plot_controller(
        paths: list[str] | None = None,
        *,
        output: str | None = None,
        theme: str = "light",
        config: str | None = None,
        verbose: bool = False,
        dashboard: bool = False,
        host: str = "127.0.0.1",
        port: int = 8050,
        mlflow_upload: bool = False,
        mlflow_tracking_uri: str | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        calls.append(
            {
                "paths": paths,
                "output": output,
                "theme": theme,
                "config": config,
                "verbose": verbose,
                "dashboard": dashboard,
                "host": host,
                "port": port,
                "mlflow_upload": mlflow_upload,
                "mlflow_tracking_uri": mlflow_tracking_uri,
                "mlflow_run_id": mlflow_run_id,
            }
        )

    monkeypatch.setattr(
        "aiperf.plot.cli_runner.run_plot_controller", fake_run_plot_controller
    )

    plot(
        paths=["/tmp/run"],
        output="/tmp/out",
        theme="dark",
        config="/tmp/plot.yaml",
        verbose=True,
        dashboard=False,
        host="127.0.0.1",
        port=8051,
    )

    assert calls == [
        {
            "paths": ["/tmp/run"],
            "output": "/tmp/out",
            "theme": "dark",
            "config": "/tmp/plot.yaml",
            "verbose": True,
            "dashboard": False,
            "host": "127.0.0.1",
            "port": 8051,
            "mlflow_upload": False,
            "mlflow_tracking_uri": None,
            "mlflow_run_id": None,
        }
    ]
