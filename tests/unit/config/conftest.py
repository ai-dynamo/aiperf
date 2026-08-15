# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the config-layer unit tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from aiperf.config.flags.cli_config import CLIConfig

GRAPH_TRACE_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "graph"
    / "adapters"
    / "fixtures"
    / "dynamo_nested"
    / "nested_2_level.jsonl.gz"
)


@pytest.fixture
def graph_trace_fixture() -> Path:
    """Path to a real nested dynamo-trace graph workload fixture."""
    return GRAPH_TRACE_FIXTURE


@pytest.fixture
def endpoint_cli_args() -> list[str]:
    """Minimal endpoint flags needed for any CLIConfig parse to succeed."""
    return ["--url", "http://localhost:8000/test", "--model", "test-model"]


@pytest.fixture
def parse_cli_args() -> Callable[[list[str]], CLIConfig]:
    """Callable that parses argv through cyclopts into a CLIConfig, without executing."""

    def _parse(argv: list[str]) -> CLIConfig:
        from cyclopts import App

        captured: dict[str, CLIConfig] = {}
        app = App(name="test_profile")

        @app.default
        def _runner(
            *, cli_config: CLIConfig
        ) -> None:  # pragma: no cover - capture only
            captured["uc"] = cli_config

        try:
            app(argv, exit_on_error=False)
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
        return captured["uc"]

    return _parse


@pytest.fixture
def write_plain_trace_file(tmp_path: Path) -> Callable[..., Path]:
    """Callable that materializes a non-graph JSONL trace file (input_file validates existence)."""

    def _write(
        line: str = '{"timestamp": 0, "input_length": 8, "output_length": 4}\n',
        name: str = "trace.jsonl",
    ) -> Path:
        path = tmp_path / name
        path.write_text(line)
        return path

    return _write
