# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Field-acceptance and CLI-flag plumbing tests for ``--auto-plot`` /
``--no-auto-plot`` / ``--plot-required``.

Covers C2 (v1 ``CLIConfig`` + v2 ``ArtifactsConfig`` field shape) and
C3 (cyclopts plumbing on the v1 ``CLIConfig``) of the auto-plot design.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.config.artifacts import ArtifactsConfig
from aiperf.config.flags.cli_config import CLIConfig


@pytest.mark.parametrize(
    "value",
    [
        param(None, id="none-default"),
        param(True, id="explicit-true"),
        param(False, id="explicit-false"),
    ],
)  # fmt: skip
def test_v1_output_accepts_auto_plot_tristate(value: bool | None) -> None:
    """v1 ``CLIConfig.auto_plot`` accepts None / True / False unchanged."""
    cfg = CLIConfig(auto_plot=value)
    assert cfg.auto_plot is value


def test_v1_output_auto_plot_default_is_none() -> None:
    """Unset ``auto_plot`` defaults to None so the converter knows to defer
    to the recipe."""
    cfg = CLIConfig()
    assert cfg.auto_plot is None
    # Crucially, ``auto_plot`` must NOT be in model_fields_set when unset --
    # build_artifacts and the resolver overlay both gate on this.
    assert "auto_plot" not in cfg.model_fields_set


@pytest.mark.parametrize(
    "value",
    [param(True, id="true"), param(False, id="false")],
)  # fmt: skip
def test_v1_output_accepts_plot_required(value: bool) -> None:
    cfg = CLIConfig(plot_required=value)
    assert cfg.plot_required is value


def test_v1_output_plot_required_default_is_false() -> None:
    cfg = CLIConfig()
    assert cfg.plot_required is False


def test_v2_artifacts_auto_plot_default_is_false() -> None:
    """v2 ``ArtifactsConfig.auto_plot`` is a plain bool defaulting to False."""
    cfg = ArtifactsConfig()
    assert cfg.auto_plot is False
    assert cfg.plot_required is False


@pytest.mark.parametrize(
    ("auto_plot", "plot_required"),
    [
        param(True, False, id="auto-on-warn"),
        param(True, True, id="auto-on-strict"),
        param(False, True, id="auto-off-strict-dormant"),
        param(False, False, id="all-off"),
    ],
)  # fmt: skip
def test_v2_artifacts_accepts_combinations(
    auto_plot: bool, plot_required: bool
) -> None:
    cfg = ArtifactsConfig(auto_plot=auto_plot, plot_required=plot_required)
    assert cfg.auto_plot is auto_plot
    assert cfg.plot_required is plot_required


def test_auto_plot_quick_start_uses_url_for_server_address() -> None:
    doc = Path("docs/tutorials/auto-plot.md").read_text()
    quick_start = doc.split("## Quick start", 1)[1].split("```", 2)[1]

    assert "--url http://vllm.internal:8000" in quick_start
    assert "--endpoint http://vllm.internal:8000" not in quick_start


# --- Cyclopts CLI flag plumbing ----------------------------------------------
# CLIConfig is the cyclopts-populated DTO; the profile command has the
# canonical App. We invoke the same code path here without actually running
# the profile body by registering CLIConfig as a parameter on a tiny App
# stub. This mirrors what ``aiperf profile`` does internally.


def _parse_cli_args(argv: list[str]) -> CLIConfig:
    """Parse ``argv`` through cyclopts into a ``CLIConfig`` (no execution)."""
    from cyclopts import App

    captured: dict[str, CLIConfig] = {}
    app = App(name="test_profile")

    @app.default
    def _runner(*, cli_config: CLIConfig) -> None:  # pragma: no cover - capture only
        captured["uc"] = cli_config

    # Cyclopts wraps the default in ``print_non_int_sys_exit``, which raises
    # ``SystemExit(0)`` after our capture runs. Swallow that so the captured
    # CLIConfig is observable.
    try:
        app(argv, exit_on_error=False)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    return captured["uc"]


def _required_endpoint_args() -> list[str]:
    """Minimal endpoint flags needed for any CLIConfig parse to succeed."""
    return [
        "--url",
        "http://localhost:8000/test",
        "--model",
        "test-model",
    ]


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        param("--auto-plot", True, id="positive"),
        param("--no-auto-plot", False, id="negative"),
    ],
)  # fmt: skip
def test_cyclopts_parses_auto_plot_flag(flag: str, expected: bool) -> None:
    uc = _parse_cli_args([*_required_endpoint_args(), flag])
    assert uc.auto_plot is expected
    assert "auto_plot" in uc.model_fields_set


def test_cyclopts_unset_auto_plot_stays_none() -> None:
    """Without an explicit flag, ``auto_plot`` is None and not in
    ``model_fields_set`` -- the converter relies on this to defer to the
    recipe default."""
    uc = _parse_cli_args(_required_endpoint_args())
    assert uc.auto_plot is None
    assert "auto_plot" not in uc.model_fields_set


def test_cyclopts_parses_plot_required_flag() -> None:
    uc = _parse_cli_args([*_required_endpoint_args(), "--plot-required"])
    assert uc.plot_required is True
    assert "plot_required" in uc.model_fields_set
