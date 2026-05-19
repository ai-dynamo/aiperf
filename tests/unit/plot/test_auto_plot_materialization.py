# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for build_auto_plot_callback's envelope materialization."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiperf.config.plot import PlotEnvelopeConfig
from aiperf.plot.auto_plot import build_auto_plot_callback


def _make_envelope() -> PlotEnvelopeConfig:
    return PlotEnvelopeConfig.model_validate(
        {
            "visualization": {
                "multi_run_defaults": ["pareto_a"],
                "multi_run_plots": {
                    "pareto_a": {
                        "type": "pareto",
                        "x": {"metric": "request_latency", "stat": "avg"},
                        "y": {
                            "metric": "output_token_throughput_per_gpu",
                            "stat": "avg",
                        },
                    },
                },
            },
        }
    )


def _fake_completed_run(artifact_dir: Path):
    run = MagicMock()
    run.artifact_dir = artifact_dir
    return run


def test_callback_with_envelope_writes_materialized_yaml(tmp_path: Path):
    """When plot_envelope is set, the callback writes
    <artifact_dir>/.aiperf-plot-config.yaml and passes its path to the plotter."""
    envelope = _make_envelope()
    cb = build_auto_plot_callback(plot_required=False, plot_envelope=envelope)
    with patch("aiperf.plot.auto_plot.run_plot_controller") as runner:
        cb(_fake_completed_run(tmp_path))

    materialized = tmp_path / ".aiperf-plot-config.yaml"
    assert materialized.exists()

    runner.assert_called_once()
    call_kwargs = runner.call_args.kwargs
    assert call_kwargs.get("config") == str(materialized)
    assert call_kwargs.get("paths") == [str(tmp_path)]


def test_callback_without_envelope_does_not_write(tmp_path: Path):
    """Without an envelope, no materialized YAML and config arg is None."""
    cb = build_auto_plot_callback(plot_required=False, plot_envelope=None)
    with patch("aiperf.plot.auto_plot.run_plot_controller") as runner:
        cb(_fake_completed_run(tmp_path))

    assert not (tmp_path / ".aiperf-plot-config.yaml").exists()
    runner.assert_called_once()
    assert runner.call_args.kwargs.get("config") is None


def test_materialized_yaml_is_round_trip_loadable(tmp_path: Path):
    """The materialized YAML can be loaded back into a PlotEnvelopeConfig
    that equals the original."""
    envelope = _make_envelope()
    cb = build_auto_plot_callback(plot_required=False, plot_envelope=envelope)
    with patch("aiperf.plot.auto_plot.run_plot_controller"):
        cb(_fake_completed_run(tmp_path))

    from aiperf.config.plot import load_plot_envelope_from_path

    reloaded = load_plot_envelope_from_path(
        tmp_path / ".aiperf-plot-config.yaml",
        source_dir=tmp_path,
    )
    assert reloaded.model_dump() == envelope.model_dump()


def test_callback_plot_required_true_reraises(tmp_path: Path):
    """When run_plot_controller raises and plot_required=True, the exception
    propagates."""
    envelope = _make_envelope()
    cb = build_auto_plot_callback(plot_required=True, plot_envelope=envelope)
    with (
        patch(
            "aiperf.plot.auto_plot.run_plot_controller",
            side_effect=RuntimeError("boom"),
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        cb(_fake_completed_run(tmp_path))


def test_callback_plot_required_false_swallows(tmp_path: Path, caplog):
    """When run_plot_controller raises and plot_required=False, the error is
    logged and the callback returns normally."""
    envelope = _make_envelope()
    cb = build_auto_plot_callback(plot_required=False, plot_envelope=envelope)
    with patch(
        "aiperf.plot.auto_plot.run_plot_controller",
        side_effect=RuntimeError("boom"),
    ):
        cb(_fake_completed_run(tmp_path))
    assert any("auto-plot failed" in rec.message for rec in caplog.records)


def test_materialize_helper_round_trips_to_disk(tmp_path: Path):
    """The internal _materialize_plot_envelope helper produces a file that
    load_plot_envelope_from_path reads back to an equal envelope. Used by both
    the per-run callback and the sweep-aggregate path (Task 6)."""
    from aiperf.config.plot import load_plot_envelope_from_path
    from aiperf.plot.auto_plot import _materialize_plot_envelope

    envelope = _make_envelope()
    dest = tmp_path / ".aiperf-plot-config.yaml"
    _materialize_plot_envelope(envelope, dest)

    reloaded = load_plot_envelope_from_path(dest, source_dir=tmp_path)
    assert reloaded.model_dump() == envelope.model_dump()


def test_plot_config_auto_detects_materialized_envelope(tmp_path: Path):
    """When PlotConfig is constructed with artifact_dirs and one of those
    dirs contains a .aiperf-plot-config.yaml, it is auto-detected at
    Priority 1.5 ahead of ~/.aiperf/plot_config.yaml."""
    from aiperf.plot.auto_plot import _materialize_plot_envelope
    from aiperf.plot.config import PlotConfig

    envelope = _make_envelope()
    art_dir = tmp_path / "run"
    art_dir.mkdir()
    _materialize_plot_envelope(envelope, art_dir / ".aiperf-plot-config.yaml")

    pc = PlotConfig(config_path=None, artifact_dirs=[art_dir])
    assert pc.resolved_path == art_dir / ".aiperf-plot-config.yaml"


def test_plot_config_explicit_custom_path_wins_over_artifact_dir(tmp_path: Path):
    """An explicit --config (custom_path) at Priority 1 still wins over the
    artifact-dir auto-detect at Priority 1.5."""
    from aiperf.plot.auto_plot import _materialize_plot_envelope
    from aiperf.plot.config import PlotConfig

    envelope = _make_envelope()
    art_dir = tmp_path / "run"
    art_dir.mkdir()
    _materialize_plot_envelope(envelope, art_dir / ".aiperf-plot-config.yaml")

    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        "visualization:\n  multi_run_defaults: []\n  multi_run_plots: {}\n",
        encoding="utf-8",
    )

    pc = PlotConfig(config_path=explicit, artifact_dirs=[art_dir])
    assert pc.resolved_path == explicit


def test_plot_config_first_artifact_dir_with_envelope_wins(tmp_path: Path):
    """When multiple artifact_dirs are passed, the first one containing a
    materialized envelope is used; later dirs are not scanned."""
    from aiperf.plot.auto_plot import _materialize_plot_envelope
    from aiperf.plot.config import PlotConfig

    envelope = _make_envelope()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _materialize_plot_envelope(envelope, first / ".aiperf-plot-config.yaml")
    _materialize_plot_envelope(envelope, second / ".aiperf-plot-config.yaml")

    pc = PlotConfig(config_path=None, artifact_dirs=[first, second])
    assert pc.resolved_path == first / ".aiperf-plot-config.yaml"
