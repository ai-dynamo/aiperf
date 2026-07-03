# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write->read round-trip tests for the auto-plot materialized envelope.

The auto-plot callback writes ``<artifact_dir>/.aiperf-plot-config.yaml`` via
``aiperf.plot.auto_plot._materialize_envelope``; ``aiperf plot <dir>`` (and
``aiperf plot --config <file>``) reads it back through ``PlotConfig``. These
tests exercise the REAL writer and REAL reader against a real file on disk so
a key-casing drift between the two (the writer once emitted camelCase aliases
the reader could not find) fails loudly instead of silently producing zero
plots.
"""

from pathlib import Path

from aiperf.config.plot import PlotEnvelopeConfig
from aiperf.plot.auto_plot import _materialize_envelope
from aiperf.plot.config import PlotConfig
from aiperf.plot.constants import MATERIALIZED_PLOT_CONFIG_NAME
from aiperf.plot.core.plot_specs import PlotType


def _make_envelope() -> PlotEnvelopeConfig:
    """Build a realistic envelope with one multi-run preset and custom settings."""
    return PlotEnvelopeConfig.model_validate(
        {
            "visualization": {
                "multi_run_defaults": ["throughput_vs_latency"],
                "single_run_defaults": [],
                "multi_run_plots": {
                    "throughput_vs_latency": {
                        "type": "scatter_line",
                        "x": "request_latency_p50",
                        "y": "request_throughput_avg",
                        "title": "Throughput vs Latency",
                    }
                },
                "single_run_plots": {},
            },
            "settings": {
                "server_metrics_downsampling": {
                    "enabled": False,
                    "window_size_seconds": 10.0,
                    "aggregation_method": "max",
                }
            },
        }
    )


class TestMaterializedEnvelopeRoundTrip:
    """The written envelope must be re-readable by PlotConfig, unchanged."""

    def test_materialized_file_uses_shared_constant_name(self, tmp_path: Path):
        envelope = _make_envelope()
        target = _materialize_envelope(tmp_path, envelope)
        assert target == tmp_path / MATERIALIZED_PLOT_CONFIG_NAME
        assert target.exists()

    def test_roundtrip_via_artifact_dir_detection(self, tmp_path: Path, monkeypatch):
        # Isolate from the real user home so Priority 2 can't mask a 1.5 miss.
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        envelope = _make_envelope()
        _materialize_envelope(tmp_path, envelope)

        config = PlotConfig(artifact_dirs=[tmp_path])

        assert config.resolved_path == tmp_path / MATERIALIZED_PLOT_CONFIG_NAME
        specs = config.get_multi_run_plot_specs()
        assert len(specs) == 1
        assert specs[0].name == "throughput_vs_latency"
        assert specs[0].plot_type == PlotType.SCATTER_LINE

    def test_roundtrip_preserves_settings(self, tmp_path: Path):
        # Settings live under snake_case keys the reader consumes directly.
        # A camelCase writer would drop these and silently revert to defaults.
        envelope = _make_envelope()
        target = _materialize_envelope(tmp_path, envelope)

        config = PlotConfig(config_path=target)
        downsampling = config.get_downsampling_config()
        assert downsampling["enabled"] is False
        assert downsampling["window_size_seconds"] == 10.0
        assert downsampling["aggregation_method"] == "max"

    def test_roundtrip_readable_via_explicit_config_flag(self, tmp_path: Path):
        # `aiperf plot --config <materialized>` must read the same file the
        # same way (this is how the file doubles as a reproducibility receipt).
        envelope = _make_envelope()
        target = _materialize_envelope(tmp_path, envelope)

        config = PlotConfig(config_path=target)
        specs = config.get_multi_run_plot_specs()
        assert len(specs) == 1
        assert specs[0].name == "throughput_vs_latency"
