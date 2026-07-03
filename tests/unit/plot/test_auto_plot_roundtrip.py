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


def _make_single_run_envelope() -> PlotEnvelopeConfig:
    """Envelope with a populated ``single_run_plots`` preset + default."""
    return PlotEnvelopeConfig.model_validate(
        {
            "visualization": {
                "single_run_defaults": ["ttft_over_time"],
                "single_run_plots": {
                    "ttft_over_time": {
                        "type": "scatter",
                        "x": "request_number",
                        "y": "time_to_first_token",
                        "title": "TTFT Per Request Over Time",
                    }
                },
            },
        }
    )


def _make_experiment_classification_envelope() -> PlotEnvelopeConfig:
    """Envelope carrying a full ``experiment_classification`` block."""
    return PlotEnvelopeConfig.model_validate(
        {
            "visualization": {
                "multi_run_defaults": ["throughput_vs_latency"],
                "multi_run_plots": {
                    "throughput_vs_latency": {
                        "type": "scatter_line",
                        "x": "request_latency_p50",
                        "y": "request_throughput_avg",
                        "title": "Throughput vs Latency",
                    }
                },
            },
            "experiment_classification": {
                "baselines": ["*_agg_*", "*baseline*"],
                "treatments": ["*_disagg_*"],
                "default": "treatment",
                "group_extraction_pattern": r"^(baseline|treatment_\d+)",
                "group_display_names": {
                    "baseline": "Baseline",
                    "treatment_1": "Disaggregated",
                },
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

    def test_roundtrip_preserves_single_run_plots(self, tmp_path: Path):
        # A populated single_run_plots dict + single_run_defaults must survive
        # the snake_case / exclude_none writer and re-parse into a PlotSpec.
        envelope = _make_single_run_envelope()
        target = _materialize_envelope(tmp_path, envelope)

        config = PlotConfig(config_path=target)
        specs = config.get_single_run_plot_specs()
        assert len(specs) == 1
        assert specs[0].name == "ttft_over_time"
        assert specs[0].plot_type == PlotType.SCATTER

    def test_roundtrip_preserves_experiment_classification(self, tmp_path: Path):
        # experiment_classification is a snake_case top-level block the reader
        # consumes directly; the writer must not drop it or its nested fields.
        envelope = _make_experiment_classification_envelope()
        target = _materialize_envelope(tmp_path, envelope)

        config = PlotConfig(config_path=target)
        exp = config.get_experiment_classification_config()
        assert exp is not None
        assert exp.baselines == ["*_agg_*", "*baseline*"]
        assert exp.treatments == ["*_disagg_*"]
        assert exp.default == "treatment"
        assert exp.group_extraction_pattern == r"^(baseline|treatment_\d+)"
        assert exp.group_display_names == {
            "baseline": "Baseline",
            "treatment_1": "Disaggregated",
        }
