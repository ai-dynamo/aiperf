# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.config.plot envelope models."""

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config import AIPerfConfig  # noqa: F401  (used by Task 3 tests below)
from aiperf.config.loader.core import load_config_from_string
from aiperf.config.loader.errors import ConfigurationError
from aiperf.config.plot import (
    PlotEnvelopeConfig,
    PlotVisualization,
    ServerMetricsDownsampling,
    load_plot_envelope_from_path,
)


def test_plot_envelope_minimal_inline_form_loads():
    """A bare visualization block with one preset and one default loads."""
    env = PlotEnvelopeConfig.model_validate(
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
    assert env.visualization.multi_run_defaults == ["pareto_a"]
    assert "pareto_a" in env.visualization.multi_run_plots
    assert env.experiment_classification is None
    assert env.settings.server_metrics_downsampling.enabled is True


def test_plot_envelope_default_name_not_in_presets_rejected():
    """A name in multi_run_defaults that isn't in multi_run_plots is rejected."""
    with pytest.raises(ValidationError) as exc:
        PlotEnvelopeConfig.model_validate(
            {
                "visualization": {
                    "multi_run_defaults": ["missing_one"],
                    "multi_run_plots": {
                        "pareto_a": {"type": "pareto"},
                    },
                },
            }
        )
    msg = str(exc.value)
    assert "missing_one" in msg
    assert "multi_run_defaults" in msg
    assert "multi_run_plots" in msg


def test_plot_envelope_default_name_close_match_hint():
    """Typo'd default name produces a difflib 'did you mean' hint."""
    with pytest.raises(ValidationError) as exc:
        PlotEnvelopeConfig.model_validate(
            {
                "visualization": {
                    "multi_run_defaults": ["pareto_aa"],
                    "multi_run_plots": {
                        "pareto_a": {"type": "pareto"},
                    },
                },
            }
        )
    assert "pareto_a" in str(exc.value)


def test_plot_envelope_unknown_top_level_key_rejected():
    """extra='forbid' rejects unknown keys with a clear message."""
    with pytest.raises(ValidationError) as exc:
        PlotEnvelopeConfig.model_validate(
            {
                "visualization": {"multi_run_defaults": [], "multi_run_plots": {}},
                "boguskey": 1,
            }
        )
    assert "boguskey" in str(exc.value)


def test_plot_envelope_empty_dict_rejected():
    """plot: {} is rejected (visualization required)."""
    with pytest.raises(ValidationError):
        PlotEnvelopeConfig.model_validate({})


@pytest.mark.parametrize(
    "agg",
    [
        param("mean", id="mean"),
        param("max", id="max"),
        param("min", id="min"),
        param("median", id="median"),
    ],
)  # fmt: skip
def test_server_metrics_downsampling_aggregation_methods(agg: str):
    """All four documented aggregation methods are accepted."""
    s = ServerMetricsDownsampling.model_validate({"aggregation_method": agg})
    assert s.aggregation_method == agg


def test_server_metrics_downsampling_invalid_method_rejected():
    """Other strings are rejected."""
    with pytest.raises(ValidationError):
        ServerMetricsDownsampling.model_validate({"aggregation_method": "p99"})


def test_server_metrics_downsampling_window_size_must_be_positive():
    """window_size_seconds must be > 0."""
    with pytest.raises(ValidationError):
        ServerMetricsDownsampling.model_validate({"window_size_seconds": 0})


def test_plot_visualization_empty_defaults_lists_allowed():
    """Empty *_defaults lists are valid (user disabled all defaults)."""
    v = PlotVisualization.model_validate(
        {
            "multi_run_defaults": [],
            "single_run_defaults": [],
            "multi_run_plots": {},
            "single_run_plots": {},
        }
    )
    assert v.multi_run_defaults == []
    assert v.single_run_defaults == []


def test_load_plot_envelope_absolute_path(tmp_path: Path):
    """Loader reads an absolute YAML path and returns a validated envelope."""
    plot_file = tmp_path / "abs.yaml"
    plot_file.write_text(
        "visualization:\n"
        "  multi_run_defaults: [pareto_a]\n"
        "  multi_run_plots:\n"
        "    pareto_a: {type: pareto}\n"
    )
    env = load_plot_envelope_from_path(plot_file, source_dir=None)
    assert env.visualization.multi_run_defaults == ["pareto_a"]


def test_load_plot_envelope_relative_path_uses_source_dir(tmp_path: Path):
    """Loader resolves a relative path against source_dir."""
    plot_file = tmp_path / "rel.yaml"
    plot_file.write_text(
        "visualization:\n  multi_run_defaults: []\n  multi_run_plots: {}\n"
    )
    env = load_plot_envelope_from_path("./rel.yaml", source_dir=tmp_path)
    assert env.visualization.multi_run_defaults == []


def test_load_plot_envelope_relative_path_no_source_dir_rejected():
    """Loader rejects relative paths when source_dir is None."""
    from aiperf.config.loader.errors import ConfigurationError

    with pytest.raises(ConfigurationError) as exc:
        load_plot_envelope_from_path("./relative.yaml", source_dir=None)
    msg = str(exc.value)
    assert "relative" in msg
    assert "absolute path" in msg or "inline the plot section" in msg


def test_load_plot_envelope_path_not_found(tmp_path: Path):
    """Loader error includes both raw and resolved paths."""
    from aiperf.config.loader.errors import ConfigurationError

    with pytest.raises(ConfigurationError) as exc:
        load_plot_envelope_from_path("./missing.yaml", source_dir=tmp_path)
    msg = str(exc.value)
    assert "missing.yaml" in msg
    assert "not found" in msg


def test_load_plot_envelope_malformed_yaml(tmp_path: Path):
    """Loader wraps YAML parse errors in ConfigurationError."""
    from aiperf.config.loader.errors import ConfigurationError

    plot_file = tmp_path / "bad.yaml"
    plot_file.write_text(
        "visualization: [this, is, a, list, not, mapping]\nbroken: : :"
    )
    with pytest.raises(ConfigurationError) as exc:
        load_plot_envelope_from_path(plot_file, source_dir=None)
    assert "failed to parse" in str(
        exc.value
    ).lower() or "must contain a mapping" in str(exc.value)


def test_load_plot_envelope_top_level_must_be_mapping(tmp_path: Path):
    """Loader rejects YAMLs whose top level isn't a dict."""
    from aiperf.config.loader.errors import ConfigurationError

    plot_file = tmp_path / "list.yaml"
    plot_file.write_text("- a\n- b\n")
    with pytest.raises(ConfigurationError) as exc:
        load_plot_envelope_from_path(plot_file, source_dir=None)
    assert "mapping" in str(exc.value)


def test_load_config_from_string_with_file_path_threads_source_dir(tmp_path):
    """When file_path is given, source_dir is the file's parent. No assertion
    on plot here — that's Task 3. We just confirm the call shape still works
    after threading the context."""
    minimal_yaml = """
benchmark:
  models: [llama-3-8b]
  endpoint:
    type: chat
    urls: ["http://localhost:8000/v1/chat/completions"]
  datasets:
    - {name: main, type: synthetic, entries: 10}
  phases:
    - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
"""
    config = load_config_from_string(minimal_yaml, file_path=tmp_path / "x.yaml")
    assert config.benchmark.models.items[0].name == "llama-3-8b"


# Minimum viable AIPerfConfig YAML — same shape used by the threading test in
# Task 2. Field paths/keys mirror tests/unit/config/test_envelope_restructure.py.
_BASE_BENCHMARK_YAML = """
benchmark:
  models: [llama-3-8b]
  endpoint:
    type: chat
    urls: ["http://localhost:8000/v1/chat/completions"]
  datasets:
    - name: main
      type: synthetic
      entries: 10
      prompts:
        isl: 16
  phases:
    - name: profiling
      type: concurrency
      requests: 10
      concurrency: 1
"""

_INLINE_PLOT_YAML = """
plot:
  visualization:
    multi_run_defaults: [pareto_a]
    multi_run_plots:
      pareto_a:
        type: pareto
        x: {metric: request_latency, stat: avg}
        y: {metric: output_token_throughput_per_gpu, stat: avg}
"""


def test_aiperfconfig_inline_plot_block_loads():
    """Form B parses into PlotEnvelopeConfig on AIPerfConfig.plot."""
    config = load_config_from_string(_BASE_BENCHMARK_YAML + _INLINE_PLOT_YAML)
    assert config.plot is not None
    assert config.plot.visualization.multi_run_defaults == ["pareto_a"]


def test_aiperfconfig_no_plot_block_field_is_none():
    """When plot: is omitted, AIPerfConfig.plot is None."""
    config = load_config_from_string(_BASE_BENCHMARK_YAML)
    assert config.plot is None


def test_aiperfconfig_plot_path_relative_to_yaml(tmp_path: Path):
    """A bare-string plot: path resolves relative to the AIPerf YAML's dir."""
    plot_file = tmp_path / "my_plots.yaml"
    plot_file.write_text(
        "visualization:\n"
        "  multi_run_defaults: [pareto_a]\n"
        "  multi_run_plots:\n"
        "    pareto_a:\n"
        "      type: pareto\n"
    )
    main_yaml = _BASE_BENCHMARK_YAML + "\nplot: ./my_plots.yaml\n"
    main_file = tmp_path / "run.yaml"
    main_file.write_text(main_yaml)

    config = load_config_from_string(main_yaml, file_path=main_file)
    assert config.plot is not None
    assert config.plot.visualization.multi_run_defaults == ["pareto_a"]


def test_aiperfconfig_plot_path_absolute(tmp_path: Path):
    """An absolute plot: path is used as-is regardless of source_dir."""
    plot_file = tmp_path / "abs_plots.yaml"
    plot_file.write_text(
        "visualization:\n"
        "  multi_run_defaults: [pareto_a]\n"
        "  multi_run_plots:\n"
        "    pareto_a: {type: pareto}\n"
    )
    main_yaml = _BASE_BENCHMARK_YAML + f"\nplot: {plot_file}\n"
    config = load_config_from_string(main_yaml, file_path=tmp_path / "run.yaml")
    assert config.plot is not None


def test_aiperfconfig_plot_path_relative_no_source_dir_rejected():
    """load_config_from_string with no file_path rejects relative plot paths."""
    yaml_str = _BASE_BENCHMARK_YAML + "\nplot: ./not_resolvable.yaml\n"
    with pytest.raises(ConfigurationError) as exc:
        load_config_from_string(yaml_str)
    msg = str(exc.value)
    assert "relative" in msg
    assert ("absolute path" in msg) or ("inline the plot section" in msg)


def test_aiperfconfig_plot_path_not_found(tmp_path: Path):
    """A non-existent plot path produces an error with the raw filename."""
    main_yaml = _BASE_BENCHMARK_YAML + "\nplot: ./missing.yaml\n"
    with pytest.raises(ConfigurationError) as exc:
        load_config_from_string(main_yaml, file_path=tmp_path / "run.yaml")
    msg = str(exc.value)
    assert "missing.yaml" in msg
    assert "not found" in msg


def test_aiperfconfig_plot_presence_implies_auto_plot_true():
    """When plot: is set and auto_plot is unset, auto_plot flips to True."""
    config = load_config_from_string(_BASE_BENCHMARK_YAML + _INLINE_PLOT_YAML)
    assert config.benchmark.artifacts.auto_plot is True


def test_aiperfconfig_plot_presence_respects_explicit_auto_plot_false(caplog):
    """Explicit artifacts.auto_plot: false wins over plot: presence."""
    yaml_with_explicit_false = (
        _BASE_BENCHMARK_YAML.replace(
            "  endpoint:",
            "  artifacts:\n    auto_plot: false\n  endpoint:",
        )
        + _INLINE_PLOT_YAML
    )
    with caplog.at_level(logging.INFO):
        config = load_config_from_string(yaml_with_explicit_false)
    assert config.benchmark.artifacts.auto_plot is False
    assert any(
        "plot section present but artifacts.auto_plot=false" in rec.message
        for rec in caplog.records
    )


def test_aiperfconfig_no_plot_block_does_not_flip_auto_plot():
    """Without a plot: section, auto_plot keeps its existing default (False)."""
    config = load_config_from_string(_BASE_BENCHMARK_YAML)
    assert config.benchmark.artifacts.auto_plot is False


def test_aiperfconfig_plot_invalid_type_rejected():
    """plot: <int> / plot: [list] etc. are rejected with a clear message."""
    yaml_str = _BASE_BENCHMARK_YAML + "\nplot: 42\n"
    with pytest.raises(Exception) as exc:
        load_config_from_string(yaml_str)
    msg = str(exc.value)
    assert "must be null" in msg or "path string" in msg or "inline mapping" in msg


def test_build_benchmark_plan_carries_plot_envelope():
    """build_benchmark_plan copies envelope.plot onto the resulting plan."""
    from aiperf.config.loader import build_benchmark_plan

    config = load_config_from_string(_BASE_BENCHMARK_YAML + _INLINE_PLOT_YAML)
    plan = build_benchmark_plan(config)
    assert plan.plot is not None
    assert plan.plot.visualization.multi_run_defaults == ["pareto_a"]


def test_build_benchmark_plan_no_plot_section():
    """Without an envelope plot section, plan.plot is None."""
    from aiperf.config.loader import build_benchmark_plan

    config = load_config_from_string(_BASE_BENCHMARK_YAML)
    plan = build_benchmark_plan(config)
    assert plan.plot is None
