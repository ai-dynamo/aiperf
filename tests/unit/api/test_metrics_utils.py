# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the metrics utilities."""

from unittest.mock import patch

import pytest
from pytest import param

from aiperf.api.metrics_utils import build_info_labels, format_metrics_json
from aiperf.common.exceptions import MetricTypeError
from aiperf.common.models import MetricResult
from aiperf.config import AIPerfConfig

from .conftest import (
    make_info_labels,
    make_latency_metric,
    make_metric_result,
    make_throughput_metric,
)

# ---------------------------------------------------------------------------
# AIPerfConfig builder
# ---------------------------------------------------------------------------

_BASE: dict = {
    "datasets": {
        "main": {
            "type": "synthetic",
            "entries": 10,
            "prompts": {"isl": 32, "osl": 16},
        },
    },
    "phases": {
        "default": {"type": "concurrency", "requests": 10, "concurrency": 1},
    },
}


def _make_config(
    model_names: list[str] | None = None,
    endpoint_type: str = "chat",
    streaming: bool = False,
    benchmark_id: str | None = None,
) -> AIPerfConfig:
    """Build a minimal AIPerfConfig for metrics-utils tests."""
    models = model_names or ["test-model"]
    url = f"http://localhost:8000/v1/{endpoint_type}/completions"
    endpoint: dict = {"urls": [url], "type": endpoint_type, "streaming": streaming}
    kwargs: dict = {**_BASE, "models": models, "endpoint": endpoint}
    if benchmark_id is not None:
        kwargs["artifacts"] = {"benchmark_id": benchmark_id}
    return AIPerfConfig(**kwargs)


class TestBuildInfoLabels:
    """Test info label building from AIPerfConfig."""

    def test_basic_labels(self) -> None:
        """Test basic label extraction from config."""
        config = _make_config(model_names=["gpt-4"])
        labels = build_info_labels(config)

        assert labels["model"] == "gpt-4"
        assert labels["endpoint_type"] == "chat"
        assert labels["streaming"] == "false"
        assert "config" in labels

    def test_multiple_models(self) -> None:
        """Test multiple model names are comma-separated."""
        config = _make_config(model_names=["gpt-4", "gpt-3.5-turbo"])
        labels = build_info_labels(config)
        assert labels["model"] == "gpt-4,gpt-3.5-turbo"

    def test_benchmark_id_included(self) -> None:
        """Test benchmark_id is included when set."""
        config = _make_config(benchmark_id="test-bench-123")
        labels = build_info_labels(config)
        assert labels["benchmark_id"] == "test-bench-123"

    def test_benchmark_id_auto_generated(self) -> None:
        """Test benchmark_id is auto-generated when not explicitly set."""
        config = _make_config()
        labels = build_info_labels(config)
        assert "benchmark_id" in labels

    @pytest.mark.parametrize(
        "streaming,expected",
        [
            param(False, "false", id="streaming-false"),
            param(True, "true", id="streaming-true"),
        ],
    )  # fmt: skip
    def test_streaming_label(self, streaming: bool, expected: str) -> None:
        """Test streaming label reflects endpoint configuration."""
        config = _make_config(streaming=streaming)
        labels = build_info_labels(config)
        assert labels["streaming"] == expected

    @pytest.mark.parametrize(
        "endpoint_type",
        [
            param("chat", id="chat"),
            param("completions", id="completions"),
            param("embeddings", id="embeddings"),
        ],
    )  # fmt: skip
    def test_endpoint_type_label(self, endpoint_type: str) -> None:
        """Test endpoint_type label reflects configuration."""
        config = _make_config(endpoint_type=endpoint_type)
        labels = build_info_labels(config)
        assert labels["endpoint_type"] == endpoint_type

    def test_config_contains_serialized_config(self) -> None:
        """Test that config label contains serialized AIPerfConfig."""
        config = _make_config(
            benchmark_id="test-123",
            model_names=["test-model"],
            streaming=True,
        )
        labels = build_info_labels(config)

        assert isinstance(labels["config"], dict)

    def test_benchmark_id_excluded_when_empty(self) -> None:
        """Test benchmark_id is not included when it's an empty string."""
        config = _make_config(benchmark_id="will-be-replaced")
        config.artifacts.benchmark_id = ""
        labels = build_info_labels(config)
        assert "benchmark_id" not in labels

    def test_benchmark_id_excluded_when_none(self) -> None:
        """Test benchmark_id is not included when it's None."""
        config = _make_config(benchmark_id="will-be-replaced")
        config.artifacts.benchmark_id = None
        labels = build_info_labels(config)
        assert "benchmark_id" not in labels


class TestFormatMetricsJson:
    """Test JSON metrics formatting."""

    def test_empty_metrics(self) -> None:
        """Test formatting empty metrics list."""
        data = format_metrics_json([])

        assert "aiperf_version" in data
        assert data["metrics"] == {}

    def test_single_metric(self) -> None:
        """Test formatting single metric."""
        metric = make_metric_result(avg=100.0, min=50.0, max=150.0)
        data = format_metrics_json([metric])

        assert "test_metric" in data["metrics"]
        assert data["metrics"]["test_metric"]["avg"] == 100.0
        assert data["metrics"]["test_metric"]["min"] == 50.0
        assert data["metrics"]["test_metric"]["max"] == 150.0
        assert "tag" not in data["metrics"]["test_metric"]

    def test_benchmark_id_included(self) -> None:
        """Test benchmark_id is included when provided."""
        data = format_metrics_json([], benchmark_id="bench-123")
        assert data["benchmark_id"] == "bench-123"

    def test_benchmark_id_excluded_when_none(self) -> None:
        """Test benchmark_id is not included when None."""
        data = format_metrics_json([])
        assert "benchmark_id" not in data

    def test_info_labels_included(self) -> None:
        """Test info labels are merged into response."""
        labels = make_info_labels(model="gpt-4", endpoint_type="openai")
        data = format_metrics_json([], info_labels=labels)

        assert data["model"] == "gpt-4"
        assert data["endpoint_type"] == "openai"

    @pytest.mark.parametrize(
        "excluded_key",
        [
            param("config", id="config-excluded"),
            param("version", id="version-excluded"),
        ],
    )  # fmt: skip
    def test_excluded_labels(self, excluded_key: str) -> None:
        """Test that certain labels are excluded from output."""
        data = format_metrics_json(
            [], info_labels={"model": "gpt-4", excluded_key: "some_value"}
        )

        assert excluded_key not in data or data.get(excluded_key) != "some_value"
        assert data["model"] == "gpt-4"

    def test_multiple_metrics(self) -> None:
        """Test formatting multiple metrics."""
        metrics = [
            make_latency_metric(avg=100.0),
            make_throughput_metric(avg=50.0),
        ]
        data = format_metrics_json(metrics)

        assert data["metrics"]["latency"]["avg"] == 100.0
        assert data["metrics"]["throughput"]["avg"] == 50.0

    def test_metric_type_error_uses_raw_metric(self) -> None:
        """Test that MetricTypeError falls back to raw metric values."""
        metric = make_metric_result(tag="unknown_metric", avg=100.0, min=50.0)

        with patch.object(
            MetricResult,
            "to_display_unit",
            side_effect=MetricTypeError("Unknown metric"),
        ):
            data = format_metrics_json([metric])

        assert "unknown_metric" in data["metrics"]
        assert data["metrics"]["unknown_metric"]["avg"] == 100.0

    def test_none_values_excluded_from_metric_dump(self) -> None:
        """Test that None values are excluded from metric output."""
        metric = make_metric_result(avg=100.0, min=None, max=None)
        data = format_metrics_json([metric])

        metric_data = data["metrics"]["test_metric"]
        assert "avg" in metric_data
        assert "min" not in metric_data
        assert "max" not in metric_data

    def test_all_stat_values_included(self) -> None:
        """Test that all provided stat values are included."""
        metric = MetricResult(
            tag="full_metric",
            header="Full Metric",
            unit="ms",
            avg=100.0,
            min=10.0,
            max=200.0,
            sum=5000.0,
            std=25.0,
            p50=95.0,
            p95=180.0,
            p99=195.0,
        )
        data = format_metrics_json([metric])

        metric_data = data["metrics"]["full_metric"]
        assert metric_data["avg"] == 100.0
        assert metric_data["min"] == 10.0
        assert metric_data["max"] == 200.0
        assert metric_data["sum"] == 5000.0
        assert metric_data["std"] == 25.0
        assert metric_data["p50"] == 95.0
        assert metric_data["p95"] == 180.0
        assert metric_data["p99"] == 195.0

    def test_info_labels_coerced(self) -> None:
        """Test that info label values are coerced via coerce_value."""
        labels = {"model": "test", "numeric": 42, "bool": True}
        data = format_metrics_json([], info_labels=labels)

        assert data["model"] == "test"
        assert data["numeric"] == 42
        assert data["bool"] is True

    def test_version_always_included(self) -> None:
        """Test that aiperf_version is always included."""
        data = format_metrics_json([])
        assert "aiperf_version" in data
        assert isinstance(data["aiperf_version"], str)
        assert len(data["aiperf_version"]) > 0

    def test_empty_info_labels_dict(self) -> None:
        """Test that empty info_labels dict does not add any labels."""
        data = format_metrics_json([], info_labels={})
        assert "aiperf_version" in data
        assert "metrics" in data
        assert len(data) == 2

    def test_info_labels_with_string_values(self) -> None:
        """Test that string values in info_labels are coerced correctly."""
        labels = {
            "model": "gpt-4",
            "str_true": "true",
            "str_false": "false",
            "str_none": "none",
            "str_null": "null",
            "str_int": "42",
            "str_negative_int": "-10",
            "str_float": "3.14",
            "str_negative_float": "-2.5",
            "plain_string": "hello",
        }
        data = format_metrics_json([], info_labels=labels)

        assert data["model"] == "gpt-4"
        assert data["str_true"] is True
        assert data["str_false"] is False
        assert data["str_none"] is None
        assert data["str_null"] is None
        assert data["str_int"] == 42
        assert data["str_negative_int"] == -10
        assert data["str_float"] == 3.14
        assert data["str_negative_float"] == -2.5
        assert data["plain_string"] == "hello"

    def test_benchmark_id_and_info_labels_combined(self) -> None:
        """Test both benchmark_id and info_labels provided together."""
        labels = make_info_labels(model="test-model", benchmark_id="label-bench-id")
        data = format_metrics_json(
            [], info_labels=labels, benchmark_id="param-bench-id"
        )

        assert data["benchmark_id"] == "label-bench-id"
        assert data["model"] == "test-model"

    def test_metric_with_all_percentiles(self) -> None:
        """Test metric with all percentile values included."""
        metric = MetricResult(
            tag="percentile_metric",
            header="Percentile Metric",
            unit="ms",
            p50=50.0,
            p90=90.0,
            p95=95.0,
            p99=99.0,
        )
        data = format_metrics_json([metric])

        metric_data = data["metrics"]["percentile_metric"]
        assert metric_data["p50"] == 50.0
        assert metric_data["p90"] == 90.0
        assert metric_data["p95"] == 95.0
        assert metric_data["p99"] == 99.0

    def test_metric_display_unit_conversion_success(self) -> None:
        """Test that metrics are converted to display units when possible."""
        metric = make_latency_metric(avg=1000.0, min=500.0, max=2000.0)

        data = format_metrics_json([metric])

        assert "latency" in data["metrics"]
        assert "avg" in data["metrics"]["latency"]
