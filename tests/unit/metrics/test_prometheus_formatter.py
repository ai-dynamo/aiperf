# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Prometheus formatter."""

import math
from unittest.mock import patch

import pytest
from prometheus_client.parser import text_string_to_metric_families
from pytest import param

from aiperf.common.exceptions import MetricTypeError
from aiperf.common.models import MetricResult
from aiperf.metrics.prometheus_formatter import (
    GAUGE_STATS,
    QUANTILE_STATS,
    _escape_help_text,
    _format_info_metric,
    _format_value,
    format_as_prometheus,
    format_labels,
    sanitize_metric_name,
)


def make_metric_result(
    tag: str = "test_metric",
    header: str = "Test Metric",
    unit: str = "ms",
    avg: float | None = None,
    min: float | None = None,
    max: float | None = None,
    sum: float | None = None,
    p50: float | None = None,
    p95: float | None = None,
    p99: float | None = None,
    std: float | None = None,
    **kwargs,
) -> MetricResult:
    """Create a MetricResult with sensible defaults."""
    return MetricResult(
        tag=tag,
        header=header,
        unit=unit,
        avg=avg,
        min=min,
        max=max,
        sum=sum,
        p50=p50,
        p95=p95,
        p99=p99,
        std=std,
        **kwargs,
    )


def make_latency_metric(
    avg: float = 100.0,
    min: float = 50.0,
    max: float = 200.0,
    p50: float = 95.0,
    p95: float = 180.0,
    p99: float = 195.0,
) -> MetricResult:
    """Create a typical latency metric for testing."""
    return MetricResult(
        tag="latency",
        header="Latency",
        unit="ms",
        avg=avg,
        min=min,
        max=max,
        p50=p50,
        p95=p95,
        p99=p99,
    )


def make_info_labels(
    model: str = "test-model",
    endpoint_type: str = "chat",
    streaming: str = "false",
    benchmark_id: str | None = None,
    concurrency: str | None = None,
    request_rate: str | None = None,
    config: dict | None = None,
) -> dict[str, str]:
    """Create info labels dict for Prometheus/JSON metrics testing."""
    labels: dict[str, str] = {
        "model": model,
        "endpoint_type": endpoint_type,
        "streaming": streaming,
    }
    if benchmark_id:
        labels["benchmark_id"] = benchmark_id
    if concurrency:
        labels["concurrency"] = concurrency
    if request_rate:
        labels["request_rate"] = request_rate
    if config:
        labels["config"] = config
    return labels


class TestSanitizeMetricName:
    """Test metric name sanitization."""

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            param("TestMetric", "testmetric", id="camelcase-to-lower"),
            param("test-metric", "test_metric", id="dash-to-underscore"),
            param("test.metric", "test_metric", id="dot-to-underscore"),
            param("test/metric", "test_metric", id="slash-to-underscore"),
            param("123metric", "_123metric", id="leading-digit-prefix"),
            param("valid_metric_123", "valid_metric_123", id="already-valid"),
            param("", "", id="empty-string"),
            param("UPPER_CASE", "upper_case", id="uppercase-to-lower"),
            param("test@metric#name", "test_metric_name", id="special-chars"),
            param("test  metric", "test__metric", id="spaces-to-underscores"),
            param("0", "_0", id="single-digit"),
            param("_already_prefixed", "_already_prefixed", id="underscore-prefix"),
        ],
    )  # fmt: skip
    def test_sanitize(self, input_name: str, expected: str) -> None:
        """Test metric name sanitization handles various inputs."""
        assert sanitize_metric_name(input_name) == expected


class TestFormatLabels:
    """Test label formatting."""

    def test_empty_labels(self) -> None:
        """Test formatting empty labels returns empty string."""
        assert format_labels({}) == ""

    def test_single_label(self) -> None:
        """Test formatting single label."""
        assert format_labels({"key": "value"}) == '{key="value"}'

    def test_multiple_labels(self) -> None:
        """Test formatting multiple labels."""
        result = format_labels({"key1": "value1", "key2": "value2"})
        assert 'key1="value1"' in result
        assert 'key2="value2"' in result

    @pytest.mark.parametrize(
        "value,expected",
        [
            param('value"with"quotes', '{key="value\\"with\\"quotes"}', id="quotes"),
            param("value\\with\\backslash", '{key="value\\\\with\\\\backslash"}', id="backslash"),
            param('mixed\\"both', '{key="mixed\\\\\\"both"}', id="mixed-escape"),
            param("line\nbreak", '{key="line\\nbreak"}', id="newline"),
            param("no_special", '{key="no_special"}', id="no-escaping-needed"),
            param("", '{key=""}', id="empty-value"),
        ],
    )  # fmt: skip
    def test_escape_special_chars(self, value: str, expected: str) -> None:
        """Test that special characters in values are escaped correctly."""
        assert format_labels({"key": value}) == expected

    def test_numeric_value_coerced(self) -> None:
        """Test that numeric values are coerced to strings."""
        result = format_labels({"count": 42})
        assert result == '{count="42"}'


class TestFormatInfoMetric:
    """Test _format_info_metric helper function."""

    def test_empty_info_labels(self) -> None:
        """Test formatting with empty info labels returns empty string."""
        assert _format_info_metric({}) == ""

    def test_includes_version(self) -> None:
        """Test that aiperf version is included in info metric."""
        result = _format_info_metric({"model": "test"})
        assert "aiperf_info" in result
        assert 'version="' in result

    def test_includes_all_labels(self) -> None:
        """Test that all provided labels are included."""
        labels = make_info_labels(
            model="gpt-4",
            endpoint_type="chat",
            benchmark_id="bench-123",
        )
        result = _format_info_metric(labels)

        assert 'model="gpt-4"' in result
        assert 'endpoint_type="chat"' in result
        assert 'benchmark_id="bench-123"' in result

    def test_help_and_type_lines(self) -> None:
        """Test that HELP and TYPE lines are generated."""
        result = _format_info_metric({"model": "test"})
        assert "# HELP aiperf_info" in result
        assert "# TYPE aiperf_info gauge" in result
        assert "aiperf_info{" in result
        assert "} 1" in result

    @pytest.mark.parametrize(
        "value,expected",
        [
            param('value"with"quotes', 'model="value\\"with\\"quotes"', id="quotes"),
            param("value\\with\\backslash", 'model="value\\\\with\\\\backslash"', id="backslash"),
            param('mixed\\"both', 'model="mixed\\\\\\"both"', id="mixed-escape"),
        ],
    )  # fmt: skip
    def test_escape_special_chars_in_labels(self, value: str, expected: str) -> None:
        """Test that special characters in info metric labels are escaped."""
        result = _format_info_metric({"model": value})
        assert expected in result

    def test_numeric_label_value_coerced(self) -> None:
        """Test that numeric label values are coerced to strings."""
        result = _format_info_metric({"count": 42})
        assert 'count="42"' in result

    def test_ends_with_newline(self) -> None:
        """Test that output ends with a newline."""
        result = _format_info_metric({"model": "test"})
        assert result.endswith("\n")


class TestFormatAsPrometheus:
    """Test Prometheus format generation."""

    def test_empty_metrics(self) -> None:
        """Test formatting empty metrics list returns empty string."""
        assert format_as_prometheus([]) == ""

    def test_gauge_metric_has_correct_type(self) -> None:
        """Test that gauge-only metrics are typed as gauge."""
        metric = make_metric_result(avg=100.0, min=50.0, max=150.0)
        result = format_as_prometheus([metric])

        assert "# TYPE aiperf_test_metric_avg_ms gauge" in result
        assert "aiperf_test_metric_avg_ms 100.0" in result

    def test_info_labels_included(self) -> None:
        """Test that info labels are included in output."""
        metric = make_metric_result(avg=100.0)
        labels = make_info_labels(model="gpt-4", endpoint_type="openai")
        result = format_as_prometheus([metric], info_labels=labels)

        assert "aiperf_info" in result
        assert 'model="gpt-4"' in result
        assert 'endpoint_type="openai"' in result

    def test_config_and_version_excluded_from_metric_labels(self) -> None:
        """Test that config and version are excluded from metric labels."""
        metric = make_metric_result(avg=100.0)
        labels = {
            "model": "test",
            "config": {"some": "config"},
            "version": "1.0.0",
        }
        result = format_as_prometheus([metric], info_labels=labels)

        lines = result.split("\n")
        metric_lines = [line for line in lines if line.startswith("aiperf_test_metric")]
        for line in metric_lines:
            assert "config=" not in line

    def test_percentiles_emit_summary_type(self) -> None:
        """Test that percentile stats produce a summary metric type."""
        metric = make_metric_result(p50=95.0, p95=180.0, p99=195.0)
        result = format_as_prometheus([metric])

        assert "# TYPE aiperf_test_metric_ms summary" in result
        assert 'quantile="0.5"' in result
        assert 'quantile="0.95"' in result
        assert 'quantile="0.99"' in result

    def test_summary_count_emitted(self) -> None:
        """Test that _count suffix is emitted when count is present with quantiles."""
        metric = MetricResult(
            tag="latency", header="Latency", unit="ms", p50=100.0, count=500
        )
        result = format_as_prometheus([metric])

        assert "# TYPE aiperf_latency_ms summary" in result
        assert "aiperf_latency_ms_count 500" in result

    def test_summary_sum_emitted(self) -> None:
        """Test that _sum suffix is emitted when sum is present with quantiles."""
        metric = make_metric_result(p50=100.0, sum=50000.0)
        result = format_as_prometheus([metric])

        assert "aiperf_test_metric_ms_sum 50000.0" in result

    def test_gauge_only_count_emitted_as_gauge(self) -> None:
        """Test that count is emitted as a gauge when no quantiles exist."""
        metric = make_metric_result(avg=100.0, count=500)
        result = format_as_prometheus([metric])

        assert "# TYPE aiperf_test_metric_count_ms gauge" in result
        assert "aiperf_test_metric_count_ms 500" in result
        assert "summary" not in result

    def test_gauge_only_sum_emitted_as_gauge(self) -> None:
        """Test that sum is emitted as a gauge when no quantiles exist."""
        metric = make_metric_result(avg=100.0, sum=50000.0)
        result = format_as_prometheus([metric])

        assert "# TYPE aiperf_test_metric_sum_ms gauge" in result
        assert "aiperf_test_metric_sum_ms 50000.0" in result
        assert "summary" not in result

    def test_count_sum_not_duplicated_with_quantiles(self) -> None:
        """Test that count/sum appear only in Summary when quantiles exist."""
        metric = MetricResult(
            tag="latency",
            header="Latency",
            unit="ms",
            p50=100.0,
            count=500,
            sum=50000.0,
        )
        result = format_as_prometheus([metric])

        assert "aiperf_latency_ms_count 500" in result
        assert "aiperf_latency_ms_sum 50000.0" in result
        assert "aiperf_latency_count_ms" not in result
        assert "aiperf_latency_sum_ms" not in result

    def test_gauge_stats_display_names_in_help(self) -> None:
        """Test that gauge stat display names appear in HELP text."""
        metric = make_metric_result(avg=100.0, min=10.0, max=200.0, std=25.0)
        result = format_as_prometheus([metric])

        assert "average" in result
        assert "minimum" in result
        assert "maximum" in result
        assert "standard deviation" in result

    def test_multiple_metrics(self) -> None:
        """Test formatting multiple metrics."""
        metrics = [
            make_latency_metric(avg=100.0),
            make_metric_result(
                tag="throughput", header="Throughput", unit="req/s", avg=50.0
            ),
        ]
        result = format_as_prometheus(metrics)

        assert "aiperf_latency" in result
        assert "aiperf_throughput" in result

    def test_metric_type_error_uses_raw_metric(self) -> None:
        """Test that MetricTypeError falls back to raw metric."""
        metric = make_metric_result(tag="unknown_metric", avg=100.0)

        with patch.object(
            MetricResult,
            "to_display_unit",
            side_effect=MetricTypeError("Test error"),
        ):
            result = format_as_prometheus([metric])

        assert "aiperf_unknown_metric" in result
        assert "100.0" in result

    def test_none_stat_values_skipped(self) -> None:
        """Test that None stat values are not included in output."""
        metric = make_metric_result(avg=100.0, min=None, max=None)
        result = format_as_prometheus([metric])

        assert "avg" in result
        assert "aiperf_test_metric_avg_ms 100.0" in result
        assert "min" not in result
        assert "max" not in result

    def test_all_percentiles_as_quantile_labels(self) -> None:
        """Test that all percentile stats are formatted with quantile labels."""
        metric = MetricResult(
            tag="latency",
            header="Latency",
            unit="ms",
            p1=10.0,
            p5=20.0,
            p10=30.0,
            p25=50.0,
            p50=100.0,
            p75=150.0,
            p90=180.0,
            p95=190.0,
            p99=199.0,
        )
        result = format_as_prometheus([metric])

        assert "# TYPE aiperf_latency_ms summary" in result
        for quantile_value in QUANTILE_STATS.values():
            assert f'quantile="{quantile_value}"' in result

    def test_none_info_labels_produces_no_info_metric(self) -> None:
        """Test that None info_labels produces no aiperf_info metric."""
        metric = make_metric_result(avg=100.0)
        result = format_as_prometheus([metric], info_labels=None)

        assert "aiperf_info" not in result
        assert "aiperf_test_metric" in result

    def test_empty_info_labels_produces_no_info_metric(self) -> None:
        """Test that empty dict info_labels produces no aiperf_info metric."""
        metric = make_metric_result(avg=100.0)
        result = format_as_prometheus([metric], info_labels={})

        assert "aiperf_info" not in result
        assert "aiperf_test_metric" in result

    def test_only_info_labels_no_metrics_produces_only_info(self) -> None:
        """Test that info_labels with empty metrics produces only info metric."""
        labels = make_info_labels(model="gpt-4")
        result = format_as_prometheus([], info_labels=labels)

        assert "aiperf_info" in result
        assert 'model="gpt-4"' in result

    def test_metric_labels_applied_to_all_value_lines(self) -> None:
        """Test that metric labels from info_labels are applied to all value lines."""
        metric = make_metric_result(avg=100.0, p50=95.0)
        labels = make_info_labels(model="gpt-4", endpoint_type="chat")
        result = format_as_prometheus([metric], info_labels=labels)

        lines = result.split("\n")
        value_lines = [
            line
            for line in lines
            if line.startswith("aiperf_test_metric") and "{" in line
        ]
        assert len(value_lines) > 0
        for line in value_lines:
            assert 'model="gpt-4"' in line
            assert 'endpoint_type="chat"' in line

    def test_metric_type_error_uses_raw_unit(self) -> None:
        """Test that MetricTypeError fallback uses raw metric unit."""
        metric = make_metric_result(tag="custom_metric", unit="custom_unit", avg=100.0)

        with patch.object(
            MetricResult,
            "to_display_unit",
            side_effect=MetricTypeError("Unknown metric"),
        ):
            result = format_as_prometheus([metric])

        assert "custom_unit" in result

    def test_mixed_summary_and_gauge_for_same_metric(self) -> None:
        """Test a metric with both percentiles and scalar stats."""
        metric = MetricResult(
            tag="latency",
            header="Latency",
            unit="ms",
            avg=100.0,
            min=10.0,
            max=200.0,
            p50=95.0,
            p95=185.0,
            p99=195.0,
            count=1000,
        )
        result = format_as_prometheus([metric])

        assert "# TYPE aiperf_latency_ms summary" in result
        assert "# TYPE aiperf_latency_avg_ms gauge" in result
        assert "# TYPE aiperf_latency_min_ms gauge" in result
        assert "# TYPE aiperf_latency_max_ms gauge" in result
        assert 'quantile="0.5"' in result
        assert "aiperf_latency_ms_count 1000" in result


class TestStatConstants:
    """Test QUANTILE_STATS and GAUGE_STATS constants."""

    def test_quantile_stats_cover_all_percentiles(self) -> None:
        """Verify all expected percentile stats are present."""
        expected = {"p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"}
        assert expected == set(QUANTILE_STATS.keys())

    def test_gauge_stats_cover_scalar_stats(self) -> None:
        """Verify all expected gauge stats are present."""
        expected = {"avg", "min", "max", "std", "current"}
        assert expected == set(GAUGE_STATS.keys())

    def test_quantile_values_are_valid_floats(self) -> None:
        """Verify quantile values are valid float strings between 0 and 1."""
        for stat, qval in QUANTILE_STATS.items():
            f = float(qval)
            assert 0 < f < 1, f"Quantile value for {stat} must be between 0 and 1"


class TestEscapeHelpText:
    """Test HELP docstring escaping per Prometheus spec."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            param("simple text", "simple text", id="no-escaping"),
            param("back\\slash", "back\\\\slash", id="backslash"),
            param("line\nbreak", "line\\nbreak", id="newline"),
            param("both\\\n", "both\\\\\\n", id="backslash-and-newline"),
        ],
    )  # fmt: skip
    def test_escape(self, text: str, expected: str) -> None:
        """Test HELP text escaping handles backslash and newline."""
        assert _escape_help_text(text) == expected


class TestFormatValue:
    """Test numeric value formatting per Prometheus spec."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            param(42, "42", id="int"),
            param(3.14, "3.14", id="float"),
            param(0.0, "0.0", id="zero"),
            param(float("nan"), "NaN", id="nan"),
            param(float("inf"), "+Inf", id="positive-inf"),
            param(float("-inf"), "-Inf", id="negative-inf"),
        ],
    )  # fmt: skip
    def test_format(self, value: int | float, expected: str) -> None:
        """Test value formatting produces Prometheus-compatible output."""
        assert _format_value(value) == expected


class TestSpecComplianceIntegration:
    """Integration tests verifying Prometheus spec compliance."""

    def test_nan_value_in_gauge(self) -> None:
        """Test that NaN metric values are formatted correctly."""
        metric = make_metric_result(avg=float("nan"))
        result = format_as_prometheus([metric])
        assert "NaN" in result

    def test_inf_value_in_gauge(self) -> None:
        """Test that Inf metric values are formatted correctly."""
        metric = make_metric_result(avg=float("inf"))
        result = format_as_prometheus([metric])
        assert "+Inf" in result

    def test_help_text_with_newline_escaped(self) -> None:
        """Test that newlines in metric headers are escaped in HELP text."""
        metric = MetricResult(tag="test", header="Line\nBreak", unit="ms", avg=1.0)
        result = format_as_prometheus([metric])
        assert "Line\\nBreak" in result
        assert "Line\nBreak" not in result.split("# HELP")[1].split("\n")[0]

    def test_label_value_with_newline_escaped(self) -> None:
        """Test that newlines in label values are escaped."""
        metric = make_metric_result(avg=1.0)
        labels = {"model": "line\nbreak"}
        result = format_as_prometheus([metric], info_labels=labels)
        assert 'model="line\\nbreak"' in result


class TestReplaceMap:
    """Test _UNIT_REPLACE_MAP unit suffix transformations."""

    def test_slash_replaced_with_per(self) -> None:
        """Test that forward slashes in unit suffixes become _per_."""
        from aiperf.metrics.prometheus_formatter import _UNIT_REPLACE_MAP

        assert "/" in _UNIT_REPLACE_MAP
        assert _UNIT_REPLACE_MAP["/"] == "_per_"

    def test_tokens_per_sec_replaced_with_tps(self) -> None:
        """Test that tokens_per_sec becomes tps."""
        from aiperf.metrics.prometheus_formatter import _UNIT_REPLACE_MAP

        assert "tokens_per_sec" in _UNIT_REPLACE_MAP
        assert _UNIT_REPLACE_MAP["tokens_per_sec"] == "tps"

    def test_double_underscore_collapsed(self) -> None:
        """Test that double underscores are collapsed to single."""
        from aiperf.metrics.prometheus_formatter import _UNIT_REPLACE_MAP

        assert "__" in _UNIT_REPLACE_MAP
        assert _UNIT_REPLACE_MAP["__"] == "_"

    def test_percent_replaced(self) -> None:
        """Test that % in unit becomes percent."""
        from aiperf.metrics.prometheus_formatter import _UNIT_REPLACE_MAP

        assert "%" in _UNIT_REPLACE_MAP
        assert _UNIT_REPLACE_MAP["%"] == "percent"


class TestSanitizeUnit:
    """Test _sanitize_unit applies replacements and sanitizes."""

    def test_percent_unit_produces_valid_name(self) -> None:
        """Test that % unit is sanitized to a valid Prometheus suffix."""
        metric = make_metric_result(tag="osl_mismatch_diff_pct", unit="%", avg=0.0)
        result = format_as_prometheus([metric])

        assert "aiperf_osl_mismatch_diff_pct_avg_percent" in result
        metric_names = [
            line.split("{")[0].split(" ")[0]
            for line in result.split("\n")
            if line.startswith("aiperf_")
        ]
        for name in metric_names:
            assert "%" not in name

    def test_special_chars_in_unit_sanitized(self) -> None:
        """Test that arbitrary special characters in units are sanitized."""
        metric = make_metric_result(tag="test", unit="µs/op", avg=1.0)
        result = format_as_prometheus([metric])

        for line in result.split("\n"):
            if line.startswith("aiperf_"):
                assert "µ" not in line
                assert "/" not in line


class TestRegisteredMetricFormatting:
    """Test formatting with actual registered metrics."""

    def test_request_latency_metric_formatting(self) -> None:
        """Test formatting a registered metric (request_latency) uses display unit."""
        metric = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ns",
            avg=1000000.0,
        )
        result = format_as_prometheus([metric])

        assert "aiperf_request_latency" in result
        assert "# HELP" in result
        assert "# TYPE" in result
        assert "avg" in result

    def test_output_ends_with_newline(self) -> None:
        """Test that non-empty output ends with newline."""
        metric = make_metric_result(avg=100.0)
        result = format_as_prometheus([metric])
        assert result.endswith("\n")

    def test_metric_with_all_stats_produces_complete_output(self) -> None:
        """Test that metric with all stat values produces complete output."""
        metric = MetricResult(
            tag="full_metric",
            header="Full Metric",
            unit="ms",
            avg=100.0,
            sum=1000.0,
            min=10.0,
            max=200.0,
            std=25.0,
            count=500,
            current=99.0,
            p1=11.0,
            p5=20.0,
            p10=30.0,
            p25=50.0,
            p50=95.0,
            p75=150.0,
            p90=175.0,
            p95=185.0,
            p99=195.0,
        )
        result = format_as_prometheus([metric])

        assert "# TYPE aiperf_full_metric_ms summary" in result
        assert "aiperf_full_metric_ms_sum 1000.0" in result
        assert "aiperf_full_metric_ms_count 500" in result
        for qval in QUANTILE_STATS.values():
            assert f'quantile="{qval}"' in result
        for stat in GAUGE_STATS:
            assert f"# TYPE aiperf_full_metric_{stat}_ms gauge" in result

    def test_empty_lines_not_added_between_metrics(self) -> None:
        """Test that output does not contain empty lines between metrics."""
        metrics = [
            make_metric_result(tag="metric1", avg=100.0),
            make_metric_result(tag="metric2", avg=200.0),
        ]
        result = format_as_prometheus(metrics)

        assert "\n\n" not in result


def _parse_families(pef_text: str) -> dict:
    """Parse PEF text into a dict of {metric_name: MetricFamily}."""
    return {f.name: f for f in text_string_to_metric_families(pef_text)}


class TestPrometheusParserValidation:
    """Validate output is parseable by the official prometheus_client parser."""

    def test_gauge_only_parseable(self) -> None:
        """Test that gauge-only output is valid PEF."""
        metric = make_metric_result(avg=100.0, min=50.0, max=150.0)
        families = _parse_families(format_as_prometheus([metric]))

        gauge_names = {f.name for f in families.values() if f.type == "gauge"}
        assert "aiperf_test_metric_avg_ms" in gauge_names
        assert "aiperf_test_metric_min_ms" in gauge_names
        assert "aiperf_test_metric_max_ms" in gauge_names

    def test_summary_parseable(self) -> None:
        """Test that summary output is valid PEF with correct quantiles."""
        metric = MetricResult(
            tag="latency",
            header="Latency",
            unit="ms",
            p50=95.0,
            p95=185.0,
            p99=195.0,
            count=1000,
            sum=50000.0,
        )
        families = _parse_families(format_as_prometheus([metric]))

        assert "aiperf_latency_ms" in families
        family = families["aiperf_latency_ms"]
        assert family.type == "summary"

        samples_by_name = {}
        for sample in family.samples:
            samples_by_name.setdefault(sample.name, []).append(sample)

        quantile_samples = samples_by_name["aiperf_latency_ms"]
        quantiles = {s.labels["quantile"]: s.value for s in quantile_samples}
        assert quantiles["0.5"] == 95.0
        assert quantiles["0.95"] == 185.0
        assert quantiles["0.99"] == 195.0

        count_samples = samples_by_name["aiperf_latency_ms_count"]
        assert count_samples[0].value == 1000.0

        sum_samples = samples_by_name["aiperf_latency_ms_sum"]
        assert sum_samples[0].value == 50000.0

    def test_mixed_summary_and_gauges_parseable(self) -> None:
        """Test that mixed summary + gauge output for one metric is valid PEF."""
        metric = MetricResult(
            tag="latency",
            header="Latency",
            unit="ms",
            avg=100.0,
            min=10.0,
            max=200.0,
            p50=95.0,
            p95=185.0,
            count=500,
        )
        families = _parse_families(format_as_prometheus([metric]))

        assert families["aiperf_latency_ms"].type == "summary"
        assert families["aiperf_latency_avg_ms"].type == "gauge"
        assert families["aiperf_latency_min_ms"].type == "gauge"
        assert families["aiperf_latency_max_ms"].type == "gauge"

        avg_value = families["aiperf_latency_avg_ms"].samples[0].value
        assert avg_value == 100.0

    def test_multiple_metrics_parseable(self) -> None:
        """Test that multiple metrics produce valid PEF."""
        metrics = [
            make_metric_result(tag="latency", avg=100.0, p50=95.0),
            make_metric_result(tag="throughput", unit="req/s", avg=50.0),
        ]
        families = _parse_families(format_as_prometheus(metrics))

        assert "aiperf_latency_ms" in families
        assert "aiperf_latency_avg_ms" in families
        assert "aiperf_throughput_avg_req_per_s" in families

    def test_info_metric_parseable(self) -> None:
        """Test that aiperf_info metric is valid PEF."""
        metric = make_metric_result(avg=100.0)
        labels = make_info_labels(model="gpt-4", endpoint_type="chat")
        families = _parse_families(format_as_prometheus([metric], info_labels=labels))

        assert "aiperf_info" in families
        info_sample = families["aiperf_info"].samples[0]
        assert info_sample.labels["model"] == "gpt-4"
        assert info_sample.value == 1.0

    def test_labels_propagated_to_samples(self) -> None:
        """Test that metric labels appear on parsed samples."""
        metric = make_metric_result(avg=100.0, p50=95.0)
        labels = make_info_labels(model="gpt-4")
        families = _parse_families(format_as_prometheus([metric], info_labels=labels))

        gauge_sample = families["aiperf_test_metric_avg_ms"].samples[0]
        assert gauge_sample.labels["model"] == "gpt-4"

        summary_samples = [
            s
            for s in families["aiperf_test_metric_ms"].samples
            if "quantile" in s.labels
        ]
        for s in summary_samples:
            assert s.labels["model"] == "gpt-4"

    def test_nan_value_parseable(self) -> None:
        """Test that NaN values are parseable by the Prometheus parser."""
        metric = make_metric_result(avg=float("nan"))
        families = _parse_families(format_as_prometheus([metric]))

        value = families["aiperf_test_metric_avg_ms"].samples[0].value
        assert math.isnan(value)

    def test_inf_value_parseable(self) -> None:
        """Test that Inf values are parseable by the Prometheus parser."""
        metric = make_metric_result(avg=float("inf"))
        families = _parse_families(format_as_prometheus([metric]))

        value = families["aiperf_test_metric_avg_ms"].samples[0].value
        assert math.isinf(value) and value > 0

    def test_gauge_only_count_sum_parseable(self) -> None:
        """Test that count/sum as gauges (no quantiles) are valid PEF."""
        metric = make_metric_result(avg=100.0, count=500, sum=50000.0)
        families = _parse_families(format_as_prometheus([metric]))

        assert "aiperf_test_metric_count_ms" in families
        assert families["aiperf_test_metric_count_ms"].type == "gauge"
        assert families["aiperf_test_metric_count_ms"].samples[0].value == 500.0

        assert "aiperf_test_metric_sum_ms" in families
        assert families["aiperf_test_metric_sum_ms"].type == "gauge"
        assert families["aiperf_test_metric_sum_ms"].samples[0].value == 50000.0

        assert "aiperf_test_metric_ms" not in families

    def test_full_metric_round_trips(self) -> None:
        """Test that a fully-populated metric round-trips through the parser."""
        metric = MetricResult(
            tag="full_metric",
            header="Full Metric",
            unit="ms",
            avg=100.0,
            sum=1000.0,
            min=10.0,
            max=200.0,
            std=25.0,
            count=500,
            current=99.0,
            p1=11.0,
            p5=20.0,
            p10=30.0,
            p25=50.0,
            p50=95.0,
            p75=150.0,
            p90=175.0,
            p95=185.0,
            p99=195.0,
        )
        families = _parse_families(format_as_prometheus([metric]))

        summary = families["aiperf_full_metric_ms"]
        assert summary.type == "summary"

        quantile_samples = [
            s for s in summary.samples if s.name == "aiperf_full_metric_ms"
        ]
        assert len(quantile_samples) == len(QUANTILE_STATS)

        for stat in GAUGE_STATS:
            gauge_name = f"aiperf_full_metric_{stat}_ms"
            assert gauge_name in families, f"Missing gauge: {gauge_name}"
            assert families[gauge_name].type == "gauge"

        assert "aiperf_full_metric_count_ms" not in families
        assert "aiperf_full_metric_sum_ms" not in families
