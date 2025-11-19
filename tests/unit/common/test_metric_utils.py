# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.metric_utils import (
    build_hostname_aware_prometheus_endpoints,
    compute_histogram_delta,
    normalize_metrics_endpoint_url,
)


class TestNormalizeMetricsEndpointUrl:
    """Test URL normalization for metrics endpoints."""

    @pytest.mark.parametrize(
        "input_url,expected",  # fmt: skip
        [
            ("http://localhost:9400", "http://localhost:9400/metrics"),
            ("http://localhost:9400/", "http://localhost:9400/metrics"),
            ("http://localhost:9400/metrics", "http://localhost:9400/metrics"),
            ("http://localhost:9400/metrics/", "http://localhost:9400/metrics"),
            ("http://node1:8081", "http://node1:8081/metrics"),
            ("https://secure:443", "https://secure:443/metrics"),
            ("http://10.0.0.1:9090", "http://10.0.0.1:9090/metrics"),
        ],
    )
    def test_normalize_url_variations(self, input_url: str, expected: str):
        """Test URL normalization handles various input formats correctly."""
        assert normalize_metrics_endpoint_url(input_url) == expected

    def test_normalize_preserves_scheme(self):
        """Test that URL normalization preserves https scheme."""
        result = normalize_metrics_endpoint_url("https://secure:9400")
        assert result.startswith("https://")
        assert result == "https://secure:9400/metrics"

    def test_normalize_removes_trailing_slashes(self):
        """Test that multiple trailing slashes are removed."""
        result = normalize_metrics_endpoint_url("http://localhost:9400///")
        assert result == "http://localhost:9400/metrics"


class TestBuildHostnameAwarePrometheusEndpoints:
    """Test hostname-aware Prometheus endpoint URL generation."""

    def test_basic_endpoint_generation(self):
        """Test generating endpoints from inference URL and default ports."""
        endpoints = build_hostname_aware_prometheus_endpoints(
            "http://localhost:8000/v1/chat", [9400, 9401]
        )
        assert len(endpoints) == 2
        assert "http://localhost:9400/metrics" in endpoints
        assert "http://localhost:9401/metrics" in endpoints

    def test_preserves_scheme(self):
        """Test that generated endpoints use the same scheme as inference endpoint."""
        endpoints = build_hostname_aware_prometheus_endpoints(
            "https://secure-server:8000", [8081]
        )
        assert len(endpoints) == 1
        assert endpoints[0] == "https://secure-server:8081/metrics"

    def test_extracts_hostname_from_url(self):
        """Test hostname extraction from various URL formats."""
        test_cases = [
            ("http://node1:8000/v1/chat", [8081], "http://node1:8081/metrics"),
            ("http://10.0.0.5:8000", [9090], "http://10.0.0.5:9090/metrics"),
            (
                "https://api.example.com:443/inference",
                [8081],
                "https://api.example.com:8081/metrics",
            ),
        ]
        for inference_url, ports, expected_endpoint in test_cases:
            endpoints = build_hostname_aware_prometheus_endpoints(inference_url, ports)
            assert expected_endpoint in endpoints

    def test_multiple_ports(self):
        """Test generating multiple endpoints from multiple ports."""
        endpoints = build_hostname_aware_prometheus_endpoints(
            "http://server:8000", [8081, 6880, 9090]
        )
        assert len(endpoints) == 3
        assert "http://server:8081/metrics" in endpoints
        assert "http://server:6880/metrics" in endpoints
        assert "http://server:9090/metrics" in endpoints

    def test_empty_ports_list(self):
        """Test behavior with empty ports list."""
        endpoints = build_hostname_aware_prometheus_endpoints(
            "http://localhost:8000", []
        )
        assert len(endpoints) == 0

    def test_url_without_port(self):
        """Test handling URLs without explicit port."""
        endpoints = build_hostname_aware_prometheus_endpoints(
            "http://localhost/v1/chat", [8081]
        )
        assert len(endpoints) == 1
        assert "http://localhost:8081/metrics" in endpoints


class TestComputeHistogramDelta:
    """Test histogram delta computation for cumulative counters."""

    def test_normal_histogram_delta(self):
        """Test computing delta between two histogram snapshots."""
        start = {"0.1": 10.0, "0.5": 25.0, "1.0": 50.0, "+Inf": 100.0}
        end = {"0.1": 15.0, "0.5": 35.0, "1.0": 65.0, "+Inf": 120.0}

        delta = compute_histogram_delta(start, end)

        assert delta["0.1"] == 5.0
        assert delta["0.5"] == 10.0
        assert delta["1.0"] == 15.0
        assert delta["+Inf"] == 20.0

    def test_zero_delta(self):
        """Test delta computation when no change occurred."""
        start = {"0.1": 10.0, "0.5": 25.0, "1.0": 50.0}
        end = {"0.1": 10.0, "0.5": 25.0, "1.0": 50.0}

        delta = compute_histogram_delta(start, end)

        assert delta["0.1"] == 0.0
        assert delta["0.5"] == 0.0
        assert delta["1.0"] == 0.0

    def test_counter_reset_detection(self):
        """Test handling of counter resets (server restart)."""
        start = {"0.1": 100.0, "0.5": 250.0, "1.0": 500.0}
        end = {"0.1": 5.0, "0.5": 15.0, "1.0": 30.0}

        delta = compute_histogram_delta(start, end)

        assert delta["0.1"] == 5.0
        assert delta["0.5"] == 15.0
        assert delta["1.0"] == 30.0

    def test_mismatched_buckets_raises_error(self):
        """Test that mismatched bucket boundaries raise ValueError."""
        start = {"0.1": 10.0, "0.5": 25.0}
        end = {"0.1": 15.0, "1.0": 35.0}

        with pytest.raises(ValueError, match="bucket boundaries don't match"):
            compute_histogram_delta(start, end)

    def test_empty_histograms(self):
        """Test delta computation with empty histograms."""
        delta = compute_histogram_delta({}, {})
        assert delta == {}

    def test_fractional_values(self):
        """Test delta computation with fractional counter values."""
        start = {"0.1": 10.5, "0.5": 25.3}
        end = {"0.1": 12.7, "0.5": 28.1}

        delta = compute_histogram_delta(start, end)

        assert pytest.approx(delta["0.1"]) == 2.2
        assert pytest.approx(delta["0.5"]) == 2.8
