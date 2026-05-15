# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.metric_utils import normalize_metrics_endpoint_url


class TestNormalizeMetricsEndpointUrl:
    """Test URL normalization for metrics endpoints."""

    @pytest.mark.parametrize(
        "input_url,expected",
        [
            ("http://localhost:9400", "http://localhost:9400/metrics"),
            ("http://localhost:9400/", "http://localhost:9400/metrics"),
            ("http://localhost:9400/metrics", "http://localhost:9400/metrics"),
            ("http://localhost:9400/metrics/", "http://localhost:9400/metrics"),
            ("http://node1:8081", "http://node1:8081/metrics"),
            ("https://secure:443", "https://secure:443/metrics"),
            ("http://10.0.0.1:9090", "http://10.0.0.1:9090/metrics"),
            ("localhost:9400", "http://localhost:9400/metrics"),
            ("localhost:9400/", "http://localhost:9400/metrics"),
            ("localhost:9400/metrics", "http://localhost:9400/metrics"),
            ("10.0.0.1:9090", "http://10.0.0.1:9090/metrics"),
        ],
    )  # fmt: skip
    def test_normalize_url_variations(self, input_url: str, expected: str):
        """Test URL normalization handles various input formats correctly."""
        assert normalize_metrics_endpoint_url(input_url) == expected

    def test_normalize_injects_http_when_scheme_missing(self):
        """Scheme-less URLs get http:// prepended so downstream HTTP clients accept them."""
        assert (
            normalize_metrics_endpoint_url("localhost:9400")
            == "http://localhost:9400/metrics"
        )

    def test_normalize_does_not_double_prefix_https(self):
        """https:// URLs keep their scheme and are not coerced to http://."""
        assert (
            normalize_metrics_endpoint_url("https://secure:9400")
            == "https://secure:9400/metrics"
        )

    def test_normalize_preserves_scheme(self):
        """Test that URL normalization preserves https scheme."""
        result = normalize_metrics_endpoint_url("https://secure:9400")
        assert result.startswith("https://")
        assert result == "https://secure:9400/metrics"

    def test_normalize_removes_trailing_slashes(self):
        """Test that multiple trailing slashes are removed."""
        result = normalize_metrics_endpoint_url("http://localhost:9400///")
        assert result == "http://localhost:9400/metrics"
