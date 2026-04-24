# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator models module."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.operator.models import AIPerfJobSpec, K8sEndpointConfig, MetricsSummary

# =============================================================================
# Test MetricsSummary
# =============================================================================


class TestMetricsSummary:
    """Tests for MetricsSummary model."""

    def test_creates_empty_summary(self) -> None:
        """Verify creates empty summary with all None values."""
        summary = MetricsSummary()
        assert summary.throughput_rps is None
        assert summary.latency_avg_ms is None
        assert summary.total_requests is None

    def test_from_metrics_with_empty_dict(self) -> None:
        """Verify returns empty summary for empty metrics dict."""
        summary = MetricsSummary.from_metrics({})
        assert summary.throughput_rps is None

    def test_from_metrics_with_none(self) -> None:
        """Verify handles None input."""
        summary = MetricsSummary.from_metrics(None)  # type: ignore
        assert summary.throughput_rps is None

    def test_from_metrics_extracts_throughput(self) -> None:
        """Verify extracts throughput metrics."""
        metrics = {
            "metrics": [
                {"tag": "request_throughput", "avg": 100.5},
            ]
        }
        summary = MetricsSummary.from_metrics(metrics)
        assert summary.throughput_rps == 100.5

    def test_from_metrics_extracts_token_throughput(self) -> None:
        """Verify extracts token throughput."""
        metrics = {
            "metrics": [
                {"tag": "output_token_throughput", "avg": 500.0},
            ]
        }
        summary = MetricsSummary.from_metrics(metrics)
        assert summary.throughput_tps == 500.0

    def test_from_metrics_extracts_latency_stats(self) -> None:
        """Verify extracts latency percentiles."""
        metrics = {
            "metrics": [
                {"tag": "request_latency", "avg": 50.0, "p50": 45.0, "p99": 120.0},
            ]
        }
        summary = MetricsSummary.from_metrics(metrics)
        assert summary.latency_avg_ms == 50.0
        assert summary.latency_p50_ms == 45.0
        assert summary.latency_p99_ms == 120.0

    def test_from_metrics_extracts_ttft(self) -> None:
        """Verify extracts time to first token."""
        metrics = {
            "metrics": [
                {"tag": "time_to_first_token", "avg": 100.0, "p50": 90.0, "p99": 200.0},
            ]
        }
        summary = MetricsSummary.from_metrics(metrics)
        assert summary.ttft_avg_ms == 100.0
        assert summary.ttft_p50_ms == 90.0
        assert summary.ttft_p99_ms == 200.0

    def test_from_metrics_extracts_total_requests(self) -> None:
        """Verify extracts total requests count."""
        metrics = {
            "metrics": [
                {"tag": "total_requests", "avg": 1000.0},
            ]
        }
        summary = MetricsSummary.from_metrics(metrics)
        assert summary.total_requests == 1000

    def test_from_metrics_extracts_error_rate(self) -> None:
        """Verify extracts error rate."""
        metrics = {
            "metrics": [
                {"tag": "error_rate", "avg": 0.05},
            ]
        }
        summary = MetricsSummary.from_metrics(metrics)
        assert summary.error_rate == 0.05

    def test_from_metrics_uses_value_field(self) -> None:
        """Verify falls back to 'value' field when stat not present."""
        metrics = {
            "metrics": [
                {"tag": "request_throughput", "value": 200.0},
            ]
        }
        summary = MetricsSummary.from_metrics(metrics)
        assert summary.throughput_rps == 200.0

    def test_from_metrics_handles_total_requests_zero(self) -> None:
        """Verify handles zero total requests."""
        metrics = {
            "metrics": [
                {"tag": "total_requests", "avg": 0},
            ]
        }
        summary = MetricsSummary.from_metrics(metrics)
        assert summary.total_requests is None

    def test_to_status_dict_excludes_none(self) -> None:
        """Verify to_status_dict excludes None values."""
        summary = MetricsSummary(throughput_rps=100.0, latency_avg_ms=50.0)
        result = summary.to_status_dict()

        assert "throughput_rps" in result
        assert "latency_avg_ms" in result
        assert "throughput_tps" not in result
        assert "error_rate" not in result

    def test_to_status_dict_empty_for_all_none(self) -> None:
        """Verify returns empty dict when all values are None."""
        summary = MetricsSummary()
        result = summary.to_status_dict()
        assert result == {}


# =============================================================================
# Test K8sEndpointConfig
# =============================================================================


class TestEndpointConfig:
    """Tests for K8sEndpointConfig model."""

    def test_creates_with_valid_url(self) -> None:
        """Verify creates with valid HTTP URL."""
        config = K8sEndpointConfig(url="http://localhost:8000")
        assert config.url == "http://localhost:8000"

    def test_creates_with_https_url(self) -> None:
        """Verify creates with HTTPS URL."""
        config = K8sEndpointConfig(url="https://api.example.com/v1")
        assert config.url == "https://api.example.com/v1"

    def test_creates_with_model_and_api_type(self) -> None:
        """Verify creates with optional fields."""
        config = K8sEndpointConfig(
            url="http://localhost:8000",
            model="gpt-4",
            api_type="openai",
        )
        assert config.model == "gpt-4"
        assert config.api_type == "openai"

    def test_default_api_type(self) -> None:
        """Verify default api_type is openai."""
        config = K8sEndpointConfig(url="http://localhost:8000")
        assert config.api_type == "openai"

    @pytest.mark.parametrize(
        "url,error_msg",
        [
            param("", "Endpoint URL is required", id="empty"),
            param("localhost:8000", "must start with http://", id="no_scheme"),
            param("ftp://example.com", "must start with http://", id="wrong_scheme"),
        ],
    )  # fmt: skip
    def test_rejects_invalid_url(self, url: str, error_msg: str) -> None:
        """Verify rejects invalid URLs."""
        with pytest.raises(ValidationError) as exc_info:
            K8sEndpointConfig(url=url)
        assert error_msg in str(exc_info.value)


# =============================================================================
# Test AIPerfJobSpec
# =============================================================================


class TestAIPerfJobSpec:
    """Tests for AIPerfJobSpec model."""

    @pytest.fixture
    def valid_spec(self) -> dict[str, Any]:
        """Create a valid nested spec dict."""
        return {
            "image": "aiperf:latest",
            "benchmark": {"endpoint": {"url": "http://localhost:8000"}},
        }

    def test_creates_with_minimal_config(self, valid_spec: dict[str, Any]) -> None:
        """Verify creates with minimal valid configuration."""
        spec = AIPerfJobSpec.from_crd_spec(valid_spec)
        assert spec.image == "aiperf:latest"
        assert spec.image_pull_policy is None
        assert spec.cancel is False

    def test_creates_with_full_config(self) -> None:
        """Verify creates with all optional fields."""
        spec = AIPerfJobSpec.from_crd_spec(
            {
                "image": "aiperf:v1.0",
                "imagePullPolicy": "Always",
                "ttlSecondsAfterFinished": 3600,
                "resultsTtlDays": 30,
                "cancel": True,
                "benchmark": {"endpoint": {"url": "http://localhost:8000"}},
            }
        )
        assert spec.image == "aiperf:v1.0"
        assert spec.image_pull_policy == "Always"
        assert spec.ttl_seconds_after_finished == 3600
        assert spec.results_ttl_days == 30
        assert spec.cancel is True

    @pytest.mark.parametrize(
        "image",
        [
            param("", id="empty"),
            param("   ", id="whitespace"),
        ],
    )  # fmt: skip
    def test_rejects_empty_image(self, image: str) -> None:
        """Verify rejects empty image."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfJobSpec.from_crd_spec(
                {
                    "image": image,
                    "benchmark": {"endpoint": {"url": "http://localhost:8000"}},
                }
            )
        assert "Image is required" in str(exc_info.value)

    @pytest.mark.parametrize(
        "policy",
        [
            param("invalid", id="invalid"),
            param("", id="empty"),
        ],
    )  # fmt: skip
    def test_rejects_invalid_pull_policy(self, policy: str) -> None:
        """Verify rejects invalid imagePullPolicy."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfJobSpec.from_crd_spec(
                {
                    "image": "aiperf:latest",
                    "imagePullPolicy": policy,
                    "benchmark": {"endpoint": {"url": "http://localhost:8000"}},
                }
            )
        assert "image_pull_policy" in str(exc_info.value)

    def test_rejects_missing_endpoint(self) -> None:
        """Verify rejects spec with no endpoint in benchmark."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfJobSpec.from_crd_spec(
                {
                    "image": "aiperf:latest",
                    "benchmark": {},
                }
            )
        assert "endpoint.url or endpoint.urls is required" in str(exc_info.value)

    def test_rejects_missing_endpoint_url(self) -> None:
        """Verify rejects endpoint without url or urls."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfJobSpec.from_crd_spec(
                {
                    "image": "aiperf:latest",
                    "benchmark": {"endpoint": {"type": "openai"}},
                }
            )
        assert "endpoint.url or endpoint.urls is required" in str(exc_info.value)

    def test_accepts_urls_array(self) -> None:
        """Verify accepts urls array instead of url."""
        spec = AIPerfJobSpec.from_crd_spec(
            {
                "image": "aiperf:latest",
                "benchmark": {
                    "endpoint": {
                        "urls": ["http://localhost:8000", "http://localhost:8001"]
                    }
                },
            }
        )
        assert spec.get_endpoint_url() == "http://localhost:8000"

    def test_get_endpoint_url_from_url(self, valid_spec: dict[str, Any]) -> None:
        """Verify get_endpoint_url extracts URL."""
        spec = AIPerfJobSpec.from_crd_spec(valid_spec)
        assert spec.get_endpoint_url() == "http://localhost:8000"

    def test_get_endpoint_url_from_urls_array(self) -> None:
        """Verify get_endpoint_url extracts first URL from array."""
        spec = AIPerfJobSpec.from_crd_spec(
            {
                "image": "aiperf:latest",
                "benchmark": {
                    "endpoint": {"urls": ["http://first:8000", "http://second:8000"]}
                },
            }
        )
        assert spec.get_endpoint_url() == "http://first:8000"

    def test_get_endpoint_url_prefers_url_over_urls(self) -> None:
        """Verify get_endpoint_url prefers url over urls array."""
        spec = AIPerfJobSpec.from_crd_spec(
            {
                "image": "aiperf:latest",
                "benchmark": {
                    "endpoint": {
                        "url": "http://primary:8000",
                        "urls": ["http://backup:8000"],
                    }
                },
            }
        )
        assert spec.get_endpoint_url() == "http://primary:8000"

    def test_get_endpoint_url_with_empty_urls(self) -> None:
        """Verify handles empty urls array gracefully."""
        spec = AIPerfJobSpec.from_crd_spec(
            {
                "image": "aiperf:latest",
                "benchmark": {"endpoint": {"url": "http://localhost:8000", "urls": []}},
            }
        )
        assert spec.get_endpoint_url() == "http://localhost:8000"

    @pytest.mark.parametrize(
        "pull_policy",
        [
            param("Always", id="always"),
            param("IfNotPresent", id="if_not_present"),
            param("Never", id="never"),
        ],
    )  # fmt: skip
    def test_accepts_valid_pull_policies(self, pull_policy: str) -> None:
        """Verify accepts all valid pull policies."""
        spec = AIPerfJobSpec.from_crd_spec(
            {
                "image": "aiperf:latest",
                "imagePullPolicy": pull_policy,
                "benchmark": {"endpoint": {"url": "http://localhost:8000"}},
            }
        )
        assert spec.image_pull_policy == pull_policy
