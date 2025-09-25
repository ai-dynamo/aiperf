# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from aiperf.common.config import UserConfig
from aiperf.common.enums import EndpointType, MetricFlags, MetricType
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor
from tests.post_processors.conftest import (
    setup_mock_registry_for_metrics,
)


class TestBaseMetricsProcessor:
    """Test cases for BaseMetricsProcessor."""

    def test_initialization(self, mock_user_config: UserConfig) -> None:
        """Test processor initialization stores user config."""
        processor = BaseMetricsProcessor(mock_user_config)
        assert processor.user_config == mock_user_config

    @pytest.mark.parametrize(
        "endpoint_type,streaming,expected_supported_flags",
        [
            # Completions endpoint - supports tokens but streaming disabled
            (EndpointType.COMPLETIONS, False, [MetricFlags.PRODUCES_TOKENS_ONLY]),
            # Completions with streaming enabled - supports tokens and streaming
            (
                EndpointType.COMPLETIONS,
                True,
                [MetricFlags.PRODUCES_TOKENS_ONLY, MetricFlags.STREAMING_ONLY],
            ),
            # Embeddings endpoint - no tokens or streaming
            (EndpointType.EMBEDDINGS, False, []),
            # Chat endpoint without streaming - supports tokens, audio, and images but not streaming
            (
                EndpointType.CHAT,
                False,
                [
                    MetricFlags.PRODUCES_TOKENS_ONLY,
                    MetricFlags.SUPPORTS_AUDIO_ONLY,
                    MetricFlags.SUPPORTS_IMAGE_ONLY,
                ],
            ),
            # Chat endpoint with streaming - supports all capabilities
            (
                EndpointType.CHAT,
                True,
                [
                    MetricFlags.PRODUCES_TOKENS_ONLY,
                    MetricFlags.STREAMING_ONLY,
                    MetricFlags.SUPPORTS_AUDIO_ONLY,
                    MetricFlags.SUPPORTS_IMAGE_ONLY,
                ],
            ),
        ],
    )
    def test_get_filters(
        self,
        mock_user_config: UserConfig,
        endpoint_type: EndpointType,
        streaming: bool,
        expected_supported_flags: list[MetricFlags],
    ) -> None:
        """Test filter generation based on endpoint capabilities."""
        mock_user_config.endpoint.type = endpoint_type
        mock_user_config.endpoint.streaming = streaming

        processor = BaseMetricsProcessor(mock_user_config)
        required_flags, disallowed_flags = processor.get_filters()

        assert required_flags == MetricFlags.NONE

        # Check that expected supported flags are NOT in disallowed flags
        for supported_flag in expected_supported_flags:
            assert not (disallowed_flags & supported_flag), (
                f"Expected supported flag {supported_flag} should not be disallowed"
            )

    def test_setup_metrics_basic(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
    ) -> None:
        """Test basic metric setup."""
        metric_types = [RequestLatencyMetric, RequestCountMetric]
        metric_tags = setup_mock_registry_for_metrics(
            mock_metric_registry, metric_types
        )

        processor = BaseMetricsProcessor(mock_user_config)
        metrics = processor._setup_metrics(MetricType.RECORD, MetricType.AGGREGATE)

        assert len(metrics) == len(metric_tags)
        assert metrics[0].tag == RequestLatencyMetric.tag
        assert metrics[1].tag == RequestCountMetric.tag

        expected_required = MetricFlags.NONE
        expected_disallowed = (
            MetricFlags.SUPPORTS_AUDIO_ONLY
            | MetricFlags.SUPPORTS_IMAGE_ONLY
            | MetricFlags.STREAMING_ONLY
        )

        # Verify registry calls
        mock_metric_registry.tags_applicable_to.assert_called_once_with(
            expected_required,
            expected_disallowed,
            MetricType.RECORD,
            MetricType.AGGREGATE,
        )
        mock_metric_registry.create_dependency_order_for.assert_called_once_with(
            metric_tags
        )

    @pytest.mark.parametrize(
        "error_metrics_only,exclude_error_metrics,expected_required,expected_disallowed",
        [
            # Test error_metrics_only=True
            (
                True,
                False,
                MetricFlags.ERROR_ONLY,
                MetricFlags.SUPPORTS_AUDIO_ONLY
                | MetricFlags.SUPPORTS_IMAGE_ONLY
                | MetricFlags.STREAMING_ONLY,
            ),
            # Test exclude_error_metrics=True
            (
                False,
                True,
                MetricFlags.NONE,
                MetricFlags.SUPPORTS_AUDIO_ONLY
                | MetricFlags.SUPPORTS_IMAGE_ONLY
                | MetricFlags.STREAMING_ONLY
                | MetricFlags.ERROR_ONLY,
            ),
            # Test both flags (error_metrics_only takes precedence)
            (
                True,
                True,
                MetricFlags.ERROR_ONLY,
                MetricFlags.SUPPORTS_AUDIO_ONLY
                | MetricFlags.SUPPORTS_IMAGE_ONLY
                | MetricFlags.STREAMING_ONLY,
            ),
        ],
    )
    def test_setup_metrics_error_flag_scenarios(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
        error_metrics_only: bool,
        exclude_error_metrics: bool,
        expected_required: MetricFlags,
        expected_disallowed: MetricFlags,
    ) -> None:
        """Test metric setup with various error flag combinations."""
        metric_type = (
            ErrorRequestCountMetric if error_metrics_only else RequestLatencyMetric
        )
        metric_tags = setup_mock_registry_for_metrics(
            mock_metric_registry, [metric_type]
        )

        processor = BaseMetricsProcessor(mock_user_config)
        metrics = processor._setup_metrics(
            MetricType.RECORD,
            error_metrics_only=error_metrics_only,
            exclude_error_metrics=exclude_error_metrics,
        )

        assert len(metrics) == len(metric_tags)
        assert isinstance(metrics[0], metric_type)

        mock_metric_registry.tags_applicable_to.assert_called_once_with(
            expected_required,
            expected_disallowed,
            MetricType.RECORD,
        )

    def test_setup_metrics_empty_result(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test metric setup when no applicable metrics found."""
        processor = BaseMetricsProcessor(mock_user_config)
        metrics = processor._setup_metrics(MetricType.RECORD)

        assert metrics == []

    def test_setup_metrics_multiple_types(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
    ) -> None:
        """Test metric setup with multiple metric types."""
        metric_tags = setup_mock_registry_for_metrics(
            mock_metric_registry,
            [
                RequestLatencyMetric,
                RequestCountMetric,
                RequestThroughputMetric,
            ],
        )

        processor = BaseMetricsProcessor(mock_user_config)
        metrics = processor._setup_metrics(
            MetricType.RECORD, MetricType.AGGREGATE, MetricType.DERIVED
        )

        assert len(metrics) == len(metric_tags)

        mock_metric_registry.tags_applicable_to.assert_called_once_with(
            MetricFlags.NONE,
            MetricFlags.SUPPORTS_AUDIO_ONLY
            | MetricFlags.SUPPORTS_IMAGE_ONLY
            | MetricFlags.STREAMING_ONLY,
            MetricType.RECORD,
            MetricType.AGGREGATE,
            MetricType.DERIVED,
        )
