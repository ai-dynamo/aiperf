# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for testing AIPerf post processors."""

from unittest.mock import Mock

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import EndpointType
from aiperf.common.models import (
    ErrorDetails,
    ParsedResponse,
    ParsedResponseRecord,
    RequestRecord,
    TextResponseData,
)
from aiperf.metrics.base_metric import BaseMetric
from aiperf.post_processors.metric_results_processor import MetricResultsProcessor

# Constants for test data
DEFAULT_START_TIME_NS = 1_000_000
DEFAULT_FIRST_RESPONSE_NS = 1_050_000
DEFAULT_LAST_RESPONSE_NS = 1_100_000
DEFAULT_INPUT_TOKENS = 5
DEFAULT_OUTPUT_TOKENS = 2


@pytest.fixture
def mock_user_config() -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.COMPLETIONS,
            streaming=False,
        )
    )


@pytest.fixture
def sample_request_record() -> RequestRecord:
    """Create a sample RequestRecord for testing."""
    return RequestRecord(
        conversation_id="test-conversation",
        turn_index=0,
        model_name="test-model",
        start_perf_ns=DEFAULT_START_TIME_NS,
        timestamp_ns=DEFAULT_START_TIME_NS,
        end_perf_ns=DEFAULT_LAST_RESPONSE_NS,
        error=None,
    )


@pytest.fixture
def sample_parsed_record(sample_request_record: RequestRecord) -> ParsedResponseRecord:
    """Create a valid ParsedResponseRecord for testing."""
    responses = [
        ParsedResponse(
            perf_ns=DEFAULT_FIRST_RESPONSE_NS,
            data=TextResponseData(text="Hello"),
        ),
        ParsedResponse(
            perf_ns=DEFAULT_LAST_RESPONSE_NS,
            data=TextResponseData(text=" world"),
        ),
    ]

    return ParsedResponseRecord(
        request=sample_request_record,
        responses=responses,
        input_token_count=DEFAULT_INPUT_TOKENS,
        output_token_count=DEFAULT_OUTPUT_TOKENS,
    )


@pytest.fixture
def error_parsed_record() -> ParsedResponseRecord:
    """Create an error ParsedResponseRecord for testing."""
    error_details = ErrorDetails(code=500, message="Internal server error")

    request = RequestRecord(
        conversation_id="test-conversation-error",
        turn_index=0,
        model_name="test-model",
        start_perf_ns=DEFAULT_START_TIME_NS,
        timestamp_ns=DEFAULT_START_TIME_NS,
        end_perf_ns=DEFAULT_START_TIME_NS,
        error=error_details,
    )

    return ParsedResponseRecord(
        request=request,
        responses=[],
        input_token_count=None,
        output_token_count=None,
    )


def setup_mock_registry_for_metrics(
    mock_registry: Mock, metric_types: list[type[BaseMetric]]
) -> list[str]:
    """Setup mock registry for metric types, creating instances automatically.

    Args:
        mock_registry: The mock registry to configure
        metric_types: list of metric class types to configure

    Returns:
        list of metric tags in the same order as input
    """
    metric_tags = [metric_type.tag for metric_type in metric_types]
    metric_instances = {metric_type.tag: metric_type() for metric_type in metric_types}

    mock_registry.tags_applicable_to.return_value = metric_tags
    mock_registry.create_dependency_order_for.return_value = metric_tags
    mock_registry.get_instance.side_effect = lambda tag: metric_instances[tag]

    return metric_tags


def setup_mock_registry_sequences(
    mock_registry: Mock,
    valid_metric_types: list[type[BaseMetric]],
    error_metric_types: list[type[BaseMetric]],
) -> tuple[list[str], list[str]]:
    """Setup mock registry for processors that need both valid and error metrics.

    Args:
        mock_registry: The mock registry to configure
        valid_metric_types: list of valid metric class types
        error_metric_types: list of error metric class types

    Returns:
        tuple of (valid_tags, error_tags)
    """
    valid_tags = [metric_type.tag for metric_type in valid_metric_types]
    error_tags = [metric_type.tag for metric_type in error_metric_types]

    # Create lookup map for all metric instances
    all_metric_instances = {
        metric_type.tag: metric_type()
        for metric_type in valid_metric_types + error_metric_types
    }

    mock_registry.tags_applicable_to.side_effect = [valid_tags, error_tags]
    mock_registry.create_dependency_order_for.side_effect = [valid_tags, error_tags]
    mock_registry.get_instance.side_effect = lambda tag: all_metric_instances[tag]

    return valid_tags, error_tags


def create_results_processor_with_metrics(
    user_config: UserConfig, *metrics: type[BaseMetric]
) -> MetricResultsProcessor:
    """Create a MetricResultsProcessor with pre-configured metrics.

    Args:
        user_config: User configuration for the processor
        metrics: list of metric classes

    Returns:
        Configured MetricResultsProcessor instance
    """

    processor = MetricResultsProcessor(user_config)
    processor._tags_to_types = {metric.tag: metric.type for metric in metrics}
    processor._instances_map = {metric.tag: metric() for metric in metrics}
    return processor


@pytest.fixture
def mock_metric_registry(monkeypatch):
    """Provide a unified mocked MetricRegistry that represents the singleton properly.

    Uses monkeypatch to inject the same mock instance at all import locations,
    ensuring consistent singleton behavior across the entire test.
    """
    mock_registry = Mock()
    mock_registry.tags_applicable_to.return_value = []
    mock_registry.create_dependency_order_for.return_value = []
    mock_registry.get_instance.return_value = Mock()
    mock_registry.all_classes.return_value = []
    mock_registry.all_tags.return_value = []

    monkeypatch.setattr("aiperf.metrics.metric_registry.MetricRegistry", mock_registry)
    monkeypatch.setattr(
        "aiperf.post_processors.base_metrics_processor.MetricRegistry", mock_registry
    )
    monkeypatch.setattr(
        "aiperf.post_processors.metric_results_processor.MetricRegistry", mock_registry
    )

    return mock_registry
