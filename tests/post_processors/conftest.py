# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for testing AIPerf post processors."""

from unittest.mock import Mock

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import CreditPhase, EndpointType, MessageType
from aiperf.common.enums.metric_enums import MetricValueTypeT
from aiperf.common.messages import MetricRecordsMessage
from aiperf.common.models import (
    ErrorDetails,
    ParsedResponse,
    ParsedResponseRecord,
    RequestRecord,
    TextResponseData,
)
from aiperf.common.models.record_models import MetricRecordMetadata
from aiperf.common.types import MetricTagT
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


def create_metric_metadata(
    conversation_id: str | None = None,
    turn_index: int = 0,
    timestamp_ns: int = 1_000_000_000,
    worker_id: str = "worker-1",
    record_processor_id: str = "processor-1",
    credit_phase: CreditPhase = CreditPhase.PROFILING,
    x_request_id: str | None = None,
    x_correlation_id: str | None = None,
    error: ErrorDetails | None = None,
) -> MetricRecordMetadata:
    """
    Create a MetricRecordMetadata object with sensible defaults.

    Args:
        conversation_id: Conversation ID (optional)
        turn_index: Turn index in conversation
        timestamp_ns: Timestamp in nanoseconds
        worker_id: Worker ID
        record_processor_id: Record processor ID
        credit_phase: Credit phase
        x_request_id: X-Request-ID header value (optional)
        x_correlation_id: X-Correlation-ID header value (optional)
        error: Error details if any

    Returns:
        MetricRecordMetadata object
    """
    return MetricRecordMetadata(
        conversation_id=conversation_id,
        turn_index=turn_index,
        timestamp_ns=timestamp_ns,
        worker_id=worker_id,
        record_processor_id=record_processor_id,
        credit_phase=credit_phase,
        x_request_id=x_request_id,
        x_correlation_id=x_correlation_id,
        error=error,
    )


def create_metric_records_message(
    service_id: str = "test-processor",
    results: list[dict[MetricTagT, MetricValueTypeT]] | None = None,
    error: ErrorDetails | None = None,
    metadata: MetricRecordMetadata | None = None,
    x_request_id: str | None = None,
    **metadata_kwargs,
) -> MetricRecordsMessage:
    """
    Create a MetricRecordsMessage with sensible defaults.

    Args:
        service_id: Service ID
        results: List of metric result dictionaries
        error: Error details if any
        metadata: Pre-built metadata, or None to build from kwargs
        x_request_id: Record ID (will be set as x_request_id in metadata if provided)
        **metadata_kwargs: Arguments to pass to create_metric_metadata if metadata is None

    Returns:
        MetricRecordsMessage object
    """
    if results is None:
        results = []

    if metadata is None:
        # If x_request_id is provided, use it as x_request_id
        if x_request_id is not None and "x_request_id" not in metadata_kwargs:
            metadata_kwargs["x_request_id"] = x_request_id
        metadata = create_metric_metadata(**metadata_kwargs)

    return MetricRecordsMessage(
        message_type=MessageType.METRIC_RECORDS,
        service_id=service_id,
        metadata=metadata,
        results=results,
        error=error,
    )
