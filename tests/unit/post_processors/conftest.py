# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for testing AIPerf post processors."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import Mock

import pytest

from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.enums import (
    CreditPhase,
    ExportLevel,
    MessageType,
    MetricValueTypeT,
    ModelSelectionStrategy,
)
from aiperf.common.messages import MetricRecordsMessage
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    ErrorDetails,
    ParsedResponse,
    ParsedResponseRecord,
    RequestInfo,
    RequestRecord,
    TelemetryMetrics,
    TelemetryRecord,
    TextResponse,
)
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import (
    MetricRecordMetadata,
    ProfileResults,
    TokenCounts,
)
from aiperf.common.types import MetricTagT
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.base_metric import BaseMetric
from aiperf.plugin.enums import EndpointType
from aiperf.post_processors.metric_results_processor import MetricResultsProcessor
from aiperf.post_processors.raw_record_writer_processor import RawRecordWriterProcessor
from tests.unit.conftest import (
    DEFAULT_FIRST_RESPONSE_NS,
    DEFAULT_INPUT_TOKENS,
    DEFAULT_LAST_RESPONSE_NS,
    DEFAULT_OUTPUT_TOKENS,
    DEFAULT_START_TIME_NS,
)

T = TypeVar("T", bound=AIPerfLifecycleMixin)


@asynccontextmanager
async def aiperf_lifecycle(instance: T) -> T:
    """Generic async context manager for any AIPerfLifecycleMixin lifecycle.

    Handles initialize, start, and stop automatically for any component
    implementing the AIPerfLifecycleMixin interface.

    Usage:
        async with aiperf_lifecycle(processor) as proc:
            await proc.process_record(record, metadata)
    """
    await instance.initialize()
    await instance.start()
    try:
        yield instance
    finally:
        await instance.stop()


@asynccontextmanager
async def raw_record_processor(service_id: str, user_config: UserConfig):
    """Async context manager for RawRecordWriterProcessor lifecycle.

    Handles initialize, start, and stop automatically.

    Usage:
        async with raw_record_processor("processor-1", user_config) as processor:
            await processor.process_record(record, metadata)
    """

    processor = RawRecordWriterProcessor(
        service_id=service_id,
        user_config=user_config,
    )
    async with aiperf_lifecycle(processor) as proc:
        yield proc


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
def user_config_raw(tmp_artifact_dir: Path) -> UserConfig:
    """Create a UserConfig for raw record testing."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            streaming=False,
        ),
        output=OutputConfig(
            artifact_directory=tmp_artifact_dir,
            export_level=ExportLevel.RAW,
        ),
    )


def _create_test_request_info(
    model_name: str = "test-model",
    conversation_id: str = "test-conversation",
    turn_index: int = 0,
    turns: list | None = None,
) -> RequestInfo:
    """Create a RequestInfo for testing post processors."""
    return RequestInfo(
        model_endpoint=ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name=model_name)],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000/v1/test",
            ),
        ),
        turns=turns or [],
        turn_index=turn_index,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="test-request-id",
        x_correlation_id="test-correlation-id",
        conversation_id=conversation_id,
    )


@pytest.fixture
def sample_parsed_record_with_raw_responses() -> ParsedResponseRecord:
    """Create a sample ParsedResponseRecord with raw responses for raw record testing.

    This fixture includes raw TextResponse objects in the request, which is needed
    for raw record serialization tests.
    """
    from aiperf.common.models import Text, TextResponseData, Turn

    turns = [
        Turn(
            texts=[Text(contents=["Hello, how are you?"])],
            role="user",
            model="test-model",
        )
    ]

    raw_responses = [
        TextResponse(text="Hello", perf_ns=DEFAULT_FIRST_RESPONSE_NS),
        TextResponse(text=" world", perf_ns=DEFAULT_LAST_RESPONSE_NS),
    ]

    request = RequestRecord(
        request_info=_create_test_request_info(
            conversation_id="conv-123",
            turns=turns,
        ),
        model_name="test-model",
        start_perf_ns=DEFAULT_START_TIME_NS,
        timestamp_ns=DEFAULT_START_TIME_NS,
        end_perf_ns=DEFAULT_LAST_RESPONSE_NS,
        status=200,
        request_headers={"Content-Type": "application/json"},
        responses=raw_responses,
        error=None,
    )

    parsed_responses = [
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
        request=request,
        responses=parsed_responses,
        token_counts=TokenCounts(
            input=DEFAULT_INPUT_TOKENS,
            output=DEFAULT_OUTPUT_TOKENS,
            reasoning=None,
        ),
    )


@pytest.fixture
def error_parsed_record() -> ParsedResponseRecord:
    """Create an error ParsedResponseRecord for testing."""
    from aiperf.common.models import Text, Turn

    error_details = ErrorDetails(code=500, message="Internal server error")

    turns = [
        Turn(
            texts=[Text(contents=["This will fail"])],
            role="user",
            model="test-model",
        )
    ]

    request = RequestRecord(
        request_info=_create_test_request_info(
            conversation_id="test-conversation-error",
            turns=turns,
        ),
        model_name="test-model",
        start_perf_ns=DEFAULT_START_TIME_NS,
        timestamp_ns=DEFAULT_START_TIME_NS,
        end_perf_ns=DEFAULT_START_TIME_NS,
        status=500,
        error=error_details,
    )

    return ParsedResponseRecord(
        request=request,
        responses=[],
        token_counts=TokenCounts(
            input=None,
            output=None,
            reasoning=None,
        ),
    )


def create_exporter_config(user_config: UserConfig) -> ExporterConfig:
    """Helper to create standard ExporterConfig for aggregator tests."""
    return ExporterConfig(
        user_config=user_config,
        service_config=ServiceConfig(),
        results=ProfileResults(
            records=None,
            completed=0,
            start_ns=DEFAULT_START_TIME_NS,
            end_ns=DEFAULT_LAST_RESPONSE_NS,
        ),
        telemetry_results=None,
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
    session_num: int = 0,
    conversation_id: str | None = None,
    turn_index: int = 0,
    request_start_ns: int = 1_000_000_000,
    request_ack_ns: int | None = None,
    request_end_ns: int = 1_100_000_000,
    worker_id: str = "worker-1",
    record_processor_id: str = "processor-1",
    benchmark_phase: CreditPhase = CreditPhase.PROFILING,
    x_request_id: str | None = None,
    x_correlation_id: str | None = None,
) -> MetricRecordMetadata:
    """
    Create a MetricRecordMetadata object with sensible defaults.

    Args:
        session_num: Sequential session number in the benchmark
        conversation_id: Conversation ID (optional)
        turn_index: Turn index in conversation
        request_start_ns: Request start timestamp in nanoseconds
        request_ack_ns: Request acknowledgement timestamp in nanoseconds (optional)
        request_end_ns: Request end timestamp in nanoseconds (optional)
        worker_id: Worker ID
        record_processor_id: Record processor ID
        benchmark_phase: Benchmark phase (warmup or profiling)
        x_request_id: X-Request-ID header value (optional)
        x_correlation_id: X-Correlation-ID header value (optional)

    Returns:
        MetricRecordMetadata object
    """
    return MetricRecordMetadata(
        session_num=session_num,
        conversation_id=conversation_id,
        turn_index=turn_index,
        request_start_ns=request_start_ns,
        request_ack_ns=request_ack_ns,
        request_end_ns=request_end_ns,
        worker_id=worker_id,
        record_processor_id=record_processor_id,
        benchmark_phase=benchmark_phase,
        x_request_id=x_request_id,
        x_correlation_id=x_correlation_id,
    )


def create_metric_records_message(
    service_id: str = "test-processor",
    results: list[dict[MetricTagT, MetricValueTypeT]] | None = None,
    error: ErrorDetails | None = None,
    metadata: MetricRecordMetadata | None = None,
    x_request_id: str | None = None,
    trace_data: Any | None = None,
    **metadata_kwargs,
) -> MetricRecordsMessage:
    """
    Create a MetricRecordsMessage with sensible defaults.

    Args:
        service_id: Service ID
        results: List of metric result dictionaries
        error: Error details if any
        metadata: Pre-built metadata, or None to build from kwargs
        x_request_id: Record ID (set as x_request_id in metadata if provided)
        trace_data: HTTP trace data for the request (optional)
        **metadata_kwargs: Args passed to create_metric_metadata if metadata is None

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
        trace_data=trace_data,
    )


def make_telemetry_record(
    *,
    timestamp_ns: int = 1_000_000_000,
    dcgm_url: str = "http://node1:9401/metrics",
    gpu_index: int = 0,
    gpu_uuid: str = "GPU-test",
    gpu_model_name: str = "Test GPU",
    hostname: str = "node1",
    pci_bus_id: str | None = None,
    device: str | None = None,
    gpu_power_usage: float | None = 100.0,
    gpu_utilization: float | None = None,
    energy_consumption: float | None = None,
    gpu_memory_used: float | None = None,
    gpu_temperature: float | None = None,
    xid_errors: float | None = None,
    power_violation: float | None = None,
) -> TelemetryRecord:
    """Factory for creating TelemetryRecord instances with sensible defaults."""
    return TelemetryRecord(
        timestamp_ns=timestamp_ns,
        dcgm_url=dcgm_url,
        gpu_index=gpu_index,
        gpu_uuid=gpu_uuid,
        gpu_model_name=gpu_model_name,
        hostname=hostname,
        pci_bus_id=pci_bus_id,
        device=device,
        telemetry_data=TelemetryMetrics(
            gpu_power_usage=gpu_power_usage,
            gpu_utilization=gpu_utilization,
            energy_consumption=energy_consumption,
            gpu_memory_used=gpu_memory_used,
            gpu_temperature=gpu_temperature,
            xid_errors=xid_errors,
            power_violation=power_violation,
        ),
    )
