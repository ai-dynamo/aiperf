# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for testing AIPerf metrics.

"""

from aiperf.common.enums import (
    AggregationKind,
    CreditPhase,
    MetricType,
    ModelSelectionStrategy,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import (
    ErrorDetails,
    ParsedResponse,
    ParsedResponseRecord,
    RequestInfo,
    RequestRecord,
)
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import TextResponseData, TokenCounts
from aiperf.common.types import MetricTagT
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.plugin.enums import EndpointType


def _create_test_request_info(model_name: str = "test-model") -> RequestInfo:
    """Create a RequestInfo for testing metrics."""
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
        turns=[],
        turn_index=0,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="test-request-id",
        x_correlation_id="test-correlation-id",
        conversation_id="test-conversation",
    )


def create_record(
    start_ns: int = 100,
    responses: list[int] | None = None,
    input_tokens: int | None = None,
    output_tokens_per_response: int = 1,
    error: ErrorDetails | None = None,
) -> ParsedResponseRecord:
    """
    Simple helper to create test records with sensible defaults.

    If no responses are provided, a single response is created with a latency of 50ns.
    If output tokens per response are provided, each response will have that many tokens,
        meaning that the total output token count will be the number of responses times
        the output tokens per response.
    The end_perf_ns is the last response timestamp, or the start_ns if no responses are provided.
    """
    responses = responses or [start_ns + 50]  # Single response 50ns later

    request = RequestRecord(
        request_info=_create_test_request_info(),
        model_name="test-model",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=responses[-1] if responses else start_ns,
        error=error,
    )

    response_data = []
    for perf_ns in responses:
        response_data.append(
            ParsedResponse(
                perf_ns=perf_ns,
                data=TextResponseData(text="test"),
            )
        )

    return ParsedResponseRecord(
        request=request,
        responses=response_data,
        token_counts=TokenCounts(
            input=input_tokens,
            output=len(responses) * output_tokens_per_response,
            reasoning=None,
        ),
    )


def run_simple_metrics_pipeline(
    records: list[ParsedResponseRecord], *metrics_to_test: MetricTagT
) -> MetricResultsDict:
    """Run a simple metrics triple stage pipeline for a list of records and return the results.

    This function will:
    - Determine all of the metrics that are needed to compute the metrics to test
    - Sort all of the metrics by dependency order and create an instance of each
    - Parse the input metrics for each record
    - Aggregate the values of the aggregate metrics
    - Compute the derived metrics

    Returns:
        MetricResultsDict: A dictionary of the results
    """
    # first, sort the metrics by dependency order, and create an instance of each
    metrics = [
        MetricRegistry.get_class(tag)()
        for tag in MetricRegistry.create_dependency_order_for(metrics_to_test)
    ]

    metric_results = MetricResultsDict()
    # Per-record AGGREGATE values, folded once at the end via aggregation_kind
    # exactly as MetricsAccumulator combines the numpy column.
    aggregate_values: dict[MetricTagT, list] = {}
    for record in records:
        # STAGE 1: Parse the metrics for each record, and store the values in a dict
        metric_dict = MetricRecordDict()
        for metric in metrics:
            if metric.type in [MetricType.RECORD, MetricType.AGGREGATE]:
                try:
                    value = metric.parse_record(record, metric_dict)
                    metric_dict[metric.tag] = value
                except NoMetricValue:
                    # If a metric can't be calculated, skip it for this record
                    pass

        # STAGE 2: Collect per-record aggregate values; append record-metric values.
        for metric in metrics:
            if metric.type == MetricType.AGGREGATE:
                if metric.tag in metric_dict:
                    aggregate_values.setdefault(metric.tag, []).append(
                        metric_dict[metric.tag]
                    )
            elif metric.type == MetricType.RECORD and metric.tag in metric_dict:
                metric_results.setdefault(metric.tag, []).append(
                    metric_dict[metric.tag]
                )

    # STAGE 2b: Fold the aggregate columns via aggregation_kind (SUM/MIN/MAX).
    _fold = {
        AggregationKind.SUM: sum,
        AggregationKind.MIN: min,
        AggregationKind.MAX: max,
    }
    for metric in metrics:
        if metric.type == MetricType.AGGREGATE and metric.tag in aggregate_values:
            metric_results[metric.tag] = _fold[metric.aggregation_kind](
                aggregate_values[metric.tag]
            )

    # STAGE 3: Compute all of the derived metrics
    for metric in metrics:
        if metric.type == MetricType.DERIVED:
            metric_results[metric.tag] = metric.derive_value(metric_results)

    return metric_results
