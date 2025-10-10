# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponse, ParsedResponseRecord, RequestRecord
from aiperf.common.models.record_models import ReasoningResponseData, TextResponseData
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.time_to_first_output_metric import TimeToFirstOutputMetric
from tests.metrics.conftest import create_record, run_simple_metrics_pipeline


def create_response_record(
    start_ns: int,
    responses: list[tuple[int, str | None, str | None]],
) -> ParsedResponseRecord:
    """Helper to create test records with text and/or reasoning responses.

    Response type is automatically determined:
    - If reasoning is None: TextResponseData with content as text
    - If reasoning is provided: ReasoningResponseData (with optional content)

    Args:
        start_ns: Start timestamp in nanoseconds
        responses: List of (timestamp, reasoning, content) tuples
    """
    request = RequestRecord(
        conversation_id="test-conversation",
        turn_index=0,
        model_name="test-model",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=responses[-1][0],
    )

    return ParsedResponseRecord(
        request=request,
        responses=[
            ParsedResponse(
                perf_ns=perf_ns,
                data=(
                    TextResponseData(text=content or "")
                    if reasoning is None
                    else ReasoningResponseData(reasoning=reasoning, content=content)
                ),
            )
            for perf_ns, reasoning, content in responses
        ],
        input_token_count=None,
        output_token_count=len(responses),
    )


class TestTimeToFirstOutputMetric:
    @pytest.mark.parametrize(
        "start_ns,responses,expected_ttfo",
        [
            (100, [110], 10),
            (100, [105, 115], 5),
        ],
        ids=["single_text_response", "multiple_text_responses"],
    )
    def test_ttfo_basic_text_responses(self, start_ns, responses, expected_ttfo):
        """Test basic time to first output with text responses"""
        record = create_record(start_ns=start_ns, responses=responses)
        metric_results = run_simple_metrics_pipeline(
            [record], TimeToFirstOutputMetric.tag
        )
        assert metric_results[TimeToFirstOutputMetric.tag] == [expected_ttfo]

    def test_ttfo_with_reasoning_response(self):
        """Test TTFO with reasoning response that has content"""
        record = create_response_record(
            start_ns=100, responses=[(110, "reasoning", "content")]
        )
        metric_results = run_simple_metrics_pipeline(
            [record], TimeToFirstOutputMetric.tag
        )
        assert metric_results[TimeToFirstOutputMetric.tag] == [10]

    @pytest.mark.parametrize(
        "responses,expected_ttfo",
        [
            (
                [
                    (105, "reasoning1", None),  # Reasoning only - skipped
                    (110, "reasoning2", ""),  # Empty content - skipped
                    (120, None, "first content"),  # First valid content (text)
                ],
                20,
            ),
            (
                [
                    (105, "reasoning", None),  # Reasoning only - skipped
                    (110, "reasoning", "first content"),  # First content (in reasoning)
                    (115, None, "text content"),  # Text after content found
                ],
                10,
            ),
        ],
        ids=["skip_reasoning_and_empty", "reasoning_content_before_text"],
    )
    def test_ttfo_reasoning_handling(self, responses, expected_ttfo):
        """Test TTFO correctly handles reasoning tokens and finds first valid content"""
        record = create_response_record(start_ns=100, responses=responses)
        metric_results = run_simple_metrics_pipeline(
            [record], TimeToFirstOutputMetric.tag
        )
        assert metric_results[TimeToFirstOutputMetric.tag] == [expected_ttfo]

    def test_ttfo_multiple_records(self):
        """Test processing multiple records with mixed response types"""
        records = [
            create_record(start_ns=100, responses=[110]),  # 10ns TTFO (text)
            create_response_record(
                start_ns=200, responses=[(210, None, "content")]
            ),  # 10ns TTFO (text with content)
            create_response_record(
                start_ns=300,
                responses=[(305, "reasoning", None), (315, None, "content")],
            ),  # 15ns TTFO (skip reasoning-only)
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            TimeToFirstOutputMetric.tag,
        )
        assert metric_results[TimeToFirstOutputMetric.tag] == [10, 10, 15]

    def test_ttfo_invalid_timestamp(self):
        """Test error when first non-reasoning token timestamp is before request start"""
        record = create_record(start_ns=100, responses=[90])
        metric = TimeToFirstOutputMetric()
        with pytest.raises(NoMetricValue, match="Invalid Record"):
            metric.parse_record(record, MetricRecordDict())

    def test_ttfo_no_responses(self):
        """Test error when no responses are present"""
        record = create_record(start_ns=100)
        record.responses = []
        metric = TimeToFirstOutputMetric()
        with pytest.raises(NoMetricValue, match="Invalid Record"):
            metric.parse_record(record, MetricRecordDict())

    @pytest.mark.parametrize(
        "responses",
        [
            [(110, "reasoning1", None), (120, "reasoning2", None)],
            [(110, "reasoning1", ""), (120, "reasoning2", "")],
        ],
        ids=["no_content", "empty_content"],
    )
    def test_ttfo_no_valid_content(self, responses):
        """Test error when no valid content tokens are present"""
        record = create_response_record(start_ns=100, responses=responses)
        metric = TimeToFirstOutputMetric()
        with pytest.raises(
            NoMetricValue,
            match="Record must have at least one non-reasoning token",
        ):
            metric.parse_record(record, MetricRecordDict())
