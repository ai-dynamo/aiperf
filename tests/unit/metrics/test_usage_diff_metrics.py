# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponse, ParsedResponseRecord, RequestRecord
from aiperf.common.models.record_models import TextResponseData, TokenCounts
from aiperf.common.models.usage_models import Usage
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.usage_diff_metrics import (
    UsageCompletionTokensDiffMetric,
    UsageDiscrepancyCountMetric,
    UsagePromptTokensDiffMetric,
    UsageReasoningTokensDiffMetric,
)
from aiperf.metrics.types.usage_metrics import (
    UsagePromptTokensMetric,
)
from tests.unit.metrics.conftest import run_simple_metrics_pipeline


def create_record_with_usage(
    start_ns: int = 100,
    input_tokens: int = 100,
    output_tokens: int = 50,
    output_local_tokens: int | None = None,
    reasoning_tokens: int = 0,
    reasoning_local_tokens: int | None = None,
    usage_prompt_tokens: int = 100,
    usage_completion_tokens: int = 50,
    usage_reasoning_tokens: int | None = None,
) -> ParsedResponseRecord:
    """
    Create a test record with both client-computed and API-reported token counts.

    Args:
        start_ns: Start timestamp in nanoseconds
        input_tokens: Client-computed input token count
        output_tokens: Server-reported output token count (WITHOUT reasoning)
        output_local_tokens: Client-computed output token count (optional)
        reasoning_tokens: Server-reported reasoning token count (defaults to 0)
        reasoning_local_tokens: Client-computed reasoning token count (optional)
        usage_prompt_tokens: API-reported prompt token count
        usage_completion_tokens: API-reported completion token count (includes reasoning)
        usage_reasoning_tokens: API-reported reasoning token count (optional)
    """
    request = RequestRecord(
        conversation_id="test-conversation",
        turn_index=0,
        model_name="test-model",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=start_ns + 100,
    )

    # Create usage dict with API-reported values
    usage_dict = {
        "prompt_tokens": usage_prompt_tokens,
        "completion_tokens": usage_completion_tokens,
        "total_tokens": usage_prompt_tokens + usage_completion_tokens,
    }

    # Add reasoning tokens if provided
    if usage_reasoning_tokens is not None:
        usage_dict["completion_tokens_details"] = {
            "reasoning_tokens": usage_reasoning_tokens
        }

    usage = Usage(usage_dict)

    response = ParsedResponse(
        perf_ns=start_ns + 50,
        data=TextResponseData(text="test"),
        usage=usage,
    )

    return ParsedResponseRecord(
        request=request,
        responses=[response],
        token_counts=TokenCounts(
            input_local=input_tokens,
            output=output_tokens,
            output_local=output_local_tokens,
            reasoning=reasoning_tokens,
            reasoning_local=reasoning_local_tokens,
        ),
    )


class TestUsagePromptTokensDiffMetric:
    """Tests for UsagePromptTokensDiffMetric."""

    def test_exact_match(self):
        """Test when API and client token counts match exactly."""
        record = create_record_with_usage(
            input_tokens=100,
            usage_prompt_tokens=100,
        )

        # Prepare record metrics with required dependency (UsagePromptTokensMetric only)
        record_metrics = MetricRecordDict()
        record_metrics[UsagePromptTokensMetric.tag] = (
            UsagePromptTokensMetric().parse_record(record, MetricRecordDict())
        )

        metric = UsagePromptTokensDiffMetric()
        result = metric.parse_record(record, record_metrics)

        assert result == 0.0

    def test_positive_difference(self):
        """Test when API reports more tokens than client computed."""
        record = create_record_with_usage(
            input_tokens=100,
            usage_prompt_tokens=110,  # API reports 10% more
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
        )

        result = metric_results[UsagePromptTokensDiffMetric.tag][0]
        assert result == pytest.approx(10.0, rel=1e-9)

    def test_negative_difference(self):
        """Test when API reports fewer tokens than client computed."""
        record = create_record_with_usage(
            input_tokens=100,
            usage_prompt_tokens=95,  # API reports 5% less
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
        )

        result = metric_results[UsagePromptTokensDiffMetric.tag][0]
        assert result == pytest.approx(5.0, rel=1e-9)

    def test_large_discrepancy(self):
        """Test with a large discrepancy between API and client."""
        record = create_record_with_usage(
            input_tokens=100,
            usage_prompt_tokens=150,  # API reports 50% more
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
        )

        result = metric_results[UsagePromptTokensDiffMetric.tag][0]
        assert result == pytest.approx(50.0, rel=1e-9)

    def test_zero_client_tokens_raises_error(self):
        """Test that zero client tokens results in no metric value."""
        record = create_record_with_usage(
            input_tokens=0,
            usage_prompt_tokens=10,
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
        )

        # When input tokens are 0, the metric raises NoMetricValue
        assert (
            UsagePromptTokensDiffMetric.tag not in metric_results
            or len(metric_results[UsagePromptTokensDiffMetric.tag]) == 0
        )

    def test_metric_metadata(self):
        """Test that UsagePromptTokensDiffMetric has correct metadata."""
        assert UsagePromptTokensDiffMetric.tag == "usage_prompt_tokens_diff_pct"
        assert UsagePromptTokensDiffMetric.has_flags(MetricFlags.TOKENIZES_INPUT_ONLY)
        assert UsagePromptTokensDiffMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert UsagePromptTokensDiffMetric.missing_flags(MetricFlags.EXPERIMENTAL)


class TestMultipleRecordsWithVariedDiscrepancies:
    """Test processing multiple records with different discrepancies."""

    def test_mixed_discrepancies(self):
        """Test multiple records with various discrepancy patterns."""
        records = [
            create_record_with_usage(
                start_ns=100,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=100,  # Exact match
                usage_completion_tokens=50,
            ),
            create_record_with_usage(
                start_ns=200,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=110,  # +10%
                usage_completion_tokens=55,
            ),
            create_record_with_usage(
                start_ns=300,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=95,  # -5%
                usage_completion_tokens=48,
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
        )

        prompt_diffs = metric_results[UsagePromptTokensDiffMetric.tag]

        assert prompt_diffs[0] == pytest.approx(0.0, rel=1e-9)
        assert prompt_diffs[1] == pytest.approx(10.0, rel=1e-9)
        assert prompt_diffs[2] == pytest.approx(5.0, rel=1e-9)

    def test_records_with_missing_data_excluded_from_diff_metrics(self):
        """
        Test that records with missing data (zero client tokens) are excluded from diff metrics.
        """
        records = [
            # Valid record - should produce diff metrics
            create_record_with_usage(
                start_ns=100,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=110,  # +10%
                usage_completion_tokens=55,
            ),
            # Invalid record - zero input tokens, will raise NoMetricValue for prompt diff
            create_record_with_usage(
                start_ns=200,
                input_tokens=0,
                output_tokens=50,
                usage_prompt_tokens=100,
                usage_completion_tokens=50,
            ),
            # Valid record - should produce diff metrics
            create_record_with_usage(
                start_ns=300,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=95,  # -5%
                usage_completion_tokens=48,
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
        )

        # Only 2 records should have diff metrics (the record with zero input tokens is excluded)
        prompt_diffs = metric_results[UsagePromptTokensDiffMetric.tag]

        assert len(prompt_diffs) == 2, (
            "Should only have 2 records with valid prompt diffs"
        )

        assert prompt_diffs[0] == pytest.approx(10.0, rel=1e-9)
        assert prompt_diffs[1] == pytest.approx(5.0, rel=1e-9)


class TestUsageDiscrepancyCountMetric:
    """Tests for UsageDiscrepancyCountMetric aggregate counter."""

    def test_discrepancy_count_below_threshold(self, monkeypatch):
        """Test that records below threshold are not counted."""
        monkeypatch.setattr(
            "aiperf.metrics.types.usage_diff_metrics.Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD",
            10.0,
        )

        records = [
            create_record_with_usage(
                start_ns=100,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=105,  # 5% diff - below threshold
                usage_completion_tokens=52,
            ),
            create_record_with_usage(
                start_ns=200,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=102,  # 2% diff - below threshold
                usage_completion_tokens=51,
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
            UsageDiscrepancyCountMetric.tag,
        )

        # No records should be counted as discrepancies
        assert metric_results[UsageDiscrepancyCountMetric.tag] == 0

    def test_discrepancy_count_above_threshold(self, monkeypatch):
        """Test that records above threshold are counted."""
        monkeypatch.setattr(
            "aiperf.metrics.types.usage_diff_metrics.Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD",
            10.0,
        )

        records = [
            create_record_with_usage(
                start_ns=100,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=115,  # 15% diff - above threshold
                usage_completion_tokens=50,
            ),
            create_record_with_usage(
                start_ns=200,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=112,  # 12% diff - above threshold
                usage_completion_tokens=60,
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
            UsageDiscrepancyCountMetric.tag,
        )

        # Both records should be counted as discrepancies
        assert metric_results[UsageDiscrepancyCountMetric.tag] == 2

    def test_discrepancy_count_mixed(self, monkeypatch):
        """Test mixed scenario with some records above and some below threshold."""
        monkeypatch.setattr(
            "aiperf.metrics.types.usage_diff_metrics.Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD",
            10.0,
        )

        records = [
            create_record_with_usage(
                start_ns=100,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=105,  # 5% - below
                usage_completion_tokens=52,
            ),
            create_record_with_usage(
                start_ns=200,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=120,  # 20% - above
                usage_completion_tokens=50,
            ),
            create_record_with_usage(
                start_ns=300,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=110,  # 10% - at threshold (not above)
                usage_completion_tokens=45,
            ),
            create_record_with_usage(
                start_ns=400,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=115,  # 15% - above
                usage_completion_tokens=44,
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
            UsageDiscrepancyCountMetric.tag,
        )

        # Records 2 and 4 should be counted (2 total)
        assert metric_results[UsageDiscrepancyCountMetric.tag] == 2

    def test_discrepancy_count_with_missing_data(self, monkeypatch):
        """
        Test that records with missing diff metrics (due to zero client tokens)
        are excluded from discrepancy count.
        """
        monkeypatch.setattr(
            "aiperf.metrics.types.usage_diff_metrics.Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD",
            10.0,
        )

        records = [
            # Valid record with high discrepancy - SHOULD be counted
            create_record_with_usage(
                start_ns=100,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=120,  # 20% diff - above threshold
                usage_completion_tokens=60,
            ),
            # Invalid record - zero input tokens means prompt diff can't be calculated
            create_record_with_usage(
                start_ns=200,
                input_tokens=0,
                output_tokens=50,
                usage_prompt_tokens=100,
                usage_completion_tokens=60,
            ),
            # Valid record below threshold - should NOT be counted
            create_record_with_usage(
                start_ns=300,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=105,  # 5% diff - below threshold
                usage_completion_tokens=52,
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
            UsageDiscrepancyCountMetric.tag,
        )

        # Only 1 record counted (the first one)
        assert metric_results[UsageDiscrepancyCountMetric.tag] == 1

    def test_discrepancy_count_custom_threshold(self, monkeypatch):
        """Test that custom threshold values work correctly."""
        monkeypatch.setattr(
            "aiperf.metrics.types.usage_diff_metrics.Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD",
            5.0,
        )

        records = [
            create_record_with_usage(
                start_ns=100,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=106,  # 6% diff - above 5% threshold
                usage_completion_tokens=50,
            ),
            create_record_with_usage(
                start_ns=200,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=104,  # 4% diff - below 5% threshold
                usage_completion_tokens=50,
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
            UsageDiscrepancyCountMetric.tag,
        )

        # Only first record should be counted with 5% threshold
        assert metric_results[UsageDiscrepancyCountMetric.tag] == 1


class TestUsageCompletionTokensDiffMetric:
    """Tests for UsageCompletionTokensDiffMetric."""

    def test_exact_match(self):
        """Test when server and client output token counts match exactly."""
        record = create_record_with_usage(
            output_tokens=100,
            output_local_tokens=100,
        )

        metric = UsageCompletionTokensDiffMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0.0

    def test_positive_difference(self):
        """Test when server reports more output tokens than client computed."""
        record = create_record_with_usage(
            output_tokens=110,
            output_local_tokens=100,
        )

        metric = UsageCompletionTokensDiffMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == pytest.approx(10.0, rel=1e-9)

    def test_negative_difference(self):
        """Test when server reports fewer output tokens than client computed."""
        record = create_record_with_usage(
            output_tokens=95,
            output_local_tokens=100,
        )

        metric = UsageCompletionTokensDiffMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == pytest.approx(5.0, rel=1e-9)

    def test_missing_server_value(self):
        """Test that missing server output tokens raises NoMetricValue."""
        record = create_record_with_usage(
            output_tokens=50,
            output_local_tokens=100,
        )
        record.token_counts.output = None

        metric = UsageCompletionTokensDiffMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_missing_client_value(self):
        """Test that missing client output tokens raises NoMetricValue."""
        record = create_record_with_usage(
            output_tokens=50,
        )
        # output_local_tokens defaults to None

        metric = UsageCompletionTokensDiffMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_zero_client_tokens(self):
        """Test that zero client output tokens raises NoMetricValue."""
        record = create_record_with_usage(
            output_tokens=50,
            output_local_tokens=0,
        )

        metric = UsageCompletionTokensDiffMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_metric_metadata(self):
        """Test that UsageCompletionTokensDiffMetric has correct metadata."""
        assert UsageCompletionTokensDiffMetric.tag == "usage_completion_tokens_diff_pct"
        assert UsageCompletionTokensDiffMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert UsageCompletionTokensDiffMetric.has_flags(MetricFlags.NO_CONSOLE)


class TestUsageReasoningTokensDiffMetric:
    """Tests for UsageReasoningTokensDiffMetric."""

    def test_exact_match(self):
        """Test when server and client reasoning token counts match exactly."""
        record = create_record_with_usage(
            reasoning_tokens=50,
            reasoning_local_tokens=50,
        )

        metric = UsageReasoningTokensDiffMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0.0

    def test_positive_difference(self):
        """Test when server reports more reasoning tokens than client computed."""
        record = create_record_with_usage(
            reasoning_tokens=55,
            reasoning_local_tokens=50,
        )

        metric = UsageReasoningTokensDiffMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == pytest.approx(10.0, rel=1e-9)

    def test_negative_difference(self):
        """Test when server reports fewer reasoning tokens than client computed."""
        record = create_record_with_usage(
            reasoning_tokens=45,
            reasoning_local_tokens=50,
        )

        metric = UsageReasoningTokensDiffMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == pytest.approx(10.0, rel=1e-9)

    def test_missing_server_value(self):
        """Test that missing server reasoning tokens raises NoMetricValue."""
        record = create_record_with_usage(
            reasoning_tokens=0,
            reasoning_local_tokens=50,
        )
        record.token_counts.reasoning = None

        metric = UsageReasoningTokensDiffMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_missing_client_value(self):
        """Test that missing client reasoning tokens raises NoMetricValue."""
        record = create_record_with_usage(
            reasoning_tokens=50,
        )
        # reasoning_local_tokens defaults to None

        metric = UsageReasoningTokensDiffMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_zero_client_tokens(self):
        """Test that zero client reasoning tokens raises NoMetricValue."""
        record = create_record_with_usage(
            reasoning_tokens=50,
            reasoning_local_tokens=0,
        )

        metric = UsageReasoningTokensDiffMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_metric_metadata(self):
        """Test that UsageReasoningTokensDiffMetric has correct metadata."""
        assert UsageReasoningTokensDiffMetric.tag == "usage_reasoning_tokens_diff_pct"
        assert UsageReasoningTokensDiffMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert UsageReasoningTokensDiffMetric.has_flags(MetricFlags.SUPPORTS_REASONING)
        assert UsageReasoningTokensDiffMetric.has_flags(MetricFlags.NO_CONSOLE)
