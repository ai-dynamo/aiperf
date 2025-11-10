# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags
from aiperf.common.models import ParsedResponse, ParsedResponseRecord, RequestRecord
from aiperf.common.models.record_models import TextResponseData
from aiperf.common.models.usage_models import Usage
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.input_sequence_length_metric import InputSequenceLengthMetric
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.reasoning_token_count import ReasoningTokenCountMetric
from aiperf.metrics.types.usage_diff_metrics import (
    UsageCompletionTokensDiffMetric,
    UsageDiscrepancyCountMetric,
    UsagePromptTokensDiffMetric,
    UsageReasoningTokensDiffMetric,
)
from aiperf.metrics.types.usage_metrics import (
    UsageCompletionTokensMetric,
    UsagePromptTokensMetric,
    UsageReasoningTokensMetric,
)
from tests.unit.metrics.conftest import run_simple_metrics_pipeline


def create_record_with_usage(
    start_ns: int = 100,
    input_tokens: int = 100,
    output_tokens: int = 50,
    reasoning_tokens: int = 0,
    usage_prompt_tokens: int = 100,
    usage_completion_tokens: int = 50,
    usage_reasoning_tokens: int | None = None,
) -> ParsedResponseRecord:
    """
    Create a test record with both client-computed and API-reported token counts.

    Note: API completion_tokens includes reasoning tokens, so it should be compared
    against output_sequence_length (output_tokens + reasoning_tokens).

    Args:
        start_ns: Start timestamp in nanoseconds
        input_tokens: Client-computed input token count
        output_tokens: Client-computed output token count (WITHOUT reasoning)
        reasoning_tokens: Client-computed reasoning token count (defaults to 0)
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
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        reasoning_token_count=reasoning_tokens,
    )


class TestUsagePromptTokensDiffMetric:
    """Tests for UsagePromptTokensDiffMetric."""

    def test_exact_match(self):
        """Test when API and client token counts match exactly."""
        record = create_record_with_usage(
            input_tokens=100,
            usage_prompt_tokens=100,
        )

        # Prepare record metrics with required dependencies
        record_metrics = MetricRecordDict()
        record_metrics[InputSequenceLengthMetric.tag] = (
            InputSequenceLengthMetric().parse_record(record, MetricRecordDict())
        )
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
            InputSequenceLengthMetric.tag,
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
            InputSequenceLengthMetric.tag,
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
            InputSequenceLengthMetric.tag,
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
            InputSequenceLengthMetric.tag,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
        )

        # When input tokens are 0, InputSequenceLengthMetric raises NoMetricValue,
        # which cascades to UsagePromptTokensDiffMetric also not being available
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


class TestUsageCompletionTokensDiffMetric:
    """Tests for UsageCompletionTokensDiffMetric."""

    def test_exact_match(self):
        """Test when API and client token counts match exactly."""
        record = create_record_with_usage(
            output_tokens=50,
            usage_completion_tokens=50,
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            OutputSequenceLengthMetric.tag,
            UsageCompletionTokensMetric.tag,
            UsageCompletionTokensDiffMetric.tag,
        )

        result = metric_results[UsageCompletionTokensDiffMetric.tag][0]
        assert result == 0.0

    def test_positive_difference(self):
        """Test when API reports more tokens than client computed."""
        record = create_record_with_usage(
            output_tokens=50,
            usage_completion_tokens=55,  # API reports 10% more
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            OutputSequenceLengthMetric.tag,
            UsageCompletionTokensMetric.tag,
            UsageCompletionTokensDiffMetric.tag,
        )

        result = metric_results[UsageCompletionTokensDiffMetric.tag][0]
        assert result == pytest.approx(10.0, rel=1e-9)

    def test_negative_difference(self):
        """Test when API reports fewer tokens than client computed."""
        record = create_record_with_usage(
            output_tokens=100,
            usage_completion_tokens=90,  # API reports 10% less
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            OutputSequenceLengthMetric.tag,
            UsageCompletionTokensMetric.tag,
            UsageCompletionTokensDiffMetric.tag,
        )

        result = metric_results[UsageCompletionTokensDiffMetric.tag][0]
        assert result == pytest.approx(10.0, rel=1e-9)

    def test_zero_client_tokens_raises_error(self):
        """Test that zero client tokens results in no metric value."""
        record = create_record_with_usage(
            output_tokens=0,
            usage_completion_tokens=10,
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            OutputSequenceLengthMetric.tag,
            UsageCompletionTokensMetric.tag,
            UsageCompletionTokensDiffMetric.tag,
        )

        # When output tokens are 0, OutputSequenceLengthMetric raises NoMetricValue,
        # which cascades to UsageCompletionTokensDiffMetric also not being available
        assert (
            UsageCompletionTokensDiffMetric.tag not in metric_results
            or len(metric_results[UsageCompletionTokensDiffMetric.tag]) == 0
        )

    def test_metric_metadata(self):
        """Test that UsageCompletionTokensDiffMetric has correct metadata."""
        assert UsageCompletionTokensDiffMetric.tag == "usage_completion_tokens_diff_pct"
        assert UsageCompletionTokensDiffMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert UsageCompletionTokensDiffMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert UsageCompletionTokensDiffMetric.missing_flags(MetricFlags.EXPERIMENTAL)


class TestUsageReasoningTokensDiffMetric:
    """Tests for UsageReasoningTokensDiffMetric."""

    def test_exact_match(self):
        """Test when API and client reasoning token counts match exactly."""
        record = create_record_with_usage(
            reasoning_tokens=25,
            usage_reasoning_tokens=25,
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            ReasoningTokenCountMetric.tag,
            UsageReasoningTokensMetric.tag,
            UsageReasoningTokensDiffMetric.tag,
        )

        result = metric_results[UsageReasoningTokensDiffMetric.tag][0]
        assert result == 0.0

    def test_positive_difference(self):
        """Test when API reports more reasoning tokens than client computed."""
        record = create_record_with_usage(
            reasoning_tokens=50,
            usage_reasoning_tokens=60,  # API reports 20% more
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            ReasoningTokenCountMetric.tag,
            UsageReasoningTokensMetric.tag,
            UsageReasoningTokensDiffMetric.tag,
        )

        result = metric_results[UsageReasoningTokensDiffMetric.tag][0]
        assert result == pytest.approx(20.0, rel=1e-9)

    def test_negative_difference(self):
        """Test when API reports fewer reasoning tokens than client computed."""
        record = create_record_with_usage(
            reasoning_tokens=100,
            usage_reasoning_tokens=80,  # API reports 20% less
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            ReasoningTokenCountMetric.tag,
            UsageReasoningTokensMetric.tag,
            UsageReasoningTokensDiffMetric.tag,
        )

        result = metric_results[UsageReasoningTokensDiffMetric.tag][0]
        assert result == pytest.approx(20.0, rel=1e-9)

    def test_zero_client_tokens_raises_error(self):
        """Test that zero client reasoning tokens results in no metric value."""
        record = create_record_with_usage(
            reasoning_tokens=0,
            usage_reasoning_tokens=10,
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            ReasoningTokenCountMetric.tag,
            UsageReasoningTokensMetric.tag,
            UsageReasoningTokensDiffMetric.tag,
        )

        # When reasoning tokens are 0, ReasoningTokenCountMetric raises NoMetricValue,
        # which cascades to UsageReasoningTokensDiffMetric also not being available
        assert (
            UsageReasoningTokensDiffMetric.tag not in metric_results
            or len(metric_results[UsageReasoningTokensDiffMetric.tag]) == 0
        )

    def test_metric_metadata(self):
        """Test that UsageReasoningTokensDiffMetric has correct metadata."""
        assert UsageReasoningTokensDiffMetric.tag == "usage_reasoning_tokens_diff_pct"
        assert UsageReasoningTokensDiffMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert UsageReasoningTokensDiffMetric.has_flags(MetricFlags.SUPPORTS_REASONING)
        assert UsageReasoningTokensDiffMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert UsageReasoningTokensDiffMetric.missing_flags(MetricFlags.EXPERIMENTAL)


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
                usage_completion_tokens=50,  # Exact match
            ),
            create_record_with_usage(
                start_ns=200,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=110,  # +10%
                usage_completion_tokens=55,  # +10%
            ),
            create_record_with_usage(
                start_ns=300,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=95,  # -5%
                usage_completion_tokens=48,  # -4%
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            InputSequenceLengthMetric.tag,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
            OutputSequenceLengthMetric.tag,
            UsageCompletionTokensMetric.tag,
            UsageCompletionTokensDiffMetric.tag,
        )

        prompt_diffs = metric_results[UsagePromptTokensDiffMetric.tag]
        completion_diffs = metric_results[UsageCompletionTokensDiffMetric.tag]

        assert prompt_diffs[0] == pytest.approx(0.0, rel=1e-9)
        assert prompt_diffs[1] == pytest.approx(10.0, rel=1e-9)
        assert prompt_diffs[2] == pytest.approx(5.0, rel=1e-9)

        assert completion_diffs[0] == pytest.approx(0.0, rel=1e-9)
        assert completion_diffs[1] == pytest.approx(10.0, rel=1e-9)
        assert completion_diffs[2] == pytest.approx(4.0, rel=1e-9)

    def test_records_with_missing_data_excluded_from_diff_metrics(self):
        """
        Test that records with missing data (zero client tokens) are excluded from diff metrics.

        This verifies that when a diff metric raises NoMetricValue (e.g., due to zero client tokens),
        the record is correctly excluded and only valid records are included in the results.
        """
        records = [
            # Valid record - should produce diff metrics
            create_record_with_usage(
                start_ns=100,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=110,  # +10%
                usage_completion_tokens=55,  # +10%
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
                usage_completion_tokens=48,  # -4%
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            InputSequenceLengthMetric.tag,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
            OutputSequenceLengthMetric.tag,
            UsageCompletionTokensMetric.tag,
            UsageCompletionTokensDiffMetric.tag,
        )

        # Only 2 records should have diff metrics (the record with zero input tokens is excluded)
        prompt_diffs = metric_results[UsagePromptTokensDiffMetric.tag]
        completion_diffs = metric_results[UsageCompletionTokensDiffMetric.tag]

        assert len(prompt_diffs) == 2, (
            "Should only have 2 records with valid prompt diffs"
        )
        assert len(completion_diffs) == 3, (
            "Should have 3 records with valid completion diffs"
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
                usage_completion_tokens=52,  # 4% diff - below threshold
            ),
            create_record_with_usage(
                start_ns=200,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=102,  # 2% diff - below threshold
                usage_completion_tokens=51,  # 2% diff - below threshold
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            InputSequenceLengthMetric.tag,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
            OutputSequenceLengthMetric.tag,
            UsageCompletionTokensMetric.tag,
            UsageCompletionTokensDiffMetric.tag,
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
                usage_completion_tokens=50,  # 0% diff
            ),
            create_record_with_usage(
                start_ns=200,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=100,  # 0% diff
                usage_completion_tokens=60,  # 20% diff - above threshold
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            InputSequenceLengthMetric.tag,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
            OutputSequenceLengthMetric.tag,
            UsageCompletionTokensMetric.tag,
            UsageCompletionTokensDiffMetric.tag,
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
                usage_completion_tokens=52,  # 4% - below
            ),
            create_record_with_usage(
                start_ns=200,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=120,  # 20% - above
                usage_completion_tokens=50,  # 0% - below
            ),
            create_record_with_usage(
                start_ns=300,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=100,  # 0% - below
                usage_completion_tokens=45,  # 10% - at threshold (not above)
            ),
            create_record_with_usage(
                start_ns=400,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=100,  # 0% - below
                usage_completion_tokens=44,  # 12% - above
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            InputSequenceLengthMetric.tag,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
            OutputSequenceLengthMetric.tag,
            UsageCompletionTokensMetric.tag,
            UsageCompletionTokensDiffMetric.tag,
            UsageDiscrepancyCountMetric.tag,
        )

        # Records 2 and 4 should be counted (2 total)
        assert metric_results[UsageDiscrepancyCountMetric.tag] == 2

    def test_discrepancy_count_with_missing_data(self, monkeypatch):
        """
        Test that records with missing diff metrics (due to zero client tokens)
        are excluded from discrepancy count.

        This is the GOTCHA: If the required diff metrics can't be calculated (NoMetricValue),
        then _check_metrics() will raise NoMetricValue for the discrepancy count too,
        and that record won't contribute to the count at all.
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
                usage_completion_tokens=60,  # 20% diff - above threshold
            ),
            # Invalid record - zero input tokens means prompt diff can't be calculated
            # This record will be EXCLUDED from the count entirely
            create_record_with_usage(
                start_ns=200,
                input_tokens=0,  # Zero input tokens!
                output_tokens=50,
                usage_prompt_tokens=100,  # Would be infinite% diff but raises NoMetricValue
                usage_completion_tokens=60,  # 20% diff
            ),
            # Valid record below threshold - should NOT be counted
            create_record_with_usage(
                start_ns=300,
                input_tokens=100,
                output_tokens=50,
                usage_prompt_tokens=105,  # 5% diff - below threshold
                usage_completion_tokens=52,  # 4% diff - below threshold
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            InputSequenceLengthMetric.tag,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
            OutputSequenceLengthMetric.tag,
            UsageCompletionTokensMetric.tag,
            UsageCompletionTokensDiffMetric.tag,
            UsageDiscrepancyCountMetric.tag,
        )

        # Only 1 record counted (the first one)
        # The second record is excluded because diff metrics couldn't be calculated
        # The third record is not counted because it's below threshold
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
            InputSequenceLengthMetric.tag,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
            OutputSequenceLengthMetric.tag,
            UsageCompletionTokensMetric.tag,
            UsageCompletionTokensDiffMetric.tag,
            UsageDiscrepancyCountMetric.tag,
        )

        # Only first record should be counted with 5% threshold
        assert metric_results[UsageDiscrepancyCountMetric.tag] == 1

    def test_discrepancy_count_with_reasoning_tokens(self, monkeypatch):
        """Test that reasoning token discrepancies are also counted."""
        monkeypatch.setattr(
            "aiperf.metrics.types.usage_diff_metrics.Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD",
            10.0,
        )

        records = [
            # Discrepancy in reasoning tokens only
            create_record_with_usage(
                start_ns=100,
                input_tokens=100,
                output_tokens=50,
                reasoning_tokens=50,
                usage_prompt_tokens=100,  # 0% diff
                usage_completion_tokens=100,  # 0% diff
                usage_reasoning_tokens=60,  # 20% diff - above threshold
            ),
            # No discrepancies
            create_record_with_usage(
                start_ns=200,
                input_tokens=100,
                output_tokens=50,
                reasoning_tokens=50,
                usage_prompt_tokens=100,  # 0% diff
                usage_completion_tokens=100,  # 0% diff
                usage_reasoning_tokens=50,  # 0% diff
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            InputSequenceLengthMetric.tag,
            UsagePromptTokensMetric.tag,
            UsagePromptTokensDiffMetric.tag,
            OutputSequenceLengthMetric.tag,
            UsageCompletionTokensMetric.tag,
            UsageCompletionTokensDiffMetric.tag,
            ReasoningTokenCountMetric.tag,
            UsageReasoningTokensMetric.tag,
            UsageReasoningTokensDiffMetric.tag,
            UsageDiscrepancyCountMetric.tag,
        )

        # First record should be counted due to reasoning token discrepancy
        assert metric_results[UsageDiscrepancyCountMetric.tag] == 1
