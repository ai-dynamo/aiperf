# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricConsoleGroup, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponse, ParsedResponseRecord, RequestRecord
from aiperf.common.models.record_models import TextResponseData, TokenCounts
from aiperf.common.models.usage_models import Usage
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.usage_cache_metrics import (
    UsagePromptCacheMissTokensMetric,
    UsagePromptCacheReadTokensMetric,
    UsagePromptCacheWriteTokensMetric,
)
from aiperf.metrics.types.usage_extras_metrics import (
    UsagePromptAudioSecondsMetric,
    UsageToolUsePromptTokensMetric,
)
from aiperf.metrics.types.usage_metrics import (
    UsageAcceptedPredictionTokensMetric,
    UsageCompletionAudioTokensMetric,
    UsagePromptAudioTokensMetric,
    UsageReasoningTokensMetric,
    UsageRejectedPredictionTokensMetric,
)
from aiperf.metrics.types.usage_total_metrics import (
    OverallUsagePromptCacheReadPercentMetric,
    TotalUsageAcceptedPredictionTokensMetric,
    TotalUsageCompletionAudioTokensMetric,
    TotalUsagePromptAudioSecondsMetric,
    TotalUsagePromptAudioTokensMetric,
    TotalUsagePromptCacheMissTokensMetric,
    TotalUsagePromptCacheReadTokensMetric,
    TotalUsagePromptCacheWriteTokensMetric,
    TotalUsagePromptTokensMetric,
    TotalUsageReasoningTokensMetric,
    TotalUsageRejectedPredictionTokensMetric,
    TotalUsageToolUsePromptTokensMetric,
)


def create_record_with_usage(
    start_ns: int = 100,
    completion_tokens_details: dict | None = None,
    prompt_tokens_details: dict | None = None,
    extras: dict | None = None,
    streaming: bool = False,
) -> ParsedResponseRecord:
    """Create a test record with usage details dicts.

    `extras` is merged into the top-level usage dict; pass shape-shifted
    fields like `cache_read_input_tokens` (Anthropic) here.
    """
    request = RequestRecord(
        conversation_id="test-conversation",
        turn_index=0,
        model_name="test-model",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=start_ns + 100,
    )

    usage_dict: dict = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
    if completion_tokens_details is not None:
        usage_dict["completion_tokens_details"] = completion_tokens_details
    if prompt_tokens_details is not None:
        usage_dict["prompt_tokens_details"] = prompt_tokens_details
    if extras is not None:
        usage_dict.update(extras)

    usage = Usage(usage_dict)

    if streaming:
        # Simulate streaming: first chunk has no usage, last chunk has usage
        responses = [
            ParsedResponse(
                perf_ns=start_ns + 25,
                data=TextResponseData(text="chunk1"),
                usage=None,
            ),
            ParsedResponse(
                perf_ns=start_ns + 50,
                data=TextResponseData(text="chunk2"),
                usage=usage,
            ),
        ]
    else:
        responses = [
            ParsedResponse(
                perf_ns=start_ns + 50,
                data=TextResponseData(text="test"),
                usage=usage,
            ),
        ]

    return ParsedResponseRecord(
        request=request,
        responses=responses,
        token_counts=TokenCounts(input=100, output=50, reasoning=0),
    )


class TestUsagePromptCacheReadTokensMetric:
    """Tests for UsagePromptCacheReadTokensMetric (OpenAI + Anthropic shapes)."""

    def test_extracts_from_openai_nested(self):
        record = create_record_with_usage(
            prompt_tokens_details={"cached_tokens": 42},
        )
        metric = UsagePromptCacheReadTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 42

    def test_extracts_from_anthropic_top_level(self):
        record = create_record_with_usage(
            extras={"cache_read_input_tokens": 99},
        )
        metric = UsagePromptCacheReadTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 99

    def test_returns_zero(self):
        record = create_record_with_usage(
            prompt_tokens_details={"cached_tokens": 0},
        )
        metric = UsagePromptCacheReadTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_raises_when_missing(self):
        record = create_record_with_usage()
        metric = UsagePromptCacheReadTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_streaming_takes_last_non_none(self):
        record = create_record_with_usage(
            prompt_tokens_details={"cached_tokens": 77},
            streaming=True,
        )
        metric = UsagePromptCacheReadTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 77

    def test_metadata(self):
        assert UsagePromptCacheReadTokensMetric.tag == "usage_prompt_cache_read_tokens"
        assert (
            UsagePromptCacheReadTokensMetric.console_group == MetricConsoleGroup.USAGE
        )
        assert UsagePromptCacheReadTokensMetric.has_flags(MetricFlags.LARGER_IS_BETTER)
        assert UsagePromptCacheReadTokensMetric.missing_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert UsagePromptCacheReadTokensMetric.missing_flags(
            MetricFlags.SUPPORTS_AUDIO_ONLY
        )


class TestUsagePromptCacheWriteTokensMetric:
    """Tests for UsagePromptCacheWriteTokensMetric (Anthropic-only)."""

    def test_extracts_from_anthropic_top_level(self):
        record = create_record_with_usage(
            extras={"cache_creation_input_tokens": 256},
        )
        metric = UsagePromptCacheWriteTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 256

    def test_returns_zero(self):
        record = create_record_with_usage(
            extras={"cache_creation_input_tokens": 0},
        )
        metric = UsagePromptCacheWriteTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_raises_for_openai_shape(self):
        # OpenAI does not surface cache writes; reads alone must not satisfy.
        record = create_record_with_usage(
            prompt_tokens_details={"cached_tokens": 42},
        )
        metric = UsagePromptCacheWriteTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_raises_when_missing(self):
        record = create_record_with_usage()
        metric = UsagePromptCacheWriteTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_metadata(self):
        assert (
            UsagePromptCacheWriteTokensMetric.tag == "usage_prompt_cache_write_tokens"
        )
        assert (
            UsagePromptCacheWriteTokensMetric.console_group == MetricConsoleGroup.USAGE
        )
        # Cache writes are NOT unambiguously "larger is better" — they cost more
        # than ordinary input tokens but unlock cheaper reads later.
        assert UsagePromptCacheWriteTokensMetric.missing_flags(
            MetricFlags.LARGER_IS_BETTER
        )
        assert UsagePromptCacheWriteTokensMetric.missing_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert UsagePromptCacheWriteTokensMetric.missing_flags(
            MetricFlags.SUPPORTS_AUDIO_ONLY
        )


class TestUsagePromptAudioTokensMetric:
    """Tests for UsagePromptAudioTokensMetric."""

    def test_extracts_prompt_audio_tokens(self):
        record = create_record_with_usage(
            prompt_tokens_details={"audio_tokens": 30},
        )
        metric = UsagePromptAudioTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 30

    def test_returns_zero_audio_tokens(self):
        record = create_record_with_usage(
            prompt_tokens_details={"audio_tokens": 0},
        )
        metric = UsagePromptAudioTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_raises_when_missing(self):
        record = create_record_with_usage()
        metric = UsagePromptAudioTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_streaming_takes_last_non_none(self):
        record = create_record_with_usage(
            prompt_tokens_details={"audio_tokens": 55},
            streaming=True,
        )
        metric = UsagePromptAudioTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 55

    def test_metadata(self):
        assert UsagePromptAudioTokensMetric.tag == "usage_prompt_audio_tokens"
        assert UsagePromptAudioTokensMetric.console_group == MetricConsoleGroup.USAGE
        assert UsagePromptAudioTokensMetric.has_flags(MetricFlags.LARGER_IS_BETTER)
        assert UsagePromptAudioTokensMetric.has_flags(MetricFlags.SUPPORTS_AUDIO_ONLY)
        assert UsagePromptAudioTokensMetric.missing_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )


class TestUsageCompletionAudioTokensMetric:
    """Tests for UsageCompletionAudioTokensMetric."""

    def test_extracts_completion_audio_tokens(self):
        record = create_record_with_usage(
            completion_tokens_details={"audio_tokens": 20},
        )
        metric = UsageCompletionAudioTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 20

    def test_returns_zero_audio_tokens(self):
        record = create_record_with_usage(
            completion_tokens_details={"audio_tokens": 0},
        )
        metric = UsageCompletionAudioTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_raises_when_missing(self):
        record = create_record_with_usage()
        metric = UsageCompletionAudioTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_streaming_takes_last_non_none(self):
        record = create_record_with_usage(
            completion_tokens_details={"audio_tokens": 88},
            streaming=True,
        )
        metric = UsageCompletionAudioTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 88

    def test_metadata(self):
        assert UsageCompletionAudioTokensMetric.tag == "usage_completion_audio_tokens"
        assert (
            UsageCompletionAudioTokensMetric.console_group == MetricConsoleGroup.USAGE
        )
        assert UsageCompletionAudioTokensMetric.has_flags(MetricFlags.LARGER_IS_BETTER)
        assert UsageCompletionAudioTokensMetric.has_flags(
            MetricFlags.SUPPORTS_AUDIO_ONLY
        )
        assert UsageCompletionAudioTokensMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )


class TestUsageAcceptedPredictionTokensMetric:
    """Tests for UsageAcceptedPredictionTokensMetric."""

    def test_extracts_accepted_prediction_tokens(self):
        record = create_record_with_usage(
            completion_tokens_details={"accepted_prediction_tokens": 15},
        )
        metric = UsageAcceptedPredictionTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 15

    def test_returns_zero_accepted_prediction_tokens(self):
        record = create_record_with_usage(
            completion_tokens_details={"accepted_prediction_tokens": 0},
        )
        metric = UsageAcceptedPredictionTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_raises_when_missing(self):
        record = create_record_with_usage()
        metric = UsageAcceptedPredictionTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_streaming_takes_last_non_none(self):
        record = create_record_with_usage(
            completion_tokens_details={"accepted_prediction_tokens": 99},
            streaming=True,
        )
        metric = UsageAcceptedPredictionTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 99

    def test_metadata(self):
        assert (
            UsageAcceptedPredictionTokensMetric.tag
            == "usage_accepted_prediction_tokens"
        )
        assert (
            UsageAcceptedPredictionTokensMetric.console_group
            == MetricConsoleGroup.USAGE
        )
        assert UsageAcceptedPredictionTokensMetric.has_flags(
            MetricFlags.LARGER_IS_BETTER
        )
        assert UsageAcceptedPredictionTokensMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert UsageAcceptedPredictionTokensMetric.missing_flags(
            MetricFlags.SUPPORTS_AUDIO_ONLY
        )


class TestUsageRejectedPredictionTokensMetric:
    """Tests for UsageRejectedPredictionTokensMetric."""

    def test_extracts_rejected_prediction_tokens(self):
        record = create_record_with_usage(
            completion_tokens_details={"rejected_prediction_tokens": 5},
        )
        metric = UsageRejectedPredictionTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 5

    def test_returns_zero_rejected_prediction_tokens(self):
        record = create_record_with_usage(
            completion_tokens_details={"rejected_prediction_tokens": 0},
        )
        metric = UsageRejectedPredictionTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_raises_when_missing(self):
        record = create_record_with_usage()
        metric = UsageRejectedPredictionTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_streaming_takes_last_non_none(self):
        record = create_record_with_usage(
            completion_tokens_details={"rejected_prediction_tokens": 12},
            streaming=True,
        )
        metric = UsageRejectedPredictionTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 12

    def test_metadata(self):
        assert (
            UsageRejectedPredictionTokensMetric.tag
            == "usage_rejected_prediction_tokens"
        )
        assert (
            UsageRejectedPredictionTokensMetric.console_group
            == MetricConsoleGroup.USAGE
        )
        assert UsageRejectedPredictionTokensMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert UsageRejectedPredictionTokensMetric.missing_flags(
            MetricFlags.LARGER_IS_BETTER
        )
        assert UsageRejectedPredictionTokensMetric.missing_flags(
            MetricFlags.SUPPORTS_AUDIO_ONLY
        )


class TestUsagePromptCacheMissTokensMetric:
    """Tests for UsagePromptCacheMissTokensMetric (DeepSeek-specific)."""

    def test_extracts_from_deepseek_top_level(self):
        record = create_record_with_usage(
            extras={"prompt_cache_miss_tokens": 320},
        )
        metric = UsagePromptCacheMissTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 320

    def test_returns_zero(self):
        record = create_record_with_usage(extras={"prompt_cache_miss_tokens": 0})
        metric = UsagePromptCacheMissTokensMetric()
        assert metric.parse_record(record, MetricRecordDict()) == 0

    def test_raises_for_openai_shape(self):
        record = create_record_with_usage(prompt_tokens_details={"cached_tokens": 42})
        metric = UsagePromptCacheMissTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_metadata(self):
        assert UsagePromptCacheMissTokensMetric.tag == "usage_prompt_cache_miss_tokens"
        assert (
            UsagePromptCacheMissTokensMetric.console_group == MetricConsoleGroup.USAGE
        )
        # Misses are NOT "larger is better" — they're cache misses, i.e. unhelpful.
        assert UsagePromptCacheMissTokensMetric.missing_flags(
            MetricFlags.LARGER_IS_BETTER
        )


class TestOverallUsagePromptCacheReadPercentMetric:
    """Tests for OverallUsagePromptCacheReadPercentMetric (run-aggregate cache %)."""

    def test_basic_overall_percentage(self):
        metric_results = MetricResultsDict()
        metric_results[TotalUsagePromptCacheReadTokensMetric.tag] = 250
        metric_results[TotalUsagePromptTokensMetric.tag] = 1000
        result = OverallUsagePromptCacheReadPercentMetric().derive_value(metric_results)
        assert result == pytest.approx(25.0, rel=1e-9)

    def test_zero_total_prompt_tokens_raises(self):
        metric_results = MetricResultsDict()
        metric_results[TotalUsagePromptCacheReadTokensMetric.tag] = 0
        metric_results[TotalUsagePromptTokensMetric.tag] = 0
        with pytest.raises(NoMetricValue):
            OverallUsagePromptCacheReadPercentMetric().derive_value(metric_results)

    def test_metadata(self):
        assert (
            OverallUsagePromptCacheReadPercentMetric.tag
            == "overall_usage_prompt_cache_read_pct"
        )
        assert (
            OverallUsagePromptCacheReadPercentMetric.console_group
            == MetricConsoleGroup.USAGE
        )
        assert OverallUsagePromptCacheReadPercentMetric.has_flags(
            MetricFlags.LARGER_IS_BETTER
        )
        assert OverallUsagePromptCacheReadPercentMetric.required_metrics == {
            TotalUsagePromptCacheReadTokensMetric.tag,
            TotalUsagePromptTokensMetric.tag,
        }


class TestUsageToolUsePromptTokensMetric:
    """Tests for UsageToolUsePromptTokensMetric (Gemini-specific)."""

    def test_extracts_from_gemini_envelope(self):
        record = create_record_with_usage(
            extras={"usageMetadata": {"toolUsePromptTokenCount": 30}}
        )
        metric = UsageToolUsePromptTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 30

    def test_returns_zero(self):
        record = create_record_with_usage(
            extras={"usageMetadata": {"toolUsePromptTokenCount": 0}}
        )
        metric = UsageToolUsePromptTokensMetric()
        assert metric.parse_record(record, MetricRecordDict()) == 0

    def test_raises_for_openai_shape(self):
        record = create_record_with_usage()
        metric = UsageToolUsePromptTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_metadata(self):
        assert UsageToolUsePromptTokensMetric.tag == "usage_tool_use_prompt_tokens"
        assert UsageToolUsePromptTokensMetric.console_group == MetricConsoleGroup.USAGE


class TestUsagePromptAudioSecondsMetric:
    """Tests for UsagePromptAudioSecondsMetric (Mistral-specific, returns float)."""

    def test_extracts_audio_seconds(self):
        record = create_record_with_usage(extras={"prompt_audio_seconds": 12.5})
        metric = UsagePromptAudioSecondsMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 12.5
        assert isinstance(result, float)

    def test_int_payload_returns_float(self):
        record = create_record_with_usage(extras={"prompt_audio_seconds": 12})
        result = UsagePromptAudioSecondsMetric().parse_record(
            record, MetricRecordDict()
        )
        assert result == 12.0
        assert isinstance(result, float)

    def test_returns_zero(self):
        record = create_record_with_usage(extras={"prompt_audio_seconds": 0})
        assert (
            UsagePromptAudioSecondsMetric().parse_record(record, MetricRecordDict())
            == 0.0
        )

    def test_raises_for_token_only_response(self):
        record = create_record_with_usage(prompt_tokens_details={"audio_tokens": 100})
        with pytest.raises(NoMetricValue):
            UsagePromptAudioSecondsMetric().parse_record(record, MetricRecordDict())

    def test_metadata(self):
        assert UsagePromptAudioSecondsMetric.tag == "usage_prompt_audio_seconds"
        assert UsagePromptAudioSecondsMetric.has_flags(MetricFlags.SUPPORTS_AUDIO_ONLY)


class TestTotalUsageDerivedSumMetrics:
    """Tests for Total* derived sum metrics wiring."""

    @pytest.mark.parametrize(
        "total_cls,record_cls",
        [
            (TotalUsageReasoningTokensMetric, UsageReasoningTokensMetric),
            (
                TotalUsagePromptCacheReadTokensMetric,
                UsagePromptCacheReadTokensMetric,
            ),
            (
                TotalUsagePromptCacheWriteTokensMetric,
                UsagePromptCacheWriteTokensMetric,
            ),
            (TotalUsagePromptAudioTokensMetric, UsagePromptAudioTokensMetric),
            (TotalUsageCompletionAudioTokensMetric, UsageCompletionAudioTokensMetric),
            (
                TotalUsageAcceptedPredictionTokensMetric,
                UsageAcceptedPredictionTokensMetric,
            ),
            (
                TotalUsageRejectedPredictionTokensMetric,
                UsageRejectedPredictionTokensMetric,
            ),
            (
                TotalUsagePromptCacheMissTokensMetric,
                UsagePromptCacheMissTokensMetric,
            ),
            (
                TotalUsageToolUsePromptTokensMetric,
                UsageToolUsePromptTokensMetric,
            ),
            (
                TotalUsagePromptAudioSecondsMetric,
                UsagePromptAudioSecondsMetric,
            ),
        ],
    )
    def test_derived_sum_wiring(self, total_cls, record_cls):
        assert total_cls.record_metric_type is record_cls
        assert total_cls.required_metrics == {record_cls.tag}
        assert total_cls.unit == record_cls.unit
        assert total_cls.flags == record_cls.flags
