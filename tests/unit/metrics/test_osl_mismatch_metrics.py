# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import CreditPhase, MetricFlags, ModelSelectionStrategy
from aiperf.common.environment import Environment
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import (
    ParsedResponse,
    ParsedResponseRecord,
    RequestInfo,
    RequestRecord,
)
from aiperf.common.models.dataset_models import Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import TextResponseData, TokenCounts
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.osl_mismatch_metrics import (
    OSLMismatchCountMetric,
    OSLMismatchDiffMetric,
    RequestedOSLMetric,
)
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.metrics.conftest import run_simple_metrics_pipeline


def _create_request_info_with_max_tokens(max_tokens: int | None) -> RequestInfo:
    """Create a RequestInfo with a turn that has max_tokens set."""
    turn = Turn(max_tokens=max_tokens)
    return RequestInfo(
        model_endpoint=ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000/v1/test",
            ),
        ),
        turns=[turn],
        turn_index=0,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="test-request-id",
        x_correlation_id="test-correlation-id",
        conversation_id="test-conversation",
    )


def create_record_with_osl(
    start_ns: int = 100,
    requested_osl: int | None = 100,
    actual_output_tokens: int = 100,
    reasoning_tokens: int = 0,
) -> ParsedResponseRecord:
    """
    Create a test record with requested OSL (max_tokens) and actual output token count.

    Args:
        start_ns: Start timestamp in nanoseconds
        requested_osl: The max_tokens value sent in the request (--osl value)
        actual_output_tokens: The actual number of output tokens generated
        reasoning_tokens: The number of reasoning tokens (defaults to 0)
    """
    request = RequestRecord(
        request_info=_create_request_info_with_max_tokens(requested_osl),
        model_name="test-model",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=start_ns + 100,
    )

    response = ParsedResponse(
        perf_ns=start_ns + 50,
        data=TextResponseData(text="test"),
    )

    return ParsedResponseRecord(
        request=request,
        responses=[response],
        token_counts=TokenCounts(
            input=50,
            output=actual_output_tokens,
            reasoning=reasoning_tokens,
        ),
    )


class TestRequestedOSLMetric:
    """Tests for RequestedOSLMetric."""

    def test_extracts_max_tokens(self):
        """Test that the metric correctly extracts max_tokens from request."""
        record = create_record_with_osl(requested_osl=100)

        metric = RequestedOSLMetric()
        result = metric.parse_record(record, MetricRecordDict())

        assert result == 100

    def test_raises_when_max_tokens_not_set(self):
        """Test that NoMetricValue is raised when max_tokens is None."""
        record = create_record_with_osl(requested_osl=None)

        metric = RequestedOSLMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    @pytest.mark.parametrize("max_tokens", [50, 100, 500, 1000])
    def test_different_max_tokens_values(self, max_tokens):
        """Test various max_tokens values."""
        record = create_record_with_osl(requested_osl=max_tokens)
        metric = RequestedOSLMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == max_tokens

    def test_has_correct_flags(self):
        """Test that the metric has the correct flags."""
        assert RequestedOSLMetric.has_flags(MetricFlags.PRODUCES_TOKENS_ONLY)
        assert RequestedOSLMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert RequestedOSLMetric.has_flags(MetricFlags.INTERNAL)


class TestOSLMismatchDiffMetric:
    """Tests for OSLMismatchDiffMetric."""

    def test_exact_match(self):
        """Test when requested and actual OSL match exactly."""
        record = create_record_with_osl(
            requested_osl=100,
            actual_output_tokens=100,
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestedOSLMetric.tag,
            OutputSequenceLengthMetric.tag,
            OSLMismatchDiffMetric.tag,
        )

        result = metric_results[OSLMismatchDiffMetric.tag][0]
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_stopped_early(self):
        """Test when actual output is less than requested (stopped early due to EOS)."""
        record = create_record_with_osl(
            requested_osl=100,
            actual_output_tokens=50,  # Only got 50% of requested
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestedOSLMetric.tag,
            OutputSequenceLengthMetric.tag,
            OSLMismatchDiffMetric.tag,
        )

        result = metric_results[OSLMismatchDiffMetric.tag][0]
        # ((50 - 100) / 100) * 100 = -50%
        assert result == pytest.approx(-50.0, rel=1e-9)

    def test_generated_more_than_requested(self):
        """Test when actual output exceeds requested (server ignored max_tokens)."""
        record = create_record_with_osl(
            requested_osl=100,
            actual_output_tokens=120,  # Got 20% more than requested
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestedOSLMetric.tag,
            OutputSequenceLengthMetric.tag,
            OSLMismatchDiffMetric.tag,
        )

        result = metric_results[OSLMismatchDiffMetric.tag][0]
        # ((120 - 100) / 100) * 100 = 20%
        assert result == pytest.approx(20.0, rel=1e-9)

    def test_includes_reasoning_tokens(self):
        """Test that actual OSL includes reasoning tokens."""
        record = create_record_with_osl(
            requested_osl=100,
            actual_output_tokens=40,
            reasoning_tokens=60,  # Total OSL = 40 + 60 = 100
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestedOSLMetric.tag,
            OutputSequenceLengthMetric.tag,
            OSLMismatchDiffMetric.tag,
        )

        result = metric_results[OSLMismatchDiffMetric.tag][0]
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_has_correct_flags(self):
        """Test that the metric has the correct flags."""
        assert OSLMismatchDiffMetric.has_flags(MetricFlags.PRODUCES_TOKENS_ONLY)
        assert OSLMismatchDiffMetric.has_flags(MetricFlags.NO_CONSOLE)


class TestOSLMismatchCountMetric:
    """Tests for OSLMismatchCountMetric."""

    def test_counts_mismatches_above_threshold(self, monkeypatch):
        """Test that records with mismatch above threshold are counted."""
        # Set threshold to 20%
        monkeypatch.setattr(Environment.METRICS, "OSL_MISMATCH_PCT_THRESHOLD", 20.0)

        records = [
            create_record_with_osl(
                requested_osl=100, actual_output_tokens=100
            ),  # 0% diff
            create_record_with_osl(
                requested_osl=100, actual_output_tokens=50
            ),  # -50% diff - counted
            create_record_with_osl(
                requested_osl=100, actual_output_tokens=85
            ),  # -15% diff
            create_record_with_osl(
                requested_osl=100, actual_output_tokens=70
            ),  # -30% diff - counted
            create_record_with_osl(
                requested_osl=100, actual_output_tokens=120
            ),  # 20% diff
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            RequestedOSLMetric.tag,
            OutputSequenceLengthMetric.tag,
            OSLMismatchDiffMetric.tag,
            OSLMismatchCountMetric.tag,
        )

        # Should count 2 records with >20% absolute difference
        assert metric_results[OSLMismatchCountMetric.tag] == 2

    def test_counts_positive_mismatches(self, monkeypatch):
        """Test that positive mismatches (generated more than requested) are also counted."""
        monkeypatch.setattr(Environment.METRICS, "OSL_MISMATCH_PCT_THRESHOLD", 10.0)

        records = [
            create_record_with_osl(
                requested_osl=100, actual_output_tokens=130
            ),  # 30% diff - counted
            create_record_with_osl(
                requested_osl=100, actual_output_tokens=105
            ),  # 5% diff
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            RequestedOSLMetric.tag,
            OutputSequenceLengthMetric.tag,
            OSLMismatchDiffMetric.tag,
            OSLMismatchCountMetric.tag,
        )

        assert metric_results[OSLMismatchCountMetric.tag] == 1

    def test_zero_count_when_all_within_threshold(self, monkeypatch):
        """Test that count is 0 when all records are within threshold."""
        monkeypatch.setattr(Environment.METRICS, "OSL_MISMATCH_PCT_THRESHOLD", 50.0)

        records = [
            create_record_with_osl(requested_osl=100, actual_output_tokens=100),  # 0%
            create_record_with_osl(requested_osl=100, actual_output_tokens=60),  # -40%
            create_record_with_osl(requested_osl=100, actual_output_tokens=140),  # 40%
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            RequestedOSLMetric.tag,
            OutputSequenceLengthMetric.tag,
            OSLMismatchDiffMetric.tag,
            OSLMismatchCountMetric.tag,
        )

        assert metric_results[OSLMismatchCountMetric.tag] == 0

    def test_max_token_threshold_caps_large_osl(self, monkeypatch):
        """Test that max_token_threshold caps the effective threshold for large OSL values."""
        monkeypatch.setattr(Environment.METRICS, "OSL_MISMATCH_PCT_THRESHOLD", 5.0)
        monkeypatch.setattr(Environment.METRICS, "OSL_MISMATCH_MAX_TOKEN_THRESHOLD", 50)

        records = [
            # 2000 requested: threshold = min(2000*5%=100, 50) = 50 tokens
            # actual=1940: diff=60 tokens > 50 → counted
            create_record_with_osl(requested_osl=2000, actual_output_tokens=1940),
            # actual=1960: diff=40 tokens < 50 → not counted
            create_record_with_osl(requested_osl=2000, actual_output_tokens=1960),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            RequestedOSLMetric.tag,
            OutputSequenceLengthMetric.tag,
            OSLMismatchDiffMetric.tag,
            OSLMismatchCountMetric.tag,
        )

        assert metric_results[OSLMismatchCountMetric.tag] == 1

    def test_has_correct_flags(self):
        """Test that the metric has the correct flags."""
        assert OSLMismatchCountMetric.has_flags(MetricFlags.PRODUCES_TOKENS_ONLY)
        assert OSLMismatchCountMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert OSLMismatchCountMetric.has_flags(MetricFlags.NO_INDIVIDUAL_RECORDS)
