# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for image token metrics."""

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.models.dataset_models import Image, Turn
from aiperf.common.models.modality_token_counts import ModalityTokenCounts
from aiperf.common.models.record_models import TokenCounts
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.input_image_token_metrics import (
    ImageInputTokenCountMetric,
    ImageInputTokenThroughputMetric,
    ImageTokenRatioMetric,
    TextInputTokenCountMetric,
    TokensPerImageMetric,
    TotalImageInputTokensMetric,
)
from aiperf.metrics.types.input_sequence_length_metric import InputSequenceLengthMetric
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline


def create_record_with_image_tokens(
    start_ns: int = 100,
    responses: list[int] | None = None,
    input_tokens: int = 1500,
    text_input_tokens: int = 500,
    image_input_tokens: int = 1000,
    images_per_turn: list[int] | None = None,
) -> ParsedResponseRecord:
    """Create a test record with image token counts and images in turns."""
    responses = responses or [start_ns + 50]
    images_per_turn = images_per_turn or [1]

    record = create_record(
        start_ns=start_ns, responses=responses, input_tokens=input_tokens
    )
    record.token_counts = TokenCounts(
        input=input_tokens,
        input_modalities=ModalityTokenCounts(
            text=text_input_tokens,
            image=image_input_tokens,
        ),
        input_modalities_local=ModalityTokenCounts(
            text=text_input_tokens,
            image=image_input_tokens,
        ),
        output=record.token_counts.output if record.token_counts else 1,
    )

    record.request.turns = [
        Turn(
            role="user",
            images=[
                Image(
                    name="img",
                    contents=[f"image_{j}" for j in range(num_images)],
                )
            ],
        )
        for num_images in images_per_turn
    ]

    return record


class TestImageInputTokenCountMetric:
    def test_image_input_token_count_metric_reads_image_input_tokens_returns_count(
        self,
    ):
        """Image input token count reads from token_counts.input_modalities.image."""
        record = create_record_with_image_tokens(image_input_tokens=1000)
        metric_results = run_simple_metrics_pipeline(
            [record], ImageInputTokenCountMetric.tag
        )
        assert metric_results[ImageInputTokenCountMetric.tag] == [1000]

    def test_image_input_token_count_metric_zero_image_tokens_returns_zero(self):
        """Zero image tokens is valid (client overcounted text)."""
        record = create_record_with_image_tokens(
            input_tokens=200, text_input_tokens=200, image_input_tokens=0
        )
        metric_results = run_simple_metrics_pipeline(
            [record], ImageInputTokenCountMetric.tag
        )
        assert metric_results[ImageInputTokenCountMetric.tag] == [0]

    def test_image_input_token_count_metric_multiple_records(self):
        """Image input token count across multiple records."""
        records = [
            create_record_with_image_tokens(image_input_tokens=500),
            create_record_with_image_tokens(image_input_tokens=1000),
            create_record_with_image_tokens(image_input_tokens=750),
        ]
        metric_results = run_simple_metrics_pipeline(
            records, ImageInputTokenCountMetric.tag
        )
        assert metric_results[ImageInputTokenCountMetric.tag] == [500, 1000, 750]

    def test_image_input_token_count_metric_no_modalities_raises_NoMetricValue(self):
        """NoMetricValue raised when input_modalities is None."""
        record = create_record(start_ns=100, responses=[150], input_tokens=200)
        metric = ImageInputTokenCountMetric()
        with pytest.raises(
            NoMetricValue, match="Image input token count is not available"
        ):
            metric._parse_record(record, MetricRecordDict())

    def test_image_input_token_count_metric_no_token_counts_raises_NoMetricValue(self):
        """NoMetricValue raised when token_counts is None."""
        record = create_record(start_ns=100, responses=[150])
        record.token_counts = None
        metric = ImageInputTokenCountMetric()
        with pytest.raises(
            NoMetricValue, match="Image input token count is not available"
        ):
            metric._parse_record(record, MetricRecordDict())


class TestTextInputTokenCountMetric:
    def test_reads_text_input_from_token_counts(self):
        """Text input token count reads from token_counts.input_modalities.text."""
        record = create_record_with_image_tokens(text_input_tokens=200)
        metric_results = run_simple_metrics_pipeline(
            [record], TextInputTokenCountMetric.tag
        )
        assert metric_results[TextInputTokenCountMetric.tag] == [200]

    def test_no_modalities_raises_NoMetricValue(self):
        """NoMetricValue raised when input_modalities is None."""
        record = create_record(start_ns=100, responses=[150], input_tokens=200)
        metric = TextInputTokenCountMetric()
        with pytest.raises(
            NoMetricValue, match="Text input token count is not available"
        ):
            metric._parse_record(record, MetricRecordDict())


class TestTokensPerImageMetric:
    def test_single_image(self):
        """Tokens per image with a single image."""
        record = create_record_with_image_tokens(
            image_input_tokens=1000, images_per_turn=[1]
        )
        metric_results = run_simple_metrics_pipeline([record], TokensPerImageMetric.tag)
        assert metric_results[TokensPerImageMetric.tag] == [1000.0]

    def test_multiple_images(self):
        """Tokens per image with multiple images across turns."""
        record = create_record_with_image_tokens(
            image_input_tokens=1200, images_per_turn=[2, 1]
        )
        metric_results = run_simple_metrics_pipeline([record], TokensPerImageMetric.tag)
        assert metric_results[TokensPerImageMetric.tag] == [400.0]

    def test_no_images_raises_NoMetricValue(self):
        """NoMetricValue raised when no images in turns."""
        record = create_record_with_image_tokens(image_input_tokens=1000)
        record.request.turns = [Turn(role="user")]
        metric = TokensPerImageMetric()
        record_dict = MetricRecordDict()
        record_dict[ImageInputTokenCountMetric.tag] = 1000
        with pytest.raises(NoMetricValue, match="No images found"):
            metric._parse_record(record, record_dict)


class TestImageTokenRatioMetric:
    def test_ratio_computation(self):
        """Image token ratio is image_tokens / total_input_tokens."""
        record = create_record_with_image_tokens(
            input_tokens=1000, image_input_tokens=800
        )
        record_dict = MetricRecordDict()
        record_dict[ImageInputTokenCountMetric.tag] = 800
        record_dict[InputSequenceLengthMetric.tag] = 1000
        metric = ImageTokenRatioMetric()
        result = metric._parse_record(record, record_dict)
        assert result == pytest.approx(0.8)

    def test_zero_isl_raises_NoMetricValue(self):
        """NoMetricValue raised when ISL is 0."""
        record = create_record_with_image_tokens()
        record_dict = MetricRecordDict()
        record_dict[ImageInputTokenCountMetric.tag] = 800
        record_dict[InputSequenceLengthMetric.tag] = 0
        metric = ImageTokenRatioMetric()
        with pytest.raises(NoMetricValue, match="Cannot compute"):
            metric._parse_record(record, record_dict)


class TestImageTokenMetricsIntegration:
    def test_total_image_input_tokens(self):
        """Total image input tokens sums across records."""
        records = [
            create_record_with_image_tokens(image_input_tokens=500),
            create_record_with_image_tokens(image_input_tokens=500),
        ]
        metric_results = run_simple_metrics_pipeline(
            records, ImageInputTokenCountMetric.tag
        )
        total = metric_results[ImageInputTokenCountMetric.tag]
        assert sum(total) == 1000

    def test_throughput_end_to_end(self):
        """Image input token throughput computed through record + derived metrics."""
        metric = ImageInputTokenThroughputMetric()
        metric_results = MetricResultsDict()
        metric_results[TotalImageInputTokensMetric.tag] = 1000
        metric_results[BenchmarkDurationMetric.tag] = 2 * NANOS_PER_SECOND
        result = metric._derive_value(metric_results)
        assert result == pytest.approx(500.0)

    def test_throughput_zero_duration_raises(self):
        """NoMetricValue raised when duration is zero."""
        metric = ImageInputTokenThroughputMetric()
        metric_results = MetricResultsDict()
        metric_results[TotalImageInputTokensMetric.tag] = 1000
        metric_results[BenchmarkDurationMetric.tag] = 0
        with pytest.raises(NoMetricValue):
            metric._derive_value(metric_results)

    def test_total_accumulates_correctly(self):
        """Total image input tokens accumulates from multiple records."""
        records = [
            create_record_with_image_tokens(image_input_tokens=300),
            create_record_with_image_tokens(image_input_tokens=700),
        ]
        metric_results = run_simple_metrics_pipeline(
            records, ImageInputTokenCountMetric.tag
        )
        total = sum(metric_results[ImageInputTokenCountMetric.tag])
        assert total == 1000

    def test_token_counts_consistency(self):
        """input_modalities.text + input_modalities.image == input (total ISL)."""
        input_tokens = 1500
        text_input = 300
        image_input = 1200
        record = create_record_with_image_tokens(
            input_tokens=input_tokens,
            text_input_tokens=text_input,
            image_input_tokens=image_input,
        )
        assert (
            record.token_counts.input_modalities.text
            + record.token_counts.input_modalities.image
            == record.token_counts.input
        )
