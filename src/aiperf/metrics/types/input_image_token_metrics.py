# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input image token metrics.

These metrics provide visibility into image token counts when images are present
in the input. They rely on the per-modality token breakdown in
``TokenCounts.input_modalities``, which is either:
- Scaled from AutoProcessor pre-computed estimates, or
- Derived via subtraction (server total minus client text count) as fallback.

All per-record metrics in this module are NO_CONSOLE building blocks exported
only to JSON/CSV. The derived ImageInputTokenThroughputMetric is
NO_CONSOLE as well but provides system-level throughput analysis.
"""

from aiperf.common.enums import GenericMetricUnit, MetricFlags, MetricOverTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseDerivedMetric
from aiperf.metrics.base_record_metric import BaseRecordMetric
from aiperf.metrics.derived_sum_metric import DerivedSumMetric
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.input_sequence_length_metric import InputSequenceLengthMetric


class ImageInputTokenCountMetric(BaseRecordMetric[int]):
    """Number of image input tokens for a single request.

    Reads ``token_counts.input_modalities.image`` which is either scaled from
    AutoProcessor estimates or derived by subtraction.

    Formula:
        Image Input Token Count = token_counts.input_modalities.image
    """

    tag = "image_input_token_count"
    header = "Image Input Token Count"
    short_header = "Img Inp Tok"
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.TOKENIZES_INPUT_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics: set[str] = set()

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        if (
            record.token_counts is None
            or record.token_counts.input_modalities is None
            or record.token_counts.input_modalities.image is None
        ):
            raise NoMetricValue(
                "Image input token count is not available for the record."
            )
        return record.token_counts.input_modalities.image


class TextInputTokenCountMetric(BaseRecordMetric[int]):
    """Number of text-only input tokens for a single request.

    Reads ``token_counts.input_modalities.text`` which is either scaled from
    AutoProcessor estimates or the client-side tokenized count when images are present.

    Formula:
        Text Input Token Count = token_counts.input_modalities.text
    """

    tag = "text_input_token_count"
    header = "Text Input Token Count"
    short_header = "Txt Inp Tok"
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.TOKENIZES_INPUT_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics: set[str] = set()

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        if (
            record.token_counts is None
            or record.token_counts.input_modalities is None
            or record.token_counts.input_modalities.text is None
        ):
            raise NoMetricValue(
                "Text input token count is not available for the record."
            )
        return record.token_counts.input_modalities.text


class TokensPerImageMetric(BaseRecordMetric[float]):
    """Image tokens divided by the number of images in a single request.

    Formula:
        Tokens Per Image = image_input_token_count / num_images
    """

    tag = "tokens_per_image"
    header = "Tokens Per Image"
    short_header = "Tok/Img"
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.TOKENIZES_INPUT_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = {ImageInputTokenCountMetric.tag}

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> float:
        image_tokens = record_metrics.get(ImageInputTokenCountMetric.tag)
        if image_tokens is None:
            raise NoMetricValue("Image input token count is not available.")

        num_images = sum(
            len(img.contents) for turn in record.request.turns for img in turn.images
        )
        if num_images == 0:
            raise NoMetricValue("No images found in request turns.")

        return image_tokens / num_images


class ImageTokenRatioMetric(BaseRecordMetric[float]):
    """Ratio of image tokens to total input tokens for a single request.

    Formula:
        Image Token Ratio = image_input_token_count / input_sequence_length
    """

    tag = "image_token_ratio"
    header = "Image Token Ratio"
    short_header = "Img Tok %"
    unit = GenericMetricUnit.RATIO
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.TOKENIZES_INPUT_ONLY
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = {ImageInputTokenCountMetric.tag, InputSequenceLengthMetric.tag}

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> float:
        image_tokens = record_metrics.get(ImageInputTokenCountMetric.tag)
        isl = record_metrics.get(InputSequenceLengthMetric.tag)
        if image_tokens is None or isl is None or isl == 0:
            raise NoMetricValue("Cannot compute image token ratio.")
        return image_tokens / isl


class TotalImageInputTokensMetric(DerivedSumMetric[int, ImageInputTokenCountMetric]):
    """Sum of all image input tokens across all requests."""

    tag = "total_image_input_tokens"
    header = "Total Image Input Tokens"
    short_header = "Tot Img Inp"
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.TOKENIZES_INPUT_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )


class ImageInputTokenThroughputMetric(BaseDerivedMetric[float]):
    """Image input tokens processed per second across the benchmark."""

    tag = "image_input_token_throughput"
    header = "Image Input Token Throughput"
    short_header = "Img Inp TPS"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.TOKENIZES_INPUT_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = {
        TotalImageInputTokensMetric.tag,
        BenchmarkDurationMetric.tag,
    }

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        total_image_tokens = metric_results.get_or_raise(TotalImageInputTokensMetric)
        benchmark_duration = metric_results.get_converted_or_raise(
            BenchmarkDurationMetric,
            self.unit.time_unit,  # type: ignore
        )
        if benchmark_duration == 0:
            raise NoMetricValue("Cannot compute image input token throughput.")
        return total_image_tokens / benchmark_duration  # type: ignore
