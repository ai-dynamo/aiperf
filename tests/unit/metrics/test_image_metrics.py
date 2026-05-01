# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.image_metrics import (
    ImageLatencyMetric,
    ImageThroughputMetric,
    NumImagesMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline


def run_image_metrics_pipeline(
    records: list[ParsedResponseRecord],
    *metric_tags: str,
) -> dict:
    """Run metrics pipeline for image metrics with automatic dependency inclusion."""
    all_metrics = set(metric_tags)

    if (
        ImageThroughputMetric.tag in all_metrics
        or ImageLatencyMetric.tag in all_metrics
    ):
        all_metrics.add(NumImagesMetric.tag)
        all_metrics.add(RequestLatencyMetric.tag)

    return run_simple_metrics_pipeline(records, *all_metrics)


def _record_with_image_count(
    total_images: int,
    *,
    start_ns: int = 100,
    responses: list[int] | None = None,
) -> ParsedResponseRecord:
    """Build a ``ParsedResponseRecord`` with a pre-populated media image count.

    The metrics pipeline reads image counts from
    ``record.media_counts.images`` — populated in production by
    ``InferenceResultParser`` via the endpoint's single-pass
    ``extract_payload_inputs``. Tests hoist the count directly to skip
    the payload round-trip.
    """
    record = create_record(start_ns=start_ns, responses=responses or [start_ns + 50])
    record.media_counts.images = total_images
    return record


class TestNumImagesMetric:
    @pytest.mark.parametrize(
        "images_per_turn,expected",
        [
            ([1], 1),
            ([5], 5),
            ([2, 3], 5),
        ],
        ids=["single_image", "multiple_in_turn", "multiple_turns"],
    )  # fmt: skip
    def test_num_images_counting(self, images_per_turn, expected):
        """Test counting images in various configurations."""
        record = _record_with_image_count(sum(images_per_turn))
        metric_results = run_image_metrics_pipeline([record], NumImagesMetric.tag)
        assert metric_results[NumImagesMetric.tag] == [expected]

    def test_num_images_batched_contents(self):
        """Test counting images with multiple contents in a single batch."""
        record = _record_with_image_count(3)
        metric_results = run_image_metrics_pipeline([record], NumImagesMetric.tag)
        assert metric_results[NumImagesMetric.tag] == [3]

    def test_num_images_multiple_records(self):
        """Test counting images across multiple records."""
        records = [
            _record_with_image_count(1, start_ns=10, responses=[25]),
            _record_with_image_count(2, start_ns=20, responses=[35]),
            _record_with_image_count(3, start_ns=30, responses=[50]),
        ]
        metric_results = run_image_metrics_pipeline(records, NumImagesMetric.tag)
        assert metric_results[NumImagesMetric.tag] == [1, 2, 3]

    def test_num_images_error_when_zero(self):
        """Records with zero images raise ``NoMetricValue``."""
        record = _record_with_image_count(0)
        metric = NumImagesMetric()
        with pytest.raises(NoMetricValue, match="at least one image"):
            metric.parse_record(record, MetricRecordDict())


class TestImageThroughputMetric:
    @pytest.mark.parametrize(
        "images_per_turn,latency_ns,expected_throughput",
        [
            ([1], 1_000_000_000, 1.0),
            ([10], 2_000_000_000, 5.0),
            ([3], 500_000_000, 6.0),
            ([2, 3], 1_000_000_000, 5.0),
        ],
        ids=["1img_1s", "10img_2s", "3img_0.5s", "5img_multi_turn"],
    )  # fmt: skip
    def test_image_throughput_calculation(
        self, images_per_turn, latency_ns, expected_throughput
    ):
        """Test image throughput calculation with various configurations."""
        record = _record_with_image_count(
            sum(images_per_turn), start_ns=0, responses=[latency_ns]
        )
        metric_results = run_image_metrics_pipeline([record], ImageThroughputMetric.tag)
        assert metric_results[ImageThroughputMetric.tag] == [expected_throughput]

    def test_image_throughput_multiple_records(self):
        """Test throughput across multiple records."""
        records = [
            _record_with_image_count(2, start_ns=0, responses=[1_000_000_000]),
            _record_with_image_count(3, start_ns=0, responses=[500_000_000]),
        ]
        metric_results = run_image_metrics_pipeline(records, ImageThroughputMetric.tag)
        assert metric_results[ImageThroughputMetric.tag] == [2.0, 6.0]


class TestImageLatencyMetric:
    @pytest.mark.parametrize(
        "images_per_turn,latency_ns,expected_latency_ms",
        [
            ([1], 1_000_000_000, 1000.0),
            ([10], 1_000_000_000, 100.0),
            ([3], 500_000_000, 166.666666),
            ([2, 3], 1_000_000_000, 200.0),
            ([1], 10_000_000, 10.0),
        ],
        ids=["1img_1s", "10img_1s", "3img_0.5s", "5img_multi_turn", "fast_processing"],
    )  # fmt: skip
    def test_image_latency_calculation(
        self, images_per_turn, latency_ns, expected_latency_ms
    ):
        """Test image latency calculation with various configurations."""
        record = _record_with_image_count(
            sum(images_per_turn), start_ns=0, responses=[latency_ns]
        )
        metric_results = run_image_metrics_pipeline([record], ImageLatencyMetric.tag)
        assert metric_results[ImageLatencyMetric.tag][0] == pytest.approx(
            expected_latency_ms, rel=1e-5
        )

    def test_image_latency_multiple_records(self):
        """Test latency across multiple records."""
        records = [
            _record_with_image_count(2, start_ns=0, responses=[1_000_000_000]),
            _record_with_image_count(5, start_ns=0, responses=[500_000_000]),
        ]
        metric_results = run_image_metrics_pipeline(records, ImageLatencyMetric.tag)
        assert metric_results[ImageLatencyMetric.tag] == [500.0, 100.0]


class TestImageMetricsIntegration:
    def test_image_throughput_and_latency_are_inverses(self):
        """Test that throughput and latency are mathematical inverses."""
        record = _record_with_image_count(4, start_ns=0, responses=[2_000_000_000])
        metric_results = run_image_metrics_pipeline(
            [record], ImageThroughputMetric.tag, ImageLatencyMetric.tag
        )

        throughput = metric_results[ImageThroughputMetric.tag][0]
        latency = metric_results[ImageLatencyMetric.tag][0]

        # throughput (img/s) * latency (ms/img) = ms/s = 1000
        assert abs(throughput * latency - 1000.0) < 0.001

    def test_all_metrics_together(self):
        """Test computing all image metrics together."""
        record = _record_with_image_count(5, start_ns=0, responses=[1_000_000_000])
        metric_results = run_image_metrics_pipeline(
            [record],
            NumImagesMetric.tag,
            ImageThroughputMetric.tag,
            ImageLatencyMetric.tag,
        )

        assert metric_results[NumImagesMetric.tag] == [5]
        assert metric_results[ImageThroughputMetric.tag] == [5.0]
        assert metric_results[ImageLatencyMetric.tag] == [200.0]
