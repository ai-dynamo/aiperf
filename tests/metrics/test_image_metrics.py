# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.models.dataset_models import Image, Turn
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.image_metrics import (
    ImageLatencyMetric,
    ImageThroughputMetric,
    NumImagesMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from tests.metrics.conftest import run_simple_metrics_pipeline


def run_image_metrics_pipeline(
    records: list[ParsedResponseRecord],
    *metric_tags: str,
) -> dict:
    """
    Helper to run metrics pipeline for image metrics with automatic dependency inclusion.

    Automatically includes NumImagesMetric and RequestLatencyMetric when needed by
    ImageThroughputMetric or ImageLatencyMetric.
    """
    all_metrics = set(metric_tags)

    # If any image throughput/latency metric is requested, add dependencies
    if (
        ImageThroughputMetric.tag in all_metrics
        or ImageLatencyMetric.tag in all_metrics
    ):
        all_metrics.add(NumImagesMetric.tag)
        all_metrics.add(RequestLatencyMetric.tag)

    return run_simple_metrics_pipeline(records, *all_metrics)


def create_record_with_images(
    start_ns: int = 100,
    responses: list[int] | None = None,
    images_per_turn: list[int] | None = None,
) -> ParsedResponseRecord:
    """
    Create a test record with images.

    Args:
        start_ns: Start timestamp in nanoseconds
        responses: List of response timestamps
        images_per_turn: List of image counts per turn (e.g., [2, 3] = 2 images in turn 0, 3 in turn 1)
    """
    from tests.metrics.conftest import create_record

    responses = responses or [start_ns + 50]
    images_per_turn = images_per_turn or [1]

    # Create base record
    record = create_record(start_ns=start_ns, responses=responses)

    # Add turns with images
    turns = []
    for num_images in images_per_turn:
        # Each Image object has a contents list with the actual image data
        images = [
            Image(name=f"image_{i}", contents=[f"data_{i}"]) for i in range(num_images)
        ]
        turns.append(Turn(images=images))

    # Update the request with turns containing images
    record.request.turns = turns

    return record


class TestNumImagesMetric:
    @pytest.mark.parametrize(
        "images_per_turn,expected",
        [
            ([1], 1),  # Single image
            ([5], 5),  # Multiple images in single turn
            ([2, 3], 5),  # Multiple turns (2+3=5)
        ],
    )
    def test_num_images_counting(self, images_per_turn, expected):
        """Test counting images in various configurations"""
        record = create_record_with_images(
            start_ns=100, responses=[150], images_per_turn=images_per_turn
        )

        metric_results = run_image_metrics_pipeline([record], NumImagesMetric.tag)
        assert metric_results[NumImagesMetric.tag] == [expected]

    def test_num_images_batched_contents(self):
        """Test counting images with batched contents in a single Image object"""
        from tests.metrics.conftest import create_record

        record = create_record(start_ns=100, responses=[150])

        # Create a single Image object with multiple contents (batched)
        image_with_batch = Image(name="batch", contents=["data1", "data2", "data3"])
        record.request.turns = [Turn(images=[image_with_batch])]

        metric_results = run_image_metrics_pipeline([record], NumImagesMetric.tag)
        assert metric_results[NumImagesMetric.tag] == [3]

    def test_num_images_multiple_records(self):
        """Test counting images across multiple records"""
        records = [
            create_record_with_images(start_ns=10, responses=[25], images_per_turn=[1]),
            create_record_with_images(start_ns=20, responses=[35], images_per_turn=[2]),
            create_record_with_images(start_ns=30, responses=[50], images_per_turn=[3]),
        ]

        metric_results = run_image_metrics_pipeline(records, NumImagesMetric.tag)
        assert metric_results[NumImagesMetric.tag] == [1, 2, 3]

    def test_num_images_no_images_error(self):
        """Test error when record has no images"""
        from tests.metrics.conftest import create_record

        record = create_record(start_ns=100, responses=[150])
        # Create turns without images
        record.request.turns = [Turn(images=[])]

        metric = NumImagesMetric()
        with pytest.raises(NoMetricValue, match="at least one image"):
            metric.parse_record(record, MetricRecordDict())

    def test_num_images_no_turns(self):
        """Test error when record has no turns"""
        from tests.metrics.conftest import create_record

        record = create_record(start_ns=100, responses=[150])
        record.request.turns = []

        metric = NumImagesMetric()
        with pytest.raises(NoMetricValue, match="at least one image"):
            metric.parse_record(record, MetricRecordDict())


class TestImageThroughputMetric:
    @pytest.mark.parametrize(
        "images_per_turn,latency_ns,expected_throughput",
        [
            ([1], 1_000_000_000, 1.0),    # 1 image, 1 second = 1 img/s
            ([10], 2_000_000_000, 5.0),   # 10 images, 2 seconds = 5 img/s
            ([3], 500_000_000, 6.0),      # 3 images, 0.5 seconds = 6 img/s
            ([2, 3], 1_000_000_000, 5.0), # 5 images (2+3), 1 second = 5 img/s
        ],
    )  # fmt: skip
    def test_image_throughput_calculation(
        self, images_per_turn, latency_ns, expected_throughput
    ):
        """Test image throughput calculation with various configurations"""
        record = create_record_with_images(
            start_ns=0,
            responses=[latency_ns],
            images_per_turn=images_per_turn,
        )

        metric_results = run_image_metrics_pipeline([record], ImageThroughputMetric.tag)
        assert metric_results[ImageThroughputMetric.tag] == [expected_throughput]

    def test_image_throughput_multiple_records(self):
        """Test throughput across multiple records"""
        records = [
            create_record_with_images(
                start_ns=0, responses=[1_000_000_000], images_per_turn=[2]
            ),  # 2 img/s
            create_record_with_images(
                start_ns=0, responses=[500_000_000], images_per_turn=[3]
            ),  # 6 img/s
        ]

        metric_results = run_image_metrics_pipeline(records, ImageThroughputMetric.tag)
        assert metric_results[ImageThroughputMetric.tag] == [2.0, 6.0]


class TestImageLatencyMetric:
    @pytest.mark.parametrize(
        "images_per_turn,latency_ns,expected_latency_ms",
        [
            ([1], 1_000_000_000, 1000.0),     # 1000ms / 1 image = 1000 ms/img
            ([10], 1_000_000_000, 100.0),     # 1000ms / 10 images = 100 ms/img
            ([3], 500_000_000, 166.666666),   # 500ms / 3 images = ~166.67 ms/img
            ([2, 3], 1_000_000_000, 200.0),   # 1000ms / 5 images (2+3) = 200 ms/img
            ([1], 10_000_000, 10.0),          # 10ms / 1 image = 10 ms/img (fast processing)
        ],
    )  # fmt: skip
    def test_image_latency_calculation(
        self, images_per_turn, latency_ns, expected_latency_ms
    ):
        """Test image latency calculation with various configurations"""
        record = create_record_with_images(
            start_ns=0,
            responses=[latency_ns],
            images_per_turn=images_per_turn,
        )

        metric_results = run_image_metrics_pipeline([record], ImageLatencyMetric.tag)
        # Use approx comparison for floating point values
        assert metric_results[ImageLatencyMetric.tag][0] == pytest.approx(
            expected_latency_ms, rel=1e-5
        )

    def test_image_latency_multiple_records(self):
        """Test latency across multiple records"""
        records = [
            create_record_with_images(
                start_ns=0, responses=[1_000_000_000], images_per_turn=[2]
            ),  # 500 ms/img
            create_record_with_images(
                start_ns=0, responses=[500_000_000], images_per_turn=[5]
            ),  # 100 ms/img
        ]

        metric_results = run_image_metrics_pipeline(records, ImageLatencyMetric.tag)
        assert metric_results[ImageLatencyMetric.tag] == [500.0, 100.0]


class TestImageMetricsIntegration:
    def test_image_throughput_and_latency_are_inverses(self):
        """Test that throughput and latency are mathematical inverses"""
        # When dealing with the same record, throughput * latency = images * time_unit_conversion
        # images/sec * ms/image should equal images * 1000 (converting seconds to milliseconds)
        record = create_record_with_images(
            start_ns=0,
            responses=[2_000_000_000],  # 2 seconds
            images_per_turn=[4],
        )

        metric_results = run_image_metrics_pipeline(
            [record],
            ImageThroughputMetric.tag,
            ImageLatencyMetric.tag,
        )

        throughput = metric_results[ImageThroughputMetric.tag][0]  # images/second
        latency = metric_results[ImageLatencyMetric.tag][0]  # ms/image

        # throughput (img/s) * latency (ms/img) = ms/s = 1000
        assert abs(throughput * latency - 1000.0) < 0.001

    def test_all_metrics_together(self):
        """Test computing all image metrics together"""
        record = create_record_with_images(
            start_ns=0,
            responses=[1_000_000_000],  # 1 second
            images_per_turn=[2, 3],  # 5 images total
        )

        metric_results = run_image_metrics_pipeline(
            [record],
            NumImagesMetric.tag,
            ImageThroughputMetric.tag,
            ImageLatencyMetric.tag,
        )

        assert metric_results[NumImagesMetric.tag] == [5]
        assert metric_results[ImageThroughputMetric.tag] == [5.0]  # 5 images / 1 second
        assert metric_results[ImageLatencyMetric.tag] == [200.0]  # 1000ms / 5 images
