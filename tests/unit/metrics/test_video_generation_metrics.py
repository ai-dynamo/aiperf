# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags, MetricSizeUnit, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponse, ParsedResponseRecord
from aiperf.common.models.record_models import TextResponseData, VideoResponseData
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.video_generation_metrics import (
    VideoInferenceTimeMetric,
    VideoPeakMemoryMetric,
)
from tests.unit.metrics.conftest import create_record


def create_video_record(
    start_ns: int = 100,
    response_ns: int = 150,
    inference_time_s: float | None = 5.5,
    peak_memory_mb: float | None = 8192.0,
) -> ParsedResponseRecord:
    """Create a test record with VideoResponseData."""
    record = create_record(start_ns=start_ns, responses=[response_ns])

    # Replace the response data with VideoResponseData
    record.responses = [
        ParsedResponse(
            perf_ns=response_ns,
            data=VideoResponseData(
                video_id="test-video-id",
                status="completed",
                inference_time_s=inference_time_s,
                peak_memory_mb=peak_memory_mb,
            ),
        )
    ]

    return record


class TestVideoInferenceTimeMetric:
    def test_metric_attributes(self):
        """Test that metric has correct attributes."""
        assert VideoInferenceTimeMetric.tag == "video_inference_time"
        assert VideoInferenceTimeMetric.header == "Video Inference Time"
        assert VideoInferenceTimeMetric.unit == MetricTimeUnit.SECONDS
        assert VideoInferenceTimeMetric.display_unit == MetricTimeUnit.MILLISECONDS
        assert VideoInferenceTimeMetric.has_flags(MetricFlags.PRODUCES_VIDEO_ONLY)

    @pytest.mark.parametrize(
        "inference_time_s,expected",
        [
            (5.5, 5.5),
            (0.0, 0.0),
            (123.456, 123.456),
            (0.001, 0.001),
        ],
        ids=["normal", "zero", "large", "small"],
    )
    def test_extracts_inference_time(self, inference_time_s, expected):
        """Test that metric extracts inference_time_s correctly."""
        record = create_video_record(inference_time_s=inference_time_s)
        metric = VideoInferenceTimeMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == expected

    def test_raises_when_inference_time_missing(self):
        """Test that NoMetricValue is raised when inference_time_s is None."""
        record = create_video_record(inference_time_s=None)
        metric = VideoInferenceTimeMetric()

        with pytest.raises(NoMetricValue, match="not available"):
            metric.parse_record(record, MetricRecordDict())

    def test_raises_when_no_video_response_data(self):
        """Test that NoMetricValue is raised when response is not VideoResponseData."""
        record = create_record(start_ns=100, responses=[150])
        # Default create_record uses TextResponseData
        metric = VideoInferenceTimeMetric()

        with pytest.raises(NoMetricValue, match="not available"):
            metric.parse_record(record, MetricRecordDict())

    def test_finds_video_data_in_multiple_responses(self):
        """Test that metric finds VideoResponseData among multiple responses."""
        record = create_record(start_ns=100, responses=[150])

        # Add multiple responses, with VideoResponseData in the middle
        record.responses = [
            ParsedResponse(
                perf_ns=120,
                data=TextResponseData(text="queued"),
            ),
            ParsedResponse(
                perf_ns=140,
                data=VideoResponseData(
                    video_id="test-id",
                    status="in_progress",
                    inference_time_s=None,
                ),
            ),
            ParsedResponse(
                perf_ns=150,
                data=VideoResponseData(
                    video_id="test-id",
                    status="completed",
                    inference_time_s=42.5,
                ),
            ),
        ]

        metric = VideoInferenceTimeMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 42.5

    def test_uses_first_valid_inference_time(self):
        """Test that metric uses first VideoResponseData with valid inference_time_s."""
        record = create_record(start_ns=100, responses=[150])

        record.responses = [
            ParsedResponse(
                perf_ns=120,
                data=VideoResponseData(
                    video_id="test-id",
                    status="completed",
                    inference_time_s=10.0,
                ),
            ),
            ParsedResponse(
                perf_ns=150,
                data=VideoResponseData(
                    video_id="test-id",
                    status="completed",
                    inference_time_s=20.0,
                ),
            ),
        ]

        metric = VideoInferenceTimeMetric()
        result = metric.parse_record(record, MetricRecordDict())
        # Should return first valid one
        assert result == 10.0


class TestVideoPeakMemoryMetric:
    def test_metric_attributes(self):
        """Test that metric has correct attributes."""
        assert VideoPeakMemoryMetric.tag == "video_peak_memory"
        assert VideoPeakMemoryMetric.header == "Video Peak Memory"
        assert VideoPeakMemoryMetric.unit == MetricSizeUnit.MEGABYTES
        assert VideoPeakMemoryMetric.has_flags(MetricFlags.PRODUCES_VIDEO_ONLY)

    @pytest.mark.parametrize(
        "peak_memory_mb,expected",
        [
            (8192.0, 8192.0),
            (0.0, 0.0),
            (16384.5, 16384.5),
            (1024.0, 1024.0),
        ],
        ids=["normal", "zero", "large", "small"],
    )
    def test_extracts_peak_memory(self, peak_memory_mb, expected):
        """Test that metric extracts peak_memory_mb correctly."""
        record = create_video_record(peak_memory_mb=peak_memory_mb)
        metric = VideoPeakMemoryMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == expected

    def test_raises_when_peak_memory_missing(self):
        """Test that NoMetricValue is raised when peak_memory_mb is None."""
        record = create_video_record(peak_memory_mb=None)
        metric = VideoPeakMemoryMetric()

        with pytest.raises(NoMetricValue, match="not available"):
            metric.parse_record(record, MetricRecordDict())

    def test_raises_when_no_video_response_data(self):
        """Test that NoMetricValue is raised when response is not VideoResponseData."""
        record = create_record(start_ns=100, responses=[150])
        metric = VideoPeakMemoryMetric()

        with pytest.raises(NoMetricValue, match="not available"):
            metric.parse_record(record, MetricRecordDict())
