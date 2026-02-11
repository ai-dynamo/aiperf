# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricSizeUnit, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.models.record_models import VideoResponseData
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class VideoInferenceTimeMetric(BaseRecordMetric[float]):
    """Server-reported video inference time from SGLang.

    Extracts the `inference_time_s` field from the video generation response,
    which represents the actual GPU generation time reported by the server.
    """

    tag = "video_inference_time"
    header = "Video Inference Time"
    short_header = "Inference Time"
    unit = MetricTimeUnit.SECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 310  # After request_latency (300)
    flags = MetricFlags.PRODUCES_VIDEO_ONLY

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """Extract inference_time_s from VideoResponseData."""
        for response in record.responses:
            if (
                isinstance(response.data, VideoResponseData)
                and response.data.inference_time_s is not None
            ):
                return response.data.inference_time_s

        raise NoMetricValue("Video inference time not available in response.")


class VideoPeakMemoryMetric(BaseRecordMetric[float]):
    """Server-reported peak GPU memory usage from SGLang.

    Extracts the `peak_memory_mb` field from the video generation response,
    which represents the peak GPU memory used during generation.
    """

    tag = "video_peak_memory"
    header = "Video Peak Memory"
    short_header = "Peak Memory"
    unit = MetricSizeUnit.MEGABYTES
    display_order = 311  # After video_inference_time (310)
    flags = MetricFlags.PRODUCES_VIDEO_ONLY

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """Extract peak_memory_mb from VideoResponseData."""
        for response in record.responses:
            if (
                isinstance(response.data, VideoResponseData)
                and response.data.peak_memory_mb is not None
            ):
                return response.data.peak_memory_mb

        raise NoMetricValue("Video peak memory not available in response.")
