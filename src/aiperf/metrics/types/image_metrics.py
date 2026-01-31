# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import GenericMetricUnit, MetricFlags, MetricOverTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_record_metric import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


class NumImagesMetric(BaseRecordMetric[int]):
    """Number of images metric."""

    tag = "num_images"
    header = "Number of Images"
    short_header = "Num Images"
    unit = GenericMetricUnit.IMAGES
    flags = MetricFlags.SUPPORTS_IMAGE_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        """Parse the number of images from the record by summing the number of images in each turn."""
        num_images = sum(
            len(image.contents)
            for turn in record.request.turns
            for image in turn.images
        )
        if num_images == 0:
            raise NoMetricValue(
                "Record must have at least one image in at least one turn."
            )
        return num_images


class ImageThroughputMetric(BaseRecordMetric[float]):
    """Image throughput metric."""

    tag = "image_throughput"
    header = "Image Throughput"
    short_header = "Image Throughput"
    display_order = 860
    unit = MetricOverTimeUnit.IMAGES_PER_SECOND
    flags = MetricFlags.SUPPORTS_IMAGE_ONLY
    required_metrics = {
        NumImagesMetric.tag,
        RequestLatencyMetric.tag,
    }

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> float:
        """Parse the image throughput from the record by dividing the number of images by the request latency."""
        num_images = record_metrics.get_or_raise(NumImagesMetric)
        request_latency_sec = record_metrics.get_converted_or_raise(
            RequestLatencyMetric, self.unit.time_unit
        )
        if request_latency_sec == 0:
            raise NoMetricValue("Request latency must be greater than 0.")
        return num_images / request_latency_sec


class ImageLatencyMetric(BaseRecordMetric[float]):
    """Image latency metric."""

    tag = "image_latency"
    header = "Image Latency"
    short_header = "Image Latency"
    display_order = 861
    unit = MetricOverTimeUnit.MS_PER_IMAGE
    flags = MetricFlags.SUPPORTS_IMAGE_ONLY
    required_metrics = {
        NumImagesMetric.tag,
        RequestLatencyMetric.tag,
    }

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> float:
        """Parse the image latency from the record by dividing the request latency by the number of images."""
        num_images = record_metrics.get_or_raise(NumImagesMetric)
        request_latency_ms = record_metrics.get_converted_or_raise(
            RequestLatencyMetric, self.unit.time_unit
        )
        if num_images == 0:
            raise NoMetricValue("Number of images must be greater than 0.")
        return request_latency_ms / num_images
