# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class PostProcessorType(CaseInsensitiveStrEnum):
    METRIC_SUMMARY = "metric_summary"


class StreamingPostProcessorType(CaseInsensitiveStrEnum):
    """Type of response streamer."""

    PROCESSING_STATS = "processing_stats"
    """Streamer that provides the processing stats of the records."""

    BASIC_METRICS = "basic_metrics"
    """Streamer that handles the basic metrics of the records."""

    JSONL = "jsonl"
    """Streams all parsed records to a JSONL file."""
