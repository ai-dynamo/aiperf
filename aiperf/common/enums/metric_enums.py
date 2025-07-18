# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class MetricTimeType(CaseInsensitiveStrEnum):
    """Defines the time types for metrics."""

    NANOSECONDS = "nanoseconds"
    MILLISECONDS = "milliseconds"
    SECONDS = "seconds"

    def short_name(self) -> str:
        """Get the short name for the time type."""
        _short_name_map = {
            MetricTimeType.NANOSECONDS: "ns",
            MetricTimeType.MILLISECONDS: "ms",
            MetricTimeType.SECONDS: "s",
        }
        return _short_name_map[self]


class MetricType(Enum):
    METRIC_OF_RECORDS = auto()
    METRIC_OF_METRICS = auto()
    METRIC_OF_BOTH = auto()
