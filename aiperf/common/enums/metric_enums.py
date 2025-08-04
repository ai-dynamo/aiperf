# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, Flag, auto

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


class MetricTag(CaseInsensitiveStrEnum):
    BENCHMARK_DURATION = "benchmark_duration"
    ISL = "isl"
    INTER_TOKEN_LATENCY = "inter_token_latency"
    MAX_RESPONSE = "max_response"
    MIN_REQUEST = "min_request"
    OSL = "osl"
    OUTPUT_TOKEN_COUNT = "output_token_count"
    OUTPUT_TOKEN_THROUGHPUT = "output_token_throughput"
    OUTPUT_TOKEN_THROUGHPUT_PER_USER = "output_token_throughput_per_user"
    REQUEST_COUNT = "request_count"
    REQUEST_LATENCY = "request_latency"
    REQUEST_THROUGHPUT = "request_throughput"
    TTFT = "ttft"
    TTST = "ttst"


class MetricFlags(Flag):
    """Defines the possible flags for metrics that are used to determine how they are processed or grouped.
    These flags are intended to be an easy way to group metrics, or turn on/off certain features.

    Note that the flags are a bitmask, so they can be combined using the bitwise OR operator (`|`).
    For example, to create a flag that is both `STREAMING_ONLY` and `HIDDEN`, you can do:
    ```python
    MetricFlags.STREAMING_ONLY | MetricFlags.HIDDEN
    ```

    To check if a metric has a flag, you can use the `has_flags` method.
    For example, to check if a metric has both the `STREAMING_ONLY` and `HIDDEN` flags, you can do:
    ```python
    metric.has_flags(MetricFlags.STREAMING_ONLY | MetricFlags.HIDDEN)
    ```

    To check if a metric does not have a flag(s), you can use the `missing_flags` method.
    For example, to check if a metric does not have either the `STREAMING_ONLY` or `HIDDEN` flags, you can do:
    ```python
    metric.missing_flags(MetricFlags.STREAMING_ONLY | MetricFlags.HIDDEN)
    ```
    """

    # NOTE: The flags are a bitmask, so they must be powers of 2 (or a combination thereof).

    NONE = 0
    """No flags."""

    STREAMING_ONLY = 1 << 0
    """Metrics that are only applicable to streamed responses."""

    ERROR_ONLY = 1 << 1
    """Metrics that are only applicable to error records. By default, metrics are only computed if the record is valid.
    If this flag is set, the metric will only be computed if the record is invalid."""

    PRODUCES_TOKENS_ONLY = 1 << 2
    """Metrics that are only applicable when profiling an endpoint that produces tokens."""

    HIDDEN = 1 << 3
    """Metrics that should not be displayed in the UI."""

    LARGER_IS_BETTER = 1 << 4
    """Metrics that are better when the value is larger. By default, it is assumed that metrics are
    better when the value is smaller."""

    HIDE_IF_ZERO = 1 << 5
    """Metrics that should be hidden if the value is 0, such as error counts."""

    INTERNAL = (1 << 6) | HIDDEN
    """Metrics that are internal to the system and not applicable to the user. This inherently means that the metric
    is HIDDEN as well."""

    SUPPORTS_AUDIO_ONLY = 1 << 7
    """Metrics that are only applicable to audio-based endpoints."""

    SUPPORTS_IMAGE_ONLY = 1 << 8
    """Metrics that are only applicable to image-based endpoints."""

    STREAMING_TOKENS_ONLY = STREAMING_ONLY | PRODUCES_TOKENS_ONLY
    """Metrics that are only applicable to streamed responses and token-based endpoints.
    This is a convenience flag that is the combination of the `STREAMING_ONLY` and `PRODUCES_TOKENS_ONLY` flags."""

    def has_flags(self, flags: "MetricFlags") -> bool:
        """Return True if the metric has ALL of the given flag(s) (regardless of other flags)."""
        # Bitwise AND will return the input flags only if all of the given flags are present.
        return (flags & self) == flags

    def missing_flags(self, flags: "MetricFlags") -> bool:
        """Return True if the metric does not have ANY of the given flag(s) (regardless of other flags). It will
        return False if the metric has ANY of the given flags. If the input flags are NONE, it will return True."""
        if flags == MetricFlags.NONE:
            return True  # If there are no flags to check, return True

        # Bitwise AND will return 0 (MetricFlags.NONE) if there are no common flags.
        # If there are some missing, but some found, the result will not be 0.
        return (self & flags) == MetricFlags.NONE
