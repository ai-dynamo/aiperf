# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from enum import Flag
from functools import cached_property
from typing import Any

from aiperf.common.enums.base_enums import (
    BasePydanticBackedStrEnum,
    BasePydanticEnumInfo,
    CaseInsensitiveStrEnum,
)
from aiperf.common.enums.metric_base import (
    BaseMetricUnit,
    BaseMetricUnitInfo,
    MetricDictValueTypeT,
    MetricUnitT,
    MetricValueTypeT,
    MetricValueTypeVarT,
)
from aiperf.common.enums.metric_unit_types import (
    EnergyMetricUnit,
    EnergyMetricUnitInfo,
    FrequencyMetricUnit,
    FrequencyMetricUnitInfo,
    GenericMetricUnit,
    MetricOverTimeUnit,
    MetricOverTimeUnitInfo,
    MetricSizeUnit,
    MetricSizeUnitInfo,
    MetricTimeUnit,
    MetricTimeUnitInfo,
    PowerMetricUnit,
    PowerMetricUnitInfo,
    TemperatureMetricUnit,
    TemperatureMetricUnitInfo,
)

__all__ = [
    "BaseMetricUnit",
    "BaseMetricUnitInfo",
    "EnergyMetricUnit",
    "EnergyMetricUnitInfo",
    "FrequencyMetricUnit",
    "FrequencyMetricUnitInfo",
    "GenericMetricUnit",
    "MetricDictValueTypeT",
    "MetricFlags",
    "MetricOverTimeUnit",
    "MetricOverTimeUnitInfo",
    "MetricSizeUnit",
    "MetricSizeUnitInfo",
    "MetricTimeUnit",
    "MetricTimeUnitInfo",
    "MetricType",
    "MetricUnitT",
    "MetricValueType",
    "MetricValueTypeInfo",
    "MetricValueTypeT",
    "MetricValueTypeVarT",
    "PlotMetricDirection",
    "PowerMetricUnit",
    "PowerMetricUnitInfo",
    "TemperatureMetricUnit",
    "TemperatureMetricUnitInfo",
]


class MetricType(CaseInsensitiveStrEnum):
    """Defines the possible types of metrics."""

    RECORD = "record"
    """Metrics that provide a distinct value for each request. Every request that comes in will produce a new value that is not affected by any other requests.
    These metrics can be tracked over time and compared to each other.
    Examples: request latency, ISL, ITL, OSL, etc."""

    AGGREGATE = "aggregate"
    """Metrics that keep track of one or more values over time, that are updated for each request, such as total counts, min/max values, etc.
    These metrics may or may not change each request, and are affected by other requests.
    Examples: min/max request latency, total request count, benchmark duration, etc."""

    DERIVED = "derived"
    """Metrics that are purely derived from other metrics as a summary, and do not require per-request values.
    Examples: request throughput, output token throughput, etc."""


class PlotMetricDirection(CaseInsensitiveStrEnum):
    """Direction indicating whether higher or lower metric values are better for plotting purposes."""

    HIGHER = "higher"
    """Higher values are better (e.g., throughput, accuracy)."""

    LOWER = "lower"
    """Lower values are better (e.g., latency, error rate)."""


class MetricValueTypeInfo(BasePydanticEnumInfo):
    """Information about a metric value type."""

    default_factory: Callable[[], MetricValueTypeT]
    converter: Callable[[Any], MetricValueTypeT]
    dtype: Any


class MetricValueType(BasePydanticBackedStrEnum):
    """Defines the possible types of values for metrics.

    NOTE: The string representation (tag) is important here, as it is used to automatically determine the type
    based on the python generic type definition.
    """

    FLOAT = MetricValueTypeInfo(
        tag="float",
        default_factory=float,
        converter=float,
        dtype=float,
    )
    INT = MetricValueTypeInfo(
        tag="int",
        default_factory=int,
        converter=int,
        dtype=int,
    )
    FLOAT_LIST = MetricValueTypeInfo(
        tag="list[float]",
        default_factory=list,
        converter=lambda v: [float(x) for x in v],
        dtype=float,
    )
    INT_LIST = MetricValueTypeInfo(
        tag="list[int]",
        default_factory=list,
        converter=lambda v: [int(x) for x in v],
        dtype=int,
    )

    @cached_property
    def info(self) -> MetricValueTypeInfo:
        """Get the info for the metric value type."""
        return self._info  # type: ignore

    @cached_property
    def default_factory(self) -> Callable[[], MetricValueTypeT]:
        """Get the default value generator for the metric value type."""
        return self.info.default_factory

    @cached_property
    def converter(self) -> Callable[[Any], MetricValueTypeT]:
        """Get the converter for the metric value type."""
        return self.info.converter

    @cached_property
    def dtype(self) -> Any:
        """Get the dtype for the metric value type (for numpy)."""
        return self.info.dtype

    @classmethod
    def from_python_type(cls, type: type[MetricValueTypeT]) -> "MetricValueType":
        """Get the MetricValueType for a given type."""
        # If the type is a simple type like float or int, we have to use __name__.
        # This is because using str() on float or int will return <class 'float'> or <class 'int'>, etc.
        type_name = type.__name__
        if type_name == "list":
            # However, if the type is a list, we have to use str() to get the list type as well, e.g. list[int]
            type_name = str(type)
        elif type_name == "MetricValueTypeVarT":
            type_name = "float"  # Default to float if the user did not specify a type.
        return MetricValueType(type_name)


class MetricFlags(Flag):
    """Defines the possible flags for metrics that are used to determine how they are processed or grouped.
    These flags are intended to be an easy way to group metrics, or turn on/off certain features.

    Note that the flags are a bitmask, so they can be combined using the bitwise OR operator (`|`).
    For example, to create a flag that is both `STREAMING_ONLY` and `NO_CONSOLE`, you can do:
    ```python
    MetricFlags.STREAMING_ONLY | MetricFlags.NO_CONSOLE
    ```

    To check if a metric has a flag, you can use the `has_flags` method.
    For example, to check if a metric has both the `STREAMING_ONLY` and `NO_CONSOLE` flags, you can do:
    ```python
    metric.has_flags(MetricFlags.STREAMING_ONLY | MetricFlags.NO_CONSOLE)
    ```

    To check if a metric does not have a flag(s), you can use the `missing_flags` method.
    For example, to check if a metric does not have either the `STREAMING_ONLY` or `NO_CONSOLE` flags, you can do:
    ```python
    metric.missing_flags(MetricFlags.STREAMING_ONLY | MetricFlags.NO_CONSOLE)
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

    NO_CONSOLE = 1 << 3
    """Metrics that should not be displayed in the console output, but still exported to files."""

    LARGER_IS_BETTER = 1 << 4
    """Metrics that are better when the value is larger. By default, it is assumed that metrics are
    better when the value is smaller."""

    INTERNAL = 1 << 5
    """Metrics that are internal to the system and not applicable to the user.
    They will not be displayed in the console output or exported to files without developer mode enabled."""

    SUPPORTS_AUDIO_ONLY = 1 << 6
    """Metrics that are only applicable to audio-based endpoints."""

    SUPPORTS_IMAGE_ONLY = 1 << 7
    """Metrics that are only applicable to image-based endpoints."""

    SUPPORTS_REASONING = 1 << 8
    """Metrics that are only applicable to reasoning-based models and endpoints."""

    EXPERIMENTAL = 1 << 9
    """Metrics that are experimental and are not yet ready for production use, and may be subject to change.
    They will not be displayed in the console output or exported to files without developer mode enabled."""

    STREAMING_TOKENS_ONLY = STREAMING_ONLY | PRODUCES_TOKENS_ONLY
    """Metrics that are only applicable to streamed responses and token-based endpoints.
    This is a convenience flag that is the combination of the `STREAMING_ONLY` and `PRODUCES_TOKENS_ONLY` flags."""

    GOODPUT = 1 << 10
    """Metrics that are only applicable when goodput feature is enabled"""

    NO_INDIVIDUAL_RECORDS = 1 << 11
    """Metrics that should not be exported for individual records. These are typically aggregate metrics.
    This is used to filter out metrics such as request count or min/max timestamps that are not relevant to individual records."""

    TOKENIZES_INPUT_ONLY = 1 << 12
    """Metrics that are only applicable when the endpoint tokenizes input text."""

    SUPPORTS_VIDEO_ONLY = 1 << 13
    """Metrics that are only applicable to video-based endpoints."""

    USAGE_DIFF_ONLY = 1 << 14
    """Metrics that are only applicable when client side tokenization is enabled and the usage field is used."""

    HTTP_TRACE_ONLY = 1 << 15
    """Metrics that are only applicable to HTTP trace data (AioHttpTraceData)."""

    PRODUCES_VIDEO_ONLY = 1 << 16
    """Metrics that are only applicable when profiling an endpoint that produces video output."""

    def has_flags(self, flags: "MetricFlags") -> bool:
        """Return True if the metric has ALL of the given flag(s) (regardless of other flags)."""
        # Bitwise AND will return the input flags only if all of the given flags are present.
        return (flags & self) == flags

    def has_any_flags(self, flags: "MetricFlags") -> bool:
        """Return True if the metric has ANY of the given flag(s) (regardless of other flags)."""
        return (flags & self) != MetricFlags.NONE

    def missing_flags(self, flags: "MetricFlags") -> bool:
        """Return True if the metric does not have ANY of the given flag(s) (regardless of other flags). It will
        return False if the metric has ANY of the given flags. If the input flags are NONE, it will return True."""
        if flags == MetricFlags.NONE:
            return True  # If there are no flags to check, return True

        # Bitwise AND will return 0 (MetricFlags.NONE) if there are no common flags.
        # If there are some missing, but some found, the result will not be 0.
        return (self & flags) == MetricFlags.NONE
