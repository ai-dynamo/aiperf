# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from enum import Flag
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.enums.base_enums import (
    BasePydanticBackedStrEnum,
    BasePydanticEnumInfo,
    CaseInsensitiveStrEnum,
)
from aiperf.common.exceptions import MetricUnitError

if TYPE_CHECKING:
    from aiperf.metrics.metric_dicts import MetricArray

MetricValueTypeT: TypeAlias = int | float | list[float] | list[int]
MetricValueTypeVarT = TypeVar("MetricValueTypeVarT", bound=MetricValueTypeT)
MetricDictValueTypeT: TypeAlias = (
    "MetricValueTypeT | list[MetricValueTypeT] | MetricArray"
)


class BaseMetricUnitInfo(BasePydanticEnumInfo):
    """Base class for all metric units. Provides a base implementation for converting between units which
    can be overridden by subclasses to support more complex conversions.
    """

    def convert_to(self, other_unit: "MetricUnitT", value: int | float) -> float:
        """Convert a value from this unit to another unit."""
        # If the other unit is the same as this unit, return the value. This allows for chaining conversions,
        # as well as if a type does not have a conversion method, we do not want to raise an error if the conversion is a no-op.
        if other_unit == self:
            return value

        # Otherwise, we cannot convert between the two units.
        raise MetricUnitError(
            f"Cannot convert from '{self}' to '{other_unit}'.",
        )


class BaseMetricUnit(BasePydanticBackedStrEnum):
    """Base class for all metric units."""

    def display_name(self) -> str:
        """Get the display name of the metric unit."""
        return self.name.lower().replace("_per_second", "/s")

    @cached_property
    def info(self) -> BaseMetricUnitInfo:
        """Get the info for the metric unit."""
        return self._info  # type: ignore

    def convert_to(self, other_unit: "MetricUnitT", value: int | float) -> float:
        """Convert a value from this unit to another unit. This is a passthrough to the info class."""
        return self.info.convert_to(other_unit, value)


# We allow either an actual enum unit, or an info object that can act like a unit.
MetricUnitT: TypeAlias = BaseMetricUnit | BaseMetricUnitInfo


class MetricSizeUnitInfo(BaseMetricUnitInfo):
    """Information about a size unit for metrics."""

    long_name: str
    num_bytes: int

    def convert_to(self, other_unit: "MetricUnitT", value: int | float) -> float:
        """Convert a value from this unit to another unit."""
        if not isinstance(other_unit, MetricSizeUnit | MetricSizeUnitInfo):
            return super().convert_to(other_unit, value)

        return value * (self.num_bytes / other_unit.num_bytes)


class MetricSizeUnit(BaseMetricUnit):
    """Defines the size types for metrics."""

    BYTES = MetricSizeUnitInfo(
        tag="B",
        long_name="bytes",
        num_bytes=1,
    )
    KILOBYTES = MetricSizeUnitInfo(
        tag="KB",
        long_name="kilobytes",
        num_bytes=1024,
    )
    MEGABYTES = MetricSizeUnitInfo(
        tag="MB",
        long_name="megabytes",
        num_bytes=1024 * 1024,
    )
    GIGABYTES = MetricSizeUnitInfo(
        tag="GB",
        long_name="gigabytes",
        num_bytes=1024 * 1024 * 1024,
    )
    TERABYTES = MetricSizeUnitInfo(
        tag="TB",
        long_name="terabytes",
        num_bytes=1024 * 1024 * 1024 * 1024,
    )

    @cached_property
    def info(self) -> MetricSizeUnitInfo:
        """Get the info for the metric size unit."""
        return self._info  # type: ignore

    @cached_property
    def num_bytes(self) -> int:
        """The number of bytes in the metric size unit."""
        return self.info.num_bytes

    @cached_property
    def long_name(self) -> str:
        """The long name of the metric size unit."""
        return self.info.long_name


class MetricTimeUnitInfo(BaseMetricUnitInfo):
    """Information about a time unit for metrics."""

    long_name: str
    per_second: int


class MetricTimeUnit(BaseMetricUnit):
    """Defines the various time units that can be used for metrics, as well as the conversion factor to convert to other units."""

    NANOSECONDS = MetricTimeUnitInfo(
        tag="ns",
        long_name="nanoseconds",
        per_second=1_000_000_000,
    )
    MICROSECONDS = MetricTimeUnitInfo(
        tag="us",
        long_name="microseconds",
        per_second=1_000_000,
    )
    MILLISECONDS = MetricTimeUnitInfo(
        tag="ms",
        long_name="milliseconds",
        per_second=1_000,
    )
    SECONDS = MetricTimeUnitInfo(
        tag="sec",
        long_name="seconds",
        per_second=1,
    )

    @cached_property
    def info(self) -> MetricTimeUnitInfo:
        """Get the info for the metric time unit."""
        return self._info  # type: ignore

    @cached_property
    def per_second(self) -> int:
        """How many of these units there are in one second. Used as a common conversion factor to convert to other units."""
        return self.info.per_second

    @cached_property
    def long_name(self) -> str:
        """The long name of the metric time unit."""
        return self.info.long_name

    def convert_to(self, other_unit: "MetricUnitT", value: int | float) -> float:
        """Convert a value from this unit to another unit."""
        if not isinstance(other_unit, MetricTimeUnit | MetricTimeUnitInfo):
            return super().convert_to(other_unit, value)

        return value * (other_unit.per_second / self.per_second)


# Syntactic sugar for creating BaseMetricUnitInfo instances with a tag
def _unit(tag: str) -> BaseMetricUnitInfo:
    return BaseMetricUnitInfo(tag=tag)


class GenericMetricUnit(BaseMetricUnit):
    """Defines generic units for metrics. These dont have any extra information other than the tag, which is used for display purposes."""

    BLOCKS = _unit("blocks")
    COUNT = _unit("count")
    ERRORS = _unit("errors")
    IMAGE = _unit("image")
    IMAGES = _unit("images")
    PERCENT = _unit("%")
    RATIO = _unit("ratio")
    REQUESTS = _unit("requests")
    TOKENS = _unit("tokens")
    USER = _unit("user")
    USERS = _unit("users")
    VIDEO = _unit("video")
    VIDEOS = _unit("videos")


class PowerMetricUnitInfo(BaseMetricUnitInfo):
    """Information about a power unit for metrics."""

    long_name: str
    watts: float

    def convert_to(self, other_unit: "MetricUnitT", value: int | float) -> float:
        """Convert a value from this unit to another unit."""
        if not isinstance(other_unit, PowerMetricUnit | PowerMetricUnitInfo):
            return super().convert_to(other_unit, value)

        return value * (self.watts / other_unit.watts)


class PowerMetricUnit(BaseMetricUnit):
    """Defines power units for metrics."""

    WATT = PowerMetricUnitInfo(
        tag="W",
        long_name="watts",
        watts=1.0,
    )
    MILLIWATT = PowerMetricUnitInfo(
        tag="mW",
        long_name="milliwatts",
        watts=0.001,
    )

    @cached_property
    def info(self) -> PowerMetricUnitInfo:
        """Get the info for the power unit."""
        return self._info  # type: ignore

    @cached_property
    def watts(self) -> float:
        """The number of watts in the power unit."""
        return self.info.watts

    @cached_property
    def long_name(self) -> str:
        """The long name of the power unit."""
        return self.info.long_name


class EnergyMetricUnitInfo(BaseMetricUnitInfo):
    """Information about an energy unit for metrics."""

    long_name: str
    joules: float

    def convert_to(self, other_unit: "MetricUnitT", value: int | float) -> float:
        """Convert a value from this unit to another unit."""
        if not isinstance(other_unit, EnergyMetricUnit | EnergyMetricUnitInfo):
            return super().convert_to(other_unit, value)

        return value * (self.joules / other_unit.joules)


class EnergyMetricUnit(BaseMetricUnit):
    """Defines energy units for metrics."""

    JOULE = EnergyMetricUnitInfo(
        tag="J",
        long_name="joules",
        joules=1.0,
    )
    MILLIJOULE = EnergyMetricUnitInfo(
        tag="mJ",
        long_name="millijoules",
        joules=0.001,
    )
    MEGAJOULE = EnergyMetricUnitInfo(
        tag="MJ",
        long_name="megajoules",
        joules=1_000_000.0,
    )

    @cached_property
    def info(self) -> EnergyMetricUnitInfo:
        """Get the info for the energy unit."""
        return self._info  # type: ignore

    @cached_property
    def joules(self) -> float:
        """The number of joules in the energy unit."""
        return self.info.joules

    @cached_property
    def long_name(self) -> str:
        """The long name of the energy unit."""
        return self.info.long_name


class MetricOverTimeUnitInfo(BaseMetricUnitInfo):
    """Information about a metric over time unit."""

    @model_validator(mode="after")
    def _set_tag(self: Self) -> Self:
        """Set the tag based on the existing units. ie. requests/sec, tokens/sec, etc."""
        self.tag = (
            f"{self.primary_unit}/{self.time_unit}"
            if not self.inverted
            else f"{self.time_unit}/{self.primary_unit}"
        )
        if self.third_unit:
            # If there is a third unit, add it to the tag. ie. tokens/sec/user
            self.tag += f"/{self.third_unit}"
        return self

    tag: str = Field(
        default="",
        description="The tag for the metric over time unit. This will be set automatically by the model_validator.",
    )
    primary_unit: "MetricUnitT"
    time_unit: MetricTimeUnit | MetricTimeUnitInfo
    third_unit: "MetricUnitT | None" = None
    inverted: bool = False

    def convert_to(self, other_unit: "MetricUnitT", value: int | float) -> float:
        """Convert a value from this unit to another unit."""
        # If the other unit is the same as this unit, return the value.
        if other_unit == self:
            return value

        if isinstance(other_unit, MetricOverTimeUnit | MetricOverTimeUnitInfo):
            # Chain convert each unit to the other unit.
            value = self.primary_unit.convert_to(other_unit.primary_unit, value)
            value = self.time_unit.convert_to(other_unit.time_unit, value)
            if self.third_unit and other_unit.third_unit:
                value = self.third_unit.convert_to(other_unit.third_unit, value)
            return value

        # If the other unit is a time unit, convert our time unit to the other unit.
        # TODO: Should we even allow this?
        if isinstance(other_unit, MetricTimeUnit | MetricTimeUnitInfo):
            return self.time_unit.convert_to(other_unit, value)

        # Otherwise, convert the primary unit to the other unit.
        return self.primary_unit.convert_to(other_unit, value)


class MetricOverTimeUnit(BaseMetricUnit):
    """Defines the units for metrics that are a generic unit over a specific time unit."""

    REQUESTS_PER_SECOND = MetricOverTimeUnitInfo(
        primary_unit=GenericMetricUnit.REQUESTS,
        time_unit=MetricTimeUnit.SECONDS,
    )
    TOKENS_PER_SECOND = MetricOverTimeUnitInfo(
        primary_unit=GenericMetricUnit.TOKENS,
        time_unit=MetricTimeUnit.SECONDS,
    )
    TOKENS_PER_SECOND_PER_USER = MetricOverTimeUnitInfo(
        primary_unit=GenericMetricUnit.TOKENS,
        time_unit=MetricTimeUnit.SECONDS,
        third_unit=GenericMetricUnit.USER,
    )
    IMAGES_PER_SECOND = MetricOverTimeUnitInfo(
        primary_unit=GenericMetricUnit.IMAGES,
        time_unit=MetricTimeUnit.SECONDS,
    )
    MS_PER_IMAGE = MetricOverTimeUnitInfo(
        time_unit=MetricTimeUnit.MILLISECONDS,
        primary_unit=GenericMetricUnit.IMAGE,
        inverted=True,
    )
    VIDEOS_PER_SECOND = MetricOverTimeUnitInfo(
        primary_unit=GenericMetricUnit.VIDEOS,
        time_unit=MetricTimeUnit.SECONDS,
    )
    MS_PER_VIDEO = MetricOverTimeUnitInfo(
        time_unit=MetricTimeUnit.MILLISECONDS,
        primary_unit=GenericMetricUnit.VIDEO,
        inverted=True,
    )
    MB_PER_SECOND = MetricOverTimeUnitInfo(
        primary_unit=MetricSizeUnit.MEGABYTES,
        time_unit=MetricTimeUnit.SECONDS,
    )
    GB_PER_SECOND = MetricOverTimeUnitInfo(
        primary_unit=MetricSizeUnit.GIGABYTES,
        time_unit=MetricTimeUnit.SECONDS,
    )

    @cached_property
    def info(self) -> MetricOverTimeUnitInfo:
        """Get the info for the metric over time unit."""
        return self._info  # type: ignore

    @cached_property
    def primary_unit(self) -> "MetricUnitT":
        """Get the primary unit."""
        return self.info.primary_unit

    @cached_property
    def time_unit(self) -> MetricTimeUnit | MetricTimeUnitInfo:
        """Get the time unit."""
        return self.info.time_unit

    @cached_property
    def third_unit(self) -> "MetricUnitT | None":
        """Get the third unit (if applicable)."""
        return self.info.third_unit

    @cached_property
    def inverted(self) -> bool:
        """Whether the metric is inverted (e.g. time / metric)."""
        return self.info.inverted


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


class FrequencyMetricUnitInfo(BaseMetricUnitInfo):
    """Information about a frequency unit for metrics."""

    long_name: str
    hertz: float

    def convert_to(self, other_unit: "MetricUnitT", value: int | float) -> float:
        """Convert a value from this unit to another unit."""
        if not isinstance(other_unit, FrequencyMetricUnit | FrequencyMetricUnitInfo):
            return super().convert_to(other_unit, value)

        return value * (self.hertz / other_unit.hertz)


class FrequencyMetricUnit(BaseMetricUnit):
    """Defines frequency units for metrics."""

    HERTZ = FrequencyMetricUnitInfo(
        tag="Hz",
        long_name="hertz",
        hertz=1.0,
    )
    MEGAHERTZ = FrequencyMetricUnitInfo(
        tag="MHz",
        long_name="megahertz",
        hertz=1_000_000.0,
    )
    GIGAHERTZ = FrequencyMetricUnitInfo(
        tag="GHz",
        long_name="gigahertz",
        hertz=1_000_000_000.0,
    )

    @cached_property
    def info(self) -> FrequencyMetricUnitInfo:
        """Get the info for the frequency unit."""
        return self._info  # type: ignore

    @cached_property
    def hertz(self) -> float:
        """The number of hertz in the frequency unit."""
        return self.info.hertz

    @cached_property
    def long_name(self) -> str:
        """The long name of the frequency unit."""
        return self.info.long_name


class TemperatureMetricUnitInfo(BaseMetricUnitInfo):
    """Information about a temperature unit for metrics."""

    long_name: str
    celsius: float
    offset: float = 0.0

    def convert_to(self, other_unit: "MetricUnitT", value: int | float) -> float:
        """Convert a value from this unit to another unit."""
        if not isinstance(
            other_unit, TemperatureMetricUnit | TemperatureMetricUnitInfo
        ):
            return super().convert_to(other_unit, value)

        # Convert to Celsius first, then to target unit
        celsius_value = (value + self.offset) * self.celsius
        return (celsius_value / other_unit.celsius) - other_unit.offset


class TemperatureMetricUnit(BaseMetricUnit):
    """Defines temperature units for metrics."""

    CELSIUS = TemperatureMetricUnitInfo(
        tag="°C",
        long_name="celsius",
        celsius=1.0,
        offset=0.0,
    )
    FAHRENHEIT = TemperatureMetricUnitInfo(
        tag="°F",
        long_name="fahrenheit",
        celsius=5.0 / 9.0,
        offset=-32.0,
    )
    KELVIN = TemperatureMetricUnitInfo(
        tag="K",
        long_name="kelvin",
        celsius=1.0,
        offset=-273.15,
    )

    @cached_property
    def info(self) -> TemperatureMetricUnitInfo:
        """Get the info for the temperature unit."""
        return self._info  # type: ignore

    @cached_property
    def celsius(self) -> float:
        """The celsius conversion factor."""
        return self.info.celsius

    @cached_property
    def offset(self) -> float:
        """The offset for temperature conversion."""
        return self.info.offset

    @cached_property
    def long_name(self) -> str:
        """The long name of the temperature unit."""
        return self.info.long_name


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
