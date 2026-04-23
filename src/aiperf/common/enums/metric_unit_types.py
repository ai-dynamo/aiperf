# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.enums.metric_base import (
    BaseMetricUnit,
    BaseMetricUnitInfo,
    MetricUnitT,
)


class MetricSizeUnitInfo(BaseMetricUnitInfo):
    """Information about a size unit for metrics."""

    long_name: str
    num_bytes: int

    def convert_to(self, other_unit: "MetricUnitT", value: int | float) -> float:
        """Convert a value from this unit to another unit."""
        if not isinstance(other_unit, (MetricSizeUnit, MetricSizeUnitInfo)):
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
        if not isinstance(other_unit, (MetricTimeUnit, MetricTimeUnitInfo)):
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
        if not isinstance(other_unit, (PowerMetricUnit, PowerMetricUnitInfo)):
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
        if not isinstance(other_unit, (EnergyMetricUnit, EnergyMetricUnitInfo)):
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

        if isinstance(other_unit, (MetricOverTimeUnit, MetricOverTimeUnitInfo)):
            # Chain convert each unit to the other unit.
            value = self.primary_unit.convert_to(other_unit.primary_unit, value)
            value = self.time_unit.convert_to(other_unit.time_unit, value)
            if self.third_unit and other_unit.third_unit:
                value = self.third_unit.convert_to(other_unit.third_unit, value)
            return value

        # If the other unit is a time unit, convert our time unit to the other unit.
        # TODO: Should we even allow this?
        if isinstance(other_unit, (MetricTimeUnit, MetricTimeUnitInfo)):
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


class FrequencyMetricUnitInfo(BaseMetricUnitInfo):
    """Information about a frequency unit for metrics."""

    long_name: str
    hertz: float

    def convert_to(self, other_unit: "MetricUnitT", value: int | float) -> float:
        """Convert a value from this unit to another unit."""
        if not isinstance(other_unit, (FrequencyMetricUnit, FrequencyMetricUnitInfo)):
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
            other_unit, (TemperatureMetricUnit, TemperatureMetricUnitInfo)
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
