# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from aiperf.common.enums.base_enums import (
    BasePydanticBackedStrEnum,
    BasePydanticEnumInfo,
)
from aiperf.common.exceptions import MetricUnitError

if TYPE_CHECKING:
    from aiperf.metrics.metric_dicts import MetricSeriesProtocol

MetricValueTypeT: TypeAlias = int | float | list[float] | list[int]
MetricValueTypeVarT = TypeVar("MetricValueTypeVarT", bound=MetricValueTypeT)
MetricDictValueTypeT: TypeAlias = (
    "MetricValueTypeT | list[MetricValueTypeT] | MetricSeriesProtocol"
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
