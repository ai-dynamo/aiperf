# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from abc import ABC
from typing import ClassVar, Generic, get_args, get_origin

from aiperf.common.enums import (
    MetricFlags,
    MetricType,
    MetricUnitT,
    MetricValueType,
    MetricValueTypeVarT,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.types import MetricTagT
from aiperf.metrics.metric_dicts import (
    MetricRecordDict,
    MetricResultsDict,
)


class BaseMetric(Generic[MetricValueTypeVarT], ABC):
    """A definition of a metric type.

    This class is not meant to be instantiated directly or subclassed directly.
    It is meant to be a common base for all of the base metric classes by type.

    The class attributes are:
    - tag: The tag of the metric. This must be a non-empty string that uniquely identifies the metric type.
    - header: The header of the metric. This is the user-friendly name of the metric that will be displayed in the Console Export.
    - short_header: The short header of the metric. This is the shortened user-friendly name of the metric for display in the Dashboard.
    - unit: The unit of the internal representation of the metric. This is used for converting to other units and for display.
    - display_unit: The unit of the metric that is used for display (if different from the unit). None means use the unit for display.
    - short_header_hide_unit: If True, the unit will not be displayed in the Dashboard short header.
    - display_order: The display order in the ConsoleExporter. Lower numbers are displayed first. None means unordered after any ordered metrics.
    - flags: The flags of the metric that determine how and when it is computed and displayed.
    - required_metrics: The metrics that must be available to compute the metric. This is a set of metric tags.
    """

    # User-defined attributes to be overridden by subclasses
    tag: ClassVar[MetricTagT]
    header: ClassVar[str] = ""
    short_header: ClassVar[str | None] = None
    short_header_hide_unit: ClassVar[bool] = False
    unit: ClassVar[MetricUnitT]
    display_unit: ClassVar[MetricUnitT | None] = None
    display_order: ClassVar[int | None] = None
    flags: ClassVar[MetricFlags] = MetricFlags.NONE
    required_metrics: ClassVar[set[MetricTagT] | None] = None

    # Auto-derived attributes
    value_type: ClassVar[MetricValueType]  # Auto set based on generic type parameter
    type: ClassVar[MetricType]  # Set by base subclasses

    def __init_subclass__(cls, **kwargs):
        """
        This method is called when a class is subclassed from Metric.
        It automatically registers the subclass in the MetricRegistry
        dictionary using the `tag` class attribute.
        The `tag` attribute must be a non-empty string that uniquely identifies the
        metric type. Only concrete (non-abstract) classes will be registered.
        """

        super().__init_subclass__(**kwargs)

        # Only register concrete classes (not abstract ones)
        if inspect.isabstract(cls) or (
            hasattr(cls, "__is_abstract__") and cls.__is_abstract__
        ):
            return

        # Verify that the class is a valid metric type
        # Make sure to do this after checking for abstractness, so that the imports are available
        cls._verify_base_class()

        # Import MetricRegistry here to avoid circular imports
        from aiperf.metrics.metric_registry import MetricRegistry

        # Enforce that subclasses define a non-empty tag
        if not cls.tag or not isinstance(cls.tag, str):
            raise TypeError(
                f"Concrete metric class {cls.__name__} must define a non-empty 'tag' class attribute"
            )

        # Auto-detect value type from generic parameter
        cls.value_type = cls._detect_value_type()

        MetricRegistry.register_metric(cls)

    @classmethod
    def _verify_base_class(cls) -> None:
        """Verify that the class is a subclass of BaseRecordMetric, BaseAggregateMetric, or BaseDerivedMetric.
        This is done to ensure that the class is a valid metric type.
        """
        # Use direct imports to avoid circular import issues during metric discovery
        # (aiperf.metrics.__init__ may not be fully loaded when this is called)
        from aiperf.metrics.base_aggregate_metric import BaseAggregateMetric
        from aiperf.metrics.base_derived_metric import BaseDerivedMetric
        from aiperf.metrics.base_record_metric import BaseRecordMetric

        # Enforce that concrete subclasses are a subclass of BaseRecordMetric, BaseAggregateMetric, or BaseDerivedMetric
        valid_base_classes = {
            BaseRecordMetric,
            BaseAggregateMetric,
            BaseDerivedMetric,
        }
        if not any(issubclass(cls, base) for base in valid_base_classes):
            raise TypeError(
                f"Concrete metric class {cls.__name__} must be a subclass of BaseRecordMetric, BaseAggregateMetric, or BaseDerivedMetric"
            )

    @classmethod
    def _detect_value_type(cls) -> MetricValueType:
        """Automatically detect the MetricValueType from the generic type parameter."""
        # Look through the class hierarchy for the first Generic[Type] definition
        for base in cls.__orig_bases__:  # type: ignore
            if get_origin(base) is not None:
                args = get_args(base)
                if args:
                    # the first argument is the generic type
                    generic_type = args[0]
                    return MetricValueType.from_python_type(generic_type)

        raise ValueError(
            f"Unable to detect the value type for {cls.__name__}. Please check the generic type parameter."
        )

    def _require_valid_record(self, record: ParsedResponseRecord) -> None:
        """Check that the record is valid."""
        if (not record or not record.valid) and not self.has_flags(
            MetricFlags.ERROR_ONLY
        ):
            raise NoMetricValue("Invalid Record")

    def _check_metrics(self, metrics: MetricRecordDict | MetricResultsDict) -> None:
        """Check that the required metrics are available."""
        if self.required_metrics is None:
            return
        for tag in self.required_metrics:
            if tag not in metrics:
                raise NoMetricValue(f"Missing required metric: '{tag}'")

    @classmethod
    def has_flags(cls, flags: MetricFlags) -> bool:
        """Return True if the metric has the given flag(s) (regardless of other flags)."""
        return cls.flags.has_flags(flags)

    @classmethod
    def has_any_flags(cls, flags: MetricFlags) -> bool:
        """Return True if the metric has ANY of the given flag(s) (regardless of other flags)."""
        return cls.flags.has_any_flags(flags)

    @classmethod
    def missing_flags(cls, flags: MetricFlags) -> bool:
        """Return True if the metric does not have the given flag(s) (regardless of other flags). It will
        return False if the metric has ANY of the given flags."""
        return cls.flags.missing_flags(flags)
