#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import importlib
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from aiperf.common.enums import MetricTimeType
from aiperf.common.exceptions import MetricTypeError
from aiperf.services.records_manager.records import Record


class BaseMetric(ABC):
    "Base class for all metricss with automatic subclass registration."

    # Class attributes that subclasses must override
    tag: ClassVar[str] = ""
    unit: ClassVar[MetricTimeType] = MetricTimeType.NANOSECONDS
    larger_is_better: ClassVar[bool] = True
    header: ClassVar[str] = ""

    metric_interfaces: dict[str, type["BaseMetric"]] = {}

    def __init_subclass__(cls, **kwargs):
        """
        This method is called when a class is subclassed from Metric.
        It automatically registers the subclass in the metric_interfaces
        dictionary using the `tag` class attribute.
        The `tag` attribute must be a non-empty string that uniquely identifies the
        metric type. Only concrete (non-abstract) classes will be registered.
        """

        super().__init_subclass__(**kwargs)

        # Only register concrete classes (not abstract ones)
        if inspect.isabstract(cls):
            return

        # Enforce that subclasses define a non-empty tag
        if not cls.tag or not isinstance(cls.tag, str):
            raise TypeError(
                f"Concrete metric class {cls.__name__} must define a non-empty 'tag' class attribute"
            )

        # Check for duplicate tags
        if cls.tag in cls.metric_interfaces:
            raise ValueError(
                f"Metric tag '{cls.tag}' is already registered by {cls.metric_interfaces[cls.tag].__name__}"
            )

        cls.metric_interfaces[cls.tag] = cls

    @classmethod
    def get_all(cls) -> dict[str, type["BaseMetric"]]:
        """
        Returns the dictionary of all registered metric interfaces.

        This method dynamically imports all metric type modules from the 'types'
        directory to ensure all metric classes are registered via __init_subclass__.

        Returns:
            dict[str, type[Metric]]: Mapping of metric tags to their corresponding classes

        Raises:
            MetricTypeError: If there's an error importing metric type modules
        """
        # Get the types directory path
        types_dir = Path(__file__).parent / "types"

        # Import all metric type modules to trigger registration
        if types_dir.exists():
            for python_file in types_dir.glob("*.py"):
                if python_file.name != "__init__.py":
                    module_name = python_file.stem  # Get filename without extension
                    try:
                        importlib.import_module(
                            f"aiperf.services.records_manager.metrics.types.{module_name}"
                        )
                    except ImportError as err:
                        raise MetricTypeError(
                            f"Error importing metric type module '{module_name}'"
                        ) from err

        return cls.metric_interfaces

    @abstractmethod
    def add_record(self, record: Record) -> None:
        """
        Adds a new record and calculates the new metric value.
        """

    @abstractmethod
    def get_metrics(self) -> list[Any]:
        """
        Returns the list of calculated metrics.
        """

    @staticmethod
    @abstractmethod
    def _check_record(record: Record) -> None:
        """
        Checks if the record is valid for metric calculation.

        Raises:
            ValueError: If the record does not meet the required conditions.
        """

    def get_converted_metrics(self, unit: MetricTimeType) -> list[Any]:
        if not isinstance(unit, MetricTimeType):
            raise MetricTypeError("Invalid metric time type for conversion.")

        scale_factor = self.unit.value - unit.value

        return [metric / 10**scale_factor for metric in self.get_metrics()]
