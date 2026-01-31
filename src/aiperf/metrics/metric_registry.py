# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import graphlib
import importlib
from collections.abc import Iterable
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from aiperf.cli_utils import exit_on_error
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import (
    MetricFlags,
    MetricType,
)
from aiperf.common.exceptions import MetricTypeError
from aiperf.common.types import MetricTagT

if TYPE_CHECKING:
    # This is used to avoid circular imports
    from aiperf.metrics.base_metric import BaseMetric


_logger = AIPerfLogger(__name__)


class MetricRegistry:
    """
    A registry for metrics.

    This is used to store all the metrics that are available to the system.
    It is used to lookup metrics by their tag, and to get all the metrics that are available.
    It also provides methods to get metrics by their type, flag, and to create a dependency order for the metrics.
    It is also used to create instances of metrics.

    This class is not meant to be instantiated directly. It is meant to be used as a singleton via classmethods.
    """

    # Map of metric tags to their classes
    _metrics_map: dict[MetricTagT, type["BaseMetric"]] = {}

    # Map of metric tags to their instances
    _instances_map: dict[MetricTagT, "BaseMetric"] = {}
    _instance_lock = Lock()

    def __init__(self) -> None:
        raise TypeError(
            "MetricRegistry is a singleton and cannot be instantiated directly"
        )

    @classmethod
    def _discover_metrics(cls) -> None:
        """
        This method dynamically imports all metric type modules from the 'types' directory to ensure
        all metric classes are registered via __init_subclass__. This will be called once when the
        module is imported.
        """
        # Get the types directory path
        types_dir = Path(__file__).parent / "types"

        # Ensure that the types directory exists
        if not types_dir.exists() or not types_dir.is_dir():
            raise MetricTypeError(
                f"Types directory '{types_dir.resolve()}' does not exist or is not a directory"
            )

        # Get the module prefix for the types directory, which is the parent of
        # this module, plus the types directory name.
        # For example, `aiperf.metrics.metric_registry` will become `aiperf.metrics.types`
        module_prefix = ".".join([*cls.__module__.split(".")[:-1], "types"])
        _logger.debug(
            f"Importing metric type modules from '{types_dir.resolve()}' with module prefix '{module_prefix}'"
        )
        # Import all metric type modules to trigger registration
        cls._import_metric_type_modules(types_dir, module_prefix)

    @classmethod
    def _import_metric_type_modules(cls, types_dir: Path, module_prefix: str) -> None:
        """Import all metric type modules from the given directory. This will raise an error if the module cannot be imported."""
        for python_file in types_dir.glob("*.py"):
            if python_file.name != "__init__.py":
                module_name = python_file.stem  # Get filename without extension
                module_path = f"{module_prefix}.{module_name}"
                try:
                    _logger.debug(
                        f"Importing metric type module: '{module_path}' from '{python_file.resolve()}'"
                    )
                    importlib.import_module(module_path)
                except ImportError as err:
                    raise MetricTypeError(
                        f"Error importing metric type module '{module_path}' from '{python_file.resolve()}'"
                    ) from err

    @classmethod
    def register_metric(cls, metric: type["BaseMetric"]):
        """Register a metric class with the registry. This will raise a MetricTypeError if the class is already registered.

        This method is called automatically via the __init_subclass__ method of the BaseMetric class, so there is no need
        to call it manually.
        """
        if metric.tag in cls._metrics_map:
            # TODO: Should we consider adding an override_priority parameter to the metric class similar to the plugin system?
            #       This would allow the user to override built-in metrics with custom implementations, without requiring
            #       them to modify the built-in metric classes.
            raise MetricTypeError(
                f"Metric class with tag {metric.tag} already registered by {cls._metrics_map[metric.tag].__name__}"
            )

        cls._metrics_map[metric.tag] = metric

    @classmethod
    def get_class(cls, tag: MetricTagT) -> type["BaseMetric"]:
        """Get a metric class by its tag.

        Raises:
            MetricTypeError: If the metric class is not found.
        """
        try:
            return cls._metrics_map[tag]
        except KeyError as e:
            raise MetricTypeError(f"Metric class with tag '{tag}' not found") from e

    @classmethod
    def get_instance(cls, tag: MetricTagT) -> "BaseMetric":
        """Get an instance of a metric class by its tag. This will create a new instance if it does not exist.

        Raises:
            MetricTypeError: If the metric class is not found.
        """
        # Check first without acquiring the lock for performance reasons. Since this is a hot path, we want to avoid
        # acquiring the lock if we can. We can do this because we have added a secondary check after acquiring the lock.
        if tag not in cls._instances_map:
            with cls._instance_lock:
                # Check again after acquiring the lock
                if tag not in cls._instances_map:
                    metric_class = cls.get_class(tag)
                    cls._instances_map[tag] = metric_class()
        return cls._instances_map[tag]

    @classmethod
    def tags_applicable_to(
        cls,
        required_flags: MetricFlags,
        disallowed_flags: MetricFlags,
        *types: MetricType,
    ) -> list[MetricTagT]:
        """Get metrics tags that are applicable to the given arguments.

        This method is used to filter the metrics that are applicable to a given set of flags and types.
        For instance, this can be used to only get all DERIVED metrics, or only get metrics that are
        applicable to non-streaming endpoints, etc.

        Arguments:
            required_flags: The flags that the metric must have ALL of. If MetricFlags.NONE, no flags are required.
            disallowed_flags: The flags that the metric must not have ANY of.
            types: The types of metrics to include. If not provided, all types will be included.

        Returns:
            A list of metric tags that are applicable to the given arguments.
        """
        return [
            tag
            for tag, metric_class in cls._metrics_map.items()
            if (
                required_flags == MetricFlags.NONE
                or metric_class.has_flags(required_flags)
            )
            and metric_class.missing_flags(disallowed_flags)
            and (not types or metric_class.type in types)
        ]

    @classmethod
    def all_tags(cls) -> list[MetricTagT]:
        """Get all of the tags of the defined metric classes."""
        return list(cls._metrics_map.keys())

    @classmethod
    def all_classes(cls) -> list[type["BaseMetric"]]:
        """Get all of the classes of the defined metric classes."""
        return list(cls._metrics_map.values())

    @classmethod
    def classes_for(cls, tags: Iterable[MetricTagT]) -> list[type["BaseMetric"]]:
        """Get the classes for the given tags.

        Raises:
            MetricTypeError: If a tag is not found.
        """
        return [cls.get_class(tag) for tag in tags]

    @classmethod
    def _validate_dependencies(cls) -> None:
        """Validate that all dependencies are registered.

        Raises:
            MetricTypeError: If a dependency is not registered.
        """
        all_tags = cls._metrics_map.keys()
        all_classes = cls._metrics_map.values()

        # Map of metric types to the types of metrics they can have dependencies on
        _allowed_dependencies_by_type = {
            # Record metrics can only depend on other record metrics
            MetricType.RECORD: {MetricType.RECORD},
            # Aggregate metrics can depend on other record or aggregate metrics
            MetricType.AGGREGATE: {MetricType.RECORD, MetricType.AGGREGATE},
            # Derived metrics can depend on any other metric type
            MetricType.DERIVED: {
                MetricType.RECORD,
                MetricType.AGGREGATE,
                MetricType.DERIVED,
            },
        }

        # Validate that all required metrics are registered, and that the dependencies are allowed
        for metric in all_classes:
            for required_tag in metric.required_metrics or set():
                # Validate that the dependency is registered
                if required_tag not in all_tags:
                    raise MetricTypeError(
                        f"Metric '{metric.tag}' depends on '{required_tag}', which is not registered"
                    )

                # Validate that the dependency is allowed
                required_metric_type = cls._metrics_map[required_tag].type
                if (
                    required_metric_type
                    not in _allowed_dependencies_by_type[metric.type]
                ):
                    raise MetricTypeError(
                        f"Metric '{metric.tag}' is a {metric.type} metric, but depends on '{required_tag}', which is a {required_metric_type} metric"
                    )

    @classmethod
    def _get_all_required_tags(cls, tags: Iterable[MetricTagT]) -> set[MetricTagT]:
        """Get all of the required tags, recursively, for a given list of metric tags."""
        required_tags = set(tags)
        for metric_class in cls.classes_for(tags):
            for required_tag in metric_class.required_metrics or set():
                if required_tag not in required_tags:
                    required_tags.add(required_tag)
                    required_tags.update(cls._get_all_required_tags([required_tag]))
        return required_tags

    @classmethod
    def create_dependency_order(cls) -> list[MetricTagT]:
        """
        Create a dependency order for all available metrics using topological sort.

        See :meth:`create_dependency_order_for` for more details.
        """
        return cls.create_dependency_order_for()

    @classmethod
    def create_dependency_order_for(
        cls,
        tags: Iterable[MetricTagT] | None = None,
    ) -> list[MetricTagT]:
        """
        Create a dependency order for the given metrics using topological sort.

        This ensures that all dependencies are computed before their dependents.
        If `tags` is provided, the returned order will include all metrics needed
        to compute the requested tags (including their recursive dependencies).

        Arguments:
            tags: The tags of the metrics to compute the dependency order for. If not provided, all metrics will be included.

        Returns:
            List of metric tags in dependency order (dependencies first).

        Raises:
            MetricTypeError: If there are unregistered dependencies or circular dependencies.
        """
        if tags is None:
            tags = cls._metrics_map.keys()

        # Expand tags to include all recursive dependencies
        all_required_tags = cls._get_all_required_tags(tags)

        # Build the dependency graph
        sorter = graphlib.TopologicalSorter()

        for metric in cls.classes_for(all_required_tags):
            # Add the metric with its required dependencies
            sorter.add(metric.tag, *(metric.required_metrics or set()))

        try:
            # Get the dependency order
            order = list(sorter.static_order())

            # Return only the tags needed (requested + their dependencies)
            return [tag for tag in order if tag in all_required_tags]

        except graphlib.CycleError as e:
            raise MetricTypeError(
                f"Circular dependency detected among metrics: {e}"
            ) from e


# Ensure that the metrics are discovered when the module is imported.
with exit_on_error(
    MetricTypeError,
    title="Error Discovering Metrics",
):
    MetricRegistry._discover_metrics()

# Ensure that the dependencies are validated, and no circular dependencies are detected
with exit_on_error(
    MetricTypeError,
    title="Error Validating Metric Dependencies",
):
    MetricRegistry._validate_dependencies()
    MetricRegistry.create_dependency_order()
