#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import importlib
import os
from abc import ABCMeta, abstractmethod

from aiperf.common.exceptions import MetricTypeError
from aiperf.services.records_manager.records import Record


class MetricInterface(ABCMeta):
    metric_interfaces: dict[str, "MetricInterface"] = {}

    def __new__(cls, name, base, namespace):
        """
        This function is called upon declaration of any classes of type
        MetricInferface. It registers the class in the metric_interface
        dictionary using the `tag` attribute of the class as the key.
        The `tag` attribute should be a string that uniquely identifies the
        metric type. If the `tag` is not a string, it will not be registered.

        """

        metric_interface = super().__new__(cls, name, base, namespace)

        if isinstance(metric_interface.tag, str):
            cls.metric_interfaces[metric_interface.tag] = metric_interface
        return metric_interface

    @classmethod
    def get_all(
        cls,
    ) -> dict[str, "MetricInterface"]:
        """
        Returns the dictionary of all registered metric interfaces.
        """
        type_module_directory = os.path.join(
            globals()["__spec__"].origin.rsplit("/", 1)[0], "types"
        )
        for filename in os.listdir(type_module_directory):
            if filename != "__init__.py" and filename.endswith(".py"):
                try:
                    importlib.import_module(
                        f"aiperf.services.records_manager.metrics.types.{filename[:-3]}"
                    )
                except AttributeError as err:
                    raise MetricTypeError("Error retrieving all metric types") from err

        return cls.metric_interfaces


class Metric(metaclass=MetricInterface):
    @property
    @abstractmethod
    def tag(self) -> str:
        """
        Returns
        -------
        str
            the name tag of the record type.
        """

    @abstractmethod
    def add_record(self, record: Record) -> None:
        """
        Adds a new record and calculates the new metric value.
        """

    @abstractmethod
    def get_metrics(self) -> list[int]:
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
