# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from aiperf.common.enums import StrEnum
from aiperf.common.exceptions import FactoryCreationError

if TYPE_CHECKING:
    from aiperf.common.comms.base import (
        BaseCommunication,  # noqa: F401 - for type checking
    )
    from aiperf.common.enums import (
        CommunicationBackend,  # noqa: F401 - for type checking
    )

ClassEnumT = TypeVar("ClassEnumT", bound=StrEnum, infer_variance=True)
ClassProtocolT = TypeVar("ClassProtocolT", bound=Any, infer_variance=True)

__all__ = [
    "FactoryMixin",
    "CommunicationFactory",
]

################################################################################
# Generic Base Factory Mixin
################################################################################


class FactoryMixin(Generic[ClassEnumT, ClassProtocolT]):
    """Defines a mixin for all factories, which supports registering and creating instances of classes.

    This mixin is used to create a factory for a given class type and protocol.

    Example:
    ```python
        # Define a new enum for the expected implementation types
        # This is optional, but recommended for type safety.
        class DatasetLoaderType(StrEnum):
            FILE = "file"
            S3 = "s3"

        # Define a new class protocol.
        class DatasetLoaderProtocol(Protocol):
            def load(self) -> Dataset:
                pass

        # Create a new factory for a given class type and protocol.
        class DatasetFactory(FactoryMixin[DatasetLoaderType, DatasetLoaderProtocol]):
            pass

        # Register a new class type mapping to its corresponding class. It should implement the class protocol.
        @DatasetFactory.register(DatasetLoaderType.FILE)
        class FileDatasetLoader:
            def __init__(self, filename: str):
                self.filename = filename

            def load(self) -> Dataset:
                return Dataset.from_file(self.filename)

        DatasetConfig = {
            "type": DatasetLoaderType.FILE,
            "filename": "data.csv"
        }

        # Create a new instance of the class.
        if DatasetConfig["type"] == DatasetLoaderType.FILE:
            dataset_instance = DatasetFactory.create_instance(DatasetLoaderType.FILE, filename=DatasetConfig["filename"])
        else:
            raise ValueError(f"Unsupported dataset loader type: {DatasetConfig['type']}")

        dataset_instance.load()
    ```
    """

    logger = logging.getLogger(__name__)

    _registry: dict[ClassEnumT | str, type[ClassProtocolT]] = {}
    _override_priorities: dict[ClassEnumT | str, int] = {}

    @classmethod
    def register(
        cls, class_type: ClassEnumT | str, override_priority: int = 0
    ) -> Callable:
        """Register a new class type mapping to its corresponding class.

        Args:
            class_type: The type of class to register
            override_priority: The priority of the override. The higher the priority,
                the more precedence the override has when multiple classes are registered
                for the same class type. Built-in classes have a priority of 0.

        Returns:
            Decorator for the class that implements the class protocol
        """

        def decorator(class_cls: type[ClassProtocolT]) -> type[ClassProtocolT]:
            existing_priority = cls._override_priorities.get(class_type, -1)
            if class_type in cls._registry and existing_priority >= override_priority:
                # TODO: Will logging be initialized before this method is called?
                cls.logger.warning(
                    f"{class_type!r} class {cls._registry[class_type].__name__} already registered with same or higher priority "
                    f"({existing_priority}). The new registration of class {class_cls.__name__} with priority "
                    f"{override_priority} will be ignored."
                )
                return class_cls

            if class_type not in cls._registry:
                cls.logger.debug(
                    f"{class_type!r} class {class_cls.__name__} registered with priority {override_priority}."
                )
            else:
                cls.logger.debug(
                    f"{class_type!r} class {class_cls.__name__} with priority {override_priority} overrides "
                    f"already registered class {cls._registry[class_type].__name__} with lower priority ({existing_priority})."
                )
            cls._registry[class_type] = class_cls
            cls._override_priorities[class_type] = override_priority
            return class_cls

        return decorator

    @classmethod
    def create_instance(
        cls,
        class_type: ClassEnumT | str,
        config: Any | None = None,
        **kwargs: Any,
    ) -> ClassProtocolT:
        """Create a new class instance.

        Args:
            class_type: The type of class to create
            config: The configuration for the class
            **kwargs: Additional arguments for the class

        Returns:
            The created class instance

        Raises:
            FactoryCreationError: If the class type is not registered or there is an error creating the instance
        """
        if class_type not in cls._registry:
            raise FactoryCreationError(f"No implementation found for {class_type!r}.")
        try:
            return cls._registry[class_type](config, **kwargs)
        except Exception as e:
            raise FactoryCreationError(
                f"Error creating {class_type!r} instance: {e}"
            ) from e


################################################################################
# Built-in Factories
################################################################################


class CommunicationFactory(FactoryMixin["CommunicationBackend", "BaseCommunication"]):
    """Factory for registering and creating BaseCommunication instances based on the specified communication backend.

    Example:
    ```python
        # Register a new communication backend
        @CommunicationFactory.register(CommunicationBackend.ZMQ_TCP)
        class ZMQCommunication(BaseCommunication):
            pass

        # Create a new communication instance
        communication = CommunicationFactory.create_instance(
            CommunicationBackend.ZMQ_TCP,
            config=ZMQTCPCommunicationConfig(
                host="localhost", port=5555, timeout=10.0),
        )
    """
