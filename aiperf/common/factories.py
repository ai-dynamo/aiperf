# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any, Generic

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import (
    CommClientType,
    CommunicationBackend,
    ComposerType,
    CustomDatasetType,
    DataExporterType,
    PostProcessorType,
    ServiceType,
    StreamingPostProcessorType,
    ZMQProxyType,
)
from aiperf.common.enums.endpoints_enums import EndpointType
from aiperf.common.enums.service_enums import ServiceRunType
from aiperf.common.exceptions import FactoryCreationError, InvalidOperationError
from aiperf.common.types import (
    ClassEnumT,
    ClassProtocolT,
    ServiceProtocolT,
    ServiceTypeT,
)


class AIPerfFactory(Generic[ClassEnumT, ClassProtocolT]):
    """Defines a custom factory for AIPerf components.

    This class is used to create a factory for a given class type and protocol.

    Example:
    ```python
        # Define a new enum for the expected implementation types
        # This is optional, but recommended for type safety.
        class DatasetLoaderType(CaseInsensitiveStrEnum):
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

    _logger: AIPerfLogger
    _registry: dict[ClassEnumT | str, type[ClassProtocolT]]
    _override_priorities: dict[ClassEnumT | str, int]

    def __init_subclass__(cls) -> None:
        cls._registry = {}
        cls._override_priorities = {}
        cls._logger = AIPerfLogger(cls.__name__)
        super().__init_subclass__()

    @classmethod
    def register_all(
        cls, *class_types: ClassEnumT | str, override_priority: int = 0
    ) -> Callable:
        """Register multiple class types mapping to a single corresponding class.
        This is useful if a single class implements multiple types. Currently only supports
        registering as a single override priority for all types."""

        def decorator(class_cls: type[ClassProtocolT]) -> type[ClassProtocolT]:
            for class_type in class_types:
                cls.register(class_type, override_priority)(class_cls)
            return class_cls

        return decorator

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
                cls._logger.warning(
                    f"{class_type!r} class {cls._registry[class_type].__name__} already registered with same or higher priority "
                    f"({existing_priority}). The new registration of class {class_cls.__name__} with priority "
                    f"{override_priority} will be ignored.",
                )
                return class_cls

            if class_type not in cls._registry:
                cls._logger.debug(
                    lambda: f"{class_type!r} class {class_cls.__name__} registered with priority {override_priority}.",
                )
            else:
                cls._logger.warning(
                    f"{class_type!r} class {class_cls.__name__} with priority {override_priority} overrides "
                    f"already registered class {cls._registry[class_type].__name__} with lower priority ({existing_priority}).",
                )
            cls._registry[class_type] = class_cls
            cls._override_priorities[class_type] = override_priority
            return class_cls

        return decorator

    @classmethod
    def create_instance(
        cls,
        class_type: ClassEnumT | str,
        **kwargs: Any,
    ) -> ClassProtocolT:
        """Create a new class instance.

        Args:
            class_type: The type of class to create
            **kwargs: Additional arguments for the class

        Returns:
            The created class instance

        Raises:
            FactoryCreationError: If the class type is not registered or there is an error creating the instance
        """
        if class_type not in cls._registry:
            raise FactoryCreationError(
                f"No implementation registered for {class_type!r} in {cls.__name__}."
            )
        try:
            return cls._registry[class_type](**kwargs)
        except Exception as e:
            raise FactoryCreationError(
                f"Error creating {class_type!r} instance for {cls.__name__}: {e}"
            ) from e

    @classmethod
    def get_class_from_type(cls, class_type: ClassEnumT | str) -> type[ClassProtocolT]:
        """Get the class from a class type.

        Args:
            class_type: The class type to get the class from

        Returns:
            The class for the given class type

        Raises:
            TypeError: If the class type is not registered
        """
        if class_type not in cls._registry:
            raise TypeError(
                f"No class found for {class_type!r}. Please register the class first."
            )
        return cls._registry[class_type]

    @classmethod
    def get_all_classes(cls) -> list[type[ClassProtocolT]]:
        """Get all registered classes.

        Returns:
            A list of all registered class types implementing the expected protocol
        """
        return list(cls._registry.values())

    @classmethod
    def get_all_class_types(cls) -> list[ClassEnumT | str]:
        """Get all registered class types."""
        return list(cls._registry.keys())

    @classmethod
    def get_all_classes_and_types(
        cls,
    ) -> list[tuple[type[ClassProtocolT], ClassEnumT | str]]:
        """Get all registered classes and their corresponding class types."""
        return [(cls, class_type) for class_type, cls in cls._registry.items()]


class CommunicationClientFactory(
    AIPerfFactory[CommClientType, "CommunicationClientProtocol"]
):
    """Factory for registering and creating CommunicationClientProtocol instances based on the specified communication client type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class CommunicationClientProtocolFactory(
    AIPerfFactory[CommClientType, "CommunicationClientProtocol"]
):
    """Factory for registering and creating CommunicationClientProtocol instances based on the specified communication client type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class CommunicationFactory(
    AIPerfFactory[CommunicationBackend, "CommunicationProtocol"]
):
    """Factory for registering and creating CommunicationProtocol instances based on the specified communication backend.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class ComposerFactory(AIPerfFactory[ComposerType, "BaseDatasetComposer"]):
    """Factory for registering and creating BaseDatasetComposer instances based on the specified composer type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class CustomDatasetFactory(
    AIPerfFactory[CustomDatasetType, "CustomDatasetLoaderProtocol"]
):
    """Factory for registering and creating CustomDatasetLoaderProtocol instances based on the specified custom dataset type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class DataExporterFactory(AIPerfFactory[DataExporterType, "DataExporterProtocol"]):
    """Factory for registering and creating DataExporterProtocol instances based on the specified data exporter type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class InferenceClientFactory(AIPerfFactory[EndpointType, "InferenceClientProtocol"]):
    """Factory for registering and creating InferenceClientProtocol instances based on the specified endpoint type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class PostProcessorFactory(AIPerfFactory[PostProcessorType, "PostProcessorProtocol"]):
    """Factory for registering and creating PostProcessorProtocol instances based on the specified post processor type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class RequestConverterFactory(AIPerfFactory[EndpointType, "RequestConverterProtocol"]):
    """Factory for registering and creating RequestConverterProtocol instances based on the specified request payload type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class ResponseExtractorFactory(
    AIPerfFactory[EndpointType, "ResponseExtractorProtocol"]
):
    """Factory for registering and creating ResponseExtractorProtocol instances based on the specified response extractor type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class ServiceFactory(AIPerfFactory[ServiceType, "ServiceProtocol"]):
    """Factory for registering and creating BaseService instances based on the specified service type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    @classmethod
    def register_all(
        cls, *class_types: ServiceTypeT, override_priority: int = 0
    ) -> Callable[..., Any]:
        raise InvalidOperationError(
            "ServiceFactory.register_all is not supported. A single service can only be registered with a single type."
        )

    @classmethod
    def register(
        cls, class_type: ServiceTypeT, override_priority: int = 0
    ) -> Callable[..., Any]:
        # Override the register method to set the service_type on the class
        original_decorator = super().register(class_type, override_priority)

        def decorator(class_cls: type[ServiceProtocolT]) -> type[ServiceProtocolT]:
            class_cls.service_type = class_type
            original_decorator(class_cls)
            return class_cls

        return decorator


class ServiceManagerFactory(AIPerfFactory[ServiceRunType, "ServiceManagerProtocol"]):
    """Factory for registering and creating ServiceManagerProtocol instances based on the specified service run type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class StreamingPostProcessorFactory(
    AIPerfFactory[StreamingPostProcessorType, "BaseStreamingPostProcessor"]
):
    """Factory for registering and creating BaseStreamingPostProcessor instances based on the specified streaming post processor type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class ZMQProxyFactory(AIPerfFactory[ZMQProxyType, "BaseZMQProxy"]):
    """Factory for registering and creating BaseZMQProxy instances based on the specified ZMQ proxy type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """
