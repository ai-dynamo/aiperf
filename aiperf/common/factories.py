# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING, Any, Generic

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import (
    AIPerfUIType,
    CommClientType,
    CommunicationBackend,
    ComposerType,
    ConsoleExporterType,
    CustomDatasetType,
    DataExporterType,
    EndpointType,
    OpenAIObjectType,
    RecordProcessorType,
    RequestRateMode,
    ResultsProcessorType,
    ServiceRunType,
    ServiceType,
    ZMQProxyType,
)
from aiperf.common.exceptions import (
    FactoryCreationError,
    InvalidOperationError,
    InvalidStateError,
)
from aiperf.common.types import (
    ClassEnumT,
    ClassProtocolT,
    ServiceProtocolT,
    ServiceTypeT,
)

if TYPE_CHECKING:
    # NOTE: These imports are for the factory class type hints.
    #       We do not want to import these classes directly.

    from aiperf.clients.model_endpoint_info import ModelEndpointInfo
    from aiperf.common.config import (
        BaseZMQCommunicationConfig,
        BaseZMQProxyConfig,
        ServiceConfig,
        UserConfig,
    )
    from aiperf.common.protocols import (
        CommunicationClientProtocol,
        CommunicationProtocol,
        ConsoleExporterProtocol,
        DataExporterProtocol,
        InferenceClientProtocol,
        RecordProcessorProtocol,
        RequestRateGeneratorProtocol,
        ResponseExtractorProtocol,
        ResultsProcessorProtocol,
        ServiceManagerProtocol,
    )
    from aiperf.dataset import (
        CustomDatasetLoaderProtocol,
    )
    from aiperf.dataset.composer.base import BaseDatasetComposer
    from aiperf.exporters.exporter_config import ExporterConfig
    from aiperf.timing.config import TimingManagerConfig
    from aiperf.zmq.zmq_proxy_base import BaseZMQProxy


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
            # First try to load any lazy registration
            if cls._load_lazy_registration(class_type):
                # Successfully loaded from lazy registry
                pass
            else:
                # Try to discover and load the plugin for this class type
                cls._discover_plugin(class_type)

            # Check again after discovery attempt
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
            # First try to load any lazy registration
            if cls._load_lazy_registration(class_type):
                # Successfully loaded from lazy registry
                pass
            else:
                # Try to discover and load the plugin for this class type
                cls._discover_plugin(class_type)

            # Check again after discovery attempt
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

    @classmethod
    def _discover_plugin(cls, class_type: ClassEnumT | str) -> None:
        """Attempt to discover and load a plugin for the given class type.

        Args:
            class_type: The class type to discover a plugin for
        """
        from aiperf.module_loader import discover_and_load_plugin

        cls._logger.debug(
            lambda: f"Attempting to discover plugin for {class_type!r} in {cls.__name__}"
        )

        try:
            discover_and_load_plugin(cls.__name__, class_type)
        except Exception as e:
            cls._logger.debug(
                lambda e=e: f"Failed to discover plugin for {class_type!r} in {cls.__name__}: {e}"
            )

    @classmethod
    def register_lazy(
        cls,
        class_type: ClassEnumT | str,
        module_path: str,
        class_name: str,
        override_priority: int = 0,
    ) -> None:
        """Register a class for lazy loading.

        This method allows registering a class without importing it immediately.
        The class will be imported and registered when first requested.

        Args:
            class_type: The type of class to register
            module_path: The module path where the class is defined
            class_name: The name of the class to import
            override_priority: The priority of the override
        """
        # Store lazy registration info in a special registry
        if not hasattr(cls, "_lazy_registry"):
            cls._lazy_registry = {}
        if not hasattr(cls, "_lazy_priorities"):
            cls._lazy_priorities = {}

        existing_priority = cls._lazy_priorities.get(class_type, -1)
        if class_type in cls._lazy_registry and existing_priority >= override_priority:
            cls._logger.warning(
                f"Lazy registration for {class_type!r} already exists with same or higher priority "
                f"({existing_priority}). The new registration with priority {override_priority} will be ignored."
            )
            return

        cls._lazy_registry[class_type] = (module_path, class_name)
        cls._lazy_priorities[class_type] = override_priority

        cls._logger.debug(
            lambda: f"Lazy registration for {class_type!r} -> {module_path}.{class_name} with priority {override_priority}"
        )

    @classmethod
    def _load_lazy_registration(cls, class_type: ClassEnumT | str) -> bool:
        """Load a lazy registration if it exists.

        Args:
            class_type: The class type to load

        Returns:
            True if a lazy registration was found and loaded, False otherwise
        """
        if not hasattr(cls, "_lazy_registry"):
            return False

        if class_type not in cls._lazy_registry:
            return False

        module_path, class_name = cls._lazy_registry[class_type]

        try:
            cls._logger.debug(
                f"Loading lazy registration: {module_path}.{class_name} for {class_type!r}"
            )

            # Import the module
            import importlib

            module = importlib.import_module(module_path)

            # Get the class from the module
            class_cls = getattr(module, class_name)

            # Register it normally
            priority = cls._lazy_priorities.get(class_type, 0)
            cls._registry[class_type] = class_cls
            cls._override_priorities[class_type] = priority

            # Remove from lazy registry since it's now loaded
            del cls._lazy_registry[class_type]
            del cls._lazy_priorities[class_type]

            cls._logger.debug(
                f"Successfully loaded lazy registration for {class_type!r}"
            )
            return True

        except Exception as e:
            cls._logger.warning(
                f"Failed to load lazy registration for {class_type!r}: {e}"
            )
            return False


class AIPerfSingletonFactory(AIPerfFactory[ClassEnumT, ClassProtocolT]):
    """Factory for registering and creating singleton instances of a given class type and protocol.
    This factory is useful for creating instances that are shared across the application, such as communication clients.
    Calling create_instance will create a new instance if it doesn't exist, otherwise it will return the existing instance.
    Calling get_instance will return the existing instance if it exists, otherwise it will raise an error.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    _instances: dict[ClassEnumT | str, ClassProtocolT]
    _instances_lock: Lock
    _instances_pid: dict[ClassEnumT | str, int]

    def __init_subclass__(cls) -> None:
        cls._instances = {}
        cls._instances_lock = Lock()
        cls._instances_pid = {}
        super().__init_subclass__()

    @classmethod
    def set_instance(
        cls, class_type: ClassEnumT | str, instance: ClassProtocolT
    ) -> None:
        cls._instances[class_type] = instance

    @classmethod
    def get_or_create_instance(
        cls, class_type: ClassEnumT | str, **kwargs: Any
    ) -> ClassProtocolT:
        """Syntactic sugar for create_instance, but with a more descriptive name for singleton factories."""
        return cls.create_instance(class_type, **kwargs)

    @classmethod
    def create_instance(
        cls, class_type: ClassEnumT | str, **kwargs: Any
    ) -> ClassProtocolT:
        """Create a new instance of the given class type.
        If the instance does not exist, or the process ID has changed, a new instance will be created.
        """
        # TODO: Technically, this this should handle the case where kwargs are different,
        #       but that would require a more complex implementation.
        if (
            class_type not in cls._instances
            or os.getpid() != cls._instances_pid[class_type]
        ):
            cls._logger.debug(
                lambda: f"Creating new instance for {class_type!r} in {cls.__name__}."
            )
            with cls._instances_lock:
                if (
                    class_type not in cls._instances
                    or os.getpid() != cls._instances_pid[class_type]
                ):
                    # Use the parent create_instance which now includes plugin discovery
                    cls._instances[class_type] = super().create_instance(
                        class_type, **kwargs
                    )
                    cls._instances_pid[class_type] = os.getpid()
                    cls._logger.debug(
                        lambda: f"New instance for {class_type!r} in {cls.__name__} created."
                    )
        else:
            cls._logger.debug(
                lambda: f"Instance for {class_type!r} in {cls.__name__} already exists. Returning existing instance."
            )
        return cls._instances[class_type]

    @classmethod
    def get_instance(cls, class_type: ClassEnumT | str) -> ClassProtocolT:
        if class_type not in cls._instances:
            raise InvalidStateError(
                f"No instance found for {class_type!r} in {cls.__name__}. "
                f"Ensure you call AIPerfSingletonFactory.create_instance({class_type!r}) first."
            )
        if os.getpid() != cls._instances_pid[class_type]:
            raise InvalidStateError(
                f"Instance for {class_type!r} in {cls.__name__} is not valid for the current process. "
                f"Ensure you call AIPerfSingletonFactory.create_instance({class_type!r}) first after forking."
            )
        return cls._instances[class_type]

    @classmethod
    def get_all_instances(cls) -> dict[ClassEnumT | str, ClassProtocolT]:
        return cls._instances

    @classmethod
    def has_instance(cls, class_type: ClassEnumT | str) -> bool:
        return class_type in cls._instances


class AIPerfUIFactory(AIPerfSingletonFactory[AIPerfUIType, "AIPerfUIProtocol"]):
    """Factory for registering and creating AIPerfUIProtocol instances based on the specified AIPerfUI type.
    see: :class:`aiperf.common.factories.AIPerfSingletonFactory` for more details.
    """


class CommunicationClientFactory(
    AIPerfFactory[CommClientType, "CommunicationClientProtocol"]
):
    """Factory for registering and creating CommunicationClientProtocol instances based on the specified communication client type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    @classmethod
    def create_instance(  # type: ignore[override]
        cls,
        class_type: CommClientType | str,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
        **kwargs,
    ) -> "CommunicationClientProtocol":
        return super().create_instance(
            class_type, address=address, bind=bind, socket_ops=socket_ops, **kwargs
        )


class CommunicationFactory(
    AIPerfSingletonFactory[CommunicationBackend, "CommunicationProtocol"]
):
    """Factory for registering and creating CommunicationProtocol instances based on the specified communication backend.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    @classmethod
    def create_instance(  # type: ignore[override]
        cls,
        class_type: CommunicationBackend | str,
        config: "BaseZMQCommunicationConfig",
        **kwargs,
    ) -> "CommunicationProtocol":
        return super().create_instance(class_type, config=config, **kwargs)


class ComposerFactory(AIPerfFactory[ComposerType, "BaseDatasetComposer"]):
    """Factory for registering and creating BaseDatasetComposer instances based on the specified composer type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    @classmethod
    def create_instance(  # type: ignore[override]
        cls,
        class_type: ComposerType | str,
        **kwargs,
    ) -> "BaseDatasetComposer":
        return super().create_instance(class_type, **kwargs)


class ConsoleExporterFactory(
    AIPerfFactory[ConsoleExporterType, "ConsoleExporterProtocol"]
):
    """Factory for registering and creating ConsoleExporterProtocol instances based on the specified data exporter type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    @classmethod
    def create_instance(  # type: ignore[override]
        cls,
        class_type: ConsoleExporterType | str,
        exporter_config: "ExporterConfig",
        **kwargs,
    ) -> "ConsoleExporterProtocol":
        return super().create_instance(
            class_type, exporter_config=exporter_config, **kwargs
        )


class CustomDatasetFactory(
    AIPerfFactory[CustomDatasetType, "CustomDatasetLoaderProtocol"]
):
    """Factory for registering and creating CustomDatasetLoaderProtocol instances based on the specified custom dataset type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    @classmethod
    def create_instance(  # type: ignore[override]
        cls,
        class_type: CustomDatasetType | str,
        **kwargs,
    ) -> "CustomDatasetLoaderProtocol":
        return super().create_instance(class_type, **kwargs)


class DataExporterFactory(AIPerfFactory[DataExporterType, "DataExporterProtocol"]):
    """Factory for registering and creating DataExporterProtocol instances based on the specified data exporter type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    @classmethod
    def create_instance(  # type: ignore[override]
        cls,
        class_type: DataExporterType | str,
        exporter_config: "ExporterConfig",
        **kwargs,
    ) -> "DataExporterProtocol":
        return super().create_instance(
            class_type, exporter_config=exporter_config, **kwargs
        )


class InferenceClientFactory(AIPerfFactory[EndpointType, "InferenceClientProtocol"]):
    """Factory for registering and creating InferenceClientProtocol instances based on the specified endpoint type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    @classmethod
    def create_instance(  # type: ignore[override]
        cls,
        class_type: EndpointType | str,
        model_endpoint: "ModelEndpointInfo",
        **kwargs,
    ) -> "InferenceClientProtocol":
        return super().create_instance(
            class_type, model_endpoint=model_endpoint, **kwargs
        )


class OpenAIObjectParserFactory(
    AIPerfSingletonFactory[OpenAIObjectType, "OpenAIObjectParserProtocol"]
):
    """Factory for registering and creating OpenAIObjectParserProtocol instances based on the specified object type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class RequestConverterFactory(
    AIPerfSingletonFactory[EndpointType, "RequestConverterProtocol"]
):
    """Factory for registering and creating RequestConverterProtocol instances based on the specified request payload type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """


class ResponseExtractorFactory(
    AIPerfFactory[EndpointType, "ResponseExtractorProtocol"]
):
    """Factory for registering and creating ResponseExtractorProtocol instances based on the specified response extractor type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    @classmethod
    def create_instance(  # type: ignore[override]
        cls,
        class_type: EndpointType | str,
        model_endpoint: "ModelEndpointInfo",
        **kwargs,
    ) -> "ResponseExtractorProtocol":
        return super().create_instance(
            class_type, model_endpoint=model_endpoint, **kwargs
        )


class ServiceFactory(AIPerfFactory[ServiceType, "ServiceProtocol"]):
    """Factory for registering and creating ServiceProtocol instances based on the specified service type.
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

    @classmethod
    def create_instance(  # type: ignore[override]
        cls,
        class_type: ServiceRunType | str,
        required_services: dict[ServiceTypeT, int],
        service_config: "ServiceConfig",
        user_config: "UserConfig",
        **kwargs,
    ) -> "ServiceManagerProtocol":
        return super().create_instance(
            class_type,
            required_services=required_services,
            service_config=service_config,
            user_config=user_config,
            **kwargs,
        )


class RecordProcessorFactory(
    AIPerfFactory[RecordProcessorType, "RecordProcessorProtocol"]
):
    """Factory for registering and creating RecordProcessorProtocol instances based on the specified record processor type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    @classmethod
    def create_instance(  # type: ignore[override]
        cls,
        class_type: RecordProcessorType | str,
        service_config: "ServiceConfig",
        user_config: "UserConfig",
        **kwargs,
    ) -> "RecordProcessorProtocol":
        return super().create_instance(
            class_type,
            service_config=service_config,
            user_config=user_config,
            **kwargs,
        )


class ResultsProcessorFactory(
    AIPerfFactory[ResultsProcessorType, "ResultsProcessorProtocol"]
):
    """Factory for registering and creating ResultsProcessorProtocol instances based on the specified results processor type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    @classmethod
    def create_instance(  # type: ignore[override]
        cls,
        class_type: ResultsProcessorType | str,
        service_config: "ServiceConfig",
        user_config: "UserConfig",
        **kwargs,
    ) -> "ResultsProcessorProtocol":
        return super().create_instance(
            class_type,
            service_config=service_config,
            user_config=user_config,
            **kwargs,
        )


class RequestRateGeneratorFactory(
    AIPerfFactory[RequestRateMode, "RequestRateGeneratorProtocol"]
):
    """Factory for registering and creating RequestRateGeneratorProtocol instances based on the specified RequestRateMode.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    @classmethod
    def create_instance(  # type: ignore[override]
        cls,
        config: "TimingManagerConfig",
    ) -> "RequestRateGeneratorProtocol":
        return super().create_instance(config.request_rate_mode, config=config)


class ZMQProxyFactory(AIPerfFactory[ZMQProxyType, "BaseZMQProxy"]):
    """Factory for registering and creating BaseZMQProxy instances based on the specified ZMQ proxy type.
    see: :class:`aiperf.common.factories.AIPerfFactory` for more details.
    """

    @classmethod
    def create_instance(  # type: ignore[override]
        cls,
        class_type: ZMQProxyType | str,
        zmq_proxy_config: "BaseZMQProxyConfig",
        **kwargs,
    ) -> "BaseZMQProxy":
        return super().create_instance(
            class_type, zmq_proxy_config=zmq_proxy_config, **kwargs
        )
