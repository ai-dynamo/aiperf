# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.types import ServiceTypeT


class AIPerfError(Exception):
    """Base class for all exceptions raised by AIPerf."""

    def raw_str(self) -> str:
        """Return the raw string representation of the exception."""
        return super().__str__()

    def __str__(self) -> str:
        """Return the string representation of the exception with the class name."""
        return f"{self.__class__.__name__}: {super().__str__()}"


class AIPerfMultiError(AIPerfError):
    """Exception raised when running multiple tasks and one or more fail."""

    def __init__(self, message: str, exceptions: list[Exception]) -> None:
        err_strings = [
            e.raw_str() if isinstance(e, AIPerfError) else str(e) for e in exceptions
        ]
        super().__init__(f"{message}: {','.join(err_strings)}")
        self.exceptions = exceptions


class ServiceError(AIPerfError):
    """Generic service error."""

    def __init__(
        self,
        message: str,
        service_type: ServiceTypeT,
        service_id: str,
    ) -> None:
        super().__init__(
            f"{message} for service of type {service_type} with id {service_id}"
        )
        self.service_type = service_type
        self.service_id = service_id


class CommunicationError(AIPerfError):
    """Generic communication error."""


class ConfigurationError(AIPerfError):
    """Exception raised when something fails to configure, or there is a configuration error."""


class DatasetError(AIPerfError):
    """Generic dataset error."""


class DatasetGeneratorError(AIPerfError):
    """Generic dataset generator error."""


class FactoryCreationError(AIPerfError):
    """Exception raised when a factory encounters an error while creating a class."""


class InitializationError(AIPerfError):
    """Exception raised when something fails to initialize."""


class InferenceClientError(AIPerfError):
    """Exception raised when a inference client encounters an error."""


class InvalidOperationError(AIPerfError):
    """Exception raised when an operation is invalid."""


class InvalidPayloadError(InferenceClientError):
    """Exception raised when a inference client receives an invalid payload."""


class InvalidStateError(AIPerfError):
    """Exception raised when something is in an invalid state."""


class MetricTypeError(AIPerfError):
    """Exception raised when a metric type encounters an error while creating a class."""


class NotFoundError(AIPerfError):
    """Exception raised when something is not found or not available."""


class NotInitializedError(AIPerfError):
    """Exception raised when something that should be initialized is not."""


class ProxyError(AIPerfError):
    """Exception raised when a proxy encounters an error."""


class ShutdownError(AIPerfError):
    """Exception raised when a service encounters an error while shutting down."""


class UnsupportedHookError(AIPerfError):
    """Exception raised when a hook is defined on a class that does not have any base classes that provide that hook type."""


class ValidationError(AIPerfError):
    """Exception raised when something fails validation."""
