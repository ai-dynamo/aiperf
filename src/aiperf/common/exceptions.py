# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.common.types import ServiceTypeT


class AIPerfError(Exception):
    """Base class for all exceptions raised by AIPerf."""

    def raw_str(self) -> str:
        """Return the raw string representation of the exception."""
        return super().__str__()

    def __str__(self) -> str:
        """Return the string representation of the exception with the class name."""
        return super().__str__()


class AIPerfMultiError(AIPerfError):
    """Exception raised when running multiple tasks and one or more fail."""

    def __init__(self, message: str | None, exceptions: list[Exception]) -> None:
        self.exceptions = exceptions

        err_strings = [
            e.raw_str() if isinstance(e, AIPerfError) else str(e) for e in exceptions
        ]
        if message:
            super().__init__(f"{message}: {','.join(err_strings)}")
        else:
            super().__init__(",".join(err_strings))


class HookError(AIPerfError):
    """Exception raised when a hook encounters an error."""

    def __init__(self, hook_class_name: str, hook_func_name: str, e: Exception) -> None:
        self.hook_class_name = hook_class_name
        self.hook_func_name = hook_func_name
        self.exception = e
        super().__init__(f"{hook_class_name}.{hook_func_name}: {e}")


class ServiceError(AIPerfError):
    """Generic service error."""

    def __init__(
        self,
        message: str,
        service_type: "ServiceTypeT",
        service_id: str,
    ) -> None:
        super().__init__(
            f"{message} for service of type {service_type} with id {service_id}"
        )
        self.service_type = service_type
        self.service_id = service_id


class LifecycleOperationError(AIPerfError):
    """Exception raised when a lifecycle operation fails and the lifecycle should stop gracefully."""

    def __init__(
        self,
        operation: str,
        original_exception: Exception | None,
        lifecycle_id: str,
    ) -> None:
        self.operation = operation
        self.original_exception = original_exception
        self.lifecycle_id = lifecycle_id
        super().__init__(
            str(original_exception)
            if original_exception
            else f"Failed to perform operation '{operation}'"
        )


class CommunicationError(AIPerfError):
    """Generic communication error."""


class ConfigurationError(AIPerfError):
    """Exception raised when something fails to configure, or there is a configuration error."""


class ConsoleExporterDisabled(AIPerfError):
    """Raised when initializing a console exporter to indicate to the caller that it is disabled and should not be used."""


class DataExporterDisabled(AIPerfError):
    """Raised when initializing a data exporter to indicate to the caller that it is disabled and should not be used."""


class DatasetError(AIPerfError):
    """Generic dataset error."""


class DatasetLoaderError(AIPerfError):
    """Generic dataset loader error."""


class DatasetGeneratorError(AIPerfError):
    """Generic dataset generator error."""


class InitializationError(AIPerfError):
    """Exception raised when something fails to initialize."""


class InferenceClientError(AIPerfError):
    """Exception raised when a inference client encounters an error."""


class InvalidInferenceResultError(AIPerfError):
    """Exception raised when an inference result is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.notes = []

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    def __str__(self) -> str:
        return f"{self.message}: {', '.join(self.notes)}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.__str__()}')"


class InvalidOperationError(AIPerfError):
    """Exception raised when an operation is invalid."""


class InvalidPayloadError(InferenceClientError):
    """Exception raised when a inference client receives an invalid payload."""


class InvalidStateError(AIPerfError):
    """Exception raised when something is in an invalid state."""


class MemoryMapDatasetError(AIPerfError):
    """Base exception for memory-mapped dataset errors."""


class MemoryMapSerializationError(MemoryMapDatasetError):
    """Exception raised when serialization/deserialization of mmap data fails."""


class MemoryMapFileOperationError(MemoryMapDatasetError):
    """Exception raised when file operations on mmap files fail."""


class MetricTypeError(AIPerfError):
    """Exception raised when a metric type encounters an error while creating a class."""


class MetricUnitError(AIPerfError):
    """Exception raised when trying to convert a metric to or from a unit that is does not support it."""


class NotFoundError(AIPerfError):
    """Exception raised when something is not found or not available."""


class NotInitializedError(AIPerfError):
    """Exception raised when something that should be initialized is not."""


class NoMetricValue(AIPerfError):
    """Raised when a metric value is not available."""


class PluginNotFoundError(AIPerfError):
    """Exception raised when a plugin is not found. This is used to indicate that a plugin is not found when trying to get a plugin class or metadata."""


class PostProcessorDisabled(AIPerfError):
    """Raised when initializing a post processor to indicate to the caller that it is disabled and should not be used."""


class ProxyError(AIPerfError):
    """Exception raised when a proxy encounters an error."""


class ShutdownError(AIPerfError):
    """Exception raised when a service encounters an error while shutting down."""


class SSEResponseError(AIPerfError):
    """Exception raised when a SSE response contains an error."""

    def __init__(self, message: str, error_code: int = 500) -> None:
        self.error_code = error_code
        super().__init__(message)


class TokenizerError(AIPerfError):
    """Exception raised when a tokenizer fails to load or encounters an error."""

    def __init__(self, message: str, tokenizer_name: str | None = None) -> None:
        super().__init__(message)
        self.tokenizer_name = tokenizer_name


class UnsupportedHookError(AIPerfError):
    """Exception raised when a hook is defined on a class that does not have any base classes that provide that hook type."""


class ValidationError(AIPerfError):
    """Exception raised when something fails validation."""
